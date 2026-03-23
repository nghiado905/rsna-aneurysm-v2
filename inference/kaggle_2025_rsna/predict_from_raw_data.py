import inspect
import os
from copy import deepcopy
from pathlib import Path
from typing import Union, List
import shutil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from batchgenerators.utilities.file_and_folder_operations import join, save_json
from nnunetv2.utilities.file_path_utilities import get_output_folder
from nnunetv2.utilities.helpers import empty_cache
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from nnunetv2.inference.predict_from_raw_data import (
    nnUNetPredictor,
    convert_predicted_logits_to_segmentation_with_correct_shape,
)


class nnUNetPredictorKaggle2025RSNA(nnUNetPredictor):
    def predict_from_files_sequential(self,
                           list_of_lists_or_source_folder: Union[str, List[List[str]]],
                           output_folder_or_list_of_truncated_output_files: Union[str, None, List[str]],
                           save_probabilities: bool = False,
                           overwrite: bool = True,
                           folder_with_segs_from_prev_stage: str = None,
                           ids_mapping_json: Union[str, None] = None,
                           heatmap_dir: Union[str, None] = None,
                           cases_csv: Union[str, None] = None):
        """
        Just like predict_from_files but doesn't use any multiprocessing. Slow, but sometimes necessary
        """
        if isinstance(output_folder_or_list_of_truncated_output_files, str):
            output_folder = output_folder_or_list_of_truncated_output_files
        elif isinstance(output_folder_or_list_of_truncated_output_files, list):
            output_folder = os.path.dirname(output_folder_or_list_of_truncated_output_files[0])
            if len(output_folder) == 0:  # just a file was given without a folder
                output_folder = os.path.curdir
        else:
            output_folder = None

        ########################
        # let's store the input arguments so that its clear what was used to generate the prediction
        if output_folder is not None:
            my_init_kwargs = {}
            for k in inspect.signature(self.predict_from_files_sequential).parameters.keys():
                my_init_kwargs[k] = locals()[k]
            my_init_kwargs = deepcopy(
                my_init_kwargs)  # let's not unintentionally change anything in-place. Take this as a
            recursive_fix_for_json_export(my_init_kwargs)
            save_json(my_init_kwargs, join(output_folder, 'predict_from_raw_data_args.json'))

            # we need these two if we want to do things with the predictions like for example apply postprocessing
            save_json(self.dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)
            save_json(self.plans_manager.plans, join(output_folder, 'plans.json'), sort_keys=False)
        #######################

        # check if we need a prediction from the previous stage
        if self.configuration_manager.previous_stage_name is not None:
            assert folder_with_segs_from_prev_stage is not None, \
                f'The requested configuration is a cascaded network. It requires the segmentations of the previous ' \
                f'stage ({self.configuration_manager.previous_stage_name}) as input. Please provide the folder where' \
                f' they are located via folder_with_segs_from_prev_stage'

        # optional mapping (SeriesInstanceUID <-> iarsna_xxxx)
        id_map_rev = {}
        if ids_mapping_json is not None:
            import json
            mapping = json.loads(Path(ids_mapping_json).read_text())
            id_map_rev = {v: k for k, v in mapping.items()}

        # optional cases filter
        allowed_cases = None
        if cases_csv is not None:
            df_cases = pd.read_csv(cases_csv)
            if "Filename" not in df_cases.columns:
                raise ValueError("cases_csv must contain a 'Filename' column")

            def _norm(name):
                stem = Path(name).stem
                if stem.endswith(".nii"):
                    stem = Path(stem).stem
                if stem.endswith("_0000"):
                    stem = stem[:-5]
                elif stem.endswith("_000"):
                    stem = stem[:-4]
                return stem

            allowed_cases = set(_norm(n) for n in df_cases["Filename"])
            print(f"[LOG] Cases filter | {len(allowed_cases)} entries loaded")

        # sort out input and output filenames
        list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files = \
            self._manage_input_and_output_lists(list_of_lists_or_source_folder,
                                                output_folder_or_list_of_truncated_output_files,
                                                folder_with_segs_from_prev_stage, overwrite, 0, 1,
                                                save_probabilities)
        if len(list_of_lists_or_source_folder) == 0:
            return

        label_manager = self.plans_manager.get_label_manager(self.dataset_json)
        preprocessor = self.configuration_manager.preprocessor_class(verbose=self.verbose)

        if output_filename_truncated is None:
            output_filename_truncated = [None] * len(list_of_lists_or_source_folder)
        if seg_from_prev_stage_files is None:
            seg_from_prev_stage_files = [None] * len(list_of_lists_or_source_folder)

        res = []
        label_idx_to_name = {int(v): k for k, v in self.dataset_json['labels'].items()}
        export_indices = [idx for idx, name in label_idx_to_name.items() if idx != 0 and name != "Aneurysm Present"]
        export_names = [label_idx_to_name[idx] for idx in export_indices]

        heatmap_dir = Path(heatmap_dir) if heatmap_dir else None
        if heatmap_dir:
            heatmap_dir.mkdir(parents=True, exist_ok=True)

        peak_logs = []

        for li, of, sps in zip(list_of_lists_or_source_folder, output_filename_truncated, seg_from_prev_stage_files):
            case_name = Path(li[0]).name.replace('.nii.gz', '').replace('.nii', '').replace('.npy', '')
            case_name = case_name.replace('_0000', '').replace('_000', '')
            if allowed_cases is not None and case_name not in allowed_cases:
                print(f"[LOG] Skip (not in cases_csv): {case_name}")
                continue
            series_id = id_map_rev.get(case_name, case_name)

            data, seg, data_properties = preprocessor.run_case(
                li,
                sps,
                self.plans_manager,
                self.configuration_manager,
                self.dataset_json
            )

            print(f'perform_everything_on_device: {self.perform_everything_on_device}')

            prediction = self.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()
            # convert to probs using sigmoid -> shape: [C, H, W, D]
            prediction_probs = torch.sigmoid(prediction)
            # max per selected channels
            max_per_c = torch.amax(prediction_probs, dim=(1, 2, 3)).cpu().numpy()
            export_vals = [float(max_per_c[idx]) for idx in export_indices]

            # optional heatmap/prob save + peak
            peak_coord = (None, None, None)
            peak_prob = None
            peak_label_idx = None
            if heatmap_dir:
                _, prob_resampled = convert_predicted_logits_to_segmentation_with_correct_shape(
                    prediction,
                    self.plans_manager,
                    self.configuration_manager,
                    label_manager,
                    data_properties,
                    return_probabilities=True,
                )
                prob_resampled = prob_resampled.astype(np.float32)
                prob_fname = f"{series_id}_prob.npy"
                img_fname = f"{series_id}_img.npy"
                png_fname = f"{series_id}_mip.png"
                prob_path = heatmap_dir / prob_fname
                img_path = heatmap_dir / img_fname
                png_path = heatmap_dir / png_fname
                np.save(prob_path, prob_resampled)
                np.save(img_path, data[0].astype(np.float32))

                channel_start = 1 if 0 in label_idx_to_name else 0
                sub_probs = prob_resampled[channel_start:]
                if sub_probs.size == 0:
                    channel_start = 0
                    sub_probs = prob_resampled
                flat_idx = np.argmax(sub_probs)
                sub_coord = np.unravel_index(flat_idx, sub_probs.shape)  # (c, z, y, x)
                channel_idx = int(sub_coord[0] + channel_start)
                peak_idx = sub_coord[1:]
                peak_prob = float(sub_probs[sub_coord])
                peak_coord = peak_idx
                peak_label_idx = channel_idx

                mip_img = data[0].max(axis=0) if data[0].ndim == 3 else data[0]
                mip_img = mip_img.astype(np.float32)
                mip_img = (mip_img - mip_img.min()) / (mip_img.ptp() + 1e-8)

                yy, xx = np.ogrid[:mip_img.shape[0], :mip_img.shape[1]]
                sigma = 8.0
                blob = np.exp(-(((xx - peak_idx[2]) ** 2 + (yy - peak_idx[1]) ** 2) / (2 * sigma * sigma)))
                blob = blob.astype(np.float32)
                blob = blob / (blob.max() + 1e-8)
                heat_full = blob
                alpha_full = np.clip(blob, 0, 1) * 0.8

                plt.figure(figsize=(6, 6))
                plt.imshow(mip_img, cmap="gray", origin="upper")
                if alpha_full.max() > 0:
                    plt.imshow(heat_full, cmap="hot", alpha=alpha_full, origin="upper")
                plt.scatter(peak_idx[2], peak_idx[1], s=80, c="red", marker="o", edgecolors="white", linewidths=1.2, alpha=0.95)
                plt.axis("off")
                plt.tight_layout()
                plt.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0)
                plt.close()
                if channel_idx is not None:
                    label_name = label_idx_to_name.get(channel_idx, f"class_{channel_idx}")
                    label_dir = heatmap_dir / f"{channel_idx}_{label_name}"
                    label_dir.mkdir(parents=True, exist_ok=True)
                    shutil.move(prob_path, label_dir / prob_fname)
                    shutil.move(png_path, label_dir / png_fname)
                    shutil.move(img_path, label_dir / img_fname)

                peak_logs.append(
                    {
                        "SeriesInstanceUID": series_id,
                        "peak_z": int(peak_idx[0]),
                        "peak_y": int(peak_idx[1]),
                        "peak_x": int(peak_idx[2]),
                        "peak_prob": peak_prob,
                        "peak_label": peak_label_idx,
                    }
                )

            res.append((series_id, *export_vals, peak_coord[0], peak_coord[1], peak_coord[2], peak_prob, peak_label_idx))

            submission_df = pd.DataFrame(res, columns=['SeriesInstanceUID'] + export_names + ["peak_z", "peak_y", "peak_x", "peak_prob", "peak_label_idx"])
            submission_df['Aneurysm Present'] = submission_df[export_names].max(axis=1)

            if output_folder is not None:
                submission_df.to_csv(join(output_folder, 'submission.csv'), index=False)

        # Final dataframe (remove background); compute Aneurysm Present from class maxima
        submission_df = pd.DataFrame(res, columns=['SeriesInstanceUID'] + export_names + ["peak_z", "peak_y", "peak_x", "peak_prob", "peak_label_idx"])
        submission_df['Aneurysm Present'] = submission_df[export_names].max(axis=1)
        if output_folder is not None:
            submission_df.to_csv(join(output_folder, 'submission.csv'), index=False)
        if heatmap_dir and peak_logs:
            pd.DataFrame(peak_logs).to_csv(heatmap_dir / "peaks.csv", index=False)
        # clear device cache
        empty_cache(self.device)
        return submission_df

def predict_entry_point():
    import argparse
    parser = argparse.ArgumentParser(description='Use this to run inference with nnU-Net. This function is used when '
                                                 'you want to manually specify a folder containing a trained nnU-Net '
                                                 'model. This is useful when the nnunet environment variables '
                                                 '(nnUNet_results) are not set.')
    parser.add_argument('-i', type=str, required=True,
                        help='input folder. Remember to use the correct channel numberings for your files (_0000 etc). '
                             'File endings must be the same as the training dataset!')
    parser.add_argument('-o', type=str, required=True,
                        help='Output folder. If it does not exist it will be created. Predicted segmentations will '
                             'have the same name as their source images.')
    parser.add_argument('-d', type=str, required=True,
                        help='Dataset with which you would like to predict. You can specify either dataset name or id')
    parser.add_argument('-p', type=str, required=False, default='nnUNetPlans',
                        help='Plans identifier. Specify the plans in which the desired configuration is located. '
                             'Default: nnUNetPlans')
    parser.add_argument('-tr', type=str, required=False, default='nnUNetTrainer',
                        help='What nnU-Net trainer class was used for training? Default: nnUNetTrainer')
    parser.add_argument('-c', type=str, required=True,
                        help='nnU-Net configuration that should be used for prediction. Config must be located '
                             'in the plans specified with -p')
    parser.add_argument('-f', nargs='+', type=str, required=False, default=(0, 1, 2, 3, 4),
                        help='Specify the folds of the trained model that should be used for prediction. '
                             'Default: (0, 1, 2, 3, 4)')
    parser.add_argument('-step_size', type=float, required=False, default=0.5,
                        help='Step size for sliding window prediction. The larger it is the faster but less accurate '
                             'the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.')
    parser.add_argument('--disable_tta', action='store_true', required=False, default=False,
                        help='Set this flag to disable test time data augmentation in the form of mirroring. Faster, '
                             'but less accurate inference. Not recommended.')
    parser.add_argument('--use_gaussian', action='store_true', required=False, default=False,
                        help='Set this flag to apply a gaussian weighting when aggregating the patches')
    parser.add_argument('--verbose', action='store_true', help="Set this if you like being talked to. You will have "
                                                               "to be a good listener/reader.")
    parser.add_argument('--save_probabilities', action='store_true',
                        help='Set this to export predicted class "probabilities". Required if you want to ensemble '
                             'multiple configurations.')
    parser.add_argument('--continue_prediction', action='store_true',
                        help='Continue an aborted previous prediction (will not overwrite existing files)')
    parser.add_argument('-chk', type=str, required=False, default='checkpoint_final.pth',
                        help='Name of the checkpoint you want to use. Default: checkpoint_final.pth')
    parser.add_argument('-prev_stage_predictions', type=str, required=False, default=None,
                        help='Folder containing the predictions of the previous stage. Required for cascaded models.')
    parser.add_argument('-num_parts', type=int, required=False, default=1,
                        help='Number of separate nnUNetv2_predict call that you will be making. Default: 1 (= this one '
                             'call predicts everything)')
    parser.add_argument('-part_id', type=int, required=False, default=0,
                        help='If multiple nnUNetv2_predict exist, which one is this? IDs start with 0 can end with '
                             'num_parts - 1. So when you submit 5 nnUNetv2_predict calls you need to set -num_parts '
                             '5 and use -part_id 0, 1, 2, 3 and 4. Simple, right? Note: You are yourself responsible '
                             'to make these run on separate GPUs! Use CUDA_VISIBLE_DEVICES (google, yo!)')
    parser.add_argument('-device', type=str, default='cuda', required=False,
                        help="Use this to set the device the inference should run with. Available options are 'cuda' "
                             "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
                             "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!")
    parser.add_argument('--disable_progress_bar', action='store_true', required=False, default=False,
                        help='Set this flag to disable progress bar. Recommended for HPC environments (non interactive '
                             'jobs)')
    parser.add_argument('--heatmap_dir', type=str, required=False, default=None,
                        help='Optional folder to save prob npy, img npy, mip png, and peaks.csv')
    parser.add_argument('--ids_mapping_json', type=str, required=False, default=None,
                        help='ids_mapping.json to map internal ids (iarsna_xxxx) <-> SeriesInstanceUID')
    parser.add_argument('--cases_csv', type=str, required=False, default=None,
                        help="Optional CSV with column 'Filename'; only these cases will be processed.")

    print(
        "\n#######################################################################\nPlease cite the following paper "
        "when using nnU-Net:\n"
        "Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). "
        "nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. "
        "Nature methods, 18(2), 203-211.\n#######################################################################\n")

    args = parser.parse_args()
    args.f = [i if i == 'all' else int(i) for i in args.f]

    model_folder = get_output_folder(args.d, args.tr, args.p, args.c)

    Path(args.o).mkdir(exist_ok=True, parents=True)

    # slightly passive aggressive haha
    assert args.part_id < args.num_parts, 'Do you even read the documentation? See nnUNetv2_predict -h.'

    assert args.device in ['cpu', 'cuda',
                           'mps'], f'-device must be either cpu, mps or cuda. Other devices are not tested/supported. Got: {args.device}.'
    if args.device == 'cpu':
        # let's allow torch to use hella threads
        import multiprocessing
        torch.set_num_threads(multiprocessing.cpu_count())
        device = torch.device('cpu')
    elif args.device == 'cuda':
        # multithreading in torch doesn't help nnU-Net if run on GPU
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        device = torch.device('cuda')
    else:
        device = torch.device('mps')

    predictor = nnUNetPredictorKaggle2025RSNA(tile_step_size=args.step_size,
                                use_gaussian=args.use_gaussian,
                                use_mirroring=not args.disable_tta,
                                perform_everything_on_device=True,
                                device=device,
                                verbose=args.verbose,
                                verbose_preprocessing=args.verbose,
                                allow_tqdm=not args.disable_progress_bar)
    predictor.initialize_from_trained_model_folder(
        model_folder,
        args.f,
        checkpoint_name=args.chk
    )
        
    print("Running in non-multiprocessing mode")
    predictor.predict_from_files_sequential(args.i, args.o, save_probabilities=args.save_probabilities,
                                            overwrite=not args.continue_prediction,
                                            folder_with_segs_from_prev_stage=args.prev_stage_predictions,
                                            ids_mapping_json=args.ids_mapping_json,
                                            heatmap_dir=args.heatmap_dir,
                                            cases_csv=args.cases_csv)
    

if __name__ == '__main__':
    predict_entry_point()
   

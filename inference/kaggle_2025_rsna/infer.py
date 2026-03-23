import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nnunetv2.dataset_conversion.kaggle_2025_rsna.official_data_to_nnunet import (
    load_and_crop,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        required=True,
        help="Path to directory with all DICOM data, e.g. /path/to/rsna-intracranial-aneurysm-detection/series",
    )
    p.add_argument(
        "-o",
        "--output-path",
        type=Path,
        required=True,
        help="Where to store the resulting csv, e.g. output.csv",
    )
    p.add_argument(
        "-m",
        "--model_folder",
        type=Path,
        required=True,
        help="Path to model checkpoint, e.g. path/to/downloaded-checkpoint/Dataset004_iarsna_crop/Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32",
    )

    p.add_argument(
        "-c",
        "--chk",
        type=str,
        required=True,
        help="Name of the checkpoint, e.g. checkpoint_epoch_1500.pth",
    )
    p.add_argument(
        "--fold",
        type=ast.literal_eval,
        help="tuple of fold identifiers, e.g. \"('all',)\" or (0,1,2)",
    )
    p.add_argument(
        "--step_size",
        type=float,
        required=False,
        default=0.5,
        help="Step size for sliding window prediction. The larger it is the faster but less accurate "
        "the prediction. Default: 0.5. Cannot be larger than 1. We recommend the default.",
    )
    p.add_argument(
        "--disable_tta",
        action="store_true",
        required=False,
        default=False,
        help="Set this flag to disable test time data augmentation in the form of mirroring. Faster, "
        "but less accurate inference. Not recommended.",
    )
    p.add_argument(
        "--use_gaussian",
        action="store_true",
        required=False,
        default=False,
        help="Set this flag to apply a gaussian weighting when aggregating the patches",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        required=False,
        help="Use this to set the device the inference should run with. Available options are 'cuda' "
        "(GPU), 'cpu' (CPU) and 'mps' (Apple M1/M2). Do NOT use this to set which GPU ID! "
        "Use CUDA_VISIBLE_DEVICES=X nnUNetv2_predict [...] instead!",
    )

    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=args.step_size,
        use_gaussian=args.use_gaussian,
        use_mirroring=not args.disable_tta,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False,
    )
    predictor.initialize_from_trained_model_folder(
        args.model_folder,
        [i if i == "all" else int(i) for i in args.fold],
        checkpoint_name=args.chk,
    )

    preprocessor = predictor.configuration_manager.preprocessor_class()

    labels = (
        ["SeriesInstanceUID"]
        + list(predictor.dataset_json["labels"].keys())[1:]
        + ["Aneurysm Present"]
    )
    res = []
    for series_dir in tqdm(list(args.input_dir.iterdir())):
        img, properties = load_and_crop(series_dir)
        data, _, _ = preprocessor.run_case_npy(
            np.array([img]),
            None,
            properties,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
        logits = predictor.predict_logits_from_preprocessed_data(
            torch.from_numpy(data)
        ).cpu()
        probs = torch.sigmoid(logits)

        max_per_c = torch.amax(probs, dim=(1, 2, 3)).to(
            dtype=torch.float32, device="cpu"
        )

        res.append([series_dir.name] + max_per_c.numpy().tolist())

        pd.DataFrame(res, columns=labels).to_csv(args.output_path, index=False)

    print(f"Results saved to {args.output_path}")


if __name__ == "__main__":
    main()
    # python inference.py \
    # -i /path/to/rsna-intracranial-aneurysm-detection/series \
    # -o output.csv \
    # -m /path/to/downloaded-checkpoint/Dataset004_iarsna_crop/Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32 \
    # -c checkpoint_epoch_1500.pth \
    # --fold "('all',)"
    # --disable_tta
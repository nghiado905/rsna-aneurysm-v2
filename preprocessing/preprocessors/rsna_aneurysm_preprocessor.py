import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from typing import Union

import nnunetv2
from nnunetv2.preprocessing.resampling.default_resampling import compute_new_shape
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

class RSNA_Aneurysm_Preprocessor(DefaultPreprocessor):
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        """
        Everything we need is in the plans. Those are given when run() is called.
        For RSNA Aneurysm Challenge https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/overview 
        --> Only keep 1) Z-score normalization and 2) Resampling (use torch for faster)
        --> Remove 1) Transpose and 2) Non-zero cropping
        """

    def run_case_npy(self, data: np.ndarray, seg: Union[np.ndarray, None], properties: dict,
                     plans_manager: PlansManager, configuration_manager: ConfigurationManager,
                     dataset_json: Union[dict, str]):
        # let's not mess up the inputs!
        data = data.astype(np.float32)  # this creates a copy
        if seg is not None:
            assert data.shape[1:] == seg.shape[1:], "Shape mismatch between image and segmentation. Please fix your dataset and make use of the --verify_dataset_integrity flag to ensure everything is correct"
            seg = np.copy(seg)

        has_seg = seg is not None

        original_spacing = [properties['spacing'][i] for i in plans_manager.transpose_forward]

        # crop, remember to store size before cropping!
        shape_before_cropping = data.shape[1:]
        properties['shape_before_cropping'] = shape_before_cropping
        properties['shape_after_cropping_and_before_resampling'] = data.shape[1:]

        # resample preparation
        target_spacing = configuration_manager.spacing  # this should already be transposed

        if len(target_spacing) < len(data.shape[1:]):
            # target spacing for 2d has 2 entries but the data and original_spacing have three because everything is 3d
            # in 2d configuration we do not change the spacing between slices
            target_spacing = [original_spacing[0]] + target_spacing
        new_shape = compute_new_shape(data.shape[1:], original_spacing, target_spacing)

        data = self._normalize(data, seg, configuration_manager,
                               plans_manager.foreground_intensity_properties_per_channel)

        old_shape = data.shape[1:]
        data = configuration_manager.resampling_fn_data(data, new_shape, original_spacing, target_spacing)
        if self.verbose:
            print(f'old shape: {old_shape}, new_shape: {new_shape}, old_spacing: {original_spacing}, '
                  f'new_spacing: {target_spacing}, fn_data: {configuration_manager.resampling_fn_data}')

        if has_seg:
            seg = configuration_manager.resampling_fn_seg(seg, new_shape, original_spacing, target_spacing)
            label_manager = plans_manager.get_label_manager(dataset_json)
            collect_for_this = label_manager.foreground_regions if label_manager.has_regions \
                else label_manager.foreground_labels

            if label_manager.has_ignore_label:
                collect_for_this.append([-1] + label_manager.all_labels)

            properties['class_locations'] = self._sample_foreground_locations(seg, collect_for_this,
                                                                                   verbose=self.verbose)
            seg = self.modify_seg_fn(seg, plans_manager, dataset_json, configuration_manager)
            if np.max(seg) > 127:
                seg = seg.astype(np.int16)
            else:
                seg = seg.astype(np.int8)
        return data, seg, properties

    def _normalize(self,
               data: np.ndarray,
               seg: Union[np.ndarray, None],
               configuration_manager: ConfigurationManager,
               foreground_intensity_properties_per_channel: dict) -> np.ndarray:
        for c in range(data.shape[0]):
            scheme = configuration_manager.normalization_schemes[c]
            normalizer_class = recursive_find_python_class(
                join(nnunetv2.__path__[0], "preprocessing", "normalization"),
                scheme,
                'nnunetv2.preprocessing.normalization'
            )
            if normalizer_class is None:
                raise RuntimeError(f"Unable to locate class '{scheme}' for normalization")

            use_mask = configuration_manager.use_mask_for_norm[c]
            normalizer = normalizer_class(
                use_mask_for_norm=use_mask,
                intensityproperties=foreground_intensity_properties_per_channel[str(c)]
            )

            # Only supply a seg if we really use a mask.
            if use_mask:
                if seg is None:
                    # Defensive: fail fast if any channel expects a mask but none is provided.
                    raise ValueError(
                        f"Channel {c} has use_mask_for_norm=True but 'seg' is None. "
                        f"Either provide a seg or set use_mask_for_norm=False for this channel."
                    )
                mask_seg = seg[0]  # shape (X,Y[,Z]); negative values mark 'outside' region as expected by ZScoreNormalization
            else:
                mask_seg = None  # triggers whole-image z-score in ZScoreNormalization

            data[c] = normalizer.run(data[c], mask_seg)
        return data
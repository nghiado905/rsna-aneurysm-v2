import multiprocessing
from copy import deepcopy
from time import sleep, time
from typing import Union, Tuple, List

import cc3d
import edt
import numpy as np
import pandas as pd
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import disk, ball
from functools import lru_cache
from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from nnunetv2.utilities.collate_outputs import collate_outputs

from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.utilities.file_and_folder_operations import join, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.helpers.scalar_type import sample_scalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import (
    MultiplicativeBrightnessTransform,
    BrightnessAdditiveTransform,
)
from batchgeneratorsv2.transforms.intensity.contrast import (
    ContrastTransform,
    BGContrast,
)
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.local.brightness_gradient import (
    BrightnessGradientAdditiveTransform,
)
from batchgeneratorsv2.transforms.local.local_gamma import LocalGammaTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.noise.median_filter import MedianFilterTransform
from batchgeneratorsv2.transforms.noise.sharpen import SharpeningTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import (
    SimulateLowResolutionTransform,
)
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform, OneOfTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from threadpoolctl import threadpool_limits
from torch import nn, autocast, topk
from torch import distributed as dist
from torch.nn import functional as F, BCEWithLogitsLoss

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.configuration import default_num_processes
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.training.data_augmentation.compute_initial_patch_size import (
    get_patch_size,
)
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.data_augmentation.nnUNetTrainerDA5 import (
    _brightnessadditive_localgamma_transform_scale,
    _brightness_gradient_additive_max_strength,
    _local_gamma_gamma,
)
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.helpers import dummy_context, empty_cache


# ******************************************************************************************************************************************
# ************************************************************** DATA LOADER ***************************************************************
# ******************************************************************************************************************************************


class Kaggle2025RSNALoader(nnUNetDataLoader):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, seg_prev, properties = self._data.load_case(i)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]

            bbox_lbs, bbox_ubs = self.get_bbox(
                shape, force_fg, properties["class_locations"]
            )
            bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

            # use ACVL utils for that. Cleaner.
            data_all[j] = crop_and_pad_nd(data, bbox, 0)

            seg_cropped = crop_and_pad_nd(seg, bbox, -1)
            if seg_prev is not None:
                seg_cropped = np.vstack(
                    (seg_cropped, crop_and_pad_nd(seg_prev, bbox, -1)[None])
                )
            seg_all[j] = seg_cropped

        if self.patch_size_was_2d:
            data_all = data_all[:, :, 0]
            seg_all = seg_all[:, :, 0]

        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    bboxes = []
                    target_structs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(
                            **{"image": data_all[b], "segmentation": seg_all[b]}
                        )
                        images.append(tmp["image"])
                        bboxes.append(tmp["bboxes"])
                        target_structs.append(tmp["target_struct"])
                    data_all = torch.stack(images)
                    del images
            return {
                "data": data_all,
                "keys": selected_keys,
                "target_struct": target_structs,
                "bboxes": bboxes,
            }

        return {"data": data_all, "target": seg_all, "keys": selected_keys}


@lru_cache(maxsize=5)
def build_point(radii, use_distance_transform, binarize):
    max_radius = max(radii)
    ndim = len(radii)

    # Create a spherical (or circular) structuring element with max_radius
    if ndim == 2:
        structuring_element = disk(max_radius)
    elif ndim == 3:
        structuring_element = ball(max_radius)
    else:
        raise ValueError(
            "Unsupported number of dimensions. Only 2D and 3D are supported."
        )

    # Convert the structuring element to a tensor
    structuring_element = torch.from_numpy(structuring_element.astype(np.float32))

    # Create the target shape based on the sampled radii
    target_shape = [round(2 * r + 1) for r in radii]

    if any([i != j for i, j in zip(target_shape, structuring_element.shape)]):
        structuring_element_resized = torch.nn.functional.interpolate(
            structuring_element.unsqueeze(0).unsqueeze(
                0
            ),  # Add batch and channel dimensions for interpolation
            size=target_shape,
            mode="trilinear" if ndim == 3 else "bilinear",
            align_corners=False,
        )[
            0, 0
        ]  # Remove batch and channel dimensions after interpolation
    else:
        structuring_element_resized = structuring_element

    if use_distance_transform:
        # Convert the structuring element to a binary mask for distance transform computation
        binary_structuring_element = (structuring_element_resized >= 0.5).numpy()

        # Compute the Euclidean distance transform of the binary structuring element
        structuring_element_resized = distance_transform_edt(binary_structuring_element)

        # Normalize the distance transform to have values between 0 and 1
        structuring_element_resized /= structuring_element_resized.max()
        structuring_element_resized = torch.from_numpy(structuring_element_resized)

    if binarize and not use_distance_transform:
        # Normalize the resized structuring element to binary (values near 1 are treated as the point region)
        structuring_element_resized = (structuring_element_resized >= 0.5).float()
    return structuring_element_resized


class ConvertSegToLandmarkTarget(BasicTransform):
    # This only works in 3D.
    def __init__(
        self,
        n_landmarks: int,
        edt_radius: int = 15,
    ):
        super().__init__()
        self.edt_radius = edt_radius
        self.n_landmarks = n_landmarks

    def apply(self, data_dict, **params):
        seg = data_dict["segmentation"]

        # seg must be (1, x, y, z)
        assert len(seg.shape) == 3 or seg.shape[0] == 1
        if len(seg.shape) == 4:
            seg = seg[0]

        components = torch.unique(seg)
        components = [i for i in components if i != 0]

        # now place gaussian or etd on these coordinates
        target = build_point(
            tuple([self.edt_radius] * 3), use_distance_transform=True, binarize=False
        )

        bboxes = {}

        if len(components) > 0:
            stats = cc3d.statistics(seg.numpy().astype(np.uint8))
            for ci in components:
                bbox = stats["bounding_boxes"][
                    ci
                ]  # (slice(3, 9, None), slice(4, 10, None), slice(6, 12, None))
                crop = (seg[bbox] == ci).numpy()
                dist = edt.edt(crop, black_border=True)
                center = np.unravel_index(np.argmax(dist), crop.shape)
                center = [i + j.start for i, j in zip(center, bbox)]
                insert_bbox = [
                    [i - j // 2, i - j // 2 + j] for i, j in zip(center, target.shape)
                ]
                bboxes[ci.item()] = insert_bbox

        # it would be nicer to write that into regression_target but that would require to change the nnunet dataloader so nah
        del data_dict["segmentation"]
        data_dict["bboxes"] = bboxes
        data_dict["target_struct"] = target
        return data_dict


# ********************************************************************************************************************************
# ******************************************************** LOSS FUNCTIONS ********************************************************
# ********************************************************************************************************************************


def paste_tensor_optionalMax(target, source, bbox, use_max=False):
    """
    Paste or combine a source tensor/array into a target tensor/array using a given bounding box,
    with optional pixelwise maximum.

    Supports both NumPy arrays and PyTorch tensors. Output type matches input target type.

    Args:
        target (np.ndarray or torch.Tensor): 3D volume.
        source (np.ndarray or torch.Tensor): 3D volume (same shape as bbox region).
        bbox (list or tuple): Bounding box as [[x1, x2], [y1, y2], [z1, z2]]
        use_max (bool): If True, combine using pixelwise max instead of direct paste.

    Returns:
        Same type as `target`: Modified 3D volume.
    """
    is_numpy = isinstance(target, np.ndarray)
    xp = np if is_numpy else torch

    target_shape = target.shape
    target_indices = []
    source_indices = []

    for i, (b0, b1) in enumerate(bbox):
        t_start = max(b0, 0)
        t_end = min(b1, target_shape[i])
        if t_start >= t_end:
            return target  # No overlap

        s_start = t_start - b0
        s_end = s_start + (t_end - t_start)

        target_indices.append((t_start, t_end))
        source_indices.append((s_start, s_end))

    tz0, tz1 = target_indices[0]
    ty0, ty1 = target_indices[1]
    if len(target_shape) == 2:
        # 2D
        pass
    else:
        tx0, tx1 = target_indices[2]

    sz0, sz1 = source_indices[0]
    sy0, sy1 = source_indices[1]
    if len(target_shape) == 2:
        # 2D
        pass
    else:
        sx0, sx1 = source_indices[2]

    if use_max:
        target[tz0:tz1, ty0:ty1, tx0:tx1] = xp.maximum(
            target[tz0:tz1, ty0:ty1, tx0:tx1], source[sz0:sz1, sy0:sy1, sx0:sx1]
        )
    else:
        if len(target_shape) == 2:
            # 2D
            target[tz0:tz1, ty0:ty1] = source[sz0:sz1, sy0:sy1]
        else:
            target[tz0:tz1, ty0:ty1, tx0:tx1] = source[sz0:sz1, sy0:sy1, sx0:sx1]

    return target


class BCE_topK_loss_sep_channel(nn.Module):
    # This only works in 3D.
    def __init__(self, k: RandomScalar = 100):
        super().__init__()
        self.bce = BCEWithLogitsLoss(reduction="none")
        self.k = k
        self.preallocated_dummy_target: torch.Tensor = None

    def forward(self, net_output: torch.Tensor, target_structure: torch.Tensor, bboxes):
        # net_output is b, c, x, y, z
        # target_structure is a list of tensors x, y, z
        # bboxes is a list of dicts mapping an index to a bbox
        if self.preallocated_dummy_target is None:
            self.preallocated_dummy_target = torch.zeros(
                net_output.shape, device=net_output.device, dtype=torch.float32
            )

        with torch.no_grad():
            self.preallocated_dummy_target.zero_()

            for b in range(net_output.shape[0]):
                for c in range(net_output.shape[1]):
                    # insert into preallocated_dummy_target
                    if c + 1 in bboxes[b].keys():
                        paste_tensor_optionalMax(
                            self.preallocated_dummy_target[b, c],
                            target_structure[b],
                            bboxes[b][c + 1],
                            use_max=False,
                        )
                    elif c == net_output.shape[1] - 1:
                        self.preallocated_dummy_target[b, c] = torch.amax(
                            self.preallocated_dummy_target[b, :c], dim=0
                        )
                    else:
                        pass

        loss = self.bce(net_output, self.preallocated_dummy_target)

        n = max(1, round(np.prod(loss.shape[-3:]) * sample_scalar(self.k) / 100))

        loss = loss.view((*loss.shape[:2], -1))
        loss = topk(loss, k=n, sorted=False)[0]
        loss = loss.mean()
        return loss


# ********************************************************************************************************************************
# *********************************************************** Trainer ************************************************************
# ********************************************************************************************************************************


class Kaggle2025RSNATrainer(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.blobb_radius = 65
        # disable deep supervision for landmark loss
        self.enable_deep_supervision = False

        # ------------------------------------------------------------------
        # Load optional overrides from plans configurations
        # ------------------------------------------------------------------
        cfg = plans.get("configurations", {}).get(configuration, {})
        # if hasattr(self, "initial_lr"):
        self.initial_lr = cfg.get("initial_lr", self.initial_lr)
        self.num_epochs = cfg.get("num_epochs", self.num_epochs)
        self.save_every = cfg.get("save_every", self.save_every)

        # Pretty console banner (ANSI) for quick debugging
        _c = lambda code: f"\033[{code}m"
        reset = _c("0")
        bold = _c("1")
        cyan = _c("96")
        magenta = _c("95")
        yellow = _c("93")
        green = _c("92")

        patch_size = getattr(self.configuration_manager, "patch_size", None)
        batch_size = getattr(self.configuration_manager, "batch_size", None)
        arch = getattr(self.configuration_manager, "architecture", None)

        banner = [
            f"{bold}{magenta}=== Kaggle2025RSNATrainer Config ==={reset}",
            f"{cyan}configuration{reset}: {configuration}",
            f"{cyan}fold{reset}: {fold}",
            f"{cyan}patch_size{reset}: {patch_size}",
            f"{cyan}batch_size{reset}: {batch_size}",
            f"{cyan}initial_lr{reset}: {self.initial_lr}",
            f"{cyan}num_epochs{reset}: {self.num_epochs}",
            f"{cyan}save_every{reset}: {self.save_every}",
            f"{cyan}arch_class{reset}: {getattr(arch, 'network_class_name', getattr(arch, 'get', lambda k, d=None: None)('network_class_name', None)) if isinstance(arch, dict) else arch}",
            f"{bold}{green}Plans name{reset}: {plans.get('plans_name', 'unknown')}",
            f"{bold}{yellow}Dataset{reset}: {plans.get('dataset_name', 'unknown')}",
            f"{bold}{magenta}===============================\033[0m",
        ]
        for line in banner:
            self.print_to_log_file(line)


        # palette used for fancy logging later on
        self._ansi_palette = {
            "reset": reset,
            "bold": bold,
            "cyan": cyan,
            "magenta": magenta,
            "yellow": yellow,
            "green": green,
            "red": _c("91"),
            "blue": _c("94"),
        }

    def _style_log(self, text: str, color: str = "cyan", bold: bool = False) -> str:
        palette = getattr(self, "_ansi_palette", {})
        color_code = palette.get(color, "")
        bold_code = palette.get("bold", "") if bold else ""
        reset = palette.get("reset", "")
        return f"{bold_code}{color_code}{text}{reset}"

    def _log_step(self, tag: str, message: str, color: str = "cyan"):
        tag_fmt = self._style_log(f"[{tag.upper()}]", color=color, bold=True)
        self.print_to_log_file(f"{tag_fmt} {message}")

    def _log_header(self, title: str, color: str = "magenta"):
        bar = "=" * (len(title) + 8)
        bar_fmt = self._style_log(bar, color=color, bold=True)
        title_fmt = self._style_log(f"=== {title} ===", color=color, bold=True)
        self.print_to_log_file(bar_fmt)
        self.print_to_log_file(title_fmt)
        self.print_to_log_file(bar_fmt)

    def _build_loss(self):
        loss = BCE_topK_loss_sep_channel(k=20)
        return loss

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = (-15.0 / 360 * 2.0 * np.pi, 15.0 / 360 * 2.0 * np.pi)
            else:
                rotation_for_DA = (
                    -180.0 / 360 * 2.0 * np.pi,
                    180.0 / 360 * 2.0 * np.pi,
                )
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = (
                    -180.0 / 360 * 2.0 * np.pi,
                    180.0 / 360 * 2.0 * np.pi,
                )
            else:
                rotation_for_DA = (-30.0 / 360 * 2.0 * np.pi, 30.0 / 360 * 2.0 * np.pi)
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_patch_size = get_patch_size(
            patch_size[-dim:],
            rotation_for_DA,
            rotation_for_DA,
            rotation_for_DA,
            (0.85, 1.25),
        )
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f"do_dummy_2d_data_aug: {do_dummy_2d_data_aug}")
        mirror_axes = (0, 1)  # only mirror along axis 0 and 1
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

    def get_dataloaders(self):
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

        # we use the patch size to determine whether we need 2D or 3D dataloaders. We also use it to determine whether
        # we need to use dummy 2D augmentation (in case of 3D training) and what our initial patch size should be
        patch_size = self.configuration_manager.patch_size

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?
        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size,
            rotation_for_DA,
            deep_supervision_scales,
            mirror_axes,
            do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        # validation pipeline
        val_transforms = self.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded=self.is_cascaded,
            foreground_labels=self.label_manager.foreground_labels,
            regions=(
                self.label_manager.foreground_regions
                if self.label_manager.has_regions
                else None
            ),
            ignore_label=self.label_manager.ignore_label,
        )

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        dl_tr = Kaggle2025RSNALoader(
            dataset_tr,
            self.batch_size,
            initial_patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=tr_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            # random_offset=[i // 3 for i in self.configuration_manager.patch_size]
        )
        dl_val = Kaggle2025RSNALoader(
            dataset_val,
            self.batch_size,
            self.configuration_manager.patch_size,
            self.configuration_manager.patch_size,
            self.label_manager,
            oversample_foreground_percent=self.oversample_foreground_percent,
            sampling_probabilities=None,
            pad_sides=None,
            transforms=val_transforms,
            probabilistic_oversampling=self.probabilistic_oversampling,
            # random_offset=[i // 3 for i in self.configuration_manager.patch_size]
        )

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(
                data_loader=dl_tr,
                transform=None,
                num_processes=allowed_num_processes,
                num_cached=max(6, allowed_num_processes // 2),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
            mt_gen_val = NonDetMultiThreadedAugmenter(
                data_loader=dl_val,
                transform=None,
                num_processes=max(1, allowed_num_processes // 2),
                num_cached=max(3, allowed_num_processes // 4),
                seeds=None,
                pin_memory=self.device.type == "cuda",
                wait_time=0.002,
            )
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    def get_training_transforms(
        self,
        patch_size: Union[np.ndarray, Tuple[int]],
        rotation_for_DA: RandomScalar,
        deep_supervision_scales: Union[List, Tuple, None],
        mirror_axes: Tuple[int, ...],
        do_dummy_2d_data_aug: bool,
        use_mask_for_norm: List[bool] = None,
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        matching_axes = np.array(
            [sum([i == j for j in patch_size]) for i in patch_size]
        )
        valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])
        transforms = []

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=0,
                random_crop=False,
                p_elastic_deform=0,
                p_rotation=0.3,
                rotation=rotation_for_DA,
                p_scaling=0.3,
                scaling=(0.6, 1.67),
                p_synchronize_scaling_across_axes=0.8,
                bg_style_seg_sampling=False,
                mode_seg="nearest",
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        OneOfTransform(
            [
                RandomTransform(
                    MedianFilterTransform(
                        (2, 8), p_same_for_each_channel=0.5, p_per_channel=0.5
                    ),
                    apply_probability=0.05,
                ),
                RandomTransform(
                    GaussianBlurTransform(
                        blur_sigma=(0.3, 1.5),
                        synchronize_channels=False,
                        synchronize_axes=False,
                        p_per_channel=0.5,
                        benchmark=True,
                    ),
                    apply_probability=0.05,
                ),
            ]
        )

        transforms.append(
            RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.2),
                    p_per_channel=0.5,
                    synchronize_channels=True,
                ),
                apply_probability=0.3,
            )
        )

        transforms.append(
            RandomTransform(
                BrightnessAdditiveTransform(
                    0, 0.5, per_channel=True, p_per_channel=0.5
                ),
                apply_probability=0.1,
            )
        )

        transforms.append(
            OneOfTransform(
                [
                    RandomTransform(
                        ContrastTransform(
                            contrast_range=BGContrast((0.75, 1.25)),
                            preserve_range=True,
                            synchronize_channels=False,
                            p_per_channel=0.5,
                        ),
                        apply_probability=0.3,
                    ),
                    RandomTransform(
                        MultiplicativeBrightnessTransform(
                            multiplier_range=BGContrast((0.75, 1.25)),
                            synchronize_channels=False,
                            p_per_channel=0.5,
                        ),
                        apply_probability=0.3,
                    ),
                ]
            )
        )

        transforms.append(
            RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=ignore_axes,
                    allowed_channels=None,
                    p_per_channel=0.5,
                ),
                apply_probability=0.15,
            )
        )

        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.6, 2)),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.2,
            )
        )

        transforms.append(
            RandomTransform(
                GammaTransform(
                    gamma=BGContrast((0.6, 2)),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1,
                ),
                apply_probability=0.2,
            )
        )

        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(MirrorTransform(allowed_axes=mirror_axes))

        transforms.append(
            RandomTransform(
                BrightnessGradientAdditiveTransform(
                    _brightnessadditive_localgamma_transform_scale,
                    (-0.5, 1.5),
                    max_strength=_brightness_gradient_additive_max_strength,
                    same_for_all_channels=False,
                    mean_centered=True,
                    clip_intensities=False,
                    p_per_channel=0.5,
                ),
                apply_probability=0.2,
            )
        )

        transforms.append(
            RandomTransform(
                LocalGammaTransform(
                    _brightnessadditive_localgamma_transform_scale,
                    (-0.5, 1.5),
                    _local_gamma_gamma,
                    same_for_all_channels=False,
                    p_per_channel=0.5,
                ),
                apply_probability=0.2,
            )
        )

        transforms.append(
            RandomTransform(
                SharpeningTransform(
                    (0.1, 1.5),
                    p_same_for_each_channel=0.5,
                    p_per_channel=0.5,
                    p_clamp_intensities=0.5,
                ),
                apply_probability=0.2,
            )
        )

        transforms.append(RemoveLabelTansform(-1, 0))

        transforms.append(
            ConvertSegToLandmarkTarget(
                len(self.label_manager.foreground_labels), edt_radius=self.blobb_radius
            )
        )

        transforms = ComposeTransforms(transforms)

        return transforms

    def get_validation_transforms(
        self,
        deep_supervision_scales: Union[List, Tuple, None],
        is_cascaded: bool = False,
        foreground_labels: Union[Tuple[int, ...], List[int]] = None,
        regions: List[Union[List[int], Tuple[int, ...], int]] = None,
        ignore_label: int = None,
    ) -> BasicTransform:
        transforms: ComposeTransforms = nnUNetTrainer.get_validation_transforms(
            deep_supervision_scales,
            is_cascaded,
            foreground_labels,
            regions,
            ignore_label,
        )
        transforms.transforms.append(
            ConvertSegToLandmarkTarget(
                len(self.label_manager.foreground_labels), edt_radius=self.blobb_radius
            )
        )
        return transforms

    def train_step(self, batch: dict) -> dict:
        data = batch["data"]

        data = data.to(self.device, non_blocking=True)
        target_structure = [
            i.to(self.device, non_blocking=True) for i in batch["target_struct"]
        ]

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            # import IPython;IPython.embed()
            # if False:
            #     from batchviewer import view_batch
            #     view_batch(data[0], target[0][0], F.sigmoid(output[0][0]))

        # take loss out of autocast! Sigmoid is not stable in fp16
        l = self.loss(output, target_structure, batch["bboxes"])

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {"loss": l.detach().cpu().numpy()}

    def validation_step(self, batch: dict) -> dict:
        data = batch["data"]

        data = data.to(self.device, non_blocking=True)
        target_structure = [
            i.to(self.device, non_blocking=True) for i in batch["target_struct"]
        ]

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with (
            autocast(self.device.type, enabled=True)
            if self.device.type == "cuda"
            else dummy_context()
        ):
            output = self.network(data)
            # del data
            l = self.loss(output, target_structure, batch["bboxes"])

        return {"loss": l.detach().cpu().numpy()}

    def on_validation_epoch_end(self, val_outputs: List[dict]):
        # logs val loss instead of fg dice
        outputs_collated = collate_outputs(val_outputs)

        if self.is_ddp:
            world_size = dist.get_world_size()

            losses_val = [None for _ in range(world_size)]
            dist.all_gather_object(losses_val, outputs_collated["loss"])
            loss_here = np.vstack(losses_val).mean()
        else:
            loss_here = np.mean(outputs_collated["loss"])

        # global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in zip(tp, fp, fn)]]
        # mean_fg_dice = np.nanmean(global_dc_per_class)
        self.logger.log("mean_fg_dice", loss_here, self.current_epoch)
        self.logger.log("dice_per_class_or_region", [1, 1], self.current_epoch)
        self.logger.log("val_losses", loss_here, self.current_epoch)

    def on_epoch_end(self):
        # --- logging ---
        self.logger.log("epoch_end_timestamps", time(), self.current_epoch)
        self.print_to_log_file(
            "train_loss",
            np.round(self.logger.my_fantastic_logging["train_losses"][-1], 6),
        )
        current_val_loss = self.logger.my_fantastic_logging["val_losses"][-1]
        if np.isnan(current_val_loss):
            self.print_to_log_file("val_loss", "n/a (validation skipped this epoch)")
        else:
            self.print_to_log_file("val_loss", np.round(current_val_loss, 6))
        self.print_to_log_file(
            f"Epoch time: {np.round(self.logger.my_fantastic_logging['epoch_end_timestamps'][-1] - self.logger.my_fantastic_logging['epoch_start_timestamps'][-1], 2)} s"
        )

        # --- periodic checkpointing: every 250th epoch (1-indexed: 250, 500, ...) ---
        epoch_1idx = self.current_epoch + 1
        if (
            self.local_rank == 0
            and (epoch_1idx % 250) == 0
            and self.current_epoch != (self.num_epochs - 1)
        ):
            # keep a persistent, numbered snapshot
            self.save_checkpoint(
                join(self.output_folder, f"checkpoint_epoch_{epoch_1idx:04d}.pth")
            )

        current_epoch = self.current_epoch
        if (current_epoch + 1) % self.save_every == 0 and current_epoch != (
            self.num_epochs - 1
        ):
            self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))

        # --- 'best' checkpointing (lower is better per your comment) ---
        current_ema = self.logger.my_fantastic_logging["ema_fg_dice"][-1]
        if (
            not np.isnan(current_ema)
            and (
                self._best_ema is None
                or current_ema < self._best_ema
            )
        ):
            self._best_ema = current_ema
            if self.local_rank == 0:
                self.print_to_log_file(
                    f"Yayy! New best EMA loss: {np.round(self._best_ema, 4)}"
                )
                self.save_checkpoint(join(self.output_folder, "checkpoint_best.pth"))

        # --- plots (rank 0 only) ---
        if self.local_rank == 0:
            self.logger.plot_progress_png(self.output_folder)

        self.current_epoch += 1

    @torch.inference_mode()
    def perform_actual_validation(self, save_probabilities: bool = False):
        self.set_deep_supervision_enabled(False)
        self.network.eval()

        if (
            self.is_ddp
            and self.batch_size == 1
            and self.enable_deep_supervision
            and self._do_i_compile()
        ):
            self.print_to_log_file(
                "WARNING! batch size is 1 during training and torch.compile is enabled. If you "
                "encounter crashes in validation then this is because torch.compile forgets "
                "to trigger a recompilation of the model with deep supervision disabled. "
                "This causes torch.flip to complain about getting a tuple as input. Just rerun the "
                "validation with --val (exactly the same as before) and then it will work. "
                "Why? Because --val triggers nnU-Net to ONLY run validation meaning that the first "
                "forward pass (where compile is triggered) already has deep supervision disabled. "
                "This is exactly what we need in perform_actual_validation"
            )

        dsj = deepcopy(self.dataset_json)
        n_landmarks = len(self.label_manager.foreground_labels)
        dsj["labels"] = {
            "background": 0,
            **{str(i): i for i in range(1, n_landmarks + 1)},
        }
        # don't worry about use_mirroring=True. self.inference_allowed_mirroring_axes is None.
        # we set perform_everything_on_device=False because landmark tasks often have vram issues because of how many landmarks there are
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=False,
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False,
        )
        predictor.manual_initialization(
            self.network,
            self.plans_manager,
            self.configuration_manager,
            None,
            dsj,
            self.__class__.__name__,
            self.inference_allowed_mirroring_axes,
        )

        self._log_header("Validation Predictor", color="blue")
        self._log_step(
            "INIT",
            f"tile_step=0.5 | gaussian=True | mirroring=True | device={self.device}",
            color="cyan",
        )

        with multiprocessing.get_context("spawn").Pool(
            default_num_processes
        ) as export_pool:
            worker_list = [i for i in export_pool._pool]
            validation_output_folder = join(self.output_folder, "validation")
            maybe_mkdir_p(validation_output_folder)

            # we cannot use self.get_tr_and_val_datasets() here because we might be DDP and then we have to distribute
            # the validation keys across the workers.
            _, val_keys = self.do_split()

            dataset_val = self.dataset_class(
                self.preprocessed_dataset_folder,
                val_keys,
                folder_with_segs_from_previous_stage=self.folder_with_segs_from_previous_stage,
            )

            next_stages = self.configuration_manager.next_stage_names

            if next_stages is not None:
                _ = [
                    maybe_mkdir_p(
                        join(self.output_folder_base, "predicted_next_stage", n)
                    )
                    for n in next_stages
                ]

            results = []
            csv_entries = []
            total_cases = len(dataset_val.identifiers)
            self._log_step(
                "DATA",
                f"{total_cases} validation cases | workers={default_num_processes} | fold={self.fold}",
                color="yellow",
            )
            for i, k in enumerate(
                dataset_val.identifiers
            ):  # enumerate(['tomo_4c1ca8']): #
                case_start = time()
                proceed = not check_workers_alive_and_busy(
                    export_pool, worker_list, results, allowed_num_queued=2
                )
                while not proceed:
                    sleep(0.1)
                    proceed = not check_workers_alive_and_busy(
                        export_pool, worker_list, results, allowed_num_queued=2
                    )

                case_idx = i + 1
                self._log_step(
                    "CASE",
                    f"{case_idx:03d}/{total_cases:03d} -> predicting {k}",
                    color="magenta",
                )
                data, _, seg_prev, properties = dataset_val.load_case(k)

                # we do [:] to convert blosc2 to numpy
                t_load = time()
                data = data[:]
                data = torch.from_numpy(data)
                self._log_step(
                    "LOAD",
                    f"{k} data ready {tuple(data.shape)} in {np.round(time() - t_load, 2)}s",
                    color="blue",
                )

                if self.is_cascaded:
                    raise NotImplementedError

                self._log_step(
                    "SHAPE",
                    f"{k}, tensor {tuple(data.shape)}, rank {self.local_rank}",
                    color="yellow",
                )

                # predict logits
                t_pred = time()
                with torch.no_grad():
                    prediction = predictor.predict_sliding_window_return_logits(data)
                    empty_cache(self.device)
                    prediction = F.sigmoid(prediction).float()
                    empty_cache(self.device)
                self._log_step(
                    "PRED",
                    f"{k} sliding-window logits -> probs in {np.round(time() - t_pred, 2)}s",
                    color="green",
                )

                # max value per C
                max_per_c = torch.amax(prediction, dim=(1, 2, 3))
                stats = ", ".join(
                    [f"c{ci+1}:{v:.3f}" for ci, v in enumerate(max_per_c.tolist())]
                )
                self._log_step("STAT", f"{k} max per class | {stats}", color="cyan")

                csv_entries.append((k, *max_per_c.numpy()))

                labels = list(self.dataset_json["labels"].keys())[1:] + [
                    "Aneurysm Present"
                ]  # remove background
                # Add a column for "Aneurysm Present" based on whether any aneurysm is detected
                submission_df = pd.DataFrame(
                    csv_entries, columns=["SeriesInstanceUID"] + labels
                )

                if self.output_folder_base is not None:
                    submission_df.to_csv(
                        join(self.output_folder_base, f"val_{self.fold}.csv"),
                        index=False,
                    )
                    self._log_step(
                        "SAVE",
                        f"val_{self.fold}.csv updated | rows={len(csv_entries)}",
                        color="magenta",
                    )

                self._log_step(
                    "DONE",
                    f"{k} completed in {np.round(time() - case_start, 2)}s",
                    color="green",
                )

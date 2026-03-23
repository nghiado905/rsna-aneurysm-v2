import argparse
import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nnunetv2.inference.export_prediction import (
    convert_predicted_logits_to_segmentation_with_correct_shape,
)
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def _load_local_official_module():
    """
    Always use crop pipeline from:
    rsna-aneurysm-v1/dataset_conversion/kaggle_2025_rsna/official_data_to_nnunet.py
    """
    here = Path(__file__).resolve()
    official_path = (
        here.parents[2]
        / "dataset_conversion"
        / "kaggle_2025_rsna"
        / "official_data_to_nnunet.py"
    )
    if not official_path.exists():
        raise FileNotFoundError(f"Cannot find local official_data_to_nnunet.py at {official_path}")

    spec = importlib.util.spec_from_file_location("rsna2025_local_official_data_to_nnunet", official_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import module from {official_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "load_and_crop"):
        raise AttributeError(f"Module {official_path} has no load_and_crop")
    print(f"[INFO] Using official pipeline from: {official_path}")
    return module


official_module = _load_local_official_module()
load_and_crop = official_module.load_and_crop


def extract_aneurysm_coordinates(
    probs: torch.Tensor, label_names, threshold: float = 0.5
):
    """
    Extract coordinate summary per aneurysm label from probability maps.
    Returns one record per label with:
    - max voxel coord (z,y,x)
    - thresholded component centroid + bbox if exists
    """
    recs = []
    n_labels = min(len(label_names), int(probs.shape[0]))
    for c in range(n_labels):
        p = probs[c].detach().cpu().numpy()
        max_prob = float(np.max(p))
        max_idx = np.unravel_index(np.argmax(p), p.shape)
        z_peak, y_peak, x_peak = [int(v) for v in max_idx]

        fg = p >= threshold
        if np.any(fg):
            coords = np.argwhere(fg)  # (N, 3) = (z, y, x)
            zmin, ymin, xmin = coords.min(axis=0).tolist()
            zmax, ymax, xmax = coords.max(axis=0).tolist()
            weights = p[fg]
            # weighted centroid on thresholded voxels
            wz, wy, wx = (coords * weights[:, None]).sum(axis=0) / np.sum(weights)
            cz, cy, cx = int(round(float(wz))), int(round(float(wy))), int(round(float(wx)))
            voxel_count = int(coords.shape[0])
        else:
            zmin = ymin = xmin = zmax = ymax = xmax = -1
            cz, cy, cx = z_peak, y_peak, x_peak
            voxel_count = 0

        recs.append(
            {
                "label": label_names[c],
                "max_prob": max_prob,
                "peak_z": z_peak,
                "peak_y": y_peak,
                "peak_x": x_peak,
                "coord_z": cz,
                "coord_y": cy,
                "coord_x": cx,
                "bbox_zmin": int(zmin),
                "bbox_zmax": int(zmax),
                "bbox_ymin": int(ymin),
                "bbox_ymax": int(ymax),
                "bbox_xmin": int(xmin),
                "bbox_xmax": int(xmax),
                "voxel_count_above_thr": voxel_count,
                "coord_threshold": float(threshold),
            }
        )
    return recs


def _to_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return float(default)


def extract_aneurysm_coordinates_with_prefix(
    probs: np.ndarray, label_names, threshold: float, prefix: str
):
    recs = []
    n_labels = min(len(label_names), int(probs.shape[0]))
    for c in range(n_labels):
        p = probs[c]
        max_prob = float(np.max(p))
        max_idx = np.unravel_index(np.argmax(p), p.shape)
        z_peak, y_peak, x_peak = [int(v) for v in max_idx]

        fg = p >= threshold
        if np.any(fg):
            coords = np.argwhere(fg)
            zmin, ymin, xmin = coords.min(axis=0).tolist()
            zmax, ymax, xmax = coords.max(axis=0).tolist()
            weights = p[fg]
            wz, wy, wx = (coords * weights[:, None]).sum(axis=0) / np.sum(weights)
            cz, cy, cx = int(round(float(wz))), int(round(float(wy))), int(round(float(wx)))
            voxel_count = int(coords.shape[0])
        else:
            zmin = ymin = xmin = zmax = ymax = xmax = -1
            cz, cy, cx = z_peak, y_peak, x_peak
            voxel_count = 0

        recs.append(
            {
                "label": label_names[c],
                "max_prob": max_prob,
                f"{prefix}_peak_z": z_peak,
                f"{prefix}_peak_y": y_peak,
                f"{prefix}_peak_x": x_peak,
                f"{prefix}_coord_z": cz,
                f"{prefix}_coord_y": cy,
                f"{prefix}_coord_x": cx,
                f"{prefix}_bbox_zmin": int(zmin),
                f"{prefix}_bbox_zmax": int(zmax),
                f"{prefix}_bbox_ymin": int(ymin),
                f"{prefix}_bbox_ymax": int(ymax),
                f"{prefix}_bbox_xmin": int(xmin),
                f"{prefix}_bbox_xmax": int(xmax),
                f"{prefix}_voxel_count_above_thr": voxel_count,
            }
        )
    return recs


def merge_record_dicts(base_recs, extra_recs):
    by_label = {r["label"]: dict(r) for r in base_recs}
    for r in extra_recs:
        lb = r["label"]
        if lb not in by_label:
            by_label[lb] = dict(r)
        else:
            by_label[lb].update(r)
    return [by_label[k] for k in by_label.keys()]


def undo_to_original_voxel_and_mm(
    rec: dict,
    crop_shape_zyx: tuple,
    crop_bbox_zyx,
    spacing_zyx,
    origin_xyz,
    direction_flat_xyz,
):
    """
    Undo path:
    1) preprocessed probs are converted back to cropped+flip space via
       convert_predicted_logits_to_segmentation_with_correct_shape(...)
    2) undo flip on axis=1 (Y)
    3) undo Stage-1 crop by adding crop bbox offsets
    4) convert voxel (orig) to world mm (xyz)
    """
    z_cf = int(round(_to_float(rec.get("cropflip_coord_z", 0))))
    y_cf = int(round(_to_float(rec.get("cropflip_coord_y", 0))))
    x_cf = int(round(_to_float(rec.get("cropflip_coord_x", 0))))

    # Step-1 already done outside (we are in cropped+flip space).
    # Step-2 undo flip Y (axis=1).
    y_crop = int((crop_shape_zyx[1] - 1) - y_cf)
    z_crop = z_cf
    x_crop = x_cf

    # Step-3 undo Stage-1 crop bbox.
    z0, _, y0, _, x0, _ = [int(v) for v in crop_bbox_zyx]
    z_orig = int(z_crop + z0)
    y_orig = int(y_crop + y0)
    x_orig = int(x_crop + x0)

    # Step-4 voxel(orig zyx) -> world mm xyz
    # index for SITK-style world transform is (x,y,z) in voxel.
    spacing_xyz = np.array([spacing_zyx[2], spacing_zyx[1], spacing_zyx[0]], dtype=np.float64)
    idx_xyz = np.array([x_orig, y_orig, z_orig], dtype=np.float64)
    origin_xyz = np.array(origin_xyz, dtype=np.float64)
    direction = np.array(direction_flat_xyz, dtype=np.float64).reshape(3, 3)
    world_xyz = origin_xyz + direction.dot(idx_xyz * spacing_xyz)

    # Explicit crop-space coordinates after undoing the manual Y flip.
    rec["coord_z_crop"] = z_crop
    rec["coord_y_crop"] = y_crop
    rec["coord_x_crop"] = x_crop
    # Backward-compatible alias kept for older downstream scripts.
    rec["coord_z_crop_unflip"] = z_crop
    rec["coord_y_crop_unflip"] = y_crop
    rec["coord_x_crop_unflip"] = x_crop
    rec["coord_z_orig"] = z_orig
    rec["coord_y_orig"] = y_orig
    rec["coord_x_orig"] = x_orig
    rec["coord_world_x_mm"] = float(world_xyz[0])
    rec["coord_world_y_mm"] = float(world_xyz[1])
    rec["coord_world_z_mm"] = float(world_xyz[2])
    rec["crop_bbox_z0"] = int(z0)
    rec["crop_bbox_z1"] = int(crop_bbox_zyx[1])
    rec["crop_bbox_y0"] = int(y0)
    rec["crop_bbox_y1"] = int(crop_bbox_zyx[3])
    rec["crop_bbox_x0"] = int(x0)
    rec["crop_bbox_x1"] = int(crop_bbox_zyx[5])
    rec["crop_shape_z"] = int(crop_shape_zyx[0])
    rec["crop_shape_y"] = int(crop_shape_zyx[1])
    rec["crop_shape_x"] = int(crop_shape_zyx[2])
    return rec


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
        help="Where to store the resulting csv, e.g. output.csv (if a directory is given, submission.csv is created inside it)",
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
    p.add_argument(
        "--missing-channel-mode",
        type=str,
        default="repeat",
        choices=("repeat", "zeros"),
        help="How to fill missing input channels when model expects more channels than provided.",
    )
    p.add_argument(
        "--coords-output",
        type=Path,
        default=None,
        help="Optional path for coordinate CSV. Default: <output_path_stem>_coords.csv",
    )
    p.add_argument(
        "--coord-threshold",
        type=float,
        default=0.5,
        help="Threshold for extracting aneurysm coordinate centroid/bbox from probability map.",
    )
    p.add_argument(
        "--ids-mapping-json",
        type=Path,
        default=None,
        help="Optional ids_mapping.json (SeriesInstanceUID -> iarsna_xxxx) to add case_id column.",
    )
    p.add_argument(
        "--stage1_model_dir",
        type=Path,
        required=True,
        help="Stage-1 model dir. Required: inference crop must use Stage-1 bbox.",
    )
    p.add_argument(
        "--stage1_checkpoint",
        type=str,
        default="checkpoint_final.pth",
        help="Stage-1 checkpoint filename for crop predictor.",
    )
    p.add_argument(
        "--stage1_fold",
        type=int,
        default=0,
        help="Stage-1 fold index for crop predictor.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    if args.output_path.suffix.lower() != ".csv":
        args.output_path.mkdir(parents=True, exist_ok=True)
        args.output_path = args.output_path / "submission.csv"
        print(f"Output path is a directory; writing CSV to: {args.output_path}")
    if args.coords_output is None:
        args.coords_output = args.output_path.with_name(args.output_path.stem + "_coords.csv")
    series_to_case = {}
    if args.ids_mapping_json is not None and args.ids_mapping_json.exists():
        with open(args.ids_mapping_json, "r", encoding="utf-8") as f:
            series_to_case = {str(k): str(v) for k, v in json.load(f).items()}
        print(f"Loaded ids mapping: {args.ids_mapping_json} ({len(series_to_case)} entries)")
    elif args.ids_mapping_json is not None:
        print(f"[WARN] ids mapping not found: {args.ids_mapping_json}")

    if not args.stage1_model_dir.exists():
        raise FileNotFoundError(f"Stage-1 model dir not found: {args.stage1_model_dir}")
    if not hasattr(official_module, "init_stage1_predictor"):
        raise AttributeError("Local official_data_to_nnunet.py has no init_stage1_predictor")
    print(
        f"[INFO] Init Stage-1 crop predictor | dir={args.stage1_model_dir} "
        f"ckpt={args.stage1_checkpoint} fold={args.stage1_fold}"
    )
    stage1_predictor = official_module.init_stage1_predictor(
        model_training_output_dir=args.stage1_model_dir,
        checkpoint_name=args.stage1_checkpoint,
        fold=args.stage1_fold,
    )
    print("[INFO] Stage-1 predictor ready.")

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
    expected_channels = len(predictor.dataset_json.get("channel_names", {}))
    if expected_channels <= 0:
        expected_channels = 1

    labels = (
        ["SeriesInstanceUID"]
        + list(predictor.dataset_json["labels"].keys())[1:]
        + ["Aneurysm Present"]
    )
    existing_ids = set()
    res = []
    coord_rows = []
    if args.output_path.exists():
        try:
            df_prev = pd.read_csv(args.output_path)
            if "SeriesInstanceUID" in df_prev.columns:
                existing_ids = set(df_prev["SeriesInstanceUID"].astype(str))
                res = df_prev.values.tolist()
                print(f"Skipping {len(existing_ids)} already processed cases from {args.output_path}")
        except Exception as e:
            print(f"Could not load existing results ({e}); starting fresh.")
    if args.coords_output.exists():
        try:
            coord_prev = pd.read_csv(args.coords_output)
            if len(coord_prev) > 0 and "SeriesInstanceUID" in coord_prev.columns:
                coord_rows = coord_prev.to_dict("records")
                print(f"Loaded existing coordinate rows: {len(coord_rows)} from {args.coords_output}")
        except Exception as e:
            print(f"Could not load existing coords ({e}); starting fresh.")

    series_dirs = [p for p in args.input_dir.iterdir() if p.is_dir()]
    skipped_non_dirs = [p for p in args.input_dir.iterdir() if not p.is_dir()]
    if skipped_non_dirs:
        print(f"[INFO] Skipping {len(skipped_non_dirs)} non-directory entries in input dir.", flush=True)
        for p in skipped_non_dirs[:20]:
            print(f"[SKIP] Not a series directory: {p}", flush=True)

    print(f"[INFO] Series directories to process: {len(series_dirs)}", flush=True)

    for series_dir in tqdm(series_dirs):
        if series_dir.name in existing_ids:
            print(f"Already exists, skipping: {series_dir.name}")
            continue
        try:
            img, properties = load_and_crop(
                series_dir,
                stage1_predictor=stage1_predictor,
                case_id=series_dir.name,
            )
        except Exception as e:
            print(f"[ERROR] Failed to load/crop series {series_dir.name}: {e}", flush=True)
            continue
        crop_bbox = properties.get("crop_bbox_zyx", None)
        crop_source = properties.get("crop_source", "unknown")
        print(f"[CROP] {series_dir.name} | source={crop_source} | bbox_zyx={crop_bbox}")
        img = np.flip(img, 1)
        input_data = np.array([img], dtype=np.float32)
        if expected_channels > input_data.shape[0]:
            n_missing = expected_channels - input_data.shape[0]
            if args.missing_channel_mode == "repeat":
                extra = np.repeat(input_data[:1], n_missing, axis=0)
            else:
                extra = np.zeros((n_missing,) + input_data.shape[1:], dtype=input_data.dtype)
            input_data = np.concatenate([input_data, extra], axis=0)
            print(
                f"[WARN] {series_dir.name}: model expects {expected_channels} channels, "
                f"input has 1; filled {n_missing} channel(s) with mode={args.missing_channel_mode}."
            )
        data, _, _ = preprocessor.run_case_npy(
            input_data,
            None,
            properties,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.dataset_json,
        )
        logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()
        probs = torch.sigmoid(logits)

        max_per_c = torch.amax(probs, dim=(1, 2, 3)).to(
            dtype=torch.float32, device="cpu"
        )

        row = [series_dir.name] + max_per_c.numpy().tolist()
        res.append(row)
        aneurysm_label_names = list(predictor.dataset_json["labels"].keys())[1:-1]
        # Coordinates in preprocessed space (after preprocessor crop+resample)
        coord_recs = extract_aneurysm_coordinates(probs, aneurysm_label_names, threshold=args.coord_threshold)

        # Convert logits back to cropped+flip image space (undo preprocessor resample+crop, but not our manual flip/stage1 crop)
        _, probs_cropflip = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits,
            predictor.plans_manager,
            predictor.configuration_manager,
            predictor.label_manager,
            properties,
            return_probabilities=True,
        )
        coord_recs_cropflip = extract_aneurysm_coordinates_with_prefix(
            probs_cropflip,
            aneurysm_label_names,
            threshold=args.coord_threshold,
            prefix="cropflip",
        )
        coord_recs = merge_record_dicts(coord_recs, coord_recs_cropflip)

        crop_bbox = properties.get("crop_bbox_zyx", [0, img.shape[0], 0, img.shape[1], 0, img.shape[2]])
        spacing_zyx = properties.get("spacing", [1.0, 1.0, 1.0])
        origin_xyz = properties.get("origin", [0.0, 0.0, 0.0])
        direction_flat_xyz = properties.get("direction", [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
        case_id = series_to_case.get(series_dir.name, "")
        for r in coord_recs:
            r = undo_to_original_voxel_and_mm(
                r,
                crop_shape_zyx=tuple(img.shape),
                crop_bbox_zyx=crop_bbox,
                spacing_zyx=spacing_zyx,
                origin_xyz=origin_xyz,
                direction_flat_xyz=direction_flat_xyz,
            )
            r["SeriesInstanceUID"] = series_dir.name
            r["case_id"] = case_id
            coord_rows.append(r)

        pd.DataFrame(res, columns=labels).to_csv(args.output_path, index=False)
        pd.DataFrame(coord_rows).to_csv(args.coords_output, index=False)

    print(f"Results saved to {args.output_path}")
    print(f"Coordinate table saved to {args.coords_output}")


if __name__ == "__main__":
    main()
    # python inference.py \
    # -i /path/to/rsna-intracranial-aneurysm-detection/series \
    # -o output.csv \
    # -m /path/to/downloaded-checkpoint/Dataset004_iarsna_crop/Kaggle2025RSNATrainer__nnUNetResEncUNetMPlans__3d_fullres_bs32 \
    # -c checkpoint_epoch_1500.pth \
    # --fold "('all',)"
    # --disable_tta
    
    
    
    
    # import argparse
# import ast
# from pathlib import Path
# import os
# import sys

# import numpy as np
# import pandas as pd
# import torch
# from tqdm import tqdm
# import matplotlib.pyplot as plt

# from nnunetv2.dataset_conversion.kaggle_2025_rsna.official_data_to_nnunet import (
#     load_and_crop,
# )
# from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# # --- 1. CẤU HÌNH VẼ ẢNH (MIP) ---
# def window_hu(vol, level=400.0, width=700.0):
#     """Windowing chuẩn cho mạch máu (CTA)."""
#     low = level - width / 2.0
#     high = level + width / 2.0
#     vol = np.clip(vol, low, high)
#     vol = (vol - low) / (high - low)
#     return vol

# def save_mip_viz(volume, coords, label_name, prob, output_path):
#     """
#     Vẽ MIP từ chính dữ liệu model đã nhìn thấy.
#     volume: (Z, Y, X)
#     coords: (z, y, x)
#     """
#     try:
#         # Windowing
#         vol = window_hu(volume)
#         z_peak, y_peak, x_peak = coords
#         d, h, w = vol.shape
#         slab = 10 

#         # --- TẠO 3 GÓC CHIẾU ---
        
#         # 1. AXIAL (Chiếu trục Z - Đỉnh đầu xuống)
#         z_s, z_e = max(0, int(z_peak-slab)), min(d, int(z_peak+slab))
#         if z_e <= z_s: z_e = z_s + 1
#         # Max theo trục 0 (Z) -> Ảnh còn lại (Y, X)
#         mip_ax = np.max(vol[z_s:z_e, :, :], axis=0) 
        
#         # 2. CORONAL (Chiếu trục Y - Trước sau)
#         y_s, y_e = max(0, int(y_peak-slab)), min(h, int(y_peak+slab))
#         if y_e <= y_s: y_e = y_s + 1
#         # Max theo trục 1 (Y) -> Ảnh còn lại (Z, X)
#         mip_cor = np.max(vol[:, y_s:y_e, :], axis=1)
        
#         # 3. SAGITTAL (Chiếu trục X - Trái phải)
#         x_s, x_e = max(0, int(x_peak-slab)), min(w, int(x_peak+slab))
#         if x_e <= x_s: x_e = x_s + 1
#         # Max theo trục 2 (X) -> Ảnh còn lại (Z, Y)
#         mip_sag = np.max(vol[:, :, x_s:x_e], axis=2)

#         # --- VẼ ---
#         fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='black')
#         fig.suptitle(f"{label_name} | P={prob:.4f}", color='#00FF00', fontsize=16, fontweight='bold')

#         # HÀM VẼ CON
#         def draw(ax, img, x_dot, y_dot, title):
#             # Dùng origin='lower' để toạ độ (0,0) nằm góc dưới bên trái
#             # Đây là mấu chốt để không bị lệch trục Y
#             ax.imshow(img, cmap='gray', origin='lower', vmin=0, vmax=1)
            
#             # Vẽ điểm
#             ax.scatter(x_dot, y_dot, c='red', s=100, marker='x', linewidth=2)
            
#             # Vẽ heatmap (Gaussian blob)
#             rows, cols = img.shape
#             yy, xx = np.ogrid[:rows, :cols]
#             dist = (xx - x_dot)**2 + (yy - y_dot)**2
#             blob = np.exp(-dist / (2 * 10**2))
#             ax.imshow(blob, cmap='hot', alpha=blob*0.6, origin='lower')
            
#             ax.set_title(title, color='white')
#             ax.axis('off')

#         # 1. AXIAL: Ảnh (Y, X). Matplotlib vẽ (col, row) tức là (X, Y)
#         # Điểm vẽ: x=x_peak, y=y_peak
#         draw(axes[0], mip_ax, x_peak, y_peak, f"Axial (Z={z_peak})")

#         # 2. CORONAL: Ảnh (Z, X). Matplotlib vẽ (X, Z)
#         # Điểm vẽ: x=x_peak, y=z_peak
#         draw(axes[1], mip_cor, x_peak, z_peak, f"Coronal (Y={y_peak})")

#         # 3. SAGITTAL: Ảnh (Z, Y). Matplotlib vẽ (Y, Z)
#         # Điểm vẽ: x=y_peak, y=z_peak
#         draw(axes[2], mip_sag, y_peak, z_peak, f"Sagittal (X={x_peak})")

#         plt.tight_layout()
#         plt.savefig(output_path, dpi=100, facecolor='black')
#         plt.close(fig)

#     except Exception as e:
#         print(f"Viz Error: {e}")

# # --- 2. PARSE ARGS (Giữ nguyên) ---
# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("-i", "--input-dir", type=Path, required=True, help="Input dir")
#     p.add_argument("-o", "--output-path", type=Path, required=True, help="Output csv")
#     p.add_argument("-m", "--model_folder", type=Path, required=True, help="Model path")
#     p.add_argument("-c", "--chk", type=str, required=True, help="Checkpoint")
#     p.add_argument("--fold", type=ast.literal_eval, help="Fold")
#     p.add_argument("--step_size", type=float, default=0.5)
#     p.add_argument("--disable_tta", action="store_true", default=False)
#     p.add_argument("--use_gaussian", action="store_true", default=False)
#     p.add_argument("--device", type=str, default="cuda")
    
#     # Thêm ngưỡng vẽ để đỡ spam ảnh rác
#     p.add_argument("--viz_threshold", type=float, default=0.2, help="Threshold to save MIP")
    
#     return p.parse_args()

# # --- 3. MAIN FUNCTION ---
# def main():
#     args = parse_args()

#     # Tạo folder ảnh (Cùng cấp với file output csv)
#     args.output_path.parent.mkdir(parents=True, exist_ok=True)
#     mip_dir = args.output_path.parent / (args.output_path.stem + "_mips")
#     mip_dir.mkdir(exist_ok=True, parents=True)

#     device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#     predictor = nnUNetPredictor(
#         tile_step_size=args.step_size,
#         use_gaussian=args.use_gaussian,
#         use_mirroring=not args.disable_tta,
#         device=device,
#         verbose=False, verbose_preprocessing=False, allow_tqdm=False,
#     )
#     predictor.initialize_from_trained_model_folder(
#         args.model_folder,
#         [i if i == "all" else int(i) for i in args.fold],
#         checkpoint_name=args.chk,
#     )

#     preprocessor = predictor.configuration_manager.preprocessor_class()

#     # Lấy tên labels (bỏ background)
#     labels_dict = predictor.dataset_json["labels"]
#     # Header chuẩn của bạn
#     labels = ["SeriesInstanceUID"] + list(labels_dict.keys())[1:] + ["Aneurysm Present"]
    
#     # Map index -> tên để vẽ ảnh
#     idx_to_name = {v: k for k, v in labels_dict.items() if v != 0}

#     # === LOGIC WHITELIST (GIỮ NGUYÊN TỪ YÊU CẦU TRƯỚC) ===
#     series_list = list(args.input_dir.iterdir())
#     submission_path = Path("output1/submission.csv")
#     if submission_path.exists():
#         try:
#             df_sub = pd.read_csv(submission_path)
#             whitelist = set(df_sub["SeriesInstanceUID"].astype(str).str.strip())
#             series_list = [s for s in series_list if s.name in whitelist]
#             print(f"Filtered to {len(series_list)} cases from submission.csv")
#         except: pass
    
#     # CSV Header mở rộng để lưu thêm thông tin tọa độ (nếu muốn debug)
#     # Nhưng để khớp với code của bạn, ta chỉ lưu file chính đúng format, tọa độ chỉ dùng để vẽ.
    
#     # Mở file CSV để ghi header trước (Real-time saving)
#     if not args.output_path.exists():
#         pd.DataFrame(columns=labels).to_csv(args.output_path, index=False)
#         processed_ids = set()
#     else:
#         # Resume logic
#         try:
#             processed_ids = set(pd.read_csv(args.output_path)["SeriesInstanceUID"].astype(str))
#         except:
#             processed_ids = set()

#     print(f"Start Processing {len(series_list)} cases...")

#     for series_dir in tqdm(series_list):
#         if series_dir.name in processed_ids:
#             continue

#         try:
#             # --- CODE CHUẨN CỦA BẠN (GIỮ NGUYÊN) ---
#             img, properties = load_and_crop(series_dir)
#             img = np.flip(img, 1) # <--- QUAN TRỌNG: Ảnh đã bị lật ở đây
            
#             data, _, _ = preprocessor.run_case_npy(
#                 np.array([img]),
#                 None,
#                 properties,
#                 predictor.plans_manager,
#                 predictor.configuration_manager,
#                 predictor.dataset_json,
#             )
#             logits = predictor.predict_logits_from_preprocessed_data(
#                 torch.from_numpy(data)
#             ).cpu()
#             probs = torch.sigmoid(logits)

#             max_per_c = torch.amax(probs, dim=(1, 2, 3)).to(
#                 dtype=torch.float32, device="cpu"
#             )
#             # ----------------------------------------

#             # --- PHẦN BỔ SUNG: VẼ VÀ LƯU ---
            
#             # 1. Lưu CSV ngay lập tức (Append mode)
#             res_row = [series_dir.name] + max_per_c.numpy().tolist()
#             pd.DataFrame([res_row], columns=labels).to_csv(args.output_path, mode='a', header=False, index=False)

#             # 2. Tính toán để vẽ (chỉ vẽ nếu xác suất cao)
#             # max_per_c bao gồm cả background (index 0) nếu model output có bg
#             # Thường output nnunet shape [C, Z, Y, X].
            
#             probs_np = probs.numpy() # (C, Z, Y, X)
#             fg_probs = max_per_c[1:] # Bỏ background
#             best_prob = torch.max(fg_probs).item()
            
#             if best_prob > args.viz_threshold:
#                 # Tìm vị trí (z, y, x) của điểm max
#                 best_cls_idx = torch.argmax(fg_probs).item() + 1 # +1 vì bỏ bg
#                 label_name = idx_to_name.get(best_cls_idx, "Unknown")
                
#                 # Lấy bản đồ xác suất của class đó
#                 prob_map = probs_np[best_cls_idx]
                
#                 # Tìm tọa độ đỉnh
#                 # Lưu ý: prob_map khớp với img ĐÃ FLIP (vì img -> predict -> prob_map)
#                 peak_idx = np.argmax(prob_map)
#                 z, y, x = np.unravel_index(peak_idx, prob_map.shape)
                
#                 # Vẽ ảnh
#                 # Lưu ý: Truyền vào `img` (đã flip ở dòng img = np.flip(img, 1))
#                 # thì tọa độ (z, y, x) sẽ khớp hoàn toàn.
#                 safe_name = label_name.replace(" ", "_")
#                 png_name = f"{series_dir.name}_{safe_name}_p{best_prob:.2f}.png"
#                 save_mip_viz(img, (z, y, x), label_name, best_prob, mip_dir / png_name)

#         except Exception as e:
#             print(f"Error {series_dir.name}: {e}")
#             continue

#     print(f"Done. CSV: {args.output_path}")

# if __name__ == "__main__":
#     main()

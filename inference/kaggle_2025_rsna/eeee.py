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
    existing_ids = set()
    res = []
    if args.output_path.exists():
        try:
            df_prev = pd.read_csv(args.output_path)
            if "SeriesInstanceUID" in df_prev.columns:
                existing_ids = set(df_prev["SeriesInstanceUID"].astype(str))
                res = df_prev.values.tolist()
                print(f"Skipping {len(existing_ids)} already processed cases from {args.output_path}")
        except Exception as e:
            print(f"Could not load existing results ({e}); starting fresh.")

    for series_dir in tqdm(list(args.input_dir.iterdir())):
        if series_dir.name in existing_ids:
            print(f"Already exists, skipping: {series_dir.name}")
            continue
        img, properties = load_and_crop(series_dir)
        img = np.flip(img, 1)
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
        print(max_per_c)
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

import argparse
import tempfile
import sys
import os
import shutil
import logging
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.filters import sato
from scipy.ndimage import distance_transform_edt # Nhanh hơn Sato nhiều

# ================= IMPORT NNUNET =================
try:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
except ImportError:
    print("❌ Thiếu thư viện nnunetv2.")
    sys.exit(1)

# ================= LOGGING =================
def setup_logger():
    logger = logging.getLogger("Inference_2Channel")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(handler)
    return logger

logger = setup_logger()

# ================= HÀM XỬ LÝ PHỤ TRỢ =================

def convert_dicom_to_nifti(dicom_dir, out_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not dicom_names:
        raise ValueError(f"Không tìm thấy DICOM trong {dicom_dir}")
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, str(out_path))

def get_predictor(model_root, chk, fold):
    """Load model an toàn"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    predictor.initialize_from_trained_model_folder(
        str(model_root),
        use_folds=(fold,),
        checkpoint_name=chk
    )
    return predictor

# ================= PIPELINE CHÍNH =================

def process_case_optimized(uid, input_path, output_csv, vessel_predictor, aneurysm_predictor, temp_root):
    # 1. Chuẩn bị file NIfTI
    raw_nii = temp_root / f"{uid}.nii.gz"
    
    if input_path.is_dir():
        convert_dicom_to_nifti(input_path, raw_nii)
    else:
        # Nếu đã là nifti thì dùng luôn, không cần copy nếu muốn tiết kiệm IO
        # Nhưng để an toàn type, ta đọc trực tiếp
        raw_nii = input_path

    # 2. Đọc ảnh gốc
    img_itk = sitk.ReadImage(str(raw_nii))
    img_arr = sitk.GetArrayFromImage(img_itk).astype(np.float32) # (Z, Y, X)
    spacing = img_itk.GetSpacing() # (X, Y, Z)
    # nnU-Net cần properties chứa spacing theo thứ tự (Z, Y, X) nếu dùng API low-level,
    # nhưng dùng predict_single_npy_array thì chỉ cần truyền list spacing đúng chiều image.
    
    # Properties cho nnU-Net (quan trọng để nó restore spacing)
    props = {
        'spacing': spacing[::-1], # Sitk (X,Y,Z) -> Numpy (Z,Y,X)
        'sitk_stuff': None
    }

    # ================= STAGE 1: VESSEL SEGMENTATION =================
    # Input cho nnU-Net: (C, Z, Y, X). Với vessel model 1 kênh -> (1, Z, Y, X)
    data_1ch = img_arr[np.newaxis, ...] 
    
    # Infer Vessel
    # ret: (NumClass, Z, Y, X) -> Lấy argmax luôn -> (Z, Y, X)
    vessel_pred_logits = vessel_predictor.predict_single_npy_array(
        data_1ch, props, None, None, False
    )
    # vessel_pred_logits trả về numpy array (NumClasses, Z, Y, X)
    vessel_mask = np.argmax(vessel_pred_logits, axis=0).astype(np.uint8)

    # ================= FEATURE ENGINEERING (CHANNEL 2) =================
    # Quan trọng: Logic này PHẢI GIỐNG HỆT lúc bạn train aneurysm model
    # Nếu lúc train bạn dùng Sato, thì ở đây dùng Sato.
    # Nếu lúc train bạn dùng Distance Map (như mình khuyên), thì phải đổi sang Distance Map.
    
    # Cách A: SATO (Như code cũ của bạn) - Chậm
    # vesselness = sato(img_arr, sigmas=range(1, 7), black_ridges=False)
    # vesselness = np.clip(vesselness, 0, 1)
    # channel1 = vesselness * (vessel_mask > 0)
    
    # Cách B: DISTANCE MAP (Khuyên dùng nếu bạn đã train lại theo cách này) - Nhanh
    if np.sum(vessel_mask) > 0:
        dist_map = distance_transform_edt(vessel_mask > 0)
        dist_map = dist_map / (dist_map.max() + 1e-8)
        channel1 = dist_map * 200.0 # Scale giống lúc train
    else:
        channel1 = np.zeros_like(img_arr)

    # ================= STAGE 2: ANEURYSM DETECTION =================
    # Input cho model 2 kênh: (2, Z, Y, X)
    # Kênh 0: Ảnh gốc
    # Kênh 1: Feature (Vesselness/DistanceMap)
    data_2ch = np.stack([img_arr, channel1], axis=0)

    # Infer Aneurysm
    aneurysm_logits = aneurysm_predictor.predict_single_npy_array(
        data_2ch, props, None, None, False
    )
    # aneurysm_logits: (NumClasses, Z, Y, X)
    
    # ================= POST PROCESSING =================
    # Lấy xác suất lớp 1 (Aneurysm)
    # Giả sử binary segmentation (0: bg, 1: aneurysm)
    prob_map = torch.sigmoid(torch.from_numpy(aneurysm_logits[1])) # (Z, Y, X)
    
    # Lấy max probability của cả volume làm score cho case này
    max_prob = float(prob_map.max())
    aneurysm_present = 1 if max_prob > 0.5 else 0 # Threshold 0.5

    # Lưu kết quả
    res_row = [uid, max_prob, aneurysm_present]
    labels = ["SeriesInstanceUID", "Probability", "Aneurysm_Present"]
    
    pd.DataFrame([res_row], columns=labels).to_csv(output_csv, mode='a', header=not output_csv.exists(), index=False)
    
    logger.info(f"Done {uid}: Max Prob = {max_prob:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=Path)
    parser.add_argument("-o", "--output-csv", type=Path, default=Path(r"E:\output_aneurysm.csv"))
    
    # Paths config
    parser.add_argument("--vessel-model", type=Path, required=True)
    parser.add_argument("--aneurysm-model", type=Path, required=True)
    
    args = parser.parse_args()

    # Lấy danh sách file
    if args.input.is_file():
        targets = [args.input]
    else:
        targets = sorted(list(args.input.glob("*.nii.gz")) + list(args.input.glob("*.nii")))

    # 1. Load Vessel Model TRƯỚC
    logger.info("⚙️ Loading Vessel Model...")
    vessel_predictor = get_predictor(args.vessel_model, "checkpoint_final.pth", "all") # Hoặc fold cụ thể
    
    # 2. Load Aneurysm Model SAU (Giữ cả 2 nếu VRAM > 12GB, nếu không phải load/unload)
    # Để an toàn nhất: Load cả 2 nếu chạy máy mạnh. Nếu máy yếu, phải viết lại logic loop.
    # Ở đây giả định máy có GPU ổn (RTX 3060 trở lên)
    logger.info("⚙️ Loading Aneurysm Model...")
    aneurysm_predictor = get_predictor(args.aneurysm_model, "checkpoint_final.pth", "all")

    with tempfile.TemporaryDirectory() as temp_dir:
        for target in tqdm(targets):
            uid = target.name.replace(".nii.gz", "").replace(".nii", "")
            try:
                process_case_optimized(
                    uid, target, args.output_csv, 
                    vessel_predictor, aneurysm_predictor, Path(temp_dir)
                )
            except Exception as e:
                logger.error(f"Error processing {uid}: {e}")
                # Clear cache nếu lỗi OOM
                torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
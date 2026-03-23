import argparse
import ast
import subprocess
import tempfile
import sys
import os
import shutil
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import SimpleITK as sitk
from tqdm import tqdm

# ==============================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (có thể override qua CLI)
# ==============================================================================
DEFAULT_PATH_NNUNET_RESULTS_SEG = r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\TopCoWSubmissions\nnUNet\model"
DEFAULT_SEG_DATASET_NAME = "Dataset113_CTMulSegWholeData"
DEFAULT_SEG_CHK_POINT = "checkpoint_final.pth"
DEFAULT_SEG_FOLD = "4"
DEFAULT_OVERLAY_BOOST_VAL = 200
DEFAULT_CUSTOM_TEMP_DIR = Path(r"E:\temp")

# ==============================================================================
# 2. HỆ THỐNG LOGGING
# ==============================================================================
def setup_logger(log_file="pipeline_run.log"):
    logger = logging.getLogger("RSNA_Pipeline")
    logger.setLevel(logging.DEBUG)
    
    if os.path.exists(log_file): os.remove(log_file)
    fh = logging.FileHandler(log_file, encoding='utf-8'); fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    ch = logging.StreamHandler(); ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(message)s'))
    
    logger.addHandler(fh); logger.addHandler(ch)
    return logger

logger = setup_logger()

try:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
except ImportError:
    logger.critical("❌ Thiếu thư viện nnunetv2.")
    sys.exit(1)

# ==============================================================================
# 3. CÁC HÀM XỬ LÝ CHÍNH
# ==============================================================================

def convert_dicom_to_nifti(dicom_dir, out_path):
    """Convert DICOM Series sang NIfTI"""
    logger.info(f"   [1] Convert DICOM: {dicom_dir.name}")
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not dicom_names:
        raise ValueError(f"Không tìm thấy DICOM trong {dicom_dir}")
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, str(out_path))
    return image

def run_vessel_segmentation(input_nii, output_dir, seg_model_root, seg_dataset, seg_chk, seg_fold):
    """
    Chạy lệnh nnUNetv2_predict để tạo mask mạch máu.
    """
    logger.info(f"   [2] Segment Mạch máu (Model 113)...")
    
    # Chuẩn bị tên file đúng chuẩn _0000.nii.gz
    case_id = input_nii.name.replace(".nii.gz", "").replace(".nii", "")
    temp_input_dir = input_nii.parent / "nnunet_seg_input"
    temp_input_dir.mkdir(exist_ok=True)
    
    target_input_name = f"{case_id}_0000.nii.gz"
    shutil.copy(input_nii, temp_input_dir / target_input_name)
    
    # Thiết lập biến môi trường
    env = os.environ.copy()
    env["nnUNet_results"] = seg_model_root
    env["nnUNet_raw"] = str(input_nii.parent) 
    env["nnUNet_preprocessed"] = str(input_nii.parent)

    cmd = [
        "nnUNetv2_predict",
        "-d", seg_dataset,
        "-i", str(temp_input_dir),
        "-o", str(output_dir),
        "-f", seg_fold,
        "-tr", "nnUNetTrainer",
        "-c", "3d_fullres",
        "-p", "nnUNetPlans",
        "-chk", seg_chk,
        "--disable_tta",
        "-device", "cuda"
    ]
    
    try:
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Lỗi nnUNet Segmentation: {e.stderr.decode()}")
        # Dọn dẹp trước khi raise
        shutil.rmtree(temp_input_dir)
        raise

    expected_mask = output_dir / f"{case_id}.nii.gz"
    shutil.rmtree(temp_input_dir) # Dọn dẹp input temp
    
    if expected_mask.exists():
        return expected_mask
    else:
        raise FileNotFoundError(f"Không thấy mask đầu ra: {expected_mask}")

def apply_overlay_in_memory(img_nii_path, mask_nii_path, overlay_boost):
    """Cộng pixel mạch máu vào ảnh gốc"""
    logger.info(f"   [3] Overlay (+{overlay_boost} HU)...")
    
    img_itk = sitk.ReadImage(str(img_nii_path))
    img_arr = sitk.GetArrayFromImage(img_itk).astype(np.float32)
    
    mask_itk = sitk.ReadImage(str(mask_nii_path))
    mask_arr = sitk.GetArrayFromImage(mask_itk)
    
    if img_arr.shape != mask_arr.shape:
        logger.warning(f"⚠️ Mismatch Shape: Img {img_arr.shape} vs Mask {mask_arr.shape}. Skip Overlay.")
    else:
        mask_binary = mask_arr > 0
        if np.sum(mask_binary) > 0:
            img_arr[mask_binary] += overlay_boost
            img_arr = np.clip(img_arr, -1024, 3000)
        else:
            logger.warning("⚠️ Mask rỗng, không có mạch máu nào được detect.")

    # Spacing cho nnU-Net (Z, Y, X)
    spacing = np.array(img_itk.GetSpacing())[::-1]
    
    return img_arr, spacing

# ==============================================================================
# 4. CLASSIFIER
# ==============================================================================
class Classifier:
    def __init__(self, model_folder, chk, fold, step_size=0.5, use_gaussian=False, disable_tta=False):
        logger.info("🛠️  Đang load Model Phình Mạch (Dataset004)...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.predictor = nnUNetPredictor(
            tile_step_size=step_size,
            use_gaussian=use_gaussian,
            use_mirroring=not disable_tta,
            device=device,
            verbose=False, verbose_preprocessing=False, allow_tqdm=False
        )
        self.predictor.initialize_from_trained_model_folder(
            str(model_folder),
            [i if i == "all" else int(i) for i in fold],
            checkpoint_name=chk,
        )
        self.prep = self.predictor.configuration_manager.preprocessor_class()
        labels_dict = self.predictor.dataset_json["labels"]
        self.labels = ["SeriesInstanceUID"] + list(labels_dict.keys())[1:] + ["Aneurysm Present"]
        logger.info("✅ Model Loaded.")

    def predict(self, img_array, spacing, uid):
        logger.info(f"   [4] Predict Phình Mạch: {uid}")
        
        # --- BƯỚC QUAN TRỌNG: FLIP ẢNH ---
        # Lật trục Y (Axis 1) để khớp với model đã train
        img_array = np.flip(img_array, 1) 
        
        # Add channel dim
        input_data = img_array[np.newaxis, ...]
        
        props = {
            "spacing": spacing,
            "shape_before_cropping": img_array.shape,
            "sitk_stuff": None
        }
        
        data, _, _ = self.prep.run_case_npy(
            input_data, None, props,
            self.predictor.plans_manager,
            self.predictor.configuration_manager,
            self.predictor.dataset_json,
        )
        
        logits = self.predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data)).cpu()
        probs = torch.sigmoid(logits)
        max_per_c = torch.amax(probs, dim=(1, 2, 3)).float()
        
        prob_val = max_per_c.numpy()[-1]
        logger.info(f"      -> Max Prob: {prob_val:.4f}")
        
        return [uid] + max_per_c.numpy().tolist()

# ==============================================================================
# 5. MAIN PIPELINE
# ==============================================================================
def process_single_case(input_path, classifier, temp_root, seg_cfg, overlay_boost):
    uid = input_path.name.replace(".nii.gz", "").replace(".nii", "")
    logger.info(f"\n🚀 XỬ LÝ CA: {uid}")
    
    # Tạo folder temp riêng cho ca này trong E:\temp
    case_temp = temp_root / uid
    case_temp.mkdir(exist_ok=True)
    
    raw_nii = case_temp / f"{uid}.nii.gz"
    mask_out_dir = case_temp / "masks"
    mask_out_dir.mkdir(exist_ok=True)
    
    try:
        # B1: Chuẩn bị NIfTI
        if input_path.is_dir():
            convert_dicom_to_nifti(input_path, raw_nii)
        else:
            shutil.copy(input_path, raw_nii)
            
        # B2: Segmentation (Mạch máu)
        mask_nii = run_vessel_segmentation(
            raw_nii, mask_out_dir,
            seg_cfg["model_root"], seg_cfg["dataset"], seg_cfg["chk"], seg_cfg["fold"]
        )
        
        # B3: Overlay
        overlay_arr, spacing = apply_overlay_in_memory(raw_nii, mask_nii, overlay_boost)
        
        # B4: Predict (Phình mạch)
        return classifier.predict(overlay_arr, spacing, uid)
        
    except Exception as e:
        logger.error(f"❌ Lỗi xử lý ca {uid}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None

def run(args):
    # Khởi tạo Classifier
    cls = Classifier(args.model_folder, args.chk, args.fold)
    seg_cfg = {
        "model_root": str(args.seg_model_root),
        "dataset": args.seg_dataset,
        "chk": args.seg_chk,
        "fold": args.seg_fold,
    }
    overlay_boost = args.overlay_boost
    
    input_path = Path(args.input)
    targets = []
    
    # Quét input
    if input_path.is_file():
        targets = [input_path]
    else:
        if list(input_path.glob("*.dcm")):
            targets = [input_path]
        else:
            targets = sorted([d for d in input_path.iterdir() if d.is_dir()])
    
    logger.info(f"Tìm thấy {len(targets)} ca cần xử lý.")
    
    rows = []
    output_csv = Path(args.output)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # [QUAN TRỌNG] Tạo folder temp gốc tại E:\temp
    CUSTOM_TEMP_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 Folder tạm thời được đặt tại: {CUSTOM_TEMP_DIR}")

    # Sử dụng E:\temp làm gốc cho tempfile
    with tempfile.TemporaryDirectory(dir=str(CUSTOM_TEMP_DIR)) as temp_root_str:
        temp_root = Path(temp_root_str)
        logger.info(f"   -> Working dir hiện tại: {temp_root}")
        
        for target in tqdm(targets):
            row = process_single_case(target, cls, temp_root)
            if row:
                rows.append(row)
                pd.DataFrame(rows, columns=cls.labels).to_csv(output_csv, index=False)
                
    logger.info(f"✅ Hoàn tất! Kết quả: {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", type=Path, required=True, help="Folder DICOM Dataset hoặc File NIfTI")
    parser.add_argument("-o", "--output", type=Path, required=True, help="File CSV output")
    parser.add_argument("-m", "--model_folder", type=Path, required=True, help="Folder model nnUNet (Dataset004)")
    parser.add_argument("-c", "--chk", type=str, required=True, help="Checkpoint name (vd: checkpoint_best.pth)")
    parser.add_argument("--fold", type=ast.literal_eval, default="('all',)")
    # segmentation config
    parser.add_argument("--seg-model-root", type=Path, default=Path(DEFAULT_PATH_NNUNET_RESULTS_SEG),
                        help="Đường dẫn nnUNet_results cho model segmentation (mặc định cấu hình trong file)")
    parser.add_argument("--seg-dataset", type=str, default=DEFAULT_SEG_DATASET_NAME,
                        help="Tên dataset segmentation (vd Dataset113_CTMulSegWholeData)")
    parser.add_argument("--seg-chk", type=str, default=DEFAULT_SEG_CHK_POINT,
                        help="Tên checkpoint segmentation (vd checkpoint_final.pth)")
    parser.add_argument("--seg-fold", type=str, default=DEFAULT_SEG_FOLD,
                        help="Fold segmentation (vd 4 hoặc '0 1 2 3 4')")
    parser.add_argument("--overlay-boost", type=int, default=DEFAULT_OVERLAY_BOOST_VAL,
                        help="Giá trị cộng thêm vào voxel tại vùng mask")
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_CUSTOM_TEMP_DIR,
                        help="Thư mục gốc cho thư mục tạm (sẽ tạo thư mục con)")

    args = parser.parse_args()

    if not args.seg_model_root.exists():
        logger.error(f"❌ Đường dẫn model segmentation không tồn tại: {args.seg_model_root}")
        sys.exit(1)
        
    # cập nhật biến dùng trong run
    global CUSTOM_TEMP_DIR
    CUSTOM_TEMP_DIR = args.temp_dir

    run(args)

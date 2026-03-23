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
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import torch.nn.functional as F

# ================= CẤU HÌNH MẶC ĐỊNH =================
DEFAULT_SEG_MODEL_ROOT = r"D:\VietRAD\kaggle-rsna-intracranial-aneurysm-detection-2025-solution\TopCoWSubmissions\nnUNet\model"
DEFAULT_SEG_CHK = "checkpoint_final.pth"
DEFAULT_SEG_FOLD = "4"
DEFAULT_OVERLAY_BOOST = 200
DEFAULT_TEMP_DIR = Path(r"E:\temp_stage1")

# ================= IMPORT NNUNET =================
try:
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
except ImportError:
    print("❌ Thiếu thư viện nnunetv2.")
    sys.exit(1)

# ================= LOGGING =================
def setup_logger():
    logger = logging.getLogger("Stage1_Direct")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(handler)
    return logger

logger = setup_logger()

# ================= HÀM XỬ LÝ =================

def convert_dicom_to_nifti(dicom_dir, out_path):
    """Convert DICOM sang NIfTI"""
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_dir))
    if not dicom_names:
        raise ValueError(f"Không tìm thấy DICOM trong {dicom_dir}")
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, str(out_path))

def run_vessel_segmentation_direct(input_nii_path, predictor):
    """
    Dùng nnUNetPredictor trực tiếp + Fix lỗi lệch size (Resample back).
    """
    # 1. Đọc ảnh Input
    img_itk = sitk.ReadImage(str(input_nii_path))
    img_npy = sitk.GetArrayFromImage(img_itk).astype(np.float32) # Shape: (Z, Y, X)
    spacing = np.array(img_itk.GetSpacing())[::-1]               # Shape: (Z, Y, X)
    
    original_shape = img_npy.shape # Lưu shape gốc (ví dụ: 276, 512, 512)

    # 2. Chuẩn bị Properties
    props = {
        'spacing': spacing,
        'sitk_stuff': None,
        'shape_before_cropping': original_shape 
    }
    
    # 3. Predict (nnU-Net sẽ tự resize ảnh input theo plans)
    # input_data: (1, Z, Y, X)
    input_data = img_npy[np.newaxis, ...] 

    # Chạy Preprocessing
    preproc = predictor.configuration_manager.preprocessor_class()
    data, _, _ = preproc.run_case_npy(
        input_data, None, props,
        predictor.plans_manager,
        predictor.configuration_manager,
        predictor.dataset_json,
    )
    
    # Chạy Prediction -> logits shape: (NumClasses, Z_new, Y_new, X_new)
    # Ví dụ output shape: (2, 184, 638, 638) - Đã bị resize
    logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data))
    
    # Lấy nhãn (Argmax) -> Shape: (Z_new, Y_new, X_new)
    prediction_tensor = torch.argmax(logits, dim=0)

    # 4. [QUAN TRỌNG] Resize Mask về kích thước gốc
    # Nếu shape hiện tại khác shape gốc thì phải resize ngược lại
    if prediction_tensor.shape != original_shape:
        # logger.info(f"   -> Resizing mask from {prediction_tensor.shape} back to {original_shape}")
        
        # Input cho interpolate phải là 5D: (Batch, Channel, D, H, W)
        # prediction_tensor: (D, H, W) -> unsqueeze(0).unsqueeze(0)
        pred_input = prediction_tensor.unsqueeze(0).unsqueeze(0).float()
        
        # Interpolate Nearest (Cho Mask)
        pred_resized = F.interpolate(
            pred_input, 
            size=original_shape, 
            mode='nearest'
        )
        
        # Bỏ batch/channel dim -> Về lại (D, H, W) = (Z, Y, X)
        prediction = pred_resized[0, 0].byte().cpu().numpy()
    else:
        prediction = prediction_tensor.byte().cpu().numpy()

    # 5. Đóng gói thành SimpleITK Image
    pred_itk = sitk.GetImageFromArray(prediction)
    # Bây giờ size đã khớp, lệnh này sẽ không lỗi nữa
    pred_itk.CopyInformation(img_itk)
    
    return pred_itk


def build_predictor(args):
    """Khởi tạo predictor (dùng cho đa luồng, mỗi worker 1 predictor riêng)."""
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
        str(args.seg_model_root),
        use_folds=(args.seg_fold,),
        checkpoint_name=args.seg_chk
    )
    return predictor


def chunk_list(seq, n):
    """Chia list seq thành n phần gần bằng nhau."""
    k, m = divmod(len(seq), n)
    return [seq[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n) if seq[i*k + min(i, m):(i+1)*k + min(i+1, m)]]

def apply_overlay_and_save(img_path, mask_itk, save_path, boost_val):
    """
    Overlay -> Save NIfTI mới
    """
    img_itk = sitk.ReadImage(str(img_path))
    img_arr = sitk.GetArrayFromImage(img_itk)
    
    mask_arr = sitk.GetArrayFromImage(mask_itk)
    
    # Kiểm tra lần cuối
    if img_arr.shape == mask_arr.shape:
        mask_binary = mask_arr > 0
        if np.sum(mask_binary) > 0:
            img_arr = img_arr.astype(np.float32)
            img_arr[mask_binary] += boost_val
            img_arr = np.clip(img_arr, -1024, 3000)
    else:
        logger.warning(f"⚠️ Vẫn lệch size: Img {img_arr.shape} vs Mask {mask_arr.shape}. Saving original.")

    # Lưu file
    new_img_itk = sitk.GetImageFromArray(img_arr)
    new_img_itk.CopyInformation(img_itk)
    sitk.WriteImage(new_img_itk, str(save_path))

def process_case(input_path, output_dir, temp_root, predictor, boost_val):
    uid = input_path.name.replace(".nii.gz", "").replace(".nii", "")
    
    # Resume check
    final_out_path = output_dir / f"{uid}.nii.gz"
    if final_out_path.exists():
        return

    # Temp setup
    case_temp = temp_root / uid
    case_temp.mkdir(exist_ok=True)
    raw_nii = case_temp / f"{uid}.nii.gz"

    try:
        if input_path.is_dir():
            convert_dicom_to_nifti(input_path, raw_nii)
        else:
            shutil.copy(input_path, raw_nii)
            
        # Segment (có tự động resize back)
        mask_itk = run_vessel_segmentation_direct(raw_nii, predictor)
        
        # Overlay & Save
        apply_overlay_and_save(raw_nii, mask_itk, final_out_path, boost_val)
        
    except Exception as e:
        logger.error(f"❌ Error {uid}: {e}")
        import traceback
        traceback.print_exc()
    
    try: shutil.rmtree(case_temp)
    except: pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=Path, help="Folder DICOM hoặc NIfTI gốc")
    parser.add_argument("-o", "--output-dir", required=True, type=Path, help="Folder lưu NIfTI đã Overlay")
    
    # Config
    parser.add_argument("--seg-model-root", type=Path, default=Path(DEFAULT_SEG_MODEL_ROOT))
    parser.add_argument("--seg-chk", type=str, default=DEFAULT_SEG_CHK)
    parser.add_argument("--seg-fold", type=str, default=DEFAULT_SEG_FOLD)
    parser.add_argument("--overlay-boost", type=int, default=DEFAULT_OVERLAY_BOOST)
    parser.add_argument("--temp-dir", type=Path, default=DEFAULT_TEMP_DIR)
    parser.add_argument("--num-workers", type=int, default=1, help="Số luồng xử lý song song (mỗi worker 1 predictor riêng)")
    
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.temp_dir.mkdir(parents=True, exist_ok=True)

    if args.input.is_file():
        targets = [args.input]
    else:
        if list(args.input.glob("*.dcm")): targets = [args.input]
        else: targets = sorted([d for d in args.input.iterdir() if d.is_dir()])
        if not targets: targets = sorted(list(args.input.glob("*.nii.gz")))

    logger.info(f"🚀 Stage 1 Start: {len(targets)} cases.")
    logger.info(f"📂 Output Dir: {args.output_dir}")
    logger.info(f"🧵 Num workers: {args.num_workers}")

    if args.num_workers <= 1 or len(targets) <= 1:
        # Single worker
        logger.info("⚙️  Loading Segmentation Model (single worker)...")
        predictor = build_predictor(args)
        logger.info("✅ Model Loaded.")

        with tempfile.TemporaryDirectory(dir=str(args.temp_dir)) as temp_root:
            for t in tqdm(targets):
                uid = t.name.replace(".nii.gz", "").replace(".nii", "")
                final_out_path = args.output_dir / f"{uid}.nii.gz"
                if final_out_path.exists():
                    logger.info(f"Skip {uid} (đã có kết quả)")
                    continue
                process_case(t, args.output_dir, Path(temp_root), predictor, args.overlay_boost)
    else:
        # Multi-threaded: mỗi worker tự load predictor riêng
        chunks = chunk_list(targets, args.num_workers)
        logger.info(f"⚙️  Loading {len(chunks)} predictors (one per worker)...")

        def worker(chunk):
            pred = build_predictor(args)
            with tempfile.TemporaryDirectory(dir=str(args.temp_dir)) as temp_root:
                for t in chunk:
                    uid = t.name.replace(".nii.gz", "").replace(".nii", "")
                    final_out_path = args.output_dir / f"{uid}.nii.gz"
                    if final_out_path.exists():
                        continue
                    process_case(t, args.output_dir, Path(temp_root), pred, args.overlay_boost)
            return len(chunk)

        with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
            futures = [ex.submit(worker, c) for c in chunks]
            for _ in tqdm(as_completed(futures), total=len(futures), desc="Workers"):
                _.result()

    logger.info("✅ Stage 1 Complete!")

if __name__ == "__main__":
    main()

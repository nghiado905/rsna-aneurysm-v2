import argparse
import ast
import sys
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import SimpleITK as sitk
from tqdm import tqdm

"""
Stage 2: Phân loại trên các file NIfTI overlay đã có.
Usage:
python inference_stage2.py \
  -i /path/to/overlay_dir_or_file \
  -o /path/to/output.csv \
  -m /path/to/model_folder \
  -c checkpoint_best.pth --fold "('all',)"
"""

# đảm bảo trainer custom có trên path
REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "nnunetv2"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
try:
    importlib.import_module("nnunetv2.training.nnUNetTrainer.kaggle2025_rsna.Kaggle2025RSNATrainer")
except Exception:
    try:
        importlib.import_module("nnunetv2.training.nnUNetTrainer.project_specific.kaggle2025_rsna.Kaggle2025RSNATrainer")
    except Exception:
        pass

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


class Classifier:
    def __init__(self, model_folder, chk, fold, step_size=0.5,
                 use_gaussian=False, disable_tta=False, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.predictor = nnUNetPredictor(
            tile_step_size=step_size,
            use_gaussian=use_gaussian,
            use_mirroring=not disable_tta,
            device=self.device,
            verbose=False, verbose_preprocessing=False, allow_tqdm=False,
        )
        self.predictor.initialize_from_trained_model_folder(
            str(model_folder),
            [i if i == "all" else int(i) for i in fold],
            checkpoint_name=chk,
        )
        self.prep = self.predictor.configuration_manager.preprocessor_class()
        labels_dict = self.predictor.dataset_json["labels"]
        self.labels = ["SeriesInstanceUID"] + list(labels_dict.keys())[1:] + ["Aneurysm Present"]

    def predict_overlay(self, overlay_path: Path):
        uid = overlay_path.name.replace(".nii.gz", "").replace(".nii", "")
        
        # 1. Đọc ảnh bằng SimpleITK để lấy metadata
        itk_img = sitk.ReadImage(str(overlay_path))
        
        # 2. Lấy dữ liệu pixel
        img = sitk.GetArrayFromImage(itk_img).astype(np.float32)
        
        # 3. Lấy spacing và đảo chiều (SimpleITK: X,Y,Z -> Numpy: Z,Y,X)
        # [QUAN TRỌNG] Đây là bước fix lỗi NoneType
        spacing = np.array(itk_img.GetSpacing())[::-1]
        
        # 4. Giữ nguyên logic flip của bạn (nếu model train trên ảnh flip)
        img = np.flip(img, 1) 
        
        input_data = img[np.newaxis, ...]
        
        # 5. Truyền spacing thực tế vào properties
        props = {
            "spacing": spacing,  # <--- Đã sửa: Truyền mảng spacing thay vì None
            "shape_before_cropping": img.shape,
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
        return [uid] + max_per_c.numpy().tolist()


def gather_overlays(path: Path):
    if path.is_file():
        return [path]
    return sorted([p for p in path.iterdir() if p.is_file() and p.suffix in (".nii", ".gz")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=Path, required=True, help="File hoặc thư mục chứa overlay NIfTI")
    ap.add_argument("-o", "--output", type=Path, required=True, help="CSV output")
    ap.add_argument("-m", "--model_folder", type=Path, required=True)
    ap.add_argument("-c", "--chk", type=str, required=True)
    ap.add_argument("--fold", type=ast.literal_eval, default="('all',)")
    ap.add_argument("--step_size", type=float, default=0.5)
    ap.add_argument("--disable_tta", action="store_true", default=False)
    ap.add_argument("--use_gaussian", action="store_true", default=False)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--num-workers", type=int, default=1, help="Số luồng song song, mỗi worker 1 predictor")
    args = ap.parse_args()

    overlays = gather_overlays(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Single worker
    if args.num_workers <= 1 or len(overlays) <= 1:
        clf = Classifier(args.model_folder, args.chk, args.fold,
                         step_size=args.step_size,
                         use_gaussian=args.use_gaussian,
                         disable_tta=args.disable_tta,
                         device=args.device)
        rows = []
        for ov in tqdm(overlays):
            rows.append(clf.predict_overlay(ov))
            pd.DataFrame(rows, columns=clf.labels).to_csv(args.output, index=False)
    else:
        # Multi-thread: mỗi worker 1 predictor riêng
        def worker(chunk):
            clf_local = Classifier(args.model_folder, args.chk, args.fold,
                                   step_size=args.step_size,
                                   use_gaussian=args.use_gaussian,
                                   disable_tta=args.disable_tta,
                                   device=args.device)
            res = []
            for ov in chunk:
                res.append(clf_local.predict_overlay(ov))
            return res

        # chia chunk
        def chunk_list(seq, n):
            k, m = divmod(len(seq), n)
            return [seq[i*k + min(i, m):(i+1)*k + min(i+1, m)] for i in range(n) if seq[i*k + min(i, m):(i+1)*k + min(i+1, m)]]

        chunks = chunk_list(overlays, args.num_workers)
        rows = []
        with ThreadPoolExecutor(max_workers=len(chunks)) as ex:
            futures = [ex.submit(worker, c) for c in chunks]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Workers"):
                rows.extend(f.result())

        pd.DataFrame(rows, columns=Classifier(args.model_folder, args.chk, args.fold,
                                             step_size=args.step_size,
                                             use_gaussian=args.use_gaussian,
                                             disable_tta=args.disable_tta,
                                             device=args.device).labels).to_csv(args.output, index=False)

    print("Saved:", args.output)


if __name__ == "__main__":
    main()

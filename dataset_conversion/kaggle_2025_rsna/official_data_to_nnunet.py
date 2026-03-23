import os
from joblib import Parallel, delayed
import numpy as np
import math
import SimpleITK as sitk
from pathlib import Path
import pydicom
import argparse
import pandas as pd
import ast
from datetime import datetime
from batchgenerators.utilities.file_and_folder_operations import save_json
from tqdm import tqdm
import torch

all_labels = [
    "Other Posterior Circulation",
    "Basilar Tip",
    "Right Posterior Communicating Artery",
    "Left Posterior Communicating Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Infraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Middle Cerebral Artery",
    "Left Middle Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Anterior Cerebral Artery",
    "Anterior Communicating Artery",
]

STAGE1_TARGET_SPACING = np.array([1.0, 0.55, 0.5], dtype=np.float32)


def log_step(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def init_stage1_predictor(model_training_output_dir: Path, checkpoint_name: str = "checkpoint_final.pth", fold: int = 0):
    log_step(f"Init Stage-1 predictor | model_dir={model_training_output_dir} checkpoint={checkpoint_name} fold={fold}")
    try:
        from nnxnet.inference.predict_from_raw_data_2D_orthogonal_planes_fast import nnXNetPredictor
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Cannot import nnxnet. Please install nnxnet or set PYTHONPATH to your nnXNet source before running official_data_to_nnunet.py"
        ) from e

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    predictor = nnXNetPredictor(
        tile_step_size=0.5,
        use_mirroring=False,
        use_gaussian=True,
        perform_everything_on_device=True,
        device=device,
        allow_tqdm=False
    )
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=str(model_training_output_dir),
        use_folds=(fold,),
        checkpoint_name=checkpoint_name,
    )
    predictor.initialize_network_and_gaussian()
    log_step("Stage-1 predictor initialized")
    return predictor


def get_stage1_bbox(img_zyx: np.ndarray, image3D: sitk.Image, stage1_predictor):
    original_spacing = np.array(image3D.GetSpacing(), dtype=np.float32)  # (x, y, z)
    input_img_np = np.ascontiguousarray(img_zyx[None])  # [1, Z, Y, X]

    with torch.no_grad():
        z_min, z_max, y_min, y_max, x_min, x_max = stage1_predictor.predict_from_multi_axial_slices(
            input_img_np,
            original_spacing,
            STAGE1_TARGET_SPACING,
            max_batch_size=16,
        )

    # clamp to image bounds
    z_min = int(max(0, min(z_min, img_zyx.shape[0])))
    z_max = int(max(0, min(z_max, img_zyx.shape[0])))
    y_min = int(max(0, min(y_min, img_zyx.shape[1])))
    y_max = int(max(0, min(y_max, img_zyx.shape[1])))
    x_min = int(max(0, min(x_min, img_zyx.shape[2])))
    x_max = int(max(0, min(x_max, img_zyx.shape[2])))

    if not (z_max > z_min and y_max > y_min and x_max > x_min):
        raise RuntimeError(f"Invalid Stage-1 bbox: {(z_min, z_max, y_min, y_max, x_min, x_max)}")

    return [z_min, z_max, y_min, y_max, x_min, x_max]


def load_and_crop(
    series_path: Path,
    stage1_predictor=None,
    case_id: str | None = None,
):
    """
    Load a DICOM series and crop ROI.

    Crop priority:
    1) Stage-1 model bbox (if stage1_predictor is provided)
    2) Fallback fixed-size bbox (get_bbox) when Stage-1 is unavailable/fails
    """
    image = process_series(series_path)
    img = sitk.GetArrayFromImage(image)  # (Z, Y, X)
    crop_source = "fixed_no_stage1"
    if stage1_predictor is not None:
        try:
            bbox = get_stage1_bbox(img, image, stage1_predictor)
            crop_source = "stage1"
        except Exception as e:
            bbox = get_bbox(img, np.flip(np.array(image.GetSpacing())))
            crop_source = "fixed_fallback"
            if case_id is not None:
                log_step(f"[{case_id}] Stage-1 bbox failed ({e}), fallback fixed bbox={bbox}")
    else:
        bbox = get_bbox(img, np.flip(np.array(image.GetSpacing())))

    # apply the bounding box to the image and save it
    cropped_img = img[bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[4] : bbox[5]]
    return cropped_img, {
        "spacing": np.flip(np.array(image.GetSpacing())),
        "direction": image.GetDirection(),
        "origin": image.GetOrigin(),
        "crop_bbox_zyx": [int(v) for v in bbox],
        "crop_source": crop_source,
    }


def create_sphere(array_shape: tuple, center: tuple, radius: float, value: int):
    """
    Create a sphere inside a 3D numpy array.

    Parameters
    ----------
    array_shape : tuple of int
        Shape of the 3D array (z, y, x).
    center : tuple of float
        Center of the sphere (cz, cy, cx).
    radius : float
        Radius of the sphere.
    value : int
        Integer value to fill the sphere with.

    Returns
    -------
    np.ndarray
        3D array with the sphere inside.
    """
    # Initialize array
    arr = np.zeros(array_shape, dtype=np.int32)

    # Create grid of coordinates
    z, y, x = np.indices(array_shape)

    # Equation of a sphere
    mask = (x - center[2]) ** 2 + (y - center[1]) ** 2 + (
        z - center[0]
    ) ** 2 <= radius**2

    # Fill sphere with the given value
    arr[mask] = value
    return arr


def check_overlaps(arrays: list, cid: str) -> np.ndarray:
    """
    Check overlaps between label arrays, set minimum value
    if overlap found

    Params
    ------
    arrays : set of arrays to compare
    cid : case ID

    Return
    ------
    combined : combined set of arrays

    """
    if len(arrays) == 1:
        return arrays[0]
    elif len(arrays) > 1:
        combined = arrays[0]
        for array in arrays[1:]:
            combined += array
            overlap = (combined > 0) & (array > 0)
            if np.any(overlap):
                values = [combined[overlap].flatten()[0], int(array[overlap].mean())]
                print(
                    f"{cid} : Overlap found, setting labels to last array found: {values[-1]} ({values}), overlap size: {overlap.sum()}"
                )
                # combined[overlap] = min(values)
                combined[overlap] = array[overlap]

        return combined


def _read_meta_minimal(f):
    """Read minimal metadata for sorting and HU conversion."""
    ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
    instance = getattr(ds, "InstanceNumber", None)
    pos = getattr(ds, "ImagePositionPatient", None)
    orient = getattr(ds, "ImageOrientationPatient", None)
    spacing = getattr(ds, "PixelSpacing", None)
    thickness = getattr(ds, "SliceThickness", None)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    slope = getattr(ds, "RescaleSlope", 1.0)
    if pos is not None:
        pos = np.array(pos, dtype=float)
    if orient is not None:
        orient = np.array(orient, dtype=float)
    if spacing is not None:
        spacing = np.array(spacing, dtype=float)
    return {
        "file": f,
        "instance": instance,
        "pos": pos,
        "orient": orient,
        "spacing": spacing,
        "thickness": float(thickness) if thickness is not None else None,
        "intercept": float(intercept),
        "slope": float(slope),
    }


def _list_dicom_files(input_folder: Path):
    dicom_files = sorted(Path(input_folder).glob("*.dcm"))
    if len(dicom_files) == 0:
        dicom_files = sorted([p for p in Path(input_folder).iterdir() if p.is_file()])
    return dicom_files


def process_series(input_folder: Path, n_jobs=-1) -> sitk.Image:
    """
    Fast DICOM series reader using pydicom + NumPy with parallel metadata reading.
    Preserves original origin, spacing, and direction, then reorients to RAS.

    Args:
        input_folder (Path): Folder containing DICOM files.
        n_jobs (int): Number of parallel jobs (-1 uses all cores).

    Returns:
        sitk.Image: 3D image in RAS orientation.
    """

    # 1️⃣ List all DICOM files
    dicom_files = _list_dicom_files(input_folder)
    if len(dicom_files) == 0:
        raise RuntimeError(
            f"No DICOM files found in series folder: {input_folder}. "
            "Expected *.dcm or regular files containing DICOM slices."
        )

    if len(dicom_files) > 1:
        metas = Parallel(n_jobs=n_jobs)(
            delayed(_read_meta_minimal)(f) for f in dicom_files
        )

        first_orient = metas[0]["orient"]
        if first_orient is not None:
            row = first_orient[:3]
            col = first_orient[3:]
            slice_normal = np.cross(row, col)
            slice_normal /= np.linalg.norm(slice_normal)
        else:
            slice_normal = np.array([0, 0, 1], dtype=float)

        # Step 3: compute z position along normal for each slice
        for m in metas:
            if m["pos"] is not None:
                m["z"] = np.dot(m["pos"], slice_normal)
            elif m["instance"] is not None:
                m["z"] = m["instance"]
            else:
                m["z"] = float("inf")

        # Step 4: sort slices along z
        metas.sort(key=lambda m: m["z"])
        sorted_files = [m["file"] for m in metas]
        ds0 = metas[0]
        z_positions = np.array([m["z"] for m in metas], dtype=float)
        del metas
    else:
        sorted_files = [dicom_files[0]]

    # 3️⃣ Read pixel data in parallel
    def read_pixel_array(f):
        ds = pydicom.dcmread(f)
        arr = ds.pixel_array.astype(np.float32)
        try:
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        except:
            slope = float(
                ds.PerFrameFunctionalGroupsSequence[0]
                .PixelValueTransformationSequence[0]
                .RescaleSlope
            )
            intercept = float(
                ds.PerFrameFunctionalGroupsSequence[0]
                .PixelValueTransformationSequence[0]
                .RescaleIntercept
            )
        arr = arr * slope + intercept
        return arr

    slices = Parallel(n_jobs=n_jobs)(delayed(read_pixel_array)(f) for f in sorted_files)

    volume = np.stack(slices, axis=0)  # (Z, Y, X)
    del slices

    # 4️⃣ Convert to SimpleITK image
    image3D = sitk.GetImageFromArray(volume.squeeze())

    # 5️⃣ Extract original spacing, origin, direction from DICOM
    if len(dicom_files) > 1:
        spacing_xy = ds0["spacing"]
        if len(z_positions) > 1:
            spacing_z = np.mean(np.diff(z_positions))
        else:
            spacing_z = ds0["thickness"] if ds0["thickness"] is not None else 1.0

    else:
        ds0 = pydicom.dcmread(sorted_files[0], stop_before_pixels=True)
        spacing_xy = (
            ds0.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence[0].PixelSpacing
        )

        pos1 = np.array(
            ds0.PerFrameFunctionalGroupsSequence[0]
            .PlanePositionSequence[0]
            .ImagePositionPatient,
            dtype=float,
        )
        pos2 = np.array(
            ds0.PerFrameFunctionalGroupsSequence[1]
            .PlanePositionSequence[0]
            .ImagePositionPatient,
            dtype=float,
        )
        spacing_z = np.linalg.norm(pos2 - pos1)

    spacing = (spacing_xy[0], spacing_xy[1], spacing_z)

    image3D.SetSpacing(spacing)

    # Origin from ImagePositionPatient of first slice
    if len(dicom_files) > 1:
        origin = np.array(ds0["pos"], dtype=float)
    else:
        origin = (
            ds0.PerFrameFunctionalGroupsSequence[0]
            .PlanePositionSequence[0]
            .ImagePositionPatient
        )
    image3D.SetOrigin(origin)

    # Direction from ImageOrientationPatient
    if len(dicom_files) > 1:
        iop = [float(x) for x in ds0["orient"]]  # 6 values
        row_cos = np.array(iop[:3])
        col_cos = np.array(iop[3:])
        slice_cos = np.cross(row_cos, col_cos)
        direction = [
            row_cos[0],
            col_cos[0],
            slice_cos[0],
            row_cos[1],
            col_cos[1],
            slice_cos[1],
            row_cos[2],
            col_cos[2],
            slice_cos[2],
        ]
    else:
        orientation = (
            ds0.SharedFunctionalGroupsSequence[0]
            .PlaneOrientationSequence[0]
            .ImageOrientationPatient
        )
        row = np.array(orientation[0:3])
        col = np.array(orientation[3:6])
        slice_dir = np.cross(row, col)
        direction = np.concatenate([row, col, slice_dir]).tolist()

    image3D.SetDirection(direction)

    return image3D


def get_bbox(img: np.ndarray, spacing: np.ndarray):
    """
    Get bounding box for CoW
    Params
    ------
    img : label image
    spacing : image spacing
    Returns
    -------
    bbox : bounding box with RoI
    """
    dims = np.array(img.shape) * spacing
    target_size = [200.0, 160.0, 160.0]
    bbox = []
    for i, d in enumerate(dims):
        if d <= target_size[i]:
            # Take the complete axis
            bbox += [0, img.shape[i] - 1]
        else:
            if i == 0:
                # Sample axial coordinate from top
                z_min = int(
                    img.shape[0]
                    - 1
                    - math.ceil(target_size[i] / (spacing[0] + np.finfo(float).eps))
                )
                z_min = max([0, z_min])  # Clipping, just in case
                bbox += [z_min, img.shape[i] - 1]
            else:
                # Sample 20.0cm around the center
                center = img.shape[i] // 2
                half_size = math.ceil(
                    (target_size[i] / 2) / (spacing[i] + np.finfo(float).eps)
                )
                bbox += [
                    max([0, center - half_size]),
                    min([img.shape[i] - 1, center + half_size]),
                ]
    return bbox


def get_id_from_instance(folder: os.PathLike, target_instance: str) -> int:
    """
    Obtain axial index from instance

    Params
    ------
    folder : folder with DICOM images

    Returns
    -------
    slice_index : slice index

    """

    # Collect all DICOM slices
    slices = []
    for fname in os.listdir(folder):
        fpath = os.path.join(folder, fname)
        try:
            ds = pydicom.dcmread(fpath, stop_before_pixels=True)
            if hasattr(ds, "SOPInstanceUID"):
                slices.append((ds.SOPInstanceUID, fpath, ds))
        except:
            pass

    if len(slices) == 0:
        print(
            f"No valid slices found for series {os.path.basename(folder)} and instance {target_instance}"
        )
        return None

    # Sort slices by z position
    slices.sort(key=lambda x: float(x[2].ImagePositionPatient[2]))

    # Map SOP UID → slice index
    uid_to_index = {uid: i for i, (uid, _, _) in enumerate(slices)}
    keys = list(uid_to_index.keys())

    if target_instance in keys:
        return uid_to_index[target_instance]
    else:
        print(
            f"No data found for series {os.path.basename(folder)} and instance {target_instance}"
        )
        return None


def create_label_files(
    img: np.ndarray,
    arrays: list,
    indexes: list,
    i: str,
    image_ref,
):
    """
    Create output label files

    Params
    ------
    img : corresponding image being analyzed
    arrays : arrays with labelled spheres, if any
    indexes : indexes of aneurysm classes found, if any
    i : case ID being analyzed
    image_ref : reference image object

    Returns
    -------
    label_array : final label array (instance segmentation)

    """

    if (len(arrays) == 0) and (len(indexes) == 0):
        label_array = np.zeros(img.shape)
        instance_dict = {"instances": {}}
    elif (len(arrays) > 0) and (len(indexes) > 0):
        label_array = check_overlaps(arrays=arrays, cid=i)
        instance_dict = {
            "instances": {
                str(index_name): ind for index_name, ind in enumerate(indexes, start=1)
            }
        }

    # Save label image
    label = sitk.GetImageFromArray(label_array)
    label.CopyInformation(image_ref)

    return label, instance_dict


def process_id(
    series_folder,
    folder,
    imagesTr,
    labelsTr,
    mapping,
    loc_df,
    workers,
    stage1_predictor,
):
    # Load series
    full_folder = series_folder / folder

    if not full_folder.exists():
        print(f"No series available for ID '{folder}'")
        return

    # Determine output ID
    out_id = mapping[folder]
    log_step(f"Process case | series={folder} id={out_id}")
    outfile = imagesTr / f"{out_id}_0000.nii"
    image3D, bbox = None, None

    # Load and process DICOM
    image3D = process_series(full_folder, n_jobs=workers)

    # Save as NIfTI .nii
    assert folder in list(mapping.keys())

    # Apply Stage-1 ROI cropping (replace fixed-size get_bbox)
    img3d = sitk.GetArrayFromImage(image3D).squeeze()
    try:
        bbox = get_stage1_bbox(img3d, image3D, stage1_predictor)
    except Exception as e:
        print(f"[{folder}] Stage-1 bbox failed ({e}), fallback to fixed get_bbox")
        spacing = np.flip(np.array(image3D.GetSpacing()))
        bbox = get_bbox(img=img3d, spacing=spacing)

    log_step(f"Crop bbox zyx={bbox}")
    img_crop = img3d[bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[-2] : bbox[-1]]
    img_crop = np.flip(img_crop, 1)
    image_crop = sitk.GetImageFromArray(img_crop)
    image_crop.SetSpacing(image3D.GetSpacing())
    image_crop.SetOrigin(image3D.GetOrigin())
    image_crop.SetDirection(image3D.GetDirection())
    sitk.WriteImage(image_crop, outfile)
    log_step(f"Saved image crop: {outfile}")

    # Process aneurysm label file
    out_label_file = labelsTr / f"{out_id}.nii"

    # Process labels
    if not out_label_file.exists():
        # Access location dataframe
        # Check if series is in location dataframe
        img = sitk.GetArrayFromImage(image3D).squeeze()
        arrays, indexes = [], []
        if folder in loc_df["SeriesInstanceUID"].values.astype(str):
            # There is an aneurysm label
            # Get row indexes
            series_df = loc_df[loc_df["SeriesInstanceUID"] == folder]
            # Obtain SOP instances (CAREFUL, ONLY COLUMN WITH "_NEW" TAG IS VALID!!)
            instances = series_df["SOPInstanceUID"].values.astype(str).tolist()
            coordinates = series_df["coordinates"].apply(ast.literal_eval)
            locations = series_df["location"].values.astype(str).tolist()

            for instance, coordinate, location in zip(
                instances, coordinates, locations
            ):
                keys = list(coordinate.keys())
                global all_labels
                ind = all_labels.index(location)
                if len(keys) == 3:
                    center = [
                        coordinate["f"],
                        coordinate["y"],
                        coordinate["x"],
                    ]
                elif len(keys) == 2:
                    # Find the axial slice index where the labelling is happening
                    axial_ind = get_id_from_instance(
                        folder=full_folder, target_instance=instance
                    )
                    if axial_ind is None:
                        print(
                            f"⚠️ Skipping label in {folder}: SOPInstanceUID {instance} "
                            f"not found (get_id_from_instance returned None)"
                        )
                        continue
                    center = [float(axial_ind), coordinate["y"], coordinate["x"]]

                label_array = create_sphere(
                    array_shape=img.shape, center=center, radius=65, value=ind + 1
                )

                arrays.append(label_array)
                indexes.append(ind)

        label_image, instance_dict = create_label_files(
            img=img,
            arrays=arrays,
            indexes=indexes,
            i=out_id,
            image_ref=image3D,
        )

        out_image_file = labelsTr / f"{out_id}.nii"
        out_json_file = labelsTr / f"{out_id}.json"

        label_img = sitk.GetArrayFromImage(label_image).squeeze()
        label_img_crop = label_img[
            bbox[0] : bbox[1], bbox[2] : bbox[3], bbox[-2] : bbox[-1]
        ]
        label_img_crop = np.flip(label_img_crop, 1)
        label_image = sitk.GetImageFromArray(label_img_crop)
        label_image.SetSpacing(image3D.GetSpacing())
        label_image.SetOrigin(image3D.GetOrigin())
        label_image.SetDirection(image3D.GetDirection())

        sitk.WriteImage(label_image, out_image_file)
        save_json(instance_dict, out_json_file)
        log_step(f"Saved label+json: {out_image_file} | {out_json_file}")


def main(
    input_folder: Path,
    output_folder: Path,
    workers: int,
    stage1_model_dir: Path,
    stage1_checkpoint: str,
    stage1_fold: int,
):
    log_step("Step 1: Validate input paths")
    label_file = input_folder / "train.csv"
    loc_file = input_folder / "train_localizers.csv"
    series_folder = input_folder / "series"
    assert input_folder.exists(), f"Input folder '{input_folder}' does not exist."
    assert (
        output_folder.parent.exists()
    ), f"Parent output folder '{output_folder.parent}' does not exist."
    assert (
        series_folder.exists()
    ), f"Folder with series '{series_folder}' does not exist."
    assert label_file.exists(), f"Label file '{label_file}' does not exist."
    assert loc_file.exists(), f"Location file '{loc_file}' does not exist."
    assert stage1_model_dir.exists(), f"Stage-1 model dir '{stage1_model_dir}' does not exist."

    log_step("Step 2: Prepare output folders")
    imagesTr = output_folder / "imagesTr"
    labelsTr = output_folder / "labelsTr"

    imagesTr.mkdir(exist_ok=True, parents=True)
    labelsTr.mkdir(exist_ok=True, parents=True)

    # Iterate through series and segmentation folders
    folders = sorted(os.listdir(series_folder))

    log_step("Step 3: Build ids_mapping and filtered labels")
    # Assign an "easier" ID to all series
    mapping = {
        folder: f"iarsna_{('0000' + str(i))[(-4):]}" for i, folder in enumerate(folders)
    }
    # Save mapping file, if it does not exist
    mapping_file = output_folder / "ids_mapping.json"

    save_json(mapping, mapping_file)

    # Process label information, including ID mapping
    label_out_file = output_folder / "labels.csv"
    label_df = pd.read_csv(label_file)
    label_df["id"] = label_df["SeriesInstanceUID"].map(mapping)
    # Keep only rows with image information available
    label_df = label_df[label_df["id"].notna()]

    # Set up location dataframe
    loc_df = pd.read_csv(loc_file)

    label_df.to_csv(label_out_file)

    log_step("Step 4: Initialize Stage-1 ROI predictor")
    # Initialize Stage-1 ROI predictor once
    stage1_predictor = init_stage1_predictor(
        model_training_output_dir=stage1_model_dir,
        checkpoint_name=stage1_checkpoint,
        fold=stage1_fold,
    )

    log_step("Step 5: Convert series to imagesTr/labelsTr")
    # Convert IDs in parallel, keep only IDs with label information
    valid_ids = label_df["SeriesInstanceUID"].values.astype(str).tolist()
    for valid_id in tqdm(valid_ids):
        process_id(
            series_folder,
            valid_id,
            imagesTr,
            labelsTr,
            mapping,
            loc_df,
            workers,
            stage1_predictor,
        )

    log_step("Step 6: Write dataset.json")
    # And write the dataset.json
    dataset_json = {
        "name": output_folder.name,
        "description": "Kaggle RSNA 2025 Aneurysm Dataset",
        "file_ending": ".nii",
        "channel_names": {"0": "MRA+CTA+T1+T2"},
        "labels": {"background": 0, **{l: i + 1 for i, l in enumerate(all_labels)}},
        "numTraining": len(valid_ids),
        "overwrite_image_reader_writer": "SimpleITKIO",
        "reference": "",
        "release": "",
    }
    save_json(dataset_json, output_folder / "dataset.json", sort_keys=False)
    log_step("Done conversion")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input_folder",
        help="Input folder which contains all the kaggle data, e.g. /path/to/rsna-intracranial-aneurysm-detection",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output_folder",
        help="Output folder, e.g. $nnUNet_raw/Dataset004_iarsna_crop",
        type=Path,
        required=True,
    )
    parser.add_argument("--np", help="Workers", type=int, default=4)
    parser.add_argument(
        "--stage1_model_dir",
        help="Path to Stage-1 nnXNet model folder (nnUNetTrainer__nnUNetPlans__2d)",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--stage1_checkpoint",
        help="Stage-1 checkpoint filename",
        type=str,
        default="checkpoint_final.pth",
    )
    parser.add_argument(
        "--stage1_fold",
        help="Stage-1 fold index",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    main(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        workers=args.np,
        stage1_model_dir=args.stage1_model_dir,
        stage1_checkpoint=args.stage1_checkpoint,
        stage1_fold=args.stage1_fold,
    )

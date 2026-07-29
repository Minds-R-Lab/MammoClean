"""
Dataset Image Processing Pipeline

Matches each CSV row to its DICOM file, normalizes the pixel data, corrects
laterality, and writes processed .npy images plus per-patient info.txt files
and a filtered CSV.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import pydicom
from pydicom.errors import InvalidDicomError


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for dataset paths and matching options."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name", default="VinDr",
        help="Short name for this dataset, e.g. 'DatasetA'. Used to name the output CSV.",
    )
    parser.add_argument(
        "--csv-path", default="csv_files/processed/VinDr-Mammo.csv",
        help="Path to the dataset's CSV file.",
    )
    parser.add_argument(
        "--raw-root", default="datasets/raw/VinDr",
        help="Root folder containing the DICOM images/folders for this dataset "
             "(the 'Image File Folder (raw)' column is relative to this).",
    )
    parser.add_argument(
        "--output-root", default="datasets/processed/VinDr",
        help="Root folder to write processed .npy files, info.txt files, and "
             "the filtered CSV to.",
    )
    parser.add_argument(
        "--has-image-id", action=argparse.BooleanOptionalAction, default=True,
        help="Whether this dataset's CSV has an 'Image ID' column to match on. "
             "Use --no-has-image-id for datasets that need header-based "
             "Laterality/View matching instead. (default: True)",
    )
    parser.add_argument(
        "--layout", choices=["patient", "image"], default="patient",
        help="Folder layout: 'patient' = one folder per patient holding all "
             "their images, 'image' = one folder per image. (default: patient)",
    )
    parser.add_argument(
        "--id-col", default="ID",
        help="Name of the patient ID column in the CSV. (default: 'ID')",
    )
    parser.add_argument(
        "--image-id-col", default="Image ID",
        help="Name of the image ID column in the CSV. (default: 'Image ID')",
    )
    return parser.parse_args()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "ID",
    "Laterality",
    "View",
    "Breast Density",
    "Diagnosis",
    "BI-RADS Assessment",
    "Image File Folder (raw)",
    "Image File Path (processed)",
]

DICOM_LIKE_EXTENSIONS = (".dcm", ".dicom", ".dic", "")  # "" = extensionless files


def detect_laterality(image: np.ndarray, window_width: int = 25) -> str:
    """Returns 'L' or 'R' based on which edge window has the higher std dev."""
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale numpy array.")

    left_window = image[:, :window_width]
    right_window = image[:, -window_width:]
    return "L" if np.std(left_window) > np.std(right_window) else "R"


def is_intensity_flipped(image: np.ndarray, window_width: int = 25) -> bool:
    """Returns True if a left-lateralized image has its intensity flipped."""
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale numpy array.")

    left_window = image[:, :window_width]
    right_window = image[:, -window_width:]
    return bool(np.mean(right_window) - np.mean(left_window) > 50)


@dataclass
class DatasetConfig:
    name: str
    csv_path: str
    raw_root: str
    output_root: str
    has_image_id: bool = True
    # "patient": one folder per patient holding all of that patient's images.
    # "image":   one folder per image.
    layout: str = "patient"
    id_col: str = "ID"
    image_id_col: str = "Image ID"


@dataclass
class DatasetStats:
    total_rows: int = 0
    matched: int = 0
    dropped_no_folder: int = 0
    dropped_no_matching_file: int = 0
    laterality_mismatches: int = 0
    intensity_flips: int = 0

    def as_dict(self):
        return self.__dict__


class DatasetProcessor:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.stats = DatasetStats()
        self._patient_output_dirs: dict[str, str] = {}
        self._patient_info: dict[str, dict[str, str]] = {}

    def run(self) -> pd.DataFrame:
        logger.info("=== Processing dataset '%s' ===", self.config.name)
        df = pd.read_csv(self.config.csv_path, dtype=str)
        self._validate_columns(df)
        self.stats.total_rows = len(df)

        keep_mask = [self._process_row(row) for _, row in df.iterrows()]
        final_df = df[pd.Series(keep_mask, index=df.index)].reset_index(drop=True)

        out_csv_path = os.path.join(
            self.config.output_root, f"{self.config.name}_processed.csv"
        )
        os.makedirs(self.config.output_root, exist_ok=True)
        final_df.to_csv(out_csv_path, index=False)

        self._log_summary(out_csv_path, len(df), len(final_df))
        return final_df

    def _process_row(self, row: pd.Series) -> bool:
        folder_rel = row.get("Image File Folder (raw)")
        if pd.isna(folder_rel):
            self.stats.dropped_no_folder += 1
            return False

        folder = os.path.join(self.config.raw_root, str(folder_rel))
        if not os.path.isdir(folder):
            logger.warning("Folder not found, dropping row: %s", folder)
            self.stats.dropped_no_folder += 1
            return False

        dicom_files = self._list_dicom_files(folder)
        if not dicom_files:
            logger.warning("No DICOM files found in folder, dropping row: %s", folder)
            self.stats.dropped_no_matching_file += 1
            return False

        image_id = row.get(self.config.image_id_col) if self.config.has_image_id else None
        if self.config.has_image_id and pd.notna(image_id) and str(image_id).strip():
            match_file = self._find_by_image_id(dicom_files, str(image_id))
        else:
            match_file = self._find_by_header(dicom_files, row)

        if match_file is None:
            logger.warning(
                "Could not match a DICOM file for row ID=%s Laterality=%s View=%s in %s",
                row.get(self.config.id_col), row.get("Laterality"), row.get("View"), folder,
            )
            self.stats.dropped_no_matching_file += 1
            return False

        try:
            img = self._load_and_normalize(match_file)
        except Exception as exc:  # corrupt / unreadable pixel data
            logger.warning("Failed to read pixel data from %s (%s); dropping row.", match_file, exc)
            self.stats.dropped_no_matching_file += 1
            return False

        img = self._fix_laterality(img, row)
        self._save_npy(img, row)

        self.stats.matched += 1
        return True

    def _list_dicom_files(self, folder: str) -> list[str]:
        candidates = []
        for path in glob.glob(os.path.join(folder, "**", "*"), recursive=True):
            if not os.path.isfile(path):
                continue
            ext = os.path.splitext(path)[1].lower()
            if ext in DICOM_LIKE_EXTENSIONS:
                candidates.append(path)
        return candidates

    def _find_by_image_id(self, files: list[str], image_id: str) -> Optional[str]:
        image_id = str(image_id).strip()
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            if image_id == stem or image_id in stem:
                return f
        return None

    def _find_by_header(self, files: list[str], row: pd.Series) -> Optional[str]:
        expected_lat = str(row.get("Laterality", "")).strip().upper()
        expected_view = str(row.get("View", "")).strip().upper()

        for f in files:
            try:
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            except (InvalidDicomError, Exception):
                continue

            lat = self._read_header_laterality(ds)
            view = self._read_header_view(ds)

            if view not in ("CC", "MLO", ""):
                logger.debug(
                    "Unrecognized view code '%s' in %s (raw ViewPosition=%r)",
                    view, f, getattr(ds, "ViewPosition", None),
                )

            if lat == expected_lat and (not expected_view or view == expected_view):
                return f

        return None

    @staticmethod
    def _read_header_laterality(ds) -> str:
        for tag in ("ImageLaterality", "Laterality"):
            val = getattr(ds, tag, None)
            if val:
                return str(val).strip().upper()
        return ""

    @staticmethod
    def _read_header_view(ds) -> str:
        # ViewPosition is frequently not a clean "CC"/"MLO" string (e.g. "RCC",
        # "LMLO", "XCCL", "ML", or absent with the real info in ViewCodeSequence
        # or SeriesDescription instead), so classify() checks all three sources
        # by substring and returns unrecognized values raw so they fail an exact
        # match rather than being mistaken for CC or MLO.
        def classify(raw: str) -> Optional[str]:
            v = raw.strip().upper()
            if not v:
                return None

            # Check oblique before the plain lateral/cranio checks, since
            # "medio-lateral oblique" contains both "LATERAL" and "OBLIQUE".
            if "MLO" in v or "OBLIQUE" in v:
                return "MLO"
            if v in ("ML", "LM"):
                return "MLO"
            if "CRANIO" in v or "CRANIAL" in v or "CAUDAL" in v:
                return "CC"
            if "CC" in v:
                return "CC"
            if "MEDIOLATERAL" in v or "MEDIO-LATERAL" in v or "LATEROMEDIAL" in v or "LATERO-MEDIAL" in v:
                return "MLO"
            return None

        val = getattr(ds, "ViewPosition", None)
        if val:
            result = classify(str(val))
            if result:
                return result

        view_code_seq = getattr(ds, "ViewCodeSequence", None)
        if view_code_seq:
            for item in view_code_seq:
                meaning = getattr(item, "CodeMeaning", "")
                if meaning:
                    result = classify(str(meaning))
                    if result:
                        return result

        for tag in ("SeriesDescription", "StudyDescription"):
            text = getattr(ds, tag, None)
            if text:
                result = classify(str(text))
                if result:
                    return result

        return str(val).strip().upper() if val else ""

    def _load_and_normalize(self, path: str) -> np.ndarray:
        ds = pydicom.dcmread(path, force=True)
        arr = ds.pixel_array.astype(np.float32)

        # MONOCHROME1 means low values = bright; flip so higher values are
        # always brighter, matching the MONOCHROME2 convention.
        photometric = getattr(ds, "PhotometricInterpretation", "")
        if photometric == "MONOCHROME1":
            arr = arr.max() - arr
            self.stats.intensity_flips += 1

        # Min-max normalize the source bit depth into 0-255.
        arr_min, arr_max = float(arr.min()), float(arr.max())
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        else:
            arr = np.zeros_like(arr)

        return arr

    def _fix_laterality(self, img: np.ndarray, row: pd.Series) -> np.ndarray:
        detected = detect_laterality(img)
        detected = str(detected).strip().upper() if detected else ""
        expected = str(row.get("Laterality", "")).strip().upper()

        if detected and expected and detected != expected:
            self.stats.laterality_mismatches += 1
            logger.debug(
                "Laterality mismatch for ID=%s: CSV=%s, detected=%s",
                row.get(self.config.id_col), expected, detected,
            )

        if detected == "R":
            img = np.fliplr(img)

        return img

    def _save_npy(self, img: np.ndarray, row: pd.Series) -> None:
        rel_path = str(row["Image File Path (processed)"])
        rel_path = os.path.splitext(rel_path)[0] + ".npy"
        out_path = os.path.join(self.config.output_root, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        try:
            np.save(out_path, img.astype(np.uint8))
        except Exception as exc:
            logger.error("Failed to save %s: %s", out_path, exc)
            raise

        logger.info("Saved image: %s", out_path)

        out_dir = os.path.dirname(out_path)
        patient_id = str(row.get(self.config.id_col, ""))
        self._patient_output_dirs.setdefault(patient_id, out_dir)
        self._update_patient_info_file(row, out_dir)

    def _update_patient_info_file(self, row: pd.Series, out_dir: str) -> None:
        # Rewrite this patient's info.txt from accumulated state on every save,
        # so it is always up to date next to the file just written.
        patient_id = str(row.get(self.config.id_col, ""))
        info = self._patient_info.setdefault(
            patient_id,
            {"L_diagnosis": "", "R_diagnosis": "", "L_birads": "", "R_birads": "", "breast_density": ""},
        )

        def clean(val) -> str:
            return "" if pd.isna(val) else str(val)

        lat = str(row.get("Laterality", "")).strip().upper()
        if lat == "L":
            info["L_diagnosis"] = clean(row.get("Diagnosis", ""))
            info["L_birads"] = clean(row.get("BI-RADS Assessment", ""))
        elif lat == "R":
            info["R_diagnosis"] = clean(row.get("Diagnosis", ""))
            info["R_birads"] = clean(row.get("BI-RADS Assessment", ""))

        density = row.get("Breast Density", "")
        if pd.notna(density) and str(density).strip():
            info["breast_density"] = str(density).strip()

        info_lines = [
            f"L_diagnosis: {info['L_diagnosis']}",
            f"R_diagnosis: {info['R_diagnosis']}",
            f"L_birads: {info['L_birads']}",
            f"R_birads: {info['R_birads']}",
            f"breast_density: {info['breast_density']}",
        ]

        os.makedirs(out_dir, exist_ok=True)
        info_path = os.path.join(out_dir, "info.txt")
        with open(info_path, "w") as fh:
            fh.write("\n".join(info_lines) + "\n")

        logger.info("Updated info file: %s", info_path)

    def _validate_columns(self, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"Dataset '{self.config.name}' CSV is missing required columns: {missing}"
            )
        if self.config.has_image_id and self.config.image_id_col not in df.columns:
            logger.warning(
                "Config says has_image_id=True but column '%s' is absent; "
                "falling back to header-based matching.",
                self.config.image_id_col,
            )
            self.config.has_image_id = False

    def _log_summary(self, out_csv_path: str, n_before: int, n_after: int) -> None:
        s = self.stats
        logger.info("--- Summary for '%s' ---", self.config.name)
        logger.info("Rows in original CSV:      %d", n_before)
        logger.info("Rows matched & processed:   %d", s.matched)
        logger.info("Rows dropped (no folder):   %d", s.dropped_no_folder)
        logger.info("Rows dropped (no match):    %d", s.dropped_no_matching_file)
        logger.info("Rows in final CSV:           %d", n_after)
        logger.info("Laterality mismatches:      %d", s.laterality_mismatches)
        logger.info("Flipped lateralities (MONOCHROME1 detected): %d", s.intensity_flips)
        logger.info("Final CSV saved to:          %s", out_csv_path)


def main() -> None:
    args = parse_arguments()

    cfg = DatasetConfig(
        name=args.name,
        csv_path=args.csv_path,
        raw_root=args.raw_root,
        output_root=args.output_root,
        has_image_id=args.has_image_id,
        layout=args.layout,
        id_col=args.id_col,
        image_id_col=args.image_id_col,
    )
    DatasetProcessor(cfg).run()


if __name__ == "__main__":
    main()

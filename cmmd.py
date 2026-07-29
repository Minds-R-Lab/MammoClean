"""
CMMD Clinical Data Processing Pipeline

Parses and normalizes clinical, imaging, and lesion attributes from the
TOMPEI-CMMD clinical Excel worksheet into a standardized multi-view format.
"""

import argparse
import csv
import logging
import os
from typing import Any, Dict, List, Set, Tuple

import openpyxl


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for files and sheets."""
    parser = argparse.ArgumentParser(
        description="Process and standardize TOMPEI-CMMD clinical Excel files."
    )
    parser.add_argument(
        "--input-xlsx",
        type=str,
        default="csv_files/raw/TOMPEI-CMMD/TOMPEI-CMMD_clinical_data_v01_20250121.xlsx",
        help="Path to the raw TOMPEI-CMMD XLSX file."
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="csv_files/processed/TOMPEI-CMMD.csv",
        help="Path where the processed output CSV will be saved."
    )
    parser.add_argument(
        "--sheet-name",
        type=str,
        default="Imaging Diagnosis Details Sheet",
        help="Name of the Excel worksheet containing clinical findings."
    )
    return parser.parse_args()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

OUTPUT_COLUMNS = [
    "ID",
    "Image ID",
    "Laterality",
    "View",
    "Age",
    "Breast Density",
    "Diagnosis",
    "BI-RADS Assessment",
    "Mass",
    "Mass Shape",
    "Mass Margin",
    "Mass Density",
    "Calcification",
    "Calcification Morphology",
    "Calcification Distribution",
    "Asymmetry",
    "Architectural Distortion",
    "Other Findings",
    "Split",
    "Image File Folder (raw)",
    "Image File Path (processed)",
]

DENSITY_MAP = {
    "fatty": "A",
    "scattered": "B",
    "heterogeneous dense": "C",
    "extremely dense": "D",
}


def clean(value: Any) -> str:
    """Returns a stripped string, or an empty string for empty/None cells."""
    if value is None:
        return ""
    return str(value).strip()


def join_unique(*values: str) -> str:
    """Joins non-empty values with ';' preserving order without deduplication."""
    parts = [v for v in values if v]
    return ";".join(parts)


def split_tokens(text: str) -> List[str]:
    """Splits free-text association comments using multiple potential delimiters."""
    if not text:
        return []
    tokens = text.replace(",", "/").split("/")
    return [t.strip() for t in tokens if t.strip()]


def extract_distortion_asymmetry_other(*texts: str) -> Tuple[str, str, List[str]]:
    """Scans commentary text for structural flags and collects other findings.

    Returns (architectural_distortion_flag, asymmetry_value, other_findings_list).
    """
    has_dist = False
    has_fad = False
    other: List[str] = []
    seen: Set[str] = set()

    for text in texts:
        for token in split_tokens(text):
            low = token.lower()
            if low == "dist":
                has_dist = True
            elif "fad" in low:
                has_fad = True
            elif token not in seen:
                seen.add(token)
                other.append(token)

    asymmetry = "Focal Asymmetry" if has_fad else ""
    distortion_flag = "1" if has_dist else "0"
    return distortion_flag, asymmetry, other


def process_clinical_sheet(input_xlsx: str, sheet_name: str) -> List[Dict[str, Any]]:
    """Reads clinical details from Excel, filters exclusions, and parses metrics."""
    logger.info("Loading workbook: %s", input_xlsx)
    wb = openpyxl.load_workbook(input_xlsx, data_only=True)

    if sheet_name not in wb.sheetnames:
        raise ValueError(f"Worksheet '{sheet_name}' not found in workbook.")

    ws = wb[sheet_name]
    rows_out = []

    # Data starts at row 3 (header offset).
    for row in ws.iter_rows(min_row=3, values_only=True):
        if row[0] is None and all(c is None for c in row):
            continue

        case_id = clean(row[0])            # Column A: ID
        left_right = clean(row[1])         # Column B: LeftRight
        age = clean(row[2])                # Column C: Age
        classification = clean(row[3])     # Column D: classification

        if not case_id:
            continue
        if classification in ("Exclusion", "Invisible"):
            continue

        breast_density_raw = clean(row[5])   # Column F

        # Mass properties (main lesion): G-L -> indexes 6-11
        mass_shape_main = clean(row[7])
        mass_margin_main = clean(row[8])
        mass_density_main = clean(row[9])
        mass_other_main = clean(row[11])

        # Calcification properties (main lesion): M-Q -> indexes 12-16
        calc_morph_main = clean(row[13])
        calc_dist_main = clean(row[14])
        calc_other_main = clean(row[16])

        # Breast-level "Other findings": R-U -> indexes 17-20
        other_breast_parenchyma = clean(row[18])  # Column S
        other_skin = clean(row[19])               # Column T
        other_lymph_nodes = clean(row[20])        # Column U

        birads = clean(row[21])                   # Column V

        # Mass properties (additional lesion): X-AC -> indexes 23-28
        mass_shape_add = clean(row[24])
        mass_margin_add = clean(row[25])
        mass_density_add = clean(row[26])
        mass_other_add = clean(row[28])

        # Calcification properties (additional lesion): AD-AG -> indexes 29-32
        calc_morph_add = clean(row[30])
        calc_dist_add = clean(row[31])

        # ---- Mass attributes ----
        mass_main_present = any([
            clean(row[6]), mass_shape_main, mass_margin_main,
            mass_density_main, clean(row[10]), mass_other_main,
        ])
        mass_add_present = any([
            clean(row[23]), mass_shape_add, mass_margin_add,
            mass_density_add, clean(row[27]), mass_other_add,
        ])
        mass_flag = "1" if (mass_main_present or mass_add_present) else "0"
        mass_shape = join_unique(mass_shape_main, mass_shape_add)
        mass_margin = join_unique(mass_margin_main, mass_margin_add)
        mass_density = join_unique(mass_density_main, mass_density_add)

        # ---- Calcification attributes ----
        calc_main_present = any([
            clean(row[12]), calc_morph_main, calc_dist_main,
            clean(row[15]), calc_other_main,
        ])
        calc_add_present = any([
            clean(row[29]), calc_morph_add, calc_dist_add, clean(row[32]),
        ])
        calc_flag = "1" if (calc_main_present or calc_add_present) else "0"
        calc_morphology = join_unique(calc_morph_main, calc_morph_add)
        calc_distribution = join_unique(calc_dist_main, calc_dist_add)

        # ---- Secondary conditions and associations ----
        arch_dist, asymmetry, other_tokens = extract_distortion_asymmetry_other(
            mass_other_main, mass_other_add, calc_other_main,
            other_breast_parenchyma, other_skin, other_lymph_nodes,
        )
        other_findings = ";".join(other_tokens)
        breast_density = DENSITY_MAP.get(breast_density_raw, "")

        # Emit one row per view (CC and MLO).
        for view in ("CC", "MLO"):
            rows_out.append({
                "ID": case_id,
                "Image ID": "",
                "Laterality": left_right,
                "View": view,
                "Age": age,
                "Breast Density": breast_density,
                "Diagnosis": classification,
                "BI-RADS Assessment": birads,
                "Mass": mass_flag,
                "Mass Shape": mass_shape,
                "Mass Margin": mass_margin,
                "Mass Density": mass_density,
                "Calcification": calc_flag,
                "Calcification Morphology": calc_morphology,
                "Calcification Distribution": calc_distribution,
                "Asymmetry": asymmetry,
                "Architectural Distortion": arch_dist,
                "Other Findings": other_findings,
                "Split": "",
                "Image File Folder (raw)": case_id,
                "Image File Path (processed)": f"{case_id}/{left_right}_{view}.npy",
            })

    return rows_out


def main() -> None:
    args = parse_arguments()

    if not os.path.exists(args.input_xlsx):
        logger.error("Input path does not exist: %s", args.input_xlsx)
        return

    try:
        processed_rows = process_clinical_sheet(args.input_xlsx, args.sheet_name)
    except Exception as e:
        logger.critical("Failed to process sheet content: %s", e, exc_info=True)
        return

    if processed_rows:
        output_dir = os.path.dirname(args.output_csv)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        logger.info("Saving output file to: %s", args.output_csv)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            writer.writerows(processed_rows)
    else:
        logger.warning("No records were exported. Output skipped.")


if __name__ == "__main__":
    main()

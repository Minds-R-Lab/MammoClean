"""
VinDr-Mammo CSV Processing Pipeline

Merges and normalizes structural findings, metadata, and breast-level
annotations from the VinDr-Mammo dataset into a standard multi-view format.
"""

import argparse
import logging
import os
import re
from typing import Any, Dict

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for dataset processing files and paths."""
    parser = argparse.ArgumentParser(
        description="Process and merge raw VinDr-Mammo metadata and annotations."
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default="csv_files/raw/VinDr-Mammo/metadata.csv",
        help="Path to metadata.csv"
    )
    parser.add_argument(
        "--findings",
        type=str,
        default="csv_files/raw/VinDr-Mammo/finding_annotations.csv",
        help="Path to finding_annotations.csv"
    )
    parser.add_argument(
        "--breast-level",
        type=str,
        default="csv_files/raw/VinDr-Mammo/breast-level_annotations.csv",
        help="Path to breast-level_annotations.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="csv_files/processed",
        help="Directory path where the processed CSV will be saved."
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="VinDr-Mammo.csv",
        help="Filename for the processed output CSV."
    )
    return parser.parse_args()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def extract_number_from_string(text: str) -> str:
    """Extracts the last numeric value from a formatted string (e.g. BI-RADS)."""
    if pd.isna(text) or text == '':
        return ''
    numbers = re.findall(r'\d+', str(text).strip())
    return numbers[-1] if numbers else ''


def extract_density_letter(text: str) -> str:
    """Extracts the ACR breast density letter (A-D) from raw annotation text."""
    if pd.isna(text) or text == '':
        return ''
    letters = re.findall(r'[A-D]', str(text).strip().upper())
    return letters[-1] if letters else ''


def get_findings_from_annotations(matching_findings: pd.DataFrame) -> Dict[str, Any]:
    """Aggregates finding categories into multi-view schema flags."""
    findings_data = {
        'mass': 0,
        'mass_shape': '',
        'mass_margin': '',
        'mass_density': '',
        'calcification': 0,
        'calc_morphology': '',
        'calc_distribution': '',
        'asymmetry': '',
        'architectural_distortion': 0,
        'other_findings': ''
    }

    if matching_findings.empty:
        return findings_data

    other_findings_list = []

    for _, finding_row in matching_findings.iterrows():
        finding_categories = finding_row.get('finding_categories', '')
        if pd.isna(finding_categories) or finding_categories == '':
            continue

        try:
            categories_str = str(finding_categories).replace("'", "").replace("[", "").replace("]", "")
            categories = [cat.strip() for cat in categories_str.split(",") if cat.strip()]

            for category in categories:
                category_upper = category.upper()
                if 'MASS' in category_upper:
                    findings_data['mass'] = 1
                elif 'CALCIFICATION' in category_upper:
                    findings_data['calcification'] = 1
                elif 'ASYMMETRY' in category_upper:
                    findings_data['asymmetry'] = category
                elif any(dist_term in category_upper for dist_term in ['ARCHITECTURAL DISTORTION', 'DISTORTION']):
                    findings_data['architectural_distortion'] = 1
                else:
                    other_findings_list.append(category)

        except Exception as e:
            logger.error("Error parsing finding categories: %s | %s", finding_categories, e)
            continue

    if other_findings_list:
        findings_data['other_findings'] = '; '.join(other_findings_list)

    return findings_data


def process_three_csv_files(
    metadata_path: str,
    finding_annotations_path: str,
    breast_level_annotations_path: str
) -> pd.DataFrame:
    """Merges metadata, findings, and breast-level annotations into per-view rows."""
    logger.info("Reading CSV files...")
    breast_level_df = pd.read_csv(breast_level_annotations_path)
    metadata_df = pd.read_csv(metadata_path)
    finding_annotations_df = pd.read_csv(finding_annotations_path)

    metadata_df = metadata_df.dropna(subset=['Series Instance UID'])

    def parse_age(age_val: Any) -> str:
        if pd.isna(age_val) or age_val == '':
            return ''
        nums = re.findall(r'\d+', str(age_val))
        return str(int(nums[0])) if nums else ''

    metadata_df['parsed_age'] = metadata_df["Patient's Age"].apply(parse_age)
    age_lookup = dict(zip(metadata_df['Series Instance UID'], metadata_df['parsed_age']))

    # Group findings once for fast per-(study, laterality, view) lookup.
    findings_grouped = dict(list(finding_annotations_df.groupby(['study_id', 'laterality', 'view_position'])))

    final_data = []
    for _, row in breast_level_df.iterrows():
        study_id = row['study_id']
        laterality = row['laterality']
        view_position = row['view_position']
        series_id = row.get('series_id', '')

        split = row.get('split', '')
        split_mapped = 'train' if str(split).lower() == 'training' else str(split).lower()

        age = age_lookup.get(series_id, '')

        finding_key = (study_id, laterality, view_position)
        matching_findings = findings_grouped.get(finding_key, pd.DataFrame())
        findings_data = get_findings_from_annotations(matching_findings)

        row_data = {
            'ID': study_id,
            'Image ID': row['image_id'],
            'Laterality': laterality,
            'View': view_position,
            'Age': age,
            'Breast Density': extract_density_letter(row.get('breast_density', '')),
            'Diagnosis': '',
            'BI-RADS Assessment': extract_number_from_string(row.get('breast_birads', '')),
            'Mass': findings_data['mass'],
            'Mass Shape': findings_data['mass_shape'],
            'Mass Margin': findings_data['mass_margin'],
            'Mass Density': findings_data['mass_density'],
            'Calcification': findings_data['calcification'],
            'Calcification Morphology': findings_data['calc_morphology'],
            'Calcification Distribution': findings_data['calc_distribution'],
            'Asymmetry': findings_data['asymmetry'],
            'Architectural Distortion': findings_data['architectural_distortion'],
            'Other Findings': findings_data['other_findings'],
            'Split': split_mapped,
            'Image File Folder (raw)': study_id,
            'Image File Path (processed)': f"{study_id}/{laterality}_{view_position}.npy"
        }

        final_data.append(row_data)

    final_columns = [
        'ID', 'Image ID', 'Laterality', 'View', 'Age', 'Breast Density',
        'Diagnosis', 'BI-RADS Assessment', 'Mass', 'Mass Shape', 'Mass Margin',
        'Mass Density', 'Calcification', 'Calcification Morphology',
        'Calcification Distribution', 'Asymmetry', 'Architectural Distortion',
        'Other Findings', 'Split', 'Image File Folder (raw)', 'Image File Path (processed)'
    ]
    return pd.DataFrame(final_data, columns=final_columns)


def main() -> None:
    args = parse_arguments()

    missing_files = [f for f in [args.metadata, args.findings, args.breast_level] if not os.path.exists(f)]
    if missing_files:
        logger.error("Execution failed. Missing input files: %s", missing_files)
        return

    final_df = process_three_csv_files(args.metadata, args.findings, args.breast_level)

    if not final_df.empty:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.output_file)
        final_df.to_csv(output_path, index=False)
        logger.info("Processing completed. File saved to: %s", output_path)
    else:
        logger.error("The pipeline returned no rows. Output skipped.")


if __name__ == "__main__":
    main()

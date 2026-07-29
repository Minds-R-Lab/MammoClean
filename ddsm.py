"""
CBIS-DDSM CSV Processing Pipeline

Aggregates and standardizes calcification and mass case descriptions from the
CBIS-DDSM dataset into a unified multi-view format.
"""

import argparse
import logging
import os
from typing import Any, Dict, List

import pandas as pd


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for dataset files and output directory."""
    parser = argparse.ArgumentParser(
        description="Process and standardize raw CBIS-DDSM metadata CSV files."
    )
    parser.add_argument(
        "--calc-test",
        type=str,
        default="csv_files/raw/CBIS-DDSM/calc_case_description_test_set.csv",
        help="Path to the calcification test set CSV."
    )
    parser.add_argument(
        "--calc-train",
        type=str,
        default="csv_files/raw/CBIS-DDSM/calc_case_description_train_set.csv",
        help="Path to the calcification train set CSV."
    )
    parser.add_argument(
        "--mass-test",
        type=str,
        default="csv_files/raw/CBIS-DDSM/mass_case_description_test_set.csv",
        help="Path to the mass test set CSV."
    )
    parser.add_argument(
        "--mass-train",
        type=str,
        default="csv_files/raw/CBIS-DDSM/mass_case_description_train_set.csv",
        help="Path to the mass train set CSV."
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
        default="CBIS-DDSM.csv",
        help="Filename for the processed output CSV."
    )
    return parser.parse_args()


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def process_single_file(df: pd.DataFrame) -> pd.DataFrame:
    """Performs localized filtering and value standardizations on one DataFrame."""
    df = df.copy()
    required_cols_views = ['patient_id', 'left or right breast', 'image view']

    if 'pathology' in df.columns:
        df['pathology'] = df['pathology'].replace('BENIGN_WITHOUT_CALLBACK', 'BENIGN')

    if all(col in df.columns for col in ['patient_id', 'left or right breast']):
        # Drop invalid assessments.
        if 'assessment' in df.columns:
            df = df[df['assessment'] != 0]

        # Keep only studies consistent in assessment and pathology per breast.
        if 'assessment' in df.columns and 'pathology' in df.columns:
            assessment_consistency = df.groupby(['patient_id', 'left or right breast'])['assessment'].nunique()
            pathology_consistency = df.groupby(['patient_id', 'left or right breast'])['pathology'].nunique()
            consistent_studies = assessment_consistency[
                (assessment_consistency == 1) & (pathology_consistency == 1)
            ].index

            df_tuples = df[['patient_id', 'left or right breast']].to_records(index=False).tolist()
            df = df[pd.Index(df_tuples).isin(consistent_studies)].reset_index(drop=True)

        # Drop studies with only a single view per breast.
        if all(col in df.columns for col in required_cols_views):
            view_counts = df.groupby(['patient_id', 'left or right breast'])['image view'].nunique()
            valid_groups = view_counts[view_counts > 1].index

            df_tuples = df[['patient_id', 'left or right breast']].to_records(index=False).tolist()
            df = df[pd.Index(df_tuples).isin(valid_groups)].reset_index(drop=True)

    return df


def process_study(study_group: pd.DataFrame, patient_id: Any, breast_side: str) -> List[Dict]:
    """Flattens study metadata and abnormalities into per-view rows."""
    study_data = []

    pathology = study_group['pathology'].iloc[0] if 'pathology' in study_group.columns else None
    assessment = study_group['assessment'].iloc[0] if 'assessment' in study_group.columns else None

    breast_density = None
    if 'breast density' in study_group.columns:
        breast_density = study_group['breast density'].iloc[0]
    elif 'breast_density' in study_group.columns:
        breast_density = study_group['breast_density'].iloc[0]

    density_mapping = {1: 'A', 2: 'B', 3: 'C', 4: 'D'}
    if breast_density is not None:
        try:
            breast_density_mapped = density_mapping.get(int(breast_density), '')
        except (ValueError, TypeError):
            breast_density_mapped = breast_density
    else:
        breast_density_mapped = ''

    laterality_mapping = {'LEFT': 'L', 'RIGHT': 'R'}
    laterality = laterality_mapping.get(str(breast_side).upper(), str(breast_side).upper())

    if 'image view' not in study_group.columns:
        return study_data

    for view in study_group['image view'].unique():
        view_group = study_group[study_group['image view'] == view]

        row_data = {
            'ID': patient_id,
            'Image ID': '1-1',
            'Laterality': laterality,
            'View': view,
            'Age': '',
            'Breast Density': breast_density_mapped,
            'Diagnosis': pathology.capitalize() if pathology else '',
            'BI-RADS Assessment': assessment,
            'Mass': 0,
            'Mass Shape': '',
            'Mass Margin': '',
            'Mass Density': '',
            'Calcification': 0,
            'Calcification Morphology': '',
            'Calcification Distribution': '',
            'Asymmetry': '',
            'Architectural Distortion': 0,
            'Other Findings': '',
            'Split': '',
            'Image File Folder (raw)': '',
            'Image File Path (processed)': f"{patient_id}/{laterality}_{view}.npy"
        }

        if 'abnormality type' in view_group.columns:
            # Calcification characteristics.
            calc_rows = view_group[view_group['abnormality type'].str.upper() == 'CALCIFICATION']
            if not calc_rows.empty:
                row_data['Calcification'] = 1
                morphologies = calc_rows['calc type'].dropna().unique().tolist() if 'calc type' in calc_rows.columns else []
                distributions = calc_rows['calc distribution'].dropna().unique().tolist() if 'calc distribution' in calc_rows.columns else []
                if morphologies:
                    row_data['Calcification Morphology'] = str(morphologies) if len(morphologies) > 1 else morphologies[0]
                if distributions:
                    row_data['Calcification Distribution'] = str(distributions) if len(distributions) > 1 else distributions[0]

            # Mass characteristics.
            mass_rows = view_group[view_group['abnormality type'].str.upper() == 'MASS']
            if not mass_rows.empty:
                row_data['Mass'] = 1
                has_architectural_distortion = False
                non_distortion_shapes = []
                asymmetry_findings = []

                # 'mass shape' can be a compound hyphen-joined label (e.g.
                # "IRREGULAR-ARCHITECTURAL_DISTORTION") carrying both a distortion
                # flag and genuine shape. Split on '-' and classify each token so
                # shape/margin are not dropped for these combined cases.
                for _, mass_row in mass_rows.iterrows():
                    if 'mass shape' in mass_row and pd.notna(mass_row['mass shape']):
                        shape = mass_row['mass shape']
                        if isinstance(shape, str):
                            remaining_tokens = []
                            for token in shape.split('-'):
                                token_upper = token.upper()
                                if 'DISTORTION' in token_upper:
                                    has_architectural_distortion = True
                                elif 'ASYMMETRIC' in token_upper:
                                    asymmetry_findings.append(token)
                                else:
                                    remaining_tokens.append(token)
                            if remaining_tokens:
                                non_distortion_shapes.append('-'.join(remaining_tokens))

                if has_architectural_distortion:
                    row_data['Architectural Distortion'] = 1

                if non_distortion_shapes:
                    row_data['Mass Shape'] = str(non_distortion_shapes) if len(non_distortion_shapes) > 1 else non_distortion_shapes[0]

                if asymmetry_findings:
                    row_data['Asymmetry'] = str(asymmetry_findings) if len(asymmetry_findings) > 1 else asymmetry_findings[0]

                # Margins are independent of shape/distortion, so capture them
                # whenever present.
                if 'mass margins' in mass_rows.columns:
                    margins = mass_rows['mass margins'].dropna().unique().tolist()
                    if margins:
                        row_data['Mass Margin'] = str(margins) if len(margins) > 1 else margins[0]

                if 'mass density' in mass_rows.columns:
                    densities = mass_rows['mass density'].dropna().unique().tolist()
                    if densities:
                        row_data['Mass Density'] = str(densities) if len(densities) > 1 else densities[0]

        for col in ['image file path', 'image_file_path']:
            if col in view_group.columns and not view_group[col].isna().all():
                file_path = view_group[col].iloc[0]
                if isinstance(file_path, str) and '/' in file_path:
                    row_data['Image File Folder (raw)'] = file_path.split('/')[0]
                    break

        study_data.append(row_data)

    return study_data


def process_combined_data(df: pd.DataFrame) -> pd.DataFrame:
    """Harmonizes fields across files and aggregates rows into per-view records."""
    df = df.copy()

    # Harmonize breast density variants.
    if 'breast_density' in df.columns and 'breast density' not in df.columns:
        df['breast density'] = df['breast_density']
    elif 'breast density' in df.columns and 'breast_density' not in df.columns:
        df['breast_density'] = df['breast density']
    elif 'breast_density' in df.columns and 'breast density' in df.columns:
        mask = df['breast density'].isna()
        df.loc[mask, 'breast density'] = df.loc[mask, 'breast_density']

    # Re-check assessment/pathology consistency across the combined set.
    required_cols_integrity = ['patient_id', 'left or right breast', 'assessment', 'pathology']
    if all(col in df.columns for col in required_cols_integrity):
        assessment_consistency = df.groupby(['patient_id', 'left or right breast'])['assessment'].nunique()
        pathology_consistency = df.groupby(['patient_id', 'left or right breast'])['pathology'].nunique()
        consistent_studies = assessment_consistency[
            (assessment_consistency == 1) & (pathology_consistency == 1)
        ].index

        df_tuples = df[['patient_id', 'left or right breast']].to_records(index=False).tolist()
        df = df[pd.Index(df_tuples).isin(consistent_studies)].reset_index(drop=True)

    final_data = []
    if all(col in df.columns for col in ['patient_id', 'left or right breast']):
        grouped = df.groupby(['patient_id', 'left or right breast'])
        for (patient_id, breast_side), study_group in grouped:
            final_data.extend(process_study(study_group, patient_id, breast_side))

    final_columns = [
        'ID', 'Image ID', 'Laterality', 'View', 'Age', 'Breast Density',
        'Diagnosis', 'BI-RADS Assessment', 'Mass', 'Mass Shape', 'Mass Margin',
        'Mass Density', 'Calcification', 'Calcification Morphology',
        'Calcification Distribution', 'Asymmetry', 'Architectural Distortion',
        'Other Findings', 'Split', 'Image File Folder (raw)', 'Image File Path (processed)'
    ]
    return pd.DataFrame(final_data, columns=final_columns)


def process_csv_files(file_paths: List[str]) -> pd.DataFrame:
    """Reads, filters, combines, and standardizes all specified CSV tables."""
    processed_dfs = []

    for file_path in file_paths:
        if not os.path.exists(file_path):
            logger.warning("File path does not exist, skipping: %s", file_path)
            continue

        logger.info("Processing file: %s", file_path)
        df = pd.read_csv(file_path)
        processed_dfs.append(process_single_file(df))

    if processed_dfs:
        combined_df = pd.concat(processed_dfs, ignore_index=True)
        return process_combined_data(combined_df)

    return pd.DataFrame()


def main() -> None:
    args = parse_arguments()

    input_files = [args.calc_test, args.calc_train, args.mass_test, args.mass_train]
    final_df = process_csv_files(input_files)

    if not final_df.empty:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.output_file)
        final_df.to_csv(output_path, index=False)
        logger.info("Processing completed. File saved to: %s", output_path)
    else:
        logger.error("Pipeline resulted in an empty dataset. Output file not generated.")


if __name__ == "__main__":
    main()

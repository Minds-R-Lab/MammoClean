# <p align=center>`MammoClean: Toward Reproducible and Bias-Aware AI in Mammography through Dataset Harmonization`</p> #

MammoClean is a preprocessing pipeline that turns three public mammography datasets (**CBIS-DDSM**, **TOMPEI-CMMD**, and **VinDr-Mammo**) into a single, uniform
format for multi-view breast-image classification. Each dataset ships its
annotations in a different layout, so the pipeline runs in two stages: first it
normalizes every dataset's annotations into one shared CSV schema, then it uses
those unified CSVs to locate, normalize, and export the actual images.

- Our Paper on IEEE Access: [MammoClean: Toward Reproducible and Bias-Aware AI in Mammography Through Dataset Harmonization](https://ieeexplore.ieee.org/document/11569500)

## Citation:
```
@ARTICLE{11569500,
  author={Zafari, Yalda and Pan, Hongyi and Durak, Gorkem and Bagci, Ulas and Rashed, Essam A. and Mabrok, Mohamed},
  journal={IEEE Access}, 
  title={MammoClean: Toward Reproducible and Bias-Aware AI in Mammography Through Dataset Harmonization}, 
  year={2026},
  volume={14},
  pages={93834-93853},
  doi={10.1109/ACCESS.2026.3704749}}
```

## The two-stage pipeline

**Stage 1: Unify the annotations.** Each dataset has its own script under
`csv_files/`. A script reads that dataset's raw metadata (an Excel sheet or one
or more CSVs) and writes a single standardized CSV with the shared column schema. This is the step that hides the per-dataset differences behind
one consistent table.


| Dataset      | Script                | Raw input (under `csv_files/raw/`)                                              | Unified output (under `csv_files/processed/`) |
| ------------ | --------------------- | ------------------------------------------------------------------------------ | --------------------------------------------- |
| CBIS-DDSM    | `ddsm.py`   | `calc_*` and `mass_*` case-description CSVs (test + train)                      | `CBIS-DDSM.csv`                               |
| TOMPEI-CMMD  | `cmmd.py`   | `TOMPEI-CMMD_clinical_data_*.xlsx`                                              | `TOMPEI-CMMD.csv`                             |
| VinDr-Mammo  | `vindr.py`  | `metadata.csv`, `finding_annotations.csv`, `breast-level_annotations.csv`       | `VinDr-Mammo.csv`                             |


**Stage 2: Process the images.** `process_datasets.py` takes one unified CSV
from Stage 1 together with that dataset's raw DICOM images. For each CSV row it
finds the matching DICOM file, normalizes the pixel data to 8-bit grayscale,
corrects the breast laterality, and writes the result as a `.npy` image. It also
writes a per-patient `info.txt` (diagnosis, BI-RADS, and density per side) and a
filtered CSV containing only the rows it could successfully match.


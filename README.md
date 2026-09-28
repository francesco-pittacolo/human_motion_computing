# Human Motion Computing with Inertial Sensing

This repository contains the code developed for the Human Motion Computing assignment based on the KSAS inertial dataset.

The analysis uses XROCKET to investigate movement classification at three levels:

1. sensor channel contribution;
2. temporal scale contribution through XROCKET dilations;
3. discriminative temporal patterns and their responses across movement classes.

## Dataset

The KSAS dataset contains inertial recordings of American Kenpo Karate Blocking Set I.

The dataset used in this project contains:

- 240 recordings;
- 20 participants;
- 6 movement classes;
- 18 sensor channels from six sensor types;
- variable-length recordings ranging from 18 to 56 samples.

The six classes considered in the analysis are:

| Class | Movement |
| ---: | --- |
| 0 | Absence of movement |
| 1 | Upward Block |
| 2 | Hammering Inward Block |
| 3 | Extended Outward Block |
| 4 | Outward Downward Block |
| 5 | Rear Elbow Block |

The dataset is available from:

https://github.com/Physical-User-Modeling-PhyUM/KSAS-Dataset

## Project Structure

```text
human_motion/
├── KSAS-Dataset/
├── xrocket/
├── src/
│   ├── analyze_dataset.py
│   ├── load_dataset.py
│   ├── main_analysis.py
│   ├── pattern_analysis.py
│   └── utils.py
├── results/
├── figures/
├── HMC_report.pdf
├── requirements.txt
└── README.md
```

## Report

The technical report is available here:

[HMC_report.pdf](HMC_report.pdf)

## Setup

The project was developed and tested using Python 3.11.

XROCKET was originally implemented for Python 3.9, while this project was successfully executed using Python 3.11. More recent Python versions may not be compatible with the XROCKET implementation used here.

The XROCKET implementation used in this project is included in the `xrocket/` directory and is based on the original repository:

https://github.com/dida-do/xrocket

Create a Python 3.11 virtual environment:

```bash
py -3.11 -m venv .venv
```

On Windows PowerShell, activate it with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

Install the local XROCKET package from the project root:

```bash
python -m pip install -e .\xrocket --no-deps
```

The `--no-deps` option is used because the required dependencies are installed separately through `requirements.txt`.

## Running the Analysis

All commands should be executed from the project root.

### Dataset inspection

```bash
python src/analyze_dataset.py
```

This script checks the dataset structure and generates summary statistics and a sequence-length figure.

### Main XROCKET analysis

```bash
python src/main_analysis.py
```

This script performs five-fold participant-independent cross-validation using `GroupKFold`.

XROCKET features are evaluated using:

- Random Forest;
- Logistic Regression.

The script also performs:

- Task 1.1: sensor channel contribution analysis;
- Task 1.2: temporal scale analysis based on XROCKET dilation.

### Discriminative pattern analysis

```bash
python src/pattern_analysis.py
```

This script performs Task 1.3 by analysing the most discriminative XROCKET pattern, sensor channel, and dilation combinations.

The responses of the highest-ranked patterns are also compared across movement classes.

## Evaluation Protocol

Participants are used as groups during five-fold cross-validation so that recordings from the same participant never appear in both the training and test sets of the same fold.

The main XROCKET configuration is:

- input channels: 18;
- maximum kernel span: 33;
- kernel length: 9;
- combination order: 1;
- dilations: 1, 2, 3, and 4.

## Outputs

Generated numerical results are stored in:

```text
results/
```

Generated figures are stored in:

```text
figures/
```

The analysis generates the following result files:

```text
results/
├── dataset_summary.csv
├── dataset_class_summary.csv
├── main_cv_fold_results.csv
├── main_cv_summary.csv
├── task_1_1_channel_importance.csv
├── task_1_2_dilation_importance.csv
├── task_1_3_pattern_importance_across_folds.csv
├── task_1_3_class_profiles.csv
└── task_1_3_pattern_summary.csv
```

The generated figures support the dataset analysis and the three explainability tasks.

## Main Results

Participant-independent classification performance:

| Classifier | Accuracy | Macro-F1 |
| --- | ---: | ---: |
| Random Forest | 0.871 ± 0.111 | 0.873 ± 0.107 |
| Logistic Regression | 0.888 ± 0.065 | 0.887 ± 0.063 |

Detailed results for Tasks 1.1, 1.2, and 1.3 are available in the `results/` directory and are discussed in the accompanying report.

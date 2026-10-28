# AI-Enhanced Predictive Maintenance for Industrial Robotics via Multi-Modal Sensor Data Analysis

![IEEE IECON 2026](https://img.shields.io/badge/IEEE-IECON%202026-blue) ![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue)

**Paper:** "AI-Enhanced Predictive Maintenance for Industrial Robotics via Multi-Modal Sensor Data Analysis"
**Authors:** Most Arina Afrin · Elham Ahmadi · Mohammed Elmusrati
**Venue:** 52nd Annual Conference of the IEEE Industrial Electronics Society (IECON 2026), Doha, Qatar. Accepted and presented; publication in the proceedings is pending.
**Institution:** School of Technology and Innovations, University of Vaasa, Finland

## What This Project Does

Industrial robots generate multi-sensor data that may contain early signals of mechanical degradation. This project presents an end-to-end predictive maintenance (PdM) pipeline that:

- Extracts 196 features from raw robot sensor streams across three domains: time, frequency, and cross-sensor correlation
- Benchmarks four classical ML models (Logistic Regression, Random Forest, SVM, XGBoost) and a stacked LSTM on a unified 80/20 chronological split
- Applies SHAP (SHapley Additive exPlanations) to identify which signals drive the anomaly detection decisions
- Maps model outputs and a kinematic-drift RUL proxy to a four-tier decision-support framework, from routine monitoring to emergency stop

### Decision-Support Framework Performance

- 93% of CRITICAL-flagged windows were labelled anomalies
- 197 / 216 ground-truth anomaly windows were captured at HIGH or CRITICAL tier
- Median proxy value at CRITICAL escalation: 4.2 ± 1.8 cycles. This is a health-index proxy, not a validated remaining useful life (see limitations below).

### Top SHAP Feature

`ToolVelocity_mean` accounts for more than 80% of total SHAP attribution. Tool velocity is therefore strongly associated with the non-compliance label. This does not by itself show that the model detects mechanical faults.

## Pipeline Overview

```
Raw Sensor Data (17 channels · 30 users)
        ↓
  Preprocessing (z-score · IQR clip · median imputation)
        ↓
  Feature Engineering (196 features: time-domain · FFT · correlation)
        ↓
  ML + LSTM Models (LR · RF · SVM · XGBoost · LSTM)
        ↓
  Anomaly Score + SHAP + RUL Proxy
        ↓
  4-Tier Maintenance Decision (LOW → MEDIUM → HIGH → CRITICAL)
```

## Dataset

Sheffield Robotics Lab Dataset: 30 users · 17 sensor channels · welding, assembly, and material transport tasks.

| Channel Group | Signals |
|---|---|
| Kinematics | ToolPosition (x,y,z), ToolVelocity, ToolAcceleration |
| Dynamics | ToolForce (x,y,z), ToolTorque (x,y,z) |
| Joint-level | JointPosition 0–6 |
| Status | isCollision, isCompliance |

Download: https://data.mendeley.com/datasets/4fr33dkrjt/3

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/arinaafrin/predictive-maintenance-robotics-iecon2026.git
   cd predictive-maintenance-robotics-iecon2026
   ```

2. Install the dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Download the dataset from Mendeley Data and place the files in the `Data/` folder.

4. Run the notebook:

   ```bash
   jupyter notebook predictive-maintenance.ipynb
   ```

   Or upload the notebook to Google Colab and mount the dataset from Google Drive.

## How to Reproduce the Results

The notebook runs sequentially. Each section corresponds to a section in the paper:

| Notebook Section | Paper Section |
|---|---|
| Data loading & preprocessing | §II-A Dataset and Anomaly Ground Truth |
| Feature engineering (196 features) | §II-B Feature Engineering |
| PCA validation + 3D scatter | §III-A Exploratory Analysis |
| LR / RF / SVM / XGBoost training | §II-C Models |
| LSTM training + threshold tuning | §II-C Models |
| Confusion matrices + ROC/PR curves | §III-B–C Classification Performance |
| SHAP bar | §III-D SHAP Feature Attribution |
| Cross-user generalisation | §III-E Cross-User Generalisation |
| RUL proxy + 4-tier framework | §IV Decision-Support Framework |

**RUL limitation:** The Sheffield dataset does not contain run-to-failure trajectories, so true remaining useful life cannot be estimated from it. The kinematic-drift proxy is a health-index surrogate, not a validated RUL estimate.

## Extension: Uncertainty Analysis (Exploratory Work)

This extension is separate from the IECON 2026 paper. It investigates model calibration, conformal prediction, and uncertainty-aware decision support on held-out users.

Initial checks found that the `non_compliant` label is associated with `Tool_Velocity_mean`. The anomaly-window mean was 0.9431, compared with 0.9999 for normal windows in the current feature dataset. Anomaly windows were spread through the recordings: the first and last 20% had a combined rate of 12.2%, compared with 11.8% in the middle 60%.

These checks show an association between tool velocity and the target label. They do not establish that the model detects robot faults or improves maintenance decisions. The meaning of the label should be confirmed using the dataset documentation and discussed with the research supervisors.

### Set up the extension

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the tests:

```bash
python tests/run_tests.py
```

### Build the feature dataset

If `Data/features.csv` has not been generated, first inspect the available labels:

```bash
python -m uncertainty.build_features --data-dir Data --diagnose
```

If the intended target is non-compliance defined as `isCompliance == False`, build the feature dataset with:

```bash
python -m uncertainty.build_features --data-dir Data --label isCompliance --invert --out Data/features.csv
```

If your study instead uses `isCollision`, use the appropriate label option and keep the target consistent in every subsequent command. Do not assume `isCollision` and `non_compliant` represent the same outcome.

### Run the quick label checks

```bash
python -m uncertainty.quick_checks --csv Data/features.csv --target non_compliant
```

### Run the uncertainty study

```bash
python -m uncertainty.uq_study --csv Data/features.csv --target non_compliant --group User_ID
```

Outputs are written under `results/uncertainty/`.

Synthetic demo data (if invoked with `--demo`) checks code execution only; it is not evidence of real-world performance. Costs in `CostModel` are placeholders unless they have been validated using real operational or maintenance data.

### Next validation steps

The extension should evaluate whether uncertainty estimates remain useful under more difficult conditions:

- Few training users: train on only 3–5 users and test on users not used for training.
- Distribution shift: test on users with unusual velocity profiles.
- Sensor noise or drift: perturb test features and evaluate calibration and conformal coverage.
- Harder target: investigate remaining useful life only if suitable run-to-failure data or a defensible target is available.

Report calibration, coverage, and deferral results only after running the corresponding experiments. Do not claim that uncertainty improves maintenance outcomes unless the results support that conclusion.

## Limitations and Responsible Interpretation

- Strong performance on the current target does not automatically mean the model detects mechanical faults.
- Tool velocity may be strongly associated with the target label; confirm what the labels represent before drawing physical conclusions.
- The dataset does not provide run-to-failure trajectories for direct RUL estimation.
- Decision thresholds and maintenance costs require validation before use in operational settings.
- Stress tests are needed to understand how calibration, coverage, and predictions behave under limited data, distribution shift, and sensor noise.

## Contact

| Name | Role | Email |
|---|---|---|
| Most Arina Afrin | First author | arinaafrin05@gmail.com |
| Elham Ahmadi | Co-supervisor | elham.ahmadi@uwasa.fi |
| Mohammed Elmusrati | Supervisor | Mohammed.elmusrati@uwasa.fi |

School of Technology and Innovations, University of Vaasa, Vaasa, Finland

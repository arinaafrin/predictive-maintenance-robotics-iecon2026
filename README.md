# AI-Enhanced Predictive Maintenance for Industrial Robotics
## via Multi-Modal Sensor Data Analysis

[![IEEE IECON 2026](https://img.shields.io/badge/IEEE-IECON%202026-blue)](https://www.iecon2026.org/)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)

> **Paper:** "AI-Enhanced Predictive Maintenance for Industrial Robotics via Multi-Modal Sensor Data Analysis"  
> **Authors:** Most Arina Afrin · Elham Ahmadi · Mohammed Elmusrati  
> **Venue:** 50th Annual Conference of the IEEE Industrial Electronics Society (IECON 2026)  
> **Institution:** School of Technology and Innovation, University of Vaasa, Finland

---

## What This Project Does

Industrial robots generate rich multi-sensor data that contains early signals of mechanical degradation — long before a fault becomes critical. This project presents a complete, end-to-end **Predictive Maintenance (PdM)** pipeline that:

- Extracts **196 features** from raw robot sensor streams across three domains: time, frequency, and cross-sensor correlation
- Benchmarks **four classical ML models** (Logistic Regression, Random Forest, SVM, XGBoost) and a **stacked LSTM** on a fair, unified 80/20 chronological split
- Applies **SHAP (SHapley Additive exPlanations)** to identify which sensor signals drive anomaly detection decisions
- Delivers a **four-tier decision-support framework** that maps model outputs and a kinematic-drift RUL proxy to concrete maintenance actions — from routine monitoring to emergency stop

### Decision-Support Framework Performance

- **93%** of CRITICAL-flagged windows were genuine anomalies
- **197 / 216** ground-truth anomaly windows captured at HIGH or CRITICAL tier
- Median RUL at CRITICAL escalation: **4.2 ± 1.8 cycles** — sufficient lead time for action

### Top SHAP Feature

`ToolVelocity_mean` accounts for **>80%** of total SHAP attribution, confirming that tool speed is the primary signal separating compliant from non-compliant robot motion.

---

## Pipeline Overview

```
Raw Sensor Data (17 channels · 30 users)
        ↓
  Preprocessing
  (z-score · IQR clip · median imputation)
        ↓
  Feature Engineering
  (196 features: time-domain · FFT · correlation)
        ↓
  ML + LSTM Models
  (LR · RF · SVM · XGBoost · LSTM)
        ↓
  Anomaly Score + SHAP + RUL Proxy
  P(anomaly) + φᵢ + RUL̂
        ↓
  4-Tier Maintenance Decision
  LOW → MEDIUM → HIGH → CRITICAL
```

---

## Dataset

**Sheffield Robotics Lab Dataset**  
30 users · 17 sensor channels · Welding, assembly, and material transport tasks

| Channel Group | Signals |
|---|---|
| Kinematics | ToolPosition (x,y,z), ToolVelocity, ToolAcceleration |
| Dynamics | ToolForce (x,y,z), ToolTorque (x,y,z) |
| Joint-level | JointPosition 0–6 |
| Status | isCollision, isCompliance |

**Download:** https://data.mendeley.com/datasets/4fr33dkrjt/3

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/predictive-maintenance-robotics-iecon2026.git
cd predictive-maintenance-robotics-iecon2026
```

### 3. Download the dataset

Download from [Mendeley Data](https://data.mendeley.com/datasets/4fr33dkrjt/3) and place the files in:


### 4. Run the notebook

```bash
jupyter notebook notebook/predictive-maintenance.ipynb
```

Or open directly in **Google Colab** by uploading the notebook and mounting the dataset from Google Drive.

---

## How to Reproduce the Results

The notebook is fully self-contained and runs sequentially. Each section corresponds directly to a section in the paper:

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

---

> Note: The Sheffield dataset does not contain run-to-failure trajectories, so true RUL estimation is not possible. The kinematic drift proxy serves as a health-index surrogate for maintenance prioritisation.

---

## Contact

| Name | Role | Email |
|---|---|---|
| Most Arina Afrin | --- | arinaafrin05@gmail.com |
| Elham Ahmadi | Supervisor | elham.ahmadi@uwasa.fi |
| Mohammed Elmusrati | Supervisor | moel@uwasa.fi |

**School of Technology and Innovation**  
University of Vaasa, Vaasa, Finland
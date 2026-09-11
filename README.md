# The Dysgraphia Detective

A desktop application and machine-learning pipeline that screens for **dysgraphia** from
**online handwriting** — the way a person writes, captured live on a graphics tablet rather
than from a scanned image.

As someone writes, the system records the full pen trajectory, turns it into a compact
**119-feature descriptor**, and uses trained classifiers to flag handwriting consistent with
dysgraphia.

## How it works

### 1. Handwriting capture
Samples are captured on an **XP-Pen Deco 01** graphics tablet through a PyQt6 application.
While the user writes, the app records, for every pen sample:

- **position** (`x`, `y`),
- **pen pressure**, and
- a **timestamp**.

Each contiguous touch of the pen on the surface is treated as one **segment** (stroke), and
pen-lifts between segments are tracked. (The Deco 01 stylus does not report tilt, so no
altitude/azimuth signals are used — all features are derived from position, pressure and time.)

### 2. Feature extraction (119 features per task)
From the captured trajectory the pipeline computes velocity, acceleration and jerk by
differentiating position over time, then summarises the writing into **119 features**:

| Group | Detail | Count |
|-------|--------|------:|
| **Vector statistics** | 10 signal families — velocity, acceleration and jerk (each as magnitude, x, and y) plus pressure — each summarised by 7 statistics (mean, median, std, max, min, 5th & 95th percentile) | 70 |
| **Segment statistics** | 6 per-stroke measures — duration, path length, vertical & horizontal extent, width, height — each summarised by 5 statistics (mean, median, std, max, min) | 30 |
| **Scalar features** | pen-lift count, velocity/acceleration extrema counts, total writing time, total path/vertical/horizontal length, baseline drift between first/last and second/penultimate strokes, and variance of per-stroke vertical position | 19 |
| | | **119** |

The extraction is implemented in `dysgraphia_ml.py` (`compute_drotar_features`).

### 3. Classification
The 119-feature vector is fed to five classifiers, trained with grid-searched
hyperparameters (`grid_search_optimize.py` → `best_hyperparameters.json`):
Random Forest, SVM (RBF kernel), Logistic Regression, Gradient Boosting, and AdaBoost.

## Results

### Internal cross-validation (own dataset)
Cross-validated accuracy of each classifier on the project's captured sessions:

| Classifier | CV accuracy |
|------------|------------:|
| SVM (RBF)         | 0.915 |
| Gradient Boosting | 0.907 |
| Random Forest     | 0.906 |
| AdaBoost          | 0.882 |
| Logistic Reg.     | 0.842 |

### External validation
To test how well the classifiers generalise beyond the data they were built on, the same
119-feature pipeline and classifiers were evaluated on an **independent public handwriting
dataset** (Drotár & Dobeš, 119 subjects) — used here purely as an **external validation set**.
Reported as mean over repeated stratified 10-fold cross-validation:

| Classifier | Accuracy | Sensitivity | Specificity | F1 |
|------------|---------:|------------:|------------:|----:|
| SVM (RBF)         | 0.723 | 0.627 | 0.811 | 0.668 |
| Logistic Reg.     | 0.704 | 0.636 | 0.763 | 0.657 |
| Random Forest     | 0.701 | 0.633 | 0.763 | 0.653 |
| Gradient Boosting | 0.700 | 0.675 | 0.722 | 0.672 |
| AdaBoost          | 0.653 | 0.618 | 0.682 | 0.614 |

The external scores are lower than the internal ones — expected, since the validation data
comes from a different population and capture setup — but stay well above chance, indicating
the features and models carry real, transferable signal.

## Project layout

| File | Purpose |
|------|---------|
| `dysgraphia_detective.py` | Main PyQt6 application — capture and analyse a handwriting sample. |
| `dysgraphia_ml.py`        | Feature extraction + model training. |
| `grid_search_optimize.py` | Grid search over classifier hyperparameters → writes `best_hyperparameters.json`. |
| `evaluate_clinical.py`    | Clinical evaluation metrics (sensitivity/specificity, etc.). |
| `drotar_loader.py`        | Parser for the external-validation dataset's pen-sample files. |
| `drotar_internal_validation.py` | Runs the external validation (repeated stratified 10-fold CV). |
| `backfill_drotar_features.py` | Re-extract the 119 features for every stored session. |
| `best_hyperparameters.json` | Chosen hyperparameters consumed by the training code. |
| `dataset/` | Captured handwriting sessions (`fluent/`, `dysgraphic/`) as JSON + trained `.pkl` models. |

## Setup

Requires **Python 3.10+**.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

## Usage

```bash
# Launch the capture/analysis GUI
python dysgraphia_detective.py

# Extract features and train / compare classifiers
python dysgraphia_ml.py

# Optimise hyperparameters
python grid_search_optimize.py

# Clinical evaluation report
python evaluate_clinical.py

# External validation on the independent dataset
python drotar_internal_validation.py
```

## Data availability

The project's own captured sessions are included under `dataset/`.

The **external validation dataset** is a third-party clinical dataset and is **not** included
in this repository — it must be obtained from its original source. The loader expects its
`.svc` files in a `drotar_dataset/` folder at the project root (git-ignored).

## License

No license is currently specified — default copyright applies. Add a license file
(e.g. MIT) if you want others to reuse this code.

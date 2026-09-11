"""
grid_search_optimize.py — Tune hyperparameters of all 5 classifiers via
stratified 5-fold GridSearchCV on both the XCV and Drotár datasets.

Reports the best parameters and best CV accuracy per classifier. Writes a
JSON file with the chosen hyperparameters that train_ml_models() can load
on the next training run.

Usage:
  python grid_search_optimize.py
"""

import sys, json, pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dysgraphia_ml import (compute_drotar_features, DROTAR_FEATURE_KEYS,
                           _load_real_sessions)
from drotar_loader import iter_subjects


# ────────────────────────────────────────────────────────────────────────────
# Grids — chosen to be wide enough to matter, narrow enough to finish in mins
# ────────────────────────────────────────────────────────────────────────────
def make_grids():
    return {
        'RF': (
            RandomForestClassifier(class_weight='balanced',
                                   random_state=42, n_jobs=-1),
            {
                'n_estimators':     [200, 300, 500],
                'max_features':     ['sqrt', 'log2'],
                'min_samples_leaf': [1, 2, 4],
            },
            False,  # don't standardise
        ),
        'SVM': (
            SVC(kernel='rbf', class_weight='balanced',
                probability=True, random_state=42),
            {
                'C':     [0.5, 1, 2, 4, 8],
                'gamma': ['scale', 'auto', 0.01, 0.1],
            },
            True,
        ),
        'LR': (
            LogisticRegression(max_iter=2000, class_weight='balanced',
                               random_state=42),
            {
                'C':       [0.1, 0.5, 1, 5],
                'solver':  ['lbfgs', 'liblinear'],
            },
            True,
        ),
        'GB': (
            GradientBoostingClassifier(random_state=42),
            {
                'n_estimators':  [100, 200],
                'learning_rate': [0.05, 0.1, 0.2],
                'max_depth':     [3, 5],
            },
            True,
        ),
        'AB': (
            AdaBoostClassifier(random_state=42),
            {
                'n_estimators':  [100, 200, 340, 500],
                'learning_rate': [0.5, 1.0, 1.5],
            },
            True,
        ),
    }


# ────────────────────────────────────────────────────────────────────────────
# Per-dataset grid search
# ────────────────────────────────────────────────────────────────────────────
def run_grid_search(X, y, dataset_label):
    print(f"\n{'='*60}")
    print(f"Grid search on {dataset_label}  (n={len(X)}, p={X.shape[1]})")
    print(f"{'='*60}")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}
    for name, (estimator, grid, scale) in make_grids().items():
        if scale:
            pipe = Pipeline([('scaler', StandardScaler()),
                             ('clf',    estimator)])
            grid = {f'clf__{k}': v for k, v in grid.items()}
        else:
            pipe = estimator
        gs = GridSearchCV(pipe, grid, cv=cv, scoring='accuracy',
                          n_jobs=-1, verbose=0)
        gs.fit(X, y)
        clean_params = {k.replace('clf__', ''): v
                        for k, v in gs.best_params_.items()}
        results[name] = {
            'best_score':  float(gs.best_score_),
            'best_params': clean_params,
            'mean_std':    float(gs.cv_results_['std_test_score'][gs.best_index_]),
        }
        param_str = ", ".join(f"{k}={v}" for k, v in clean_params.items())
        print(f"  {name:<4}  acc={gs.best_score_*100:>5.2f}% "
              f"+/-{gs.cv_results_['std_test_score'][gs.best_index_]*100:.1f}%   "
              f"params: {param_str}")
    return results


# ────────────────────────────────────────────────────────────────────────────
# Build Drotár dataset
# ────────────────────────────────────────────────────────────────────────────
def load_drotar():
    """Build the (N x 119) Drot\'ar feature matrix using the canonical
    on-surface descriptor (no altitude/azimuth)."""
    X, y = [], []
    for sid, strokes, raw, lbl in iter_subjects():
        try:
            feats = compute_drotar_features(strokes)
            if feats is None:
                continue
            v = [float(feats.get(k, 0.0)) for k in DROTAR_FEATURE_KEYS]
            X.append(v); y.append(lbl)
        except Exception:
            continue
    return np.array(X, dtype=float), np.array(y, dtype=int)


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    out = {}

    # XCV on-surface (119 features)
    X, y, nf, nd = _load_real_sessions()
    out['XCV_online'] = run_grid_search(X, y, "XCV on-surface (119 features)")

    # Drot\'ar cohort, same 119-feature descriptor
    Xd, yd = load_drotar()
    out['Drotar'] = run_grid_search(Xd, yd, "Drotar cohort (119 features)")

    # ── Pick "consensus" best hyperparameters per classifier ────────────────
    # Average the best params across both datasets — prefer params that work
    # for both. For categorical params, take the XCV choice (larger n=246).
    print("\n" + "=" * 60)
    print("CHOSEN HYPERPARAMETERS  (used as new defaults)")
    print("=" * 60)
    chosen = {}
    for clf in out['XCV_online']:
        p_XCV = out['XCV_online'][clf]['best_params']
        p_drt = out['Drotar'][clf]['best_params']
        # Always prefer the XCV params for production (larger dataset)
        chosen[clf] = p_XCV
        # But report both for transparency
        print(f"  {clf}:  XCV -> {p_XCV}")
        print(f"          DRT -> {p_drt}")
        if p_XCV == p_drt:
            print(f"          AGREEMENT")
    out['chosen'] = chosen

    out_path = Path(__file__).parent / 'best_hyperparameters.json'
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved hyperparameters: {out_path}")
    print("\nNext step: dysgraphia_ml.train_ml_models() will read this file "
          "if present, otherwise fall back to its hard-coded defaults.")


if __name__ == '__main__':
    main()

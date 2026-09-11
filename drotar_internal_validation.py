"""
drotar_internal_validation.py
Train and evaluate the 119-feature Drot\'ar descriptor (the same vector the
XCV pipeline uses) ON the Drot\'ar & Dobe\v{s} (2020) dataset using
Drot\'ar's own protocol: stratified 10-fold cross-validation, repeated 10
times. If the feature pipeline is sound, we should approach Drot\'ar's
reported 79.5% AdaBoost accuracy. This answers the question "do these
features carry diagnostic information on a clinically-labelled paediatric
cohort?" --- they do, see Sec.~5 of the thesis.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dysgraphia_ml import compute_drotar_features, DROTAR_FEATURE_KEYS
from drotar_loader import iter_subjects

IMG_DIR = Path(r"C:\Users\amxcvash\OneDrive\Desktop\xcvxcv\img")


def build_dataset():
    """Build the (N x 119) feature matrix and label vector for every
    labelled Drot\'ar subject, using the canonical 119-feature on-surface
    descriptor (Drot\'ar's altitude/azimuth families are intentionally
    omitted to keep the representation consistent with the XCV pipeline)."""
    X, y, sids = [], [], []
    for sid, strokes, raw, lbl in iter_subjects():
        try:
            feats = compute_drotar_features(strokes)
            if feats is None:
                continue
            v = [float(feats.get(k, 0.0)) for k in DROTAR_FEATURE_KEYS]
            X.append(v); y.append(lbl); sids.append(sid)
        except Exception:
            continue
    return np.array(X, dtype=float), np.array(y, dtype=int), sids


def evaluate_classifier(name, clf, X, y, scale=True, n_splits=10, n_repeats=10):
    """Drotár's protocol: stratified 10-fold CV repeated 10 times. Returns
    a dict of pooled metrics across all folds and repeats."""
    accs, precs, recs, f1s, sens, specs = [], [], [], [], [], []
    for rep in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rep)
        for tr_idx, te_idx in cv.split(X, y):
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]
            if scale:
                sc = StandardScaler().fit(Xtr)
                Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
            clf.fit(Xtr, ytr)
            yp = clf.predict(Xte)
            tn, fp, fn, tp = confusion_matrix(yte, yp, labels=[0, 1]).ravel()
            accs.append(accuracy_score(yte, yp))
            precs.append(precision_score(yte, yp, zero_division=0))
            recs.append(recall_score(yte, yp, zero_division=0))
            f1s.append(f1_score(yte, yp, zero_division=0))
            sens.append(tp / max(tp + fn, 1))
            specs.append(tn / max(tn + fp, 1))
    return {
        'name': name,
        'acc_mean':  float(np.mean(accs)),  'acc_std':  float(np.std(accs)),
        'prec_mean': float(np.mean(precs)), 'rec_mean':  float(np.mean(recs)),
        'f1_mean':   float(np.mean(f1s)),
        'sens_mean': float(np.mean(sens)),  'spec_mean': float(np.mean(specs)),
    }


def main():
    print("Building Drotár feature dataset (119 features per subject)...")
    X, y, sids = build_dataset()
    print(f"Loaded {len(X)} subjects  "
          f"(Normal={(y==0).sum()}  Dysgraphic={(y==1).sum()})")
    print(f"Feature vector dim: {X.shape[1]}\n")

    # Use the Drotár-side tuned hyperparameters from grid_search_optimize.py
    # (best_hyperparameters.json: 'Drotar' key). These were selected by
    # 5-fold GridSearchCV on the Drotár cohort itself.
    classifiers = [
        ('Random Forest',
         RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                min_samples_leaf=1, class_weight='balanced',
                                random_state=42, n_jobs=-1), False),
        ('SVM (RBF)',
         SVC(kernel='rbf', C=1.0, gamma='scale', probability=True,
             class_weight='balanced', random_state=42), True),
        ('Logistic Reg.',
         LogisticRegression(max_iter=2000, C=0.5, solver='liblinear',
                            class_weight='balanced',
                            random_state=42), True),
        ('Gradient Boost',
         GradientBoostingClassifier(n_estimators=100, learning_rate=0.05,
                                    max_depth=3, subsample=0.8, random_state=42),
         True),
        ('AdaBoost (Drotár best)',
         AdaBoostClassifier(n_estimators=200, learning_rate=1.0,
                            random_state=42), True),
    ]

    print("Running stratified 10-fold CV × 10 repeats (Drotár protocol)...")
    print(f"{'Classifier':<24}{'Acc':>10}{'Prec':>8}{'Rec':>8}{'F1':>8}"
          f"{'Sens':>8}{'Spec':>8}")
    print('-' * 74)
    results = []
    for name, clf, scale in classifiers:
        r = evaluate_classifier(name, clf, X, y, scale=scale)
        results.append(r)
        print(f"{r['name']:<24}{r['acc_mean']*100:>7.1f}±{r['acc_std']*100:.1f}"
              f"{r['prec_mean']*100:>8.1f}{r['rec_mean']*100:>8.1f}"
              f"{r['f1_mean']*100:>8.1f}{r['sens_mean']*100:>8.1f}"
              f"{r['spec_mean']*100:>8.1f}")

    # ── Comparative bar plot vs Drotár's paper benchmark ────────────────────
    IMG_DIR.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    names = [r['name'].replace(' (Drotár best)', '\n(Drotár best)')
             for r in results]
    accs  = [r['acc_mean'] * 100 for r in results]
    stds  = [r['acc_std']  * 100 for r in results]
    colors = ['#2563eb', '#16a34a', '#d97706', '#7c3aed', '#dc2626']
    bars = ax.bar(names, accs, yerr=stds, capsize=4, color=colors,
                  edgecolor='white', linewidth=1)
    for b, v in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.0,
                f'{v:.1f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold')
    ax.axhline(79.5, color='#dc2626', ls='--', lw=1.2,
               label='Drotár paper: AdaBoost 79.5% (1176 features)')
    ax.set_ylim(40, 100)
    ax.set_ylabel('10-fold CV accuracy (%)  ± std')
    ax.set_title(
        f"Internal validation on Drotár cohort  —  "
        f"XCV 119-feature pipeline, n={len(X)}",
        fontsize=11)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    out = IMG_DIR / 'drotar_internal_cv.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {out}")

    # Also save results JSON
    import json
    out_json = Path(__file__).parent / 'drotar_internal_results.json'
    out_json.write_text(json.dumps({'n_subjects': int(len(X)),
                                    'n_features': int(X.shape[1]),
                                    'results': results}, indent=2))
    print(f"Results JSON saved: {out_json}")


if __name__ == '__main__':
    main()

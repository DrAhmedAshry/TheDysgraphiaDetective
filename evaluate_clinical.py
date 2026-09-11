"""
evaluate_clinical.py — Supervisor-requested clinical evaluation script.

Computes:
  1. Precision / Recall / F1 / Sensitivity / Specificity per classifier
  2. Confusion matrices for 25-30+ subject evaluation
  3. MDTW (Multidimensional Dynamic Time Warping) golden-vs-dysgraphic comparison
  4. Topological stroke connectivity comparison (control vs dysgraphic)
  5. Saves all figures to img/ for the LaTeX thesis

Requires:
  pip install scikit-learn matplotlib numpy scipy
"""

import json
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
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, classification_report
)

DATASET_DIR = Path(__file__).resolve().parent / "dataset"
IMG_DIR     = Path(r"C:\Users\amxcvash\OneDrive\Desktop\xcvxcv\img")
IMG_DIR.mkdir(exist_ok=True, parents=True)

# =============================================================================
# 1. Load real-session feature vectors
# =============================================================================
def load_features():
    rows, labels = [], []
    for lbl_int, folder in ((0, 'fluent'), (1, 'dysgraphic')):
        jdir = DATASET_DIR / folder / 'json'
        for jf in sorted(jdir.glob('*.json')):
            d = json.loads(jf.read_text(encoding='utf-8'))
            from dysgraphia_ml import (compute_drotar_features,
                                       DROTAR_FEATURE_KEYS)
            feats = d.get('drotar_features')
            if not feats:
                strokes = d.get('strokes')
                if not strokes:
                    continue
                feats = compute_drotar_features(strokes)
                if feats is None:
                    continue
            rows.append([float(feats.get(k, 0.0)) for k in DROTAR_FEATURE_KEYS])
            labels.append(lbl_int)
    return np.array(rows), np.array(labels)

# =============================================================================
# 2. Train + Clinical Metrics
# =============================================================================
def train_and_evaluate():
    X, y = load_features()
    print(f"Loaded {len(y)} sessions: {(y==0).sum()} Fluent / {(y==1).sum()} Dysgraphic")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    classifiers = {
        'Random Forest': (RandomForestClassifier(n_estimators=300, max_features='sqrt',
                          min_samples_leaf=2, class_weight='balanced',
                          random_state=42, n_jobs=-1), X_tr, X_te),
        'SVM (RBF)':     (SVC(kernel='rbf', C=2.0, gamma='scale', probability=True,
                          class_weight='balanced', random_state=42), X_tr_s, X_te_s),
        'Logistic Reg.': (LogisticRegression(max_iter=2000, C=0.5,
                          class_weight='balanced', random_state=42), X_tr_s, X_te_s),
        'Gradient Boost':(GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                          max_depth=3, subsample=0.8, random_state=42), X_tr_s, X_te_s),
        'AdaBoost':      (AdaBoostClassifier(n_estimators=340, learning_rate=1.0,
                          random_state=42), X_tr_s, X_te_s),
    }

    metrics = {}
    cms = {}
    for name, (clf, Xt, Xtest) in classifiers.items():
        clf.fit(Xt, y_tr)
        y_pred = clf.predict(Xtest)
        cm = confusion_matrix(y_te, y_pred)
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp+fn) else 0.0   # = recall for positive class
        specificity = tn / (tn + fp) if (tn+fp) else 0.0
        metrics[name] = {
            'accuracy':    accuracy_score(y_te, y_pred),
            'precision':   precision_score(y_te, y_pred, zero_division=0),
            'recall':      recall_score(y_te, y_pred, zero_division=0),
            'f1':          f1_score(y_te, y_pred, zero_division=0),
            'sensitivity': sensitivity,
            'specificity': specificity,
            'cm':          cm,
        }
        cms[name] = cm

    # Ensemble (majority vote of all 5 classifiers: RF + SVM + LR + GB + AB)
    preds = []
    for name, (clf, Xt, Xtest) in classifiers.items():
        preds.append(clf.predict(Xtest))   # already fitted in the loop above
    preds = np.array(preds)
    ens_pred = (preds.mean(0) >= 0.5).astype(int)
    cm = confusion_matrix(y_te, ens_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics['Ensemble'] = {
        'accuracy':    accuracy_score(y_te, ens_pred),
        'precision':   precision_score(y_te, ens_pred, zero_division=0),
        'recall':      recall_score(y_te, ens_pred, zero_division=0),
        'f1':          f1_score(y_te, ens_pred, zero_division=0),
        'sensitivity': tp / (tp+fn) if (tp+fn) else 0.0,
        'specificity': tn / (tn+fp) if (tn+fp) else 0.0,
        'cm':          cm,
    }
    return metrics, len(y_te)

# =============================================================================
# 3. MDTW: Golden vs Dysgraphic letter comparison
# =============================================================================
def dtw_distance(seq_a, seq_b):
    """Classic Dynamic Time Warping, Euclidean distance per pair of (x,y) points.
    Returns (cumulative distance, warping path)."""
    n, m = len(seq_a), len(seq_b)
    inf = float('inf')
    cost = np.full((n+1, m+1), inf)
    cost[0,0] = 0.0
    for i in range(1, n+1):
        for j in range(1, m+1):
            d = np.linalg.norm(seq_a[i-1] - seq_b[j-1])
            cost[i,j] = d + min(cost[i-1,j], cost[i,j-1], cost[i-1,j-1])
    # backtrack
    i, j = n, m
    path = []
    while i>0 and j>0:
        path.append((i-1, j-1))
        c = np.argmin([cost[i-1,j-1], cost[i-1,j], cost[i,j-1]])
        if c==0: i-=1; j-=1
        elif c==1: i-=1
        else: j-=1
    return cost[n,m], list(reversed(path))

def _first_long_stroke(strokes, min_pts=30):
    """Pick a representative stroke for DTW (longest stroke with >= min_pts samples)."""
    candidates = [s for s in strokes if len(s) >= min_pts]
    if not candidates:
        candidates = strokes
    return max(candidates, key=len)

def mdtw_figure():
    flu_files = sorted((DATASET_DIR/'fluent/json').glob('*.json'))
    dys_files = sorted((DATASET_DIR/'dysgraphic/json').glob('*.json'))
    flu = json.loads(flu_files[0].read_text(encoding='utf-8'))
    dys = json.loads(dys_files[0].read_text(encoding='utf-8'))

    s_flu = np.array([[p[0], p[1]] for p in _first_long_stroke(flu['strokes'])])
    s_dys = np.array([[p[0], p[1]] for p in _first_long_stroke(dys['strokes'])])
    # normalise translation: subtract starting point
    s_flu = s_flu - s_flu[0]
    s_dys = s_dys - s_dys[0]

    d, path = dtw_distance(s_flu, s_dys)
    norm_d = d / max(len(path), 1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    # Left: trajectories (overlay)
    ax[0].plot(s_flu[:,0], -s_flu[:,1], 'b-', lw=2, label='Golden (Fluent)', alpha=0.8)
    ax[0].plot(s_dys[:,0], -s_dys[:,1], 'r-', lw=2, label='Dysgraphic attempt', alpha=0.8)
    ax[0].scatter(s_flu[0,0], -s_flu[0,1], c='blue', s=80, marker='o', zorder=5)
    ax[0].scatter(s_dys[0,0], -s_dys[0,1], c='red',  s=80, marker='o', zorder=5)
    ax[0].set_title(f'Stroke trajectories (translation-normalised)\nMDTW = {d:.1f} px (normalised {norm_d:.2f} px/pair)',
                    fontsize=11)
    ax[0].set_xlabel('x (px)'); ax[0].set_ylabel('-y (px)')
    ax[0].legend(loc='best'); ax[0].grid(alpha=0.3); ax[0].set_aspect('equal', 'datalim')

    # Right: warping path matrix
    n, m = len(s_flu), len(s_dys)
    cost = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            cost[i,j] = np.linalg.norm(s_flu[i] - s_dys[j])
    im = ax[1].imshow(cost, cmap='viridis', origin='lower', aspect='auto')
    px = [p[0] for p in path]; py = [p[1] for p in path]
    ax[1].plot(py, px, 'r-', lw=2, label='DTW warping path')
    ax[1].set_xlabel('Dysgraphic sample index')
    ax[1].set_ylabel('Fluent (golden) sample index')
    ax[1].set_title('Local Euclidean cost matrix\n+ optimal warping path')
    ax[1].legend(loc='lower right')
    plt.colorbar(im, ax=ax[1], fraction=0.04, label='Local distance (px)')

    out = IMG_DIR / 'mdtw_comparison.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"MDTW figure saved: {out}  (distance={d:.1f}, norm={norm_d:.2f})")
    return d, norm_d

# =============================================================================
# 4. Topological stroke connectivity comparison
# =============================================================================
def topological_figure():
    flu = json.loads(sorted((DATASET_DIR/'fluent/json').glob('*.json'))[0].read_text(encoding='utf-8'))
    dys = json.loads(sorted((DATASET_DIR/'dysgraphic/json').glob('*.json'))[0].read_text(encoding='utf-8'))

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for k, (data, title, colour) in enumerate([
        (flu, 'Control subject — stroke connectivity', '#2563eb'),
        (dys, 'Dysgraphic subject — stroke connectivity', '#dc2626'),
    ]):
        # Draw guidelines
        for g in data['spatial_features'].get('guidelines', [70,140,210,280,350]):
            ax[k].axhline(-g, color='#bbb', ls='--', lw=0.8, alpha=0.7)

        # Each stroke a different shade so connectivity is visible
        for si, s in enumerate(data['strokes']):
            pts = np.array([[p[0], p[1]] for p in s])
            if len(pts) < 2: continue
            ax[k].plot(pts[:,0], -pts[:,1], '-', color=colour, lw=1.5, alpha=0.85)
            ax[k].scatter(pts[0,0], -pts[0,1], c='green', s=22, marker='o', zorder=5)
            ax[k].scatter(pts[-1,0], -pts[-1,1], c='black', s=22, marker='x', zorder=5)
        n_lifts = data['kinematic_features']['n_lifts']
        in_air  = data['kinematic_features']['in_air_ratio']
        ax[k].set_title(f"{title}\nstrokes={len(data['strokes'])}  pen lifts={n_lifts}  in-air ratio={in_air:.2f}",
                        fontsize=11)
        ax[k].set_xlabel('x (px)'); ax[k].set_ylabel('-y (px)')
        ax[k].set_aspect('equal', 'datalim')
        ax[k].grid(alpha=0.2)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker='o', color='w', markerfacecolor='green', markersize=8, label='Stroke start'),
        Line2D([0],[0], marker='x', color='black', markersize=8, label='Stroke end (pen lift)'),
    ]
    fig.legend(handles=handles, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5,-0.02))

    out = IMG_DIR / 'topological_comparison.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Topological figure saved: {out}")

# =============================================================================
# 5. Confusion matrix figure
# =============================================================================
def confusion_matrix_figure(metrics):
    # Plot the 5 individual classifiers (skip the Ensemble row to keep it readable)
    panels = [(n, m) for n, m in metrics.items() if n != 'Ensemble']
    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.6 * n_panels, 4),
                              constrained_layout=True)
    if n_panels == 1:
        axes = [axes]
    labels = ['Fluent', 'Dysgraphic']
    for ax, (name, m) in zip(axes, panels):
        cm = m['cm']
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(f"{name}\nF1={m['f1']:.2f}  Sens={m['sensitivity']:.2f}  Spec={m['specificity']:.2f}",
                     fontsize=10)
        for i in range(2):
            for j in range(2):
                color = 'white' if cm[i,j] > cm.max()/2 else 'black'
                ax.text(j, i, str(cm[i,j]), ha='center', va='center',
                        color=color, fontsize=18, fontweight='bold')
    out = IMG_DIR / 'confusion_matrices.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Confusion matrices saved: {out}")

# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    print("="*70)
    print(" Clinical Evaluation — Supervisor's Required Metrics")
    print("="*70)
    metrics, n_test = train_and_evaluate()
    print(f"\nHeld-out test set: {n_test} sessions\n")
    print(f"{'Classifier':<18}{'Acc':>7}{'Prec':>7}{'Rec':>7}{'F1':>7}{'Sens':>7}{'Spec':>7}")
    print("-" * 60)
    for name, m in metrics.items():
        print(f"{name:<18}{m['accuracy']:>7.3f}{m['precision']:>7.3f}{m['recall']:>7.3f}"
              f"{m['f1']:>7.3f}{m['sensitivity']:>7.3f}{m['specificity']:>7.3f}")
        print(f"  Confusion matrix: TN={m['cm'][0,0]} FP={m['cm'][0,1]} "
              f"FN={m['cm'][1,0]} TP={m['cm'][1,1]}")

    print("\nGenerating figures...")
    confusion_matrix_figure(metrics)
    d, nd = mdtw_figure()
    topological_figure()

    # Write a LaTeX-ready metrics fragment
    out_tex = Path(r"C:\Users\amxcvash\OneDrive\Desktop\xcvxcv") / 'metrics_table.tex'
    with open(out_tex, 'w', encoding='utf-8') as f:
        f.write("% Auto-generated by evaluate_clinical.py\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Clinical evaluation metrics on the held-out validation set "
                f"($n={n_test}$ sessions, stratified 25\\% split from 246 real sessions). "
                "Sensitivity is recall for the Dysgraphic class; specificity is recall for the Fluent class.}\n")
        f.write("\\label{tab:clinical-metrics}\n")
        f.write("\\begin{tabular}{lcccccc}\n\\hline\n")
        f.write("Classifier & Accuracy & Precision & Recall & F1 & Sensitivity & Specificity \\\\\n\\hline\n")
        for name, m in metrics.items():
            f.write(f"{name} & {m['accuracy']:.3f} & {m['precision']:.3f} & "
                    f"{m['recall']:.3f} & {m['f1']:.3f} & "
                    f"{m['sensitivity']:.3f} & {m['specificity']:.3f} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"\nLaTeX table saved: {out_tex}")
    print(f"\nMDTW summary: distance = {d:.1f} px, normalised = {nd:.2f} px/pair")

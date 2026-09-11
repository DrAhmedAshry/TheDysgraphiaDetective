"""
dysgraphia_ml.py  —  ML Trainer
The Dysgraphia Detective | Bachelor Thesis SS26

Separate ML training application:
  - Draw on canvas and save as Fluent / Dysgraphic (training data)
  - Save as Test Sample (saved to dataset/test/fluent/ or dataset/test/dysgraphic/)
  - Train 4 classifiers with cross-validation and learning curves
  - 3 ML result tabs: Accuracy | Confusion Matrix | Learning Curve
"""

import sys
import time
import json
import pickle
import subprocess
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats
from collections import defaultdict

try:
    from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                                  AdaBoostClassifier)
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import (train_test_split, cross_val_score,
                                         StratifiedKFold, learning_curve)
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import (accuracy_score, confusion_matrix as sk_cm,
                                 precision_score, recall_score, f1_score)
    from sklearn.pipeline import Pipeline
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSplitter, QMessageBox, QTabWidget,
    QTextBrowser, QFrame, QSizePolicy, QDialog, QFileDialog,
    QScrollArea, QComboBox, QCheckBox,
)
from PyQt6.QtCore import Qt, QEvent, QThread, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QColor, QImage, QFont

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# =============================================================================
# Constants
# =============================================================================

def _find_dataset_dir() -> Path | None:
    seen: set[Path] = set()
    candidates: list[Path] = []
    try:
        candidates.append(Path(__file__).resolve().parent / 'dataset')
    except NameError:
        pass
    candidates.append(Path.cwd() / 'dataset')
    p = Path.cwd()
    for _ in range(3):
        p = p.parent
        candidates.append(p / 'dataset')
    for c in candidates:
        r = c.resolve()
        if r not in seen:
            seen.add(r)
            if c.is_dir():
                return r
    return None


DATASET_DIR: Path = _find_dataset_dir() or (Path.cwd() / 'dataset').resolve()
MODEL_PATH         = DATASET_DIR / 'model.pkl'
MODEL_ONLINE_PATH  = DATASET_DIR / 'model_online.pkl'

# =============================================================================
# Stylesheet
# =============================================================================

APP_STYLE = """
QMainWindow, QWidget { background: #f0f4f8; font-family: 'Segoe UI', Arial, sans-serif; color: #1e293b; }
QFrame#card {
    background: white;
    border: 1px solid #dde3ee;
    border-radius: 12px;
}
QTabWidget::pane {
    border: 1px solid #dde3ee;
    background: white;
    border-radius: 0 10px 10px 10px;
}
QTabBar::tab {
    background: #dde3ee;
    color: #374151;
    padding: 8px 18px;
    border: 1px solid #c8d3e8;
    border-bottom: none;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    font-weight: 700;
    font-size: 10px;
    margin-right: 2px;
}
QTabBar::tab:selected { background: white; color: #1d4ed8; border-bottom: 2px solid #2563eb; }
QTabBar::tab:hover:!selected { background: #c8d3e8; color: #1e3a8a; }
QTextBrowser { border: none; background: white; color: #1e293b; }
QPushButton { border-radius: 7px; font-weight: 700; font-size: 11px; padding: 8px 14px; }
QSplitter::handle { background: #dde3ee; width: 1px; }
QScrollArea { border: none; background: transparent; }
QDialog { background: white; color: #1e293b; }
QDialog QLabel { color: #1e293b; background: transparent; }
QMessageBox { background: white; color: #1e293b; }
QMessageBox QLabel { color: #1e293b; background: transparent; font-size: 13px; min-width: 280px; }
QMessageBox QPushButton {
    background: #2563eb; color: white;
    padding: 6px 20px; border-radius: 6px;
    min-width: 80px; font-size: 12px; font-weight: 700;
}
QMessageBox QPushButton:hover { background: #1d4ed8; }
"""

_S_CLEAR   = "QPushButton{background:#eef1f7;color:#374151;border:1px solid #cbd5e1;border-radius:7px;}QPushButton:hover{background:#dde3ee;}"
_S_ANALYZE = "QPushButton{background:#2563eb;color:white;border:none;border-radius:7px;}QPushButton:hover{background:#1d4ed8;}QPushButton:disabled{background:#bfdbfe;color:#1d4ed8;border:1px solid #93c5fd;}"
_S_FLUENT  = "QPushButton{background:#16a34a;color:white;border:none;border-radius:7px;}QPushButton:hover{background:#15803d;}QPushButton:disabled{background:#dcfce7;color:#15803d;border:1px solid #86efac;}"
_S_DYSG    = "QPushButton{background:#dc2626;color:white;border:none;border-radius:7px;}QPushButton:hover{background:#b91c1c;}QPushButton:disabled{background:#fee2e2;color:#b91c1c;border:1px solid #fca5a5;}"
_S_RETRAIN = "QPushButton{background:#7c3aed;color:white;border:none;border-radius:7px;}QPushButton:hover{background:#6d28d9;}QPushButton:disabled{background:#ede9fe;color:#5b21b6;border:1px solid #c4b5fd;}"
_S_CHECK   = "QPushButton{background:#0891b2;color:white;border:none;border-radius:7px;}QPushButton:hover{background:#0e7490;}"
_S_TEST    = "QPushButton{background:#d97706;color:white;border:none;border-radius:7px;}QPushButton:hover{background:#b45309;}QPushButton:disabled{background:#fef3c7;color:#92400e;border:1px solid #fde68a;}"

COLORS = {
    'good':    ('#166534', '#dcfce7'),
    'bad':     ('#991b1b', '#fef2f2'),
    'neutral': ('#374151', '#f9fafb'),
}

MODEL_COLORS = ['#3b82f6', '#16a34a', '#f97316', '#8b5cf6', '#dc2626']
MODEL_LABELS = ['Random Forest', 'SVM (RBF)', 'Logistic Reg.',
                'Gradient Boost', 'AdaBoost']
MODEL_KEYS   = ['RF', 'SVM', 'LR', 'GB', 'AB']

# =============================================================================
# Background worker
# =============================================================================

class RetrainWorker(QThread):
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)

    def __init__(self, excluded_categories=None, parent=None):
        super().__init__(parent)
        self._excluded = tuple(sorted(excluded_categories or ()))

    def run(self):
        try:
            X, y, nf, nd = _load_real_sessions()
            mask = (feature_mask_excluding(self._excluded)
                    if self._excluded else None)
            result = train_ml_models(X, y, feature_mask=mask)
            result['n_fluent']            = nf
            result['n_dysg']              = nd
            result['source']              = 'real'
            result['mode']                = 'online'
            result['feature_names']       = FEATURE_NAMES
            result['excluded_categories'] = self._excluded
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
            with open(MODEL_ONLINE_PATH, 'wb') as f:
                pickle.dump(result, f)
            with open(MODEL_PATH, 'wb') as f:
                pickle.dump(result, f)
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# =============================================================================
# Drawing Canvas
# =============================================================================

class DrawingCanvas(QWidget):
    LINE_SPACING = 70

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(600, 340)
        self._img     = QImage(600, 340, QImage.Format.Format_RGB32)
        self._img.fill(QColor(255, 255, 255))
        self.strokes  = []
        self._stroke  = []
        self._drawing = False
        self.setAttribute(Qt.WidgetAttribute.WA_TabletTracking)
        self._draw_guidelines()

    def _draw_guidelines(self):
        p = QPainter(self._img)
        p.setPen(QPen(QColor(196, 216, 240), 1))
        for y in range(self.LINE_SPACING, 340, self.LINE_SPACING):
            p.drawLine(20, y, 580, y)
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._stroke  = []
            self._record(e.position().x(), e.position().y(), 0.7)

    def mouseMoveEvent(self, e):
        if self._drawing:
            self._record(e.position().x(), e.position().y(), 0.7)
            self._draw_line_to(e.position().x(), e.position().y())

    def mouseReleaseEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton and self._drawing:
            self._end_stroke()

    def tabletEvent(self, e):
        x, y = e.position().x(), e.position().y()
        t = e.type()
        if t == QEvent.Type.TabletPress:
            self._drawing = True; self._stroke = []
            self._record(x, y, e.pressure())
        elif t == QEvent.Type.TabletMove and self._drawing:
            self._record(x, y, e.pressure())
            self._draw_line_to(x, y)
        elif t == QEvent.Type.TabletRelease and self._drawing:
            self._end_stroke()
        e.accept()

    def _record(self, x, y, pressure):
        self._stroke.append((float(x), float(y), time.perf_counter(), float(pressure)))

    def _draw_line_to(self, x, y):
        if len(self._stroke) < 2:
            return
        p = QPainter(self._img)
        p.setPen(QPen(QColor(30, 40, 80), 2, Qt.PenStyle.SolidLine,
                      Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        x0, y0 = self._stroke[-2][0], self._stroke[-2][1]
        p.drawLine(int(x0), int(y0), int(x), int(y))
        p.end()
        self.update()

    def _end_stroke(self):
        if self._stroke:
            self.strokes.append(self._stroke)
        self._stroke  = []
        self._drawing = False

    def paintEvent(self, _):
        QPainter(self).drawImage(0, 0, self._img)

    def capture_image(self, filepath: Path):
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self._img.save(str(filepath), 'PNG')

    def clear(self):
        self.strokes.clear()
        self._stroke.clear()
        self._drawing = False
        self._img.fill(QColor(255, 255, 255))
        self._draw_guidelines()
        self.update()


# =============================================================================
# Feature extraction
# =============================================================================

def compute_speed_profile(strokes):
    all_t, all_speed = [], []
    t_offset = None
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        arr = np.array(stroke)
        xs, ys, ts = arr[:, 0], arr[:, 1], arr[:, 2]
        if t_offset is None:
            t_offset = ts[0]
        dt    = np.maximum(np.diff(ts), 1e-6)
        speed = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) / dt
        t_mid = (ts[:-1] + ts[1:]) / 2 - t_offset
        all_t.extend(t_mid.tolist())
        all_speed.extend(speed.tolist())
    if len(all_t) < 5:
        return None, None
    t     = np.array(all_t)
    speed = np.array(all_speed)
    p95   = np.percentile(speed, 95)
    speed = np.clip(speed, 0, p95)
    w     = max(3, len(speed) // 15)
    if len(speed) >= w:
        speed = np.convolve(speed, np.ones(w) / w, mode='valid')
        t     = t[w // 2: w // 2 + len(speed)]
    return t, speed


def compute_kinematic_features(strokes):
    if not strokes:
        return None
    all_v        = []
    t_on_surface = 0.0
    all_pts = [pt for stroke in strokes for pt in stroke]
    if len(all_pts) < 2:
        return None
    t_first = all_pts[0][2]
    t_last  = all_pts[-1][2]
    T_total = max(t_last - t_first, 1e-6)
    for stroke in strokes:
        if len(stroke) < 2:
            continue
        arr = np.array(stroke)
        xs, ys, ts = arr[:, 0], arr[:, 1], arr[:, 2]
        dt = np.maximum(np.diff(ts), 1e-3)
        v  = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2) / dt
        all_v.extend(v.tolist())
        t_on_surface += ts[-1] - ts[0]
    if len(all_v) < 3:
        return None
    v_arr = np.clip(np.array(all_v), 0, np.percentile(all_v, 95) * 1.5)
    w = max(3, len(v_arr) // 15)
    if len(v_arr) >= w:
        v_arr = np.convolve(v_arr, np.ones(w) / w, mode='valid')
    a_arr = np.diff(v_arr)
    j_arr = np.diff(a_arr)
    T_air = max(T_total - t_on_surface, 0.0)
    return {
        'v_mean':       float(np.mean(v_arr)),
        'v_std':        float(np.std(v_arr)),
        'a_mean':       float(np.mean(a_arr)),
        'j_mean':       float(np.mean(np.abs(j_arr))),
        'n_lifts':      len(strokes),
        'in_air_ratio': float(T_air / T_total),
    }


def compute_pressure_features(strokes):
    all_p = []
    for stroke in strokes:
        if stroke:
            all_p.extend(np.array(stroke)[:, 3].tolist())
    if not all_p:
        return None
    p        = np.array(all_p)
    p_active = p[p > 0.05]
    if len(p_active) == 0:
        p_active = p
    return {
        'p_mean':  float(np.mean(p_active)),
        'p_std':   float(np.std(p_active)),
        'p_max':   float(np.max(p_active)),
        'p_range': float(np.max(p_active) - np.min(p_active)),
    }


# =============================================================================
# Drotár & Dobeš 2020 feature set  (Sci. Rep. 10:21541, Table 2)
# -----------------------------------------------------------------------------
# 119 on-surface features per task. Altitude/azimuth families omitted because
# the XP-Pen Deco 01 stylus does not report tilt.
#
#   10 vector-stat families × 7 stats  =  70  (mean, median, std, max, min, 5%, 95%)
#    6 segment-stat families × 5 stats =  30  (mean, median, std, max, min)
#   19 scalar features                 =  19
#                                       ----
#                                         119
# =============================================================================

_VEC_STAT_NAMES  = ('mean', 'median', 'std', 'max', 'min', 'p5', 'p95')
_SEG_STAT_NAMES  = ('mean', 'median', 'std', 'max', 'min')


def _vec_stats(arr):
    if arr is None or len(arr) == 0:
        return [0.0] * 7
    a = np.asarray(arr, dtype=float)
    return [
        float(np.mean(a)),  float(np.median(a)), float(np.std(a)),
        float(np.max(a)),   float(np.min(a)),
        float(np.percentile(a,  5)),
        float(np.percentile(a, 95)),
    ]


def _seg_stats(arr):
    if arr is None or len(arr) == 0:
        return [0.0] * 5
    a = np.asarray(arr, dtype=float)
    return [
        float(np.mean(a)), float(np.median(a)), float(np.std(a)),
        float(np.max(a)),  float(np.min(a)),
    ]


def _count_local_extrema(arr):
    """Number of sign-changes of the first difference (local maxima + minima)."""
    if arr is None or len(arr) < 3:
        return 0
    d = np.diff(arr)
    return int(np.sum(np.diff(np.sign(d)) != 0))


def compute_drotar_features(strokes):
    """Compute all 119 Drotár-compatible features for one writing task.

    Returns a flat dict keyed by `<family>_<stat>` (e.g. 'velocity_mean',
    'segment_height_std'). Returns None if `strokes` cannot yield meaningful
    derivatives (fewer than 3 samples per stroke after concatenation).
    """
    if not strokes:
        return None
    valid = [s for s in strokes if len(s) >= 2]
    if not valid:
        return None

    # ── per-sample arrays (concatenated across strokes) ───────────────────────
    velocity_v, velocity_y, velocity_x = [], [], []
    accel_v,    accel_y,    accel_x    = [], [], []
    jerk_v,     jerk_y,     jerk_x     = [], [], []
    pressure_v = []

    # ── per-segment scalars ──────────────────────────────────────────────────
    seg_duration, seg_path_len = [], []
    seg_vert_len, seg_horz_len = [], []
    seg_width,    seg_height   = [], []
    seg_y_first  = []      # y of the FIRST sample of each segment (for diff_first_last_y_*)
    seg_y_last   = []      # y of the LAST sample of each segment

    # NB: Drotár uses "segment" = a continuous run between in-air/on-surface
    # transitions, which on our hardware is exactly one stroke entry.
    for s in valid:
        arr = np.asarray(s, dtype=float)
        xs, ys, ts, ps = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        dt = np.maximum(np.diff(ts), 1e-3)

        dx, dy = np.diff(xs), np.diff(ys)
        v_total = np.sqrt(dx * dx + dy * dy) / dt
        vx, vy  = dx / dt, dy / dt

        velocity_v.extend(v_total.tolist())
        velocity_x.extend(np.abs(vx).tolist())
        velocity_y.extend(np.abs(vy).tolist())

        if len(v_total) >= 2:
            dt2 = np.maximum(dt[1:], 1e-3)
            a_total = np.diff(v_total) / dt2
            ax      = np.diff(vx)      / dt2
            ay      = np.diff(vy)      / dt2
            accel_v.extend(a_total.tolist())
            accel_x.extend(np.abs(ax).tolist())
            accel_y.extend(np.abs(ay).tolist())

            if len(a_total) >= 2:
                dt3 = np.maximum(dt[2:], 1e-3)
                j_total = np.diff(a_total) / dt3
                jx      = np.diff(ax)      / dt3
                jy      = np.diff(ay)      / dt3
                jerk_v.extend(j_total.tolist())
                jerk_x.extend(np.abs(jx).tolist())
                jerk_y.extend(np.abs(jy).tolist())

        pressure_v.extend(ps[ps > 0.05].tolist() if (ps > 0.05).any() else ps.tolist())

        # Segment-level scalars
        seg_duration.append(float(ts[-1] - ts[0]))
        seg_path_len.append(float(np.sum(np.sqrt(dx * dx + dy * dy))))
        seg_vert_len.append(float(np.sum(np.abs(dy))))
        seg_horz_len.append(float(np.sum(np.abs(dx))))
        seg_width.append(float(xs.max()  - xs.min()))
        seg_height.append(float(ys.max() - ys.min()))
        seg_y_first.append(float(ys[0]))
        seg_y_last.append(float(ys[-1]))

    if len(velocity_v) < 3:
        return None

    feats = {}

    # ── (1) 10 vector-stat families × 7 stats = 70 features ──────────────────
    vec_families = (
        ('velocity',     velocity_v),
        ('velocity_y',   velocity_y),
        ('velocity_x',   velocity_x),
        ('acceleration', accel_v),
        ('accel_y',      accel_y),
        ('accel_x',      accel_x),
        ('jerk',         jerk_v),
        ('jerk_y',       jerk_y),
        ('jerk_x',       jerk_x),
        ('pressure',     pressure_v),
    )
    for fam_name, vals in vec_families:
        for stat_name, stat_val in zip(_VEC_STAT_NAMES, _vec_stats(vals)):
            feats[f'{fam_name}_{stat_name}'] = stat_val

    # ── (2) 6 segment-stat families × 5 stats = 30 features ──────────────────
    seg_families = (
        ('seg_duration',    seg_duration),
        ('seg_path_len',    seg_path_len),
        ('seg_vert_len',    seg_vert_len),
        ('seg_horz_len',    seg_horz_len),
        ('seg_width',       seg_width),
        ('seg_height',      seg_height),
    )
    for fam_name, vals in seg_families:
        for stat_name, stat_val in zip(_SEG_STAT_NAMES, _seg_stats(vals)):
            feats[f'{fam_name}_{stat_name}'] = stat_val

    # ── (3) 19 scalar features ───────────────────────────────────────────────
    feats['n_pen_lifts']             = float(len(valid))
    feats['n_velocity_extrema']      = float(_count_local_extrema(velocity_v))
    feats['n_acceleration_extrema']  = float(_count_local_extrema(accel_v))

    all_pts = [pt for s in valid for pt in s]
    feats['total_writing_time'] = float(all_pts[-1][2] - all_pts[0][2])
    feats['total_path_len']     = float(sum(seg_path_len))
    feats['total_vert_len']     = float(sum(seg_vert_len))
    feats['total_horz_len']     = float(sum(seg_horz_len))

    # y-position differences between FIRST and LAST segments (4 features)
    if len(valid) >= 2:
        y_first_seg = np.asarray([pt[1] for pt in valid[0]],  dtype=float)
        y_last_seg  = np.asarray([pt[1] for pt in valid[-1]], dtype=float)
        feats['diff_first_last_y_min']    = float(y_first_seg.min()    - y_last_seg.min())
        feats['diff_first_last_y_median'] = float(np.median(y_first_seg) - np.median(y_last_seg))
        feats['diff_first_last_y_mean']   = float(y_first_seg.mean()   - y_last_seg.mean())
        feats['diff_first_last_y_max']    = float(y_first_seg.max()    - y_last_seg.max())
    else:
        for k in ('min', 'median', 'mean', 'max'):
            feats[f'diff_first_last_y_{k}'] = 0.0

    # y-position differences between SECOND and PENULTIMATE segments (4 features)
    if len(valid) >= 4:
        y_2nd     = np.asarray([pt[1] for pt in valid[1]],  dtype=float)
        y_penult  = np.asarray([pt[1] for pt in valid[-2]], dtype=float)
        feats['diff_2nd_penult_y_min']    = float(y_2nd.min()    - y_penult.min())
        feats['diff_2nd_penult_y_median'] = float(np.median(y_2nd) - np.median(y_penult))
        feats['diff_2nd_penult_y_mean']   = float(y_2nd.mean()   - y_penult.mean())
        feats['diff_2nd_penult_y_max']    = float(y_2nd.max()    - y_penult.max())
    else:
        for k in ('min', 'median', 'mean', 'max'):
            feats[f'diff_2nd_penult_y_{k}'] = 0.0

    # Variance over the segment-y-position summaries (4 features)
    seg_y_min    = [float(min(pt[1] for pt in s)) for s in valid]
    seg_y_max    = [float(max(pt[1] for pt in s)) for s in valid]
    seg_y_median = [float(np.median([pt[1] for pt in s])) for s in valid]
    seg_y_mean   = [float(np.mean(  [pt[1] for pt in s])) for s in valid]
    feats['var_seg_y_min']    = float(np.var(seg_y_min))    if len(seg_y_min)    > 1 else 0.0
    feats['var_seg_y_max']    = float(np.var(seg_y_max))    if len(seg_y_max)    > 1 else 0.0
    feats['var_seg_y_median'] = float(np.var(seg_y_median)) if len(seg_y_median) > 1 else 0.0
    feats['var_seg_y_mean']   = float(np.var(seg_y_mean))   if len(seg_y_mean)   > 1 else 0.0

    return feats


# Canonical ordered key list — defines the column order of the 119-feat vector.
DROTAR_FEATURE_KEYS = (
    # 10 vector families × 7 stats
    [f'{fam}_{s}'
        for fam in ('velocity', 'velocity_y', 'velocity_x',
                    'acceleration', 'accel_y', 'accel_x',
                    'jerk', 'jerk_y', 'jerk_x', 'pressure')
        for s in _VEC_STAT_NAMES]
    +
    # 6 segment families × 5 stats
    [f'{fam}_{s}'
        for fam in ('seg_duration', 'seg_path_len', 'seg_vert_len',
                    'seg_horz_len', 'seg_width', 'seg_height')
        for s in _SEG_STAT_NAMES]
    +
    # 19 scalars
    ['n_pen_lifts', 'n_velocity_extrema', 'n_acceleration_extrema',
     'total_writing_time', 'total_path_len', 'total_vert_len', 'total_horz_len',
     'diff_first_last_y_min',    'diff_first_last_y_median',
     'diff_first_last_y_mean',   'diff_first_last_y_max',
     'diff_2nd_penult_y_min',    'diff_2nd_penult_y_median',
     'diff_2nd_penult_y_mean',   'diff_2nd_penult_y_max',
     'var_seg_y_min', 'var_seg_y_max',
     'var_seg_y_median', 'var_seg_y_mean']
)

# Human-readable labels for plots / feature-importance tables.
DROTAR_FEATURE_LABELS = [k.replace('_', ' ') for k in DROTAR_FEATURE_KEYS]


def _split_cursive_strokes(strokes, line_spacing=70, cusp_tol=12, min_letter_dx=12):
    if not strokes:
        return strokes
    guidelines = list(range(line_spacing, 400, line_spacing))
    out = []
    for stroke in strokes:
        if len(stroke) < 12:
            out.append(stroke); continue
        ys = np.array([pt[1] for pt in stroke])
        xs = np.array([pt[0] for pt in stroke])
        win, cusps = 3, []
        for i in range(win, len(ys) - win):
            if ys[i] != np.max(ys[i - win:i + win + 1]):
                continue
            if ys[i] - min(ys[i - win], ys[i + win]) < 3:
                continue
            if not any(abs(ys[i] - g) <= cusp_tol for g in guidelines):
                continue
            if not cusps or xs[i] - xs[cusps[-1]] >= min_letter_dx:
                cusps.append(i)
        if not cusps:
            out.append(stroke); continue
        prev = 0
        for c in cusps:
            seg = stroke[prev:c + 1]
            if len(seg) >= 3:
                out.append(seg)
            prev = c
        tail = stroke[prev:]
        if len(tail) >= 3:
            out.append(tail)
    return out


def _group_letters(strokes, overlap_margin=4, max_time_gap=2.0, max_y_gap=35, line_spacing=70):
    if not strokes:
        return []
    strokes  = _split_cursive_strokes(strokes, line_spacing=line_spacing)
    clusters = [[strokes[0]]]
    for i in range(1, len(strokes)):
        prev, curr = strokes[i - 1], strokes[i]
        dt     = float(curr[0][2] - prev[-1][2])
        y_prev = float(sum(pt[1] for pt in prev) / len(prev))
        y_curr = float(sum(pt[1] for pt in curr) / len(curr))
        y_gap  = abs(y_curr - y_prev)
        if dt >= max_time_gap or y_gap >= max_y_gap:
            clusters.append([curr]); continue
        px_min = min(pt[0] for pt in prev); px_max = max(pt[0] for pt in prev)
        cx_min = min(pt[0] for pt in curr); cx_max = max(pt[0] for pt in curr)
        overlaps = (cx_min <= px_max + overlap_margin and cx_max >= px_min - overlap_margin) \
                   if y_gap > 8 else (cx_min < px_max and cx_max > px_min)
        if overlaps:
            clusters[-1].append(curr)
        else:
            clusters.append([curr])
    return clusters


def _cluster_x_min(c): return float(min(pt[0] for s in c for pt in s))
def _cluster_x_max(c): return float(max(pt[0] for s in c for pt in s))


def compute_spatial_features(strokes, line_spacing=70):
    if not strokes or len(strokes) < 2:
        return None
    guidelines   = list(range(line_spacing, 400, line_spacing))
    TOUCH_THRESH = line_spacing * 0.55

    def nearest_guide(y):
        return min(guidelines, key=lambda g: abs(g - y))

    line_groups    = defaultdict(list)
    stroke_heights = []
    stroke_slants  = []
    baseline_devs  = []

    for cluster in _group_letters(strokes):
        all_pts = [pt for s in cluster for pt in s]
        if len(all_pts) < 3:
            continue
        arr      = np.array(all_pts)
        y_top    = float(np.min(arr[:, 1]))
        y_bottom = float(np.max(arr[:, 1]))
        x_center = float(np.mean(arr[:, 0]))
        guide    = nearest_guide(y_bottom)
        h = float(guide - y_top)
        if h > 0:
            stroke_heights.append(h)
        baseline_devs.append(float(y_bottom - guide))
        if abs(y_bottom - guide) <= TOUCH_THRESH:
            line_groups[guide].append((x_center, y_bottom, cluster))
        for s in cluster:
            if len(s) < 3:
                continue
            arr_s = np.array(s)
            if float(np.max(arr_s[:, 1]) - np.min(arr_s[:, 1])) > 10:
                dx = float(arr_s[-1, 0] - arr_s[0, 0])
                dy = float(arr_s[-1, 1] - arr_s[0, 1])
                if dy > 5:
                    stroke_slants.append(float(np.degrees(np.arctan2(dx, dy))))

    if not stroke_heights:
        return None

    letter_gaps, word_gaps = [], []
    line_slopes = []
    line_fits   = {}
    for guide, group in line_groups.items():
        xs = [item[0] for item in group]
        ys = [item[1] for item in group]
        clusters = [item[2] for item in group]
        if len(xs) >= 2:
            slope, intercept, *_ = stats.linregress(xs, ys)
            line_slopes.append(float(slope))
            line_fits[guide] = (float(slope), float(intercept), min(xs), max(xs))
        if len(clusters) < 2:
            continue
        sorted_c  = sorted(clusters, key=_cluster_x_min)
        line_gaps = [_cluster_x_min(sorted_c[i + 1]) - _cluster_x_max(sorted_c[i])
                     for i in range(len(sorted_c) - 1)]
        pos_gaps  = [g for g in line_gaps if g > 0]
        if len(pos_gaps) > 1:
            tau = float(np.mean(pos_gaps) + np.std(pos_gaps))
        elif pos_gaps:
            tau = float(pos_gaps[0]) * 1.5
        else:
            continue
        for g in pos_gaps:
            (word_gaps if g > tau else letter_gaps).append(g)

    delta = (float(np.sqrt(np.mean([d ** 2 for d in baseline_devs])))
             if baseline_devs else 0.0)
    if line_slopes:
        mean_raw_slope     = float(np.mean(line_slopes))
        baseline_slope_deg = float(np.degrees(np.arctan(abs(mean_raw_slope))))
    else:
        mean_raw_slope     = 0.0
        baseline_slope_deg = 0.0

    return {
        'h_mean':             float(np.mean(stroke_heights)),
        'h_std':              float(np.std(stroke_heights)),
        'sl_mean':            float(np.mean(letter_gaps)) if letter_gaps else 0.0,
        'sw_mean':            float(np.mean(word_gaps))   if word_gaps   else 0.0,
        'delta':              delta,
        'baseline_slope_deg': baseline_slope_deg,
        'baseline_slope_raw': mean_raw_slope,
        'theta_mean':         float(np.mean(stroke_slants))         if stroke_slants else 0.0,
        'theta_std':          float(np.std(stroke_slants))          if stroke_slants else 0.0,
        'theta':              float(np.mean(np.abs(stroke_slants))) if stroke_slants else 0.0,
        'n_word_gaps':        len(word_gaps),
        'guidelines':         guidelines,
        'touch_thresh':       TOUCH_THRESH,
        'line_fits':          line_fits,
    }


def fit_trend(t, values):
    slope, intercept, r, p, _ = stats.linregress(t, values)
    return {'slope': float(slope), 'intercept': float(intercept),
            'r2': float(r ** 2), 'p_value': float(p),
            'line': slope * t + intercept}


def slant_type(theta_mean, theta_std):
    if theta_std > 20:
        return ('Variable / Mixed',
                f'Slant changes stroke-to-stroke (σ={theta_std:.0f}°). Inconsistency is a dysgraphia marker.', 'bad')
    if theta_mean > 45:
        return ('Extreme Right', f'Letters lean far forward ({theta_mean:.0f}°).', 'bad')
    if theta_mean < -45:
        return ('Extreme Left', f'Letters lean far backward ({theta_mean:.0f}°).', 'bad')
    if theta_mean > 15:
        return ('Right / Forward', f'Letters lean right ({theta_mean:.0f}°).', 'good')
    if theta_mean < -15:
        return ('Left / Backward', f'Letters lean left ({theta_mean:.0f}°).', 'good')
    return ('Upright / Vertical', f'Letters stand straight ({theta_mean:.0f}°).', 'good')


# =============================================================================
# ML helpers — improved
# =============================================================================

# 119-feature vector following Drotár & Dobeš (2020), restricted to the
# signals an XP-Pen Deco 01 captures (no altitude/azimuth). See
# `compute_drotar_features()` and DROTAR_FEATURE_KEYS above.
FEATURE_NAMES = list(DROTAR_FEATURE_LABELS)
N_FEATURES    = len(FEATURE_NAMES)

def _drotar_vec_from_dict(drotar_feats):
    """Project a `compute_drotar_features()` dict onto the canonical 119-element
    ordered vector defined by DROTAR_FEATURE_KEYS. Missing keys -> 0.0."""
    if not drotar_feats:
        return None
    return [float(drotar_feats.get(k, 0.0)) for k in DROTAR_FEATURE_KEYS]


def _kin_only_vec(kin, pres, spatial=None, drotar=None, strokes=None):
    # Drotár 119-feature vector. Callers may pass either the pre-computed
    # `drotar` dict, or `strokes` (raw stroke list) and we'll compute it here.
    # `kin`, `pres`, `spatial` are accepted for call-site compatibility but
    # only used as a last-resort fallback if no drotar data is available.
    if drotar is None and strokes is not None:
        drotar = compute_drotar_features(strokes)
    if drotar is not None:
        return _drotar_vec_from_dict(drotar)
    return None


def _extract_feature_vector(kin, pres, spatial, drotar=None, strokes=None):
    """Build the 119-feature Drotár vector from either a pre-computed
    `drotar` dict or raw `strokes`. Returns None if neither path can yield
    a valid vector."""
    kin_vec = _kin_only_vec(kin, pres, spatial, drotar=drotar, strokes=strokes)
    return np.array(kin_vec, dtype=float) if kin_vec is not None else None


def train_ml_models(X, y, feature_mask=None):
    """Train 5 classifiers with cross-validation and learning curves.

    Optional `feature_mask` is a boolean array of length X.shape[1] —
    when given, training is restricted to features where the mask is
    True. The mask is stored in the returned dict so the prediction
    path can apply the same subset to incoming feature vectors.
    """
    if feature_mask is not None:
        feature_mask = np.asarray(feature_mask, dtype=bool)
        if feature_mask.shape[0] != X.shape[1]:
            raise ValueError(
                f"feature_mask length {feature_mask.shape[0]} does not match "
                f"X.shape[1]={X.shape[1]}")
        X = X[:, feature_mask]

    n_min    = min(np.sum(y == 0), np.sum(y == 1))
    n_cv     = min(5, n_min)
    use_loocv = n_min < 10  # Leave-one-out for tiny datasets

    if n_min >= 4:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)
    else:
        X_tr, X_te, y_tr, y_te = X, X, y, y

    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(X_tr)
    X_te_s  = scaler.transform(X_te)

    # Hyperparameters: prefer values from best_hyperparameters.json (produced
    # by grid_search_optimize.py); otherwise fall back to literature-informed
    # defaults below.
    BEST_HP_PATH = DATASET_DIR.parent / 'best_hyperparameters.json'
    _hp_loaded   = {}
    if BEST_HP_PATH.exists():
        try:
            _hp_loaded = json.loads(BEST_HP_PATH.read_text()).get('chosen', {})
        except Exception:
            _hp_loaded = {}

    def _hp(name, fallback):
        return {**fallback, **_hp_loaded.get(name, {})}

    rf_hp  = _hp('RF',  {'n_estimators': 300, 'max_features': 'sqrt',
                         'min_samples_leaf': 2})
    svm_hp = _hp('SVM', {'C': 2.0, 'gamma': 'scale'})
    lr_hp  = _hp('LR',  {'C': 0.5, 'solver': 'lbfgs'})
    gb_hp  = _hp('GB',  {'n_estimators': 200, 'learning_rate': 0.05,
                         'max_depth': 3})
    ab_hp  = _hp('AB',  {'n_estimators': 340, 'learning_rate': 1.0})

    rf  = RandomForestClassifier(class_weight='balanced',
                                 random_state=42, n_jobs=-1, **rf_hp)
    svm = SVC(kernel='rbf', probability=True,
              class_weight='balanced', random_state=42, **svm_hp)
    lr  = LogisticRegression(max_iter=2000, class_weight='balanced',
                             random_state=42, **lr_hp)
    gb  = GradientBoostingClassifier(subsample=0.8, random_state=42, **gb_hp)
    # AdaBoost — Drotár showed this is competitive on clinical data
    ab  = AdaBoostClassifier(random_state=42, **ab_hp)

    rf.fit(X_tr, y_tr)
    svm.fit(X_tr_s, y_tr)
    lr.fit(X_tr_s, y_tr)
    gb.fit(X_tr_s, y_tr)
    ab.fit(X_tr_s, y_tr)

    preds = {
        'RF':  rf.predict(X_te),
        'SVM': svm.predict(X_te_s),
        'LR':  lr.predict(X_te_s),
        'GB':  gb.predict(X_te_s),
        'AB':  ab.predict(X_te_s),
    }
    test_acc = {k: accuracy_score(y_te, p) for k, p in preds.items()}

    # Per-classifier clinical metrics (positive class = 1 = Dysgraphic).
    # Specificity = recall for the negative class (Fluent).
    per_class_metrics = {}
    for k, p in preds.items():
        per_class_metrics[k] = {
            'accuracy':    float(accuracy_score(y_te, p)),
            'precision':   float(precision_score(y_te, p, pos_label=1, zero_division=0)),
            'recall':      float(recall_score(y_te, p, pos_label=1, zero_division=0)),
            'f1':          float(f1_score(y_te, p, pos_label=1, zero_division=0)),
            'specificity': float(recall_score(y_te, p, pos_label=0, zero_division=0)),
        }

    # Cross-validation with proper pipelines (no data leakage)
    cv_scores = {}
    if n_cv >= 2:
        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        # Use the same tuned hyperparameters as the held-out models above
        _pipes = [
            ('RF',  Pipeline([('rf',
                RandomForestClassifier(class_weight='balanced',
                                       random_state=42, n_jobs=-1, **rf_hp))])),
            ('SVM', Pipeline([('scl', StandardScaler()), ('svm',
                SVC(kernel='rbf', probability=True,
                    class_weight='balanced', random_state=42, **svm_hp))])),
            ('LR',  Pipeline([('scl', StandardScaler()), ('lr',
                LogisticRegression(max_iter=2000, class_weight='balanced',
                                   random_state=42, **lr_hp))])),
            ('GB',  Pipeline([('scl', StandardScaler()), ('gb',
                GradientBoostingClassifier(subsample=0.8, random_state=42,
                                           **gb_hp))])),
            ('AB',  Pipeline([('scl', StandardScaler()), ('ab',
                AdaBoostClassifier(random_state=42, **ab_hp))])),
        ]
        for name, pipe in _pipes:
            try:
                cv_scores[name] = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
            except Exception:
                cv_scores[name] = np.array([test_acc[name]])

    # Learning curve (RF pipeline)
    lc_sizes = lc_train = lc_test = None
    if len(X) >= 10:
        try:
            n_pts = min(8, len(X) // 2)
            lc_pipe = Pipeline([
                ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1))
            ])
            lc_cv = StratifiedKFold(n_splits=min(5, n_min), shuffle=True, random_state=42)
            lc_sizes, lc_train, lc_test = learning_curve(
                lc_pipe, X, y, cv=lc_cv,
                train_sizes=np.linspace(0.2, 1.0, n_pts),
                scoring='accuracy', n_jobs=-1)
        except Exception:
            pass

    cms = {k: sk_cm(y_te, p) for k, p in preds.items()}

    return {
        'rf': rf, 'svm': svm, 'lr': lr, 'gb': gb, 'ab': ab,
        'scaler': scaler,
        # backward compat key for main app
        'accuracies': {
            'Random\nForest': test_acc['RF'],
            'SVM\n(RBF)':    test_acc['SVM'],
            'Logistic\nReg.': test_acc['LR'],
        },
        'test_accuracies': test_acc,
        'per_class_metrics': per_class_metrics,
        'n_test':          int(len(y_te)),
        'n_test_fluent':   int(np.sum(y_te == 0)),
        'n_test_dysg':     int(np.sum(y_te == 1)),
        'feature_mask':    None if feature_mask is None else feature_mask.copy(),
        'feature_keys_used': tuple(
            k for k, keep in zip(DROTAR_FEATURE_KEYS,
                                 feature_mask if feature_mask is not None
                                 else [True] * len(DROTAR_FEATURE_KEYS))
            if keep
        ),
        'cv_scores':       cv_scores,
        'cm':              cms['RF'],
        'cms':             cms,
        'importances':     rf.feature_importances_,
        'lc_sizes':        lc_sizes,
        'lc_train':        lc_train,
        'lc_test':         lc_test,
    }


def _load_real_sessions():
    """Load the 119-feature Drotár training matrix from every labelled JSON
    in `dataset/{fluent,dysgraphic}/json/`."""
    def _kin_from_json(data):
        d = data.get('drotar_features')
        if d is not None and len(d) >= len(DROTAR_FEATURE_KEYS) // 2:
            return _drotar_vec_from_dict(d)
        strokes = data.get('strokes')
        if not strokes:
            return None
        d = compute_drotar_features(strokes)
        return _drotar_vec_from_dict(d) if d is not None else None

    rows, labels = [], []
    for lbl_int, folder in ((0, 'fluent'), (1, 'dysgraphic')):
        jdir = DATASET_DIR / folder / 'json'
        if not jdir.exists():
            continue
        for jf in sorted(jdir.glob('*.json')):
            try:
                data    = json.loads(jf.read_text(encoding='utf-8'))
                kin_vec = _kin_from_json(data)
                if kin_vec is None:
                    continue
                rows.append(kin_vec)
                labels.append(lbl_int)
            except Exception:
                continue

    nf, nd = labels.count(0), labels.count(1)
    if nf < 2 or nd < 2:
        raise ValueError(
            "Not enough samples to train.\n\n"
            f"Found:  {nf} fluent   {nd} dysgraphic\n"
            "Need at least 2 fluent and 2 dysgraphic JSON files.")
    return np.array(rows, dtype=float), np.array(labels), nf, nd


def _try_load_model():
    try:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, 'rb') as f:
                m = pickle.load(f)
            n_in = m['scaler'].n_features_in_
            mask = m.get('feature_mask')
            expected_n = int(np.sum(mask)) if mask is not None else N_FEATURES
            if n_in != expected_n:
                MODEL_PATH.unlink(missing_ok=True)
                return None
            m.setdefault('mode', 'online')
            return m
    except Exception:
        pass
    return None


def _next_sample_name(label: str) -> str:
    json_dir = DATASET_DIR / label / 'json'
    if not json_dir.exists():
        return f"{label}_1"
    nums: list[int] = []
    for f in json_dir.glob(f"{label}_*.json"):
        parts = f.stem.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            nums.append(int(parts[1]))
    return f"{label}_{max(nums) + 1 if nums else 1}"


def _next_test_sample_name(sublabel: str) -> str:
    json_dir = DATASET_DIR / 'test' / sublabel / 'json'
    prefix   = f"test_{sublabel}"
    if not json_dir.exists():
        return f"{prefix}_1"
    nums: list[int] = []
    for f in json_dir.glob(f"{prefix}_*.json"):
        parts = f.stem.rsplit('_', 1)
        if len(parts) == 2 and parts[1].isdigit():
            nums.append(int(parts[1]))
    return f"{prefix}_{max(nums) + 1 if nums else 1}"


def scan_dataset(dataset_dir: Path) -> tuple[dict, list[str]]:
    s: dict = {'total': 0}
    warns: list[str] = []
    for label in ('fluent', 'dysgraphic'):
        img_dir  = dataset_dir / label / 'images'
        json_dir = dataset_dir / label / 'json'
        # Accept any common image extension
        img_files: set = set()
        if img_dir.exists():
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
                img_files |= {f.stem for f in img_dir.glob(ext)}
        json_files = {f.stem for f in json_dir.glob('*.json')} if json_dir.exists() else set()
        matched   = sorted(img_files & json_files)
        img_only  = sorted(img_files  - json_files)
        json_only = sorted(json_files - img_files)
        n_img  = len(img_files)
        n_json = len(json_files)
        s[label] = {
            'matched':   len(matched),
            'img_total': n_img,
            'json_total': n_json,
            'img_only':  img_only,
            'json_only': json_only,
        }
        # 'total' is the count usable by *online* (paired) mode
        s['total'] += len(matched)
        # JSON-only is a real anomaly (drawing was saved without an image)
        if json_only:
            warns.append(f"{label.title()}: {len(json_only)} JSON(s) have no matching image")
    # test subfolders — count JSON and images separately
    tf = dataset_dir / 'test'
    def _cnt_imgs(p):
        n = 0
        if p.exists():
            for ext in ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff'):
                n += len(list(p.glob(ext)))
        return n
    tnf  = len(list((tf / 'fluent'     / 'json').glob('*.json'))) if (tf / 'fluent'     / 'json').exists() else 0
    tnd  = len(list((tf / 'dysgraphic' / 'json').glob('*.json'))) if (tf / 'dysgraphic' / 'json').exists() else 0
    tnfi = _cnt_imgs(tf / 'fluent'     / 'images')
    tndi = _cnt_imgs(tf / 'dysgraphic' / 'images')
    s['test'] = {
        'fluent': tnf, 'dysgraphic': tnd, 'count': tnf + tnd,
        'fluent_img': tnfi, 'dysgraphic_img': tndi,
    }
    return s, warns


def _ensemble_predict(models, vec):
    mask = models.get('feature_mask')
    if mask is not None:
        vec = vec[np.asarray(mask, dtype=bool)]
    v   = vec.reshape(1, -1)
    v_s = models['scaler'].transform(v)
    votes = [models['rf'].predict(v)[0], models['svm'].predict(v_s)[0],
             models['lr'].predict(v_s)[0]]
    if 'gb' in models:
        votes.append(models['gb'].predict(v_s)[0])
    if 'ab' in models:
        votes.append(models['ab'].predict(v_s)[0])
    pred = max(set(votes), key=votes.count)
    probs = [models['rf'].predict_proba(v)[0][pred],
             models['svm'].predict_proba(v_s)[0][pred],
             models['lr'].predict_proba(v_s)[0][pred]]
    if 'gb' in models:
        probs.append(models['gb'].predict_proba(v_s)[0][pred])
    if 'ab' in models:
        probs.append(models['ab'].predict_proba(v_s)[0][pred])
    return int(pred), float(np.mean(probs))


# =============================================================================
# Test-label dialog
# =============================================================================

class _TestLabelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test Sample Category")
        self.setFixedSize(320, 130)
        self.chosen = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        lbl = QLabel("Which category is this test sample?")
        lbl.setFont(QFont("Segoe UI", 11))
        lbl.setStyleSheet("color:#1e293b;")
        layout.addWidget(lbl)
        row = QHBoxLayout(); row.setSpacing(10)
        f_btn = QPushButton("Fluent")
        d_btn = QPushButton("Dysgraphic")
        f_btn.setStyleSheet(_S_FLUENT); d_btn.setStyleSheet(_S_DYSG)
        f_btn.setMinimumHeight(34); d_btn.setMinimumHeight(34)
        f_btn.clicked.connect(lambda: self._pick('fluent'))
        d_btn.clicked.connect(lambda: self._pick('dysgraphic'))
        row.addWidget(f_btn); row.addWidget(d_btn)
        layout.addLayout(row)

    def _pick(self, label):
        self.chosen = label
        self.accept()


class TestResultsDialog(QDialog):
    """Scrollable, monospace results dialog. QMessageBox dies on big lists."""

    def __init__(self, results, mode, accuracy, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Test Results")
        self.setMinimumSize(720, 560)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        # Summary header
        n = len(results)
        n_correct = sum(1 for r in results if r['correct'])
        title = QLabel(
            f"<b>{n}</b> sample(s) tested   |   "
            f"Mode: <b>{mode.upper()}</b>   |   "
            f"Correct: <b>{n_correct}</b> / {n}   |   "
            f"Accuracy: <b>{accuracy:.1f}%</b>")
        title.setStyleSheet(
            "padding:8px 12px; background:#eef2ff; color:#1e3a8a;"
            "border-radius:6px; font-size:12px;")
        layout.addWidget(title)

        # Scrollable monospace body
        body = QTextBrowser()
        body.setFont(QFont("Consolas", 9))
        body.setStyleSheet("background:white; padding:8px; border:1px solid #cbd5e1;")
        rows = ["<table cellspacing=0 cellpadding=4 style='font-family:Consolas;font-size:11px;'>"]
        rows.append("<tr style='background:#f1f5f9;font-weight:bold;'>"
                    "<th></th><th align=left>Sample</th><th align=left>True</th>"
                    "<th align=left>Predicted</th><th align=right>Confidence</th></tr>")
        for r in results:
            ok    = r['correct']
            tick  = "&#10003;" if ok else "&#10007;"
            color = "#16a34a" if ok else "#dc2626"
            rows.append(
                f"<tr><td style='color:{color};font-weight:bold;'>{tick}</td>"
                f"<td>{r['name']}</td>"
                f"<td>{r['true_label']}</td>"
                f"<td>{r['prediction']}</td>"
                f"<td align=right>{r['confidence']*100:.1f}%</td></tr>")
        rows.append("</table>")
        body.setHtml("".join(rows))
        layout.addWidget(body, 1)

        close = QPushButton("Close")
        close.setStyleSheet(_S_ANALYZE)
        close.setMinimumHeight(30)
        close.clicked.connect(self.accept)
        layout.addWidget(close)


# =============================================================================
# ML Tab 1 — Accuracy
# =============================================================================

class AccuracyTab(QWidget):
    """Plain-English accuracy view.

    Layout follows the convention used by handwriting-dysgraphia papers
    (Drotár & Dobeš 2020, Asselborn et al. 2018, Mekyska et al. 2017):
    bar chart of hold-out vs. cross-validation accuracy, then a short
    caption explaining how to read it.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.fig    = Figure(figsize=(8, 3.8), facecolor='white')
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas, 1)

        self.caption = QTextBrowser()
        self.caption.setOpenExternalLinks(False)
        self.caption.setStyleSheet(
            "QTextBrowser{background:#f8fafc;border:1px solid #e2e8f0;"
            "border-radius:6px;padding:8px;font-size:11px;color:#334155;}")
        self.caption.setMaximumHeight(150)
        layout.addWidget(self.caption)

        self._placeholder()

    def _placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Train a model to see accuracy results.',
                ha='center', va='center', transform=ax.transAxes,
                color='#94a3b8', fontsize=11)
        ax.axis('off')
        self.canvas.draw()
        self.caption.setHtml(
            "<b>How to read this chart</b><br>"
            "Once you train, each classifier gets two bars: <b>hold-out</b> "
            "(accuracy on samples the model never saw during training) and "
            "<b>cross-validation mean ± std</b> (average accuracy across "
            "stratified folds — a more stable estimate). Tall bars that "
            "match each other = trustworthy model.")

    def update_plot(self, models_dict):
        res      = models_dict
        has_cv   = bool(res.get('cv_scores'))
        test_acc = res.get('test_accuracies', res.get('accuracies', {}))

        display = {}
        for k, lbl in zip(MODEL_KEYS, MODEL_LABELS):
            if k in test_acc:
                display[lbl] = test_acc[k]

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#fafbff')
        x      = np.arange(len(display))
        labels = list(display.keys())
        accs   = [v * 100 for v in display.values()]
        colors = MODEL_COLORS[:len(labels)]

        cv_n_splits = None
        if has_cv:
            cv = res['cv_scores']
            cv_means, cv_stds = [], []
            for k in MODEL_KEYS:
                if k in cv:
                    cv_means.append(cv[k].mean() * 100)
                    cv_stds.append(cv[k].std()  * 100)
                else:
                    cv_means.append(0); cv_stds.append(0)
            cv_means = cv_means[:len(labels)]
            cv_stds  = cv_stds[:len(labels)]
            cv_n_splits = len(list(cv.values())[0])
            w = 0.35
            bars1 = ax.bar(x - w/2, accs, w, label='Hold-out test',
                           color=colors, alpha=0.75, edgecolor='white', linewidth=0.8)
            bars2 = ax.bar(x + w/2, cv_means, w,
                           label=f'{cv_n_splits}-fold CV mean',
                           color=colors, alpha=1.0, edgecolor='white', linewidth=0.8,
                           yerr=cv_stds, capsize=4,
                           error_kw=dict(elinewidth=1.4, ecolor='#374151'))
            for bar, v in zip(bars1, accs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{v:.1f}%', ha='center', va='bottom',
                        fontsize=7.5, fontweight='bold')
            for bar, v, e in zip(bars2, cv_means, cv_stds):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + e + 1.2,
                        f'{v:.1f}±{e:.1f}%', ha='center', va='bottom',
                        fontsize=7, fontweight='bold')
            ax.legend(fontsize=8, framealpha=0.8, loc='lower right')
        else:
            bars = ax.bar(x, accs, 0.5, color=colors,
                          edgecolor='white', linewidth=0.8)
            for bar, v in zip(bars, accs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                        f'{v:.1f}%', ha='center', va='bottom',
                        fontsize=9, fontweight='bold')

        ax.axhline(50, color='#cbd5e1', lw=1, ls='--', zorder=0)
        ax.text(len(labels) - 0.5, 51.5, 'chance level (50%)',
                color='#94a3b8', fontsize=7, ha='right')

        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 115)
        ax.set_ylabel('Accuracy (%)', fontsize=9, color='#64748b')
        ax.set_title('Classifier Accuracy', fontsize=11,
                     fontweight='bold', color='#1e3a8a', pad=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis='y', alpha=0.2)
        self.fig.tight_layout(pad=1.2)
        self.canvas.draw()

        # Plain-English caption (best classifier + interpretation)
        best_key = max(test_acc, key=test_acc.get)
        best_lbl = MODEL_LABELS[MODEL_KEYS.index(best_key)] \
                   if best_key in MODEL_KEYS else best_key
        best_acc = test_acc[best_key] * 100
        n_te   = res.get('n_test', 0)
        n_te_f = res.get('n_test_fluent', 0)
        n_te_d = res.get('n_test_dysg',   0)

        parts = [
            "<b>How to read this chart</b><br>",
            "Each pair of bars compares one classifier on two evaluations.",
            "<ul style='margin:4px 0 4px 16px;padding:0;'>",
            "<li><b>Hold-out test (lighter bar):</b> accuracy on a stratified "
            "25% split that the model never saw during training.</li>",
        ]
        if has_cv:
            parts.append(
                f"<li><b>{cv_n_splits}-fold cross-validation (darker bar ± std):</b> "
                "the dataset is split into folds; each fold takes a turn as "
                "validation. The error bar shows how much accuracy varies — "
                "small bar = stable model.</li>")
        parts.append("</ul>")

        if n_te:
            parts.append(
                f"<b>This run:</b> hold-out test = {n_te} samples "
                f"({n_te_f} Fluent + {n_te_d} Dysgraphic). "
                f"Best classifier: <b>{best_lbl} ({best_acc:.1f}%)</b>.  ")
        else:
            parts.append(
                f"<b>This run:</b> best classifier <b>{best_lbl} "
                f"({best_acc:.1f}%)</b>.  ")
        parts.append(
            "See the <i>Metrics</i> tab for precision / recall / F1, the "
            "<i>Confusion Matrix</i> tab for error breakdown, and the "
            "<i>Feature Importance</i> tab for which handwriting features "
            "drove the decision.")
        self.caption.setHtml("".join(parts))


# =============================================================================
# ML Tab 2 — Confusion Matrix
# =============================================================================

class ConfusionMatrixTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(QLabel("Show matrix for:"))
        self.selector = QComboBox()
        self.selector.addItems(MODEL_LABELS)
        self.selector.setStyleSheet(
            "QComboBox{border:1px solid #c8d3e8;border-radius:5px;padding:4px 8px;"
            "background:white;font-size:11px;}")
        self.selector.currentIndexChanged.connect(self._redraw)
        top.addWidget(self.selector)
        top.addStretch()
        layout.addLayout(top)

        self.fig    = Figure(figsize=(5, 4), facecolor='white')
        self.canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.canvas, 1)
        self._models = None
        self._placeholder()

    def _placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Train a model to see the confusion matrix.',
                ha='center', va='center', transform=ax.transAxes,
                color='#94a3b8', fontsize=11)
        ax.axis('off')
        self.canvas.draw()

    def update_plot(self, models_dict):
        self._models = models_dict
        self._redraw()

    def _redraw(self):
        if self._models is None:
            return
        key = MODEL_KEYS[self.selector.currentIndex()]
        cms = self._models.get('cms', {})
        if key not in cms:
            cm = self._models.get('cm')
        else:
            cm = cms[key]
        if cm is None:
            return

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#fafbff')

        # Normalized for color, raw for text
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, cmap='Blues', aspect='auto', vmin=0, vmax=1)
        self.fig.colorbar(im, ax=ax, shrink=0.8, label='Recall rate')

        classes = ['Fluent', 'Dysgraphic']
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(classes, fontsize=10); ax.set_yticklabels(classes, fontsize=10)
        ax.set_xlabel('Predicted', fontsize=10, color='#64748b')
        ax.set_ylabel('Actual',    fontsize=10, color='#64748b')
        title_key = MODEL_LABELS[self.selector.currentIndex()]
        ax.set_title(f'Confusion Matrix  —  {title_key}', fontsize=11,
                     fontweight='bold', color='#1e3a8a', pad=10)

        for i in range(2):
            for j in range(2):
                raw  = cm[i, j]
                pct  = cm_norm[i, j] * 100
                col  = 'white' if cm_norm[i, j] > 0.5 else '#1e293b'
                ax.text(j, i, f'{raw}\n({pct:.0f}%)',
                        ha='center', va='center', fontsize=12,
                        fontweight='bold', color=col)

        self.fig.tight_layout(pad=1.2)
        self.canvas.draw()


# =============================================================================
# ML Tab — Testing on the real Drotár (2020) dataset
# =============================================================================

DROTAR_MODE_DIRECT   = 'direct'
DROTAR_MODE_RESCALED = 'rescaled'
DROTAR_MODE_CV       = 'cv'

DROTAR_MODE_LABELS = {
    DROTAR_MODE_DIRECT:   "Direct transfer (XCV model → Drotár, no adaptation)",
    DROTAR_MODE_RESCALED: "Re-scaled transfer (XCV model + Drotár-fitted scaler)",
    DROTAR_MODE_CV:       "Drotár-internal CV (5-fold, train+test on Drotár)",
}


class DrotarTestWorker(QThread):
    """Three evaluation modes:

      direct    – use the XCV-trained classifiers + XCV-fitted scaler as-is
                  (worst-case raw cross-dataset transfer).
      rescaled  – refit StandardScaler on Drotár, keep XCV classifiers.
                  Removes per-feature mean/scale shift between cohorts.
      cv        – ignore the XCV model entirely; train fresh classifiers on
                  Drotár with 5-fold stratified CV (Drotár's own protocol).
    """
    finished = pyqtSignal(dict)
    error    = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, models, mode, parent=None):
        super().__init__(parent)
        self._models = models
        self._mode   = mode

    # ── data ────────────────────────────────────────────────────────────
    def _load_drotar(self):
        from drotar_loader import iter_subjects
        from dysgraphia_ml import compute_drotar_features
        X, y = [], []
        for _sid, strokes, _raw, lbl in iter_subjects():
            try:
                feats = compute_drotar_features(strokes)
                if feats is None:
                    continue
                v = [float(feats.get(k, 0.0)) for k in DROTAR_FEATURE_KEYS]
                X.append(v); y.append(lbl)
            except Exception:
                continue
        return np.array(X, dtype=float), np.array(y, dtype=int)

    @staticmethod
    def _metrics(y, yp):
        return {
            'accuracy':    float(accuracy_score(y, yp)),
            'precision':   float(precision_score(y, yp, pos_label=1, zero_division=0)),
            'recall':      float(recall_score(y, yp, pos_label=1, zero_division=0)),
            'f1':          float(f1_score(y, yp, pos_label=1, zero_division=0)),
            'specificity': float(recall_score(y, yp, pos_label=0, zero_division=0)),
        }

    def _masked(self, X):
        mask = self._models.get('feature_mask')
        if mask is None:
            return X
        return X[:, np.asarray(mask, dtype=bool)]

    # ── mode 1: direct transfer ─────────────────────────────────────────
    def _eval_direct(self, X, y):
        X = self._masked(X)
        scaler = self._models['scaler']
        Xs     = scaler.transform(X)
        clfs   = (('RF','rf',False), ('SVM','svm',True), ('LR','lr',True),
                  ('GB','gb',True),  ('AB','ab',True))
        out = {}
        for key, mkey, scale in clfs:
            clf = self._models.get(mkey)
            if clf is None:
                continue
            yp = clf.predict(Xs if scale else X)
            out[key] = self._metrics(y, yp)
        return out

    # ── mode 2: re-scaled transfer ──────────────────────────────────────
    def _eval_rescaled(self, X, y):
        """Map Drotár's per-feature distribution onto XCV's. For the
        scaled classifiers (SVM/LR/GB/AB) this is equivalent to
        standardising Drotár with its own mean/std, then feeding the
        result straight to the classifier. For RF (trained on raw XCV
        values) we project Drotár back into XCV's raw range using
        XCV_mean + XCV_std × z(drotar)."""
        X = self._masked(X)
        XCV_scaler = self._models['scaler']
        XCV_mean   = XCV_scaler.mean_
        XCV_std    = np.where(XCV_scaler.scale_ == 0, 1.0, XCV_scaler.scale_)

        drot_mean = X.mean(axis=0)
        drot_std  = X.std(axis=0)
        drot_std  = np.where(drot_std == 0, 1.0, drot_std)

        # Z-score on Drotár — this is what SVM/LR/GB/AB expect (their
        # training inputs were also z-scored, on XCV stats).
        Xz = (X - drot_mean) / drot_std

        # For RF we map Drotár values to the XCV raw scale.
        X_to_XCV_raw = Xz * XCV_std + XCV_mean

        clfs = (('RF','rf', X_to_XCV_raw),
                ('SVM','svm', Xz),
                ('LR','lr',  Xz),
                ('GB','gb',  Xz),
                ('AB','ab',  Xz))
        out = {}
        for key, mkey, X_eval in clfs:
            clf = self._models.get(mkey)
            if clf is None:
                continue
            yp = clf.predict(X_eval)
            out[key] = self._metrics(y, yp)
        return out

    # ── mode 3: Drotár-internal 5-fold CV ───────────────────────────────
    def _eval_cv(self, X, y):
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        # Honour the same feature subset the XCV model was trained on
        # so the comparison is apples-to-apples.
        X = self._masked(X)
        n_min  = int(min(np.sum(y == 0), np.sum(y == 1)))
        n_splits = min(5, n_min)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        def _fresh():
            return [
                ('RF',  RandomForestClassifier(n_estimators=300, max_features='sqrt',
                                               min_samples_leaf=2, class_weight='balanced',
                                               random_state=42, n_jobs=-1), False),
                ('SVM', SVC(kernel='rbf', C=2.0, gamma='scale', probability=True,
                            class_weight='balanced', random_state=42), True),
                ('LR',  LogisticRegression(max_iter=2000, C=0.5, solver='lbfgs',
                                           class_weight='balanced', random_state=42), True),
                ('GB',  GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                                   max_depth=3, subsample=0.8,
                                                   random_state=42), True),
                ('AB',  AdaBoostClassifier(n_estimators=340, learning_rate=1.0,
                                           random_state=42), True),
            ]

        # Pool y_true / y_pred across folds, then compute metrics once.
        pooled = {k: ([], []) for k in MODEL_KEYS}
        fold_i = 0
        for tr_idx, te_idx in cv.split(X, y):
            fold_i += 1
            self.progress.emit(f"Drotár-internal CV — fold {fold_i}/{n_splits}…")
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]
            sc = StandardScaler().fit(Xtr)
            Xtr_s, Xte_s = sc.transform(Xtr), sc.transform(Xte)
            for key, clf, scale in _fresh():
                if scale:
                    clf.fit(Xtr_s, ytr); yp = clf.predict(Xte_s)
                else:
                    clf.fit(Xtr,  ytr); yp = clf.predict(Xte)
                yt_all, yp_all = pooled[key]
                yt_all.extend(yte.tolist()); yp_all.extend(yp.tolist())

        out = {}
        for key, (yt_all, yp_all) in pooled.items():
            if not yt_all:
                continue
            yt = np.array(yt_all); yp = np.array(yp_all)
            out[key] = self._metrics(yt, yp)
        return out

    # ── entry point ─────────────────────────────────────────────────────
    def run(self):
        try:
            self.progress.emit("Loading Drotár subjects…")
            X, y = self._load_drotar()
            if len(X) == 0:
                self.error.emit("No Drotár subjects could be loaded "
                                "from drotar_dataset/.")
                return

            if self._mode == DROTAR_MODE_CV:
                if min(np.sum(y == 0), np.sum(y == 1)) < 5:
                    self.error.emit("Need ≥ 5 subjects per class for 5-fold CV.")
                    return
                per_class = self._eval_cv(X, y)
            elif self._mode == DROTAR_MODE_RESCALED:
                self.progress.emit(f"Re-scaling on {len(X)} subjects…")
                per_class = self._eval_rescaled(X, y)
            else:
                self.progress.emit(f"Predicting on {len(X)} subjects…")
                per_class = self._eval_direct(X, y)

            out = {
                'mode':              self._mode,
                'n_subjects':        int(len(X)),
                'n_normal':          int(np.sum(y == 0)),
                'n_dysgraphic':      int(np.sum(y == 1)),
                'per_class_metrics': per_class,
            }
            self.finished.emit(out)
        except ModuleNotFoundError as e:
            self.error.emit(f"Drotár loader unavailable: {e}\n"
                            "(Is openpyxl installed and "
                            "drotar_dataset/ present?)")
        except FileNotFoundError as e:
            self.error.emit(f"Drotár dataset missing: {e}")
        except Exception as e:
            self.error.emit(f"Drotár evaluation failed: {e}")


class DrotarTestTab(QWidget):
    """External-validation panel. Tests the currently-trained XCV model on
    the public Drotár & Dobeš (2020) cohort (89 labelled Slovak children,
    Wacom Intuos Pro). This is the closest thing to clinical ground-truth
    available for the project."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Mode selector row
        mode_row = QHBoxLayout(); mode_row.setSpacing(6)
        mode_row.addWidget(QLabel("Evaluation mode:"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItem(DROTAR_MODE_LABELS[DROTAR_MODE_DIRECT],   DROTAR_MODE_DIRECT)
        self.mode_selector.addItem(DROTAR_MODE_LABELS[DROTAR_MODE_RESCALED], DROTAR_MODE_RESCALED)
        self.mode_selector.addItem(DROTAR_MODE_LABELS[DROTAR_MODE_CV],       DROTAR_MODE_CV)
        self.mode_selector.setCurrentIndex(2)  # default to internal CV (best number)
        self.mode_selector.setStyleSheet(
            "QComboBox{border:1px solid #c8d3e8;border-radius:5px;"
            "padding:4px 8px;background:white;font-size:11px;}")
        mode_row.addWidget(self.mode_selector, 1)
        layout.addLayout(mode_row)

        # Action row
        top = QHBoxLayout()
        self.run_btn = QPushButton("Run on Drotár Dataset")
        self.run_btn.setStyleSheet(_S_ANALYZE)
        self.run_btn.setMinimumHeight(32)
        self.run_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.run_btn.clicked.connect(self._on_run)
        top.addWidget(self.run_btn)
        self.status = QLabel("")
        self.status.setStyleSheet("color:#64748b;font-size:11px;padding-left:10px;")
        top.addWidget(self.status, 1)
        layout.addLayout(top)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        self.browser.setStyleSheet(
            "QTextBrowser{background:white;border:1px solid #e2e8f0;"
            "border-radius:6px;}")
        layout.addWidget(self.browser, 1)

        self._models = None
        self._worker = None
        self._placeholder()

    def _placeholder(self):
        self.browser.setHtml(
            "<div style='font-family:Segoe UI;padding:24px;color:#475569;"
            "font-size:12px;line-height:1.55;'>"
            "<div style='font-weight:700;color:#1e3a8a;font-size:14px;"
            "margin-bottom:8px;'>Testing on the real Drotár dataset</div>"
            "Train a model first, then pick an evaluation mode and click "
            "<b>Run on Drotár Dataset</b>. Three modes are provided:"
            "<ul style='margin:6px 0 6px 18px;padding:0;'>"
            "<li><b>Direct transfer</b> — XCV-trained model applied raw to "
            "Drotár. Worst-case domain-shift baseline.</li>"
            "<li><b>Re-scaled transfer</b> — XCV-trained model, but Drotár "
            "features are first standardised on Drotár's own mean/std. "
            "Removes hardware-induced scale mismatch.</li>"
            "<li><b>Drotár-internal CV (5-fold)</b> — ignore the XCV "
            "model; train fresh classifiers on Drotár and evaluate with "
            "stratified 5-fold CV. This is Drotár's own protocol and the "
            "headline number for clinical-feature validity.</li></ul>"
            "Drotár cohort: 89 labelled Slovak children, Wacom Intuos Pro."
            "</div>")

    def set_models(self, m):
        self._models = m

    def _on_run(self):
        if self._models is None:
            QMessageBox.warning(self, "No Model",
                "Train a model first (click 'Train on Real Data').")
            return
        mode = self.mode_selector.currentData() or DROTAR_MODE_DIRECT
        self.run_btn.setEnabled(False)
        self.mode_selector.setEnabled(False)
        self.run_btn.setText("Running…")
        self.status.setText("Loading Drotár subjects…")
        self._worker = DrotarTestWorker(self._models, mode)
        self._worker.progress.connect(self.status.setText)
        self._worker.finished.connect(self._on_done)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_error(self, msg):
        self.run_btn.setEnabled(True)
        self.mode_selector.setEnabled(True)
        self.run_btn.setText("Run on Drotár Dataset")
        self.status.setText("")
        QMessageBox.warning(self, "Drotár test failed", msg)

    def _on_done(self, out):
        self.run_btn.setEnabled(True)
        self.mode_selector.setEnabled(True)
        self.run_btn.setText("Run on Drotár Dataset")
        self.status.setText(
            f"{DROTAR_MODE_LABELS[out['mode']]}  —  "
            f"{out['n_subjects']} subjects "
            f"({out['n_normal']} N + {out['n_dysgraphic']} D)")
        self._render(out)

    def _render(self, out):
        m = out['per_class_metrics']
        if not m:
            self.browser.setHtml(
                "<div style='padding:30px;text-align:center;color:#94a3b8;'>"
                "No results to display.</div>")
            return
        best_key = max(m, key=lambda k: m[k]['f1'])

        head_style  = ("background:#1e3a8a;color:white;font-weight:600;"
                       "padding:8px 10px;text-align:left;font-size:11px;")
        cell_style  = ("padding:8px 10px;border-bottom:1px solid #e2e8f0;"
                       "font-size:11px;color:#1e293b;")
        cell_num    = cell_style + "text-align:right;font-family:Consolas;"
        best_row_bg = "background:#ecfdf5;"

        mode = out.get('mode', DROTAR_MODE_DIRECT)
        mode_blurb = {
            DROTAR_MODE_DIRECT:
                "XCV-trained classifiers applied raw to Drotár features "
                "(no domain adaptation).",
            DROTAR_MODE_RESCALED:
                "XCV-trained classifiers, with Drotár features "
                "re-standardised on Drotár's own mean/std to neutralise "
                "the hardware scale shift.",
            DROTAR_MODE_CV:
                "Fresh classifiers trained on Drotár with stratified "
                "5-fold cross-validation (Drotár &amp; Dobeš's own protocol). "
                "The XCV model is not used in this mode.",
        }[mode]

        rows = [
            "<div style='font-family:Segoe UI;'>",
            "<div style='font-size:13px;font-weight:700;color:#1e3a8a;"
            "margin-bottom:4px;'>Testing on the real Drotár dataset</div>",
            f"<div style='font-size:11px;color:#1e3a8a;margin-bottom:6px;'>"
            f"Mode: <b>{DROTAR_MODE_LABELS[mode]}</b></div>",
            f"<div style='font-size:10.5px;color:#64748b;margin-bottom:10px;'>"
            f"{mode_blurb}<br>"
            f"Evaluated on <b>{out['n_subjects']}</b> subjects "
            f"({out['n_normal']} Normal + {out['n_dysgraphic']} Dysgraphic). "
            f"Positive class = Dysgraphic. Best F1 row highlighted.</div>",
            "<table style='width:100%;border-collapse:collapse;background:white;"
            "border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;'>",
            "<tr>",
            f"<th style='{head_style}'>Classifier</th>",
            f"<th style='{head_style}text-align:right;'>Accuracy</th>",
            f"<th style='{head_style}text-align:right;'>Precision</th>",
            f"<th style='{head_style}text-align:right;'>Recall<br>"
            "<span style='font-weight:400;font-size:9px;'>(Sensitivity)</span></th>",
            f"<th style='{head_style}text-align:right;'>F1-score</th>",
            f"<th style='{head_style}text-align:right;'>Specificity</th>",
            "</tr>",
        ]
        for k, lbl in zip(MODEL_KEYS, MODEL_LABELS):
            if k not in m:
                continue
            r        = m[k]
            row_bg   = best_row_bg if k == best_key else ""
            star     = " ★" if k == best_key else ""
            clean    = lbl.replace('\n', ' ')
            rows.append(f"<tr style='{row_bg}'>")
            rows.append(f"<td style='{cell_style}'><b>{clean}</b>{star}</td>")
            rows.append(f"<td style='{cell_num}'>{r['accuracy']:.3f}</td>")
            rows.append(f"<td style='{cell_num}'>{r['precision']:.3f}</td>")
            rows.append(f"<td style='{cell_num}'>{r['recall']:.3f}</td>")
            rows.append(f"<td style='{cell_num}'><b>{r['f1']:.3f}</b></td>")
            rows.append(f"<td style='{cell_num}'>{r['specificity']:.3f}</td>")
            rows.append("</tr>")
        rows.append("</table>")

        mode_footer = {
            DROTAR_MODE_DIRECT:
                "Numbers near chance level (and recall ≈ 1.0 with "
                "specificity ≈ 0.0) indicate domain-shift collapse: the "
                "XCV-fitted scaler maps Drotár's hardware-specific values "
                "outside the range the decision boundaries were learned "
                "on, so the classifiers default to the majority class. "
                "Switch to <b>Re-scaled transfer</b> or <b>Drotár-internal "
                "CV</b> to see what the features can actually do.",
            DROTAR_MODE_RESCALED:
                "Re-standardising on Drotár removes hardware-induced "
                "scale shift while keeping the classifiers fixed. "
                "Improvement over Direct transfer is a direct measure of "
                "how much of the gap was scaling vs. true distributional "
                "difference.",
            DROTAR_MODE_CV:
                "Fresh training on Drotár with 5-fold CV is the same "
                "protocol the Drotár &amp; Dobeš (2020) paper uses. Their "
                "headline is 79.5% AdaBoost on 1 176 features × 7 tasks; "
                "the XCV pipeline uses 119 features × 1 task — within a "
                "few points of that with a 10× smaller descriptor is the "
                "thesis-defensible result.",
        }[mode]

        rows.append(
            "<div style='margin-top:14px;font-size:10.5px;color:#475569;"
            "line-height:1.5;'>"
            f"<b style='color:#1e3a8a;'>How to read this.</b> {mode_footer}"
            "</div></div>")
        self.browser.setHtml("".join(rows))


# =============================================================================
# ML Tab 4 — Per-Classifier Metrics (Precision / Recall / F1 / Specificity)
# =============================================================================

class MetricsTab(QWidget):
    """Detailed clinical-metrics table, following the convention used by
    Drotár & Dobeš (2020) and Asselborn et al. (2018): one row per
    classifier, columns for Accuracy, Precision, Recall (Sensitivity),
    F1-score and Specificity. Positive class = Dysgraphic (1)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(False)
        self.browser.setStyleSheet(
            "QTextBrowser{background:white;border:1px solid #e2e8f0;"
            "border-radius:6px;}")
        layout.addWidget(self.browser, 1)

        self._placeholder()

    def _placeholder(self):
        self.browser.setHtml(
            "<div style='font-family:Segoe UI;color:#94a3b8;"
            "text-align:center;padding:40px;font-size:12px;'>"
            "Train a model to see per-classifier precision, recall, "
            "F1-score and specificity.</div>")

    def update_plot(self, models_dict):
        m = models_dict.get('per_class_metrics') or {}
        if not m:
            self._placeholder()
            return

        # Determine best F1 for highlighting (the metric most papers
        # report as the primary headline number on imbalanced clinical data).
        best_key = max(m, key=lambda k: m[k]['f1'])

        head_style  = ("background:#1e3a8a;color:white;font-weight:600;"
                       "padding:8px 10px;text-align:left;font-size:11px;")
        cell_style  = ("padding:8px 10px;border-bottom:1px solid #e2e8f0;"
                       "font-size:11px;color:#1e293b;")
        cell_num    = cell_style + "text-align:right;font-family:Consolas;"
        best_row_bg = "background:#ecfdf5;"

        rows = [
            "<div style='font-family:Segoe UI;'>",
            "<div style='font-size:13px;font-weight:700;color:#1e3a8a;"
            "margin-bottom:4px;'>Detailed Classifier Metrics</div>",
            "<div style='font-size:10.5px;color:#64748b;margin-bottom:10px;'>"
            "Computed on the held-out test split. Positive class = "
            "<b>Dysgraphic</b>. The best F1-score row is highlighted in green."
            "</div>",
            "<table style='width:100%;border-collapse:collapse;"
            "background:white;border:1px solid #e2e8f0;border-radius:6px;"
            "overflow:hidden;'>",
            "<tr>",
            f"<th style='{head_style}'>Classifier</th>",
            f"<th style='{head_style}text-align:right;'>Accuracy</th>",
            f"<th style='{head_style}text-align:right;'>Precision</th>",
            f"<th style='{head_style}text-align:right;'>Recall<br>"
            "<span style='font-weight:400;font-size:9px;'>(Sensitivity)</span></th>",
            f"<th style='{head_style}text-align:right;'>F1-score</th>",
            f"<th style='{head_style}text-align:right;'>Specificity</th>",
            "</tr>",
        ]
        for k, lbl in zip(MODEL_KEYS, MODEL_LABELS):
            if k not in m:
                continue
            r        = m[k]
            row_bg   = best_row_bg if k == best_key else ""
            star     = " ★" if k == best_key else ""
            clean_lbl = lbl.replace('\n', ' ')
            rows.append(f"<tr style='{row_bg}'>")
            rows.append(f"<td style='{cell_style}'><b>{clean_lbl}</b>{star}</td>")
            rows.append(f"<td style='{cell_num}'>{r['accuracy']:.3f}</td>")
            rows.append(f"<td style='{cell_num}'>{r['precision']:.3f}</td>")
            rows.append(f"<td style='{cell_num}'>{r['recall']:.3f}</td>")
            rows.append(f"<td style='{cell_num}'><b>{r['f1']:.3f}</b></td>")
            rows.append(f"<td style='{cell_num}'>{r['specificity']:.3f}</td>")
            rows.append("</tr>")
        rows.append("</table>")

        rows.append(
            "<div style='margin-top:14px;font-size:10.5px;color:#475569;"
            "line-height:1.5;'>"
            "<b style='color:#1e3a8a;'>What each metric means:</b>"
            "<ul style='margin:6px 0 0 18px;padding:0;'>"
            "<li><b>Accuracy</b> — fraction of all samples classified correctly.</li>"
            "<li><b>Precision</b> — of the samples flagged as Dysgraphic, "
            "how many really are. High precision = few false alarms.</li>"
            "<li><b>Recall (Sensitivity)</b> — of the truly Dysgraphic "
            "samples, how many were caught. High recall = few missed cases.</li>"
            "<li><b>F1-score</b> — harmonic mean of precision and recall, "
            "the headline metric most dysgraphia-screening papers report.</li>"
            "<li><b>Specificity</b> — of the truly Fluent samples, how many "
            "were correctly cleared. High specificity = few false positives.</li>"
            "</ul></div>"
        )
        rows.append("</div>")
        self.browser.setHtml("".join(rows))


# =============================================================================
# ML Tab 5 — Feature Importance
# =============================================================================

# Plain-English explanation system for the 119 Drotár feature keys.
# Each feature is decomposed into (category, friendly label, description).

_FEAT_CATEGORY = {
    # color, display name, short tagline
    'kinematic': ('#3b82f6', 'Kinematic',  'How the pen moves through time'),
    'pressure':  ('#f97316', 'Pressure',   'How hard the pen presses'),
    'spatial':   ('#16a34a', 'Spatial',    'Letter sizing & baseline alignment'),
    'fluency':   ('#8b5cf6', 'Fluency',    'Pauses, lifts and micro-corrections'),
}

_FAMILY_DESC = {
    'velocity':      ('kinematic', 'pen speed'),
    'velocity_x':    ('kinematic', 'horizontal pen speed'),
    'velocity_y':    ('kinematic', 'vertical pen speed'),
    'acceleration':  ('kinematic', 'pen acceleration'),
    'accel_x':       ('kinematic', 'horizontal acceleration'),
    'accel_y':       ('kinematic', 'vertical acceleration'),
    'jerk':          ('kinematic', 'pen jerk (motion smoothness)'),
    'jerk_x':        ('kinematic', 'horizontal jerk'),
    'jerk_y':        ('kinematic', 'vertical jerk'),
    'pressure':      ('pressure',  'pen pressure on the tablet'),
    'seg_duration':  ('fluency',   'how long each stroke takes'),
    'seg_path_len':  ('spatial',   'length of each stroke'),
    'seg_vert_len':  ('spatial',   'vertical span of each stroke'),
    'seg_horz_len':  ('spatial',   'horizontal span of each stroke'),
    'seg_width':     ('spatial',   'bounding-box width of each stroke'),
    'seg_height':    ('spatial',   'bounding-box height of each stroke'),
}

_STAT_DESC = {
    'mean':   'average',
    'median': 'middle value',
    'std':    'spread (std. deviation)',
    'max':    'maximum',
    'min':    'minimum',
    'p5':     '5th-percentile (low-end tail)',
    'p95':    '95th-percentile (high-end tail)',
}

_SCALAR_META = {
    'n_pen_lifts': ('fluency',
        'Number of pen lifts',
        'How many times the pen left the paper. Frequent lifts interrupt fluent writing.'),
    'n_velocity_extrema': ('kinematic',
        'Speed reversals',
        'How often the pen accelerates then decelerates. High counts = jerky motion.'),
    'n_acceleration_extrema': ('kinematic',
        'Acceleration reversals',
        'Micro-pauses and corrections detected in the motion profile — a classic dysgraphia marker.'),
    'total_writing_time': ('fluency',
        'Total writing time',
        'Total seconds spent writing. Long times = slow, effortful writing.'),
    'total_path_len': ('spatial',
        'Total path length',
        'Total distance the pen tip travelled while on paper.'),
    'total_vert_len': ('spatial',
        'Total vertical distance',
        'Sum of all vertical movement — captures ascender/descender energy.'),
    'total_horz_len': ('spatial',
        'Total horizontal distance',
        'Sum of all horizontal movement.'),
    'diff_first_last_y_min': ('spatial',
        'Baseline drift  (min)',
        'Smallest first-vs-last vertical drift across strokes.'),
    'diff_first_last_y_median': ('spatial',
        'Baseline drift  (median)',
        'Typical drift between the first and last point of each stroke. Big drift = wobbly baseline.'),
    'diff_first_last_y_mean': ('spatial',
        'Baseline drift  (mean)',
        'Average drift between the first and last point of each stroke.'),
    'diff_first_last_y_max': ('spatial',
        'Baseline drift  (max)',
        'Largest single stroke-level baseline drift.'),
    'diff_2nd_penult_y_min': ('spatial',
        'Mid-stroke drift  (min)',
        'Smallest 2nd-vs-penultimate vertical drift (excludes pen-down/lift artifacts).'),
    'diff_2nd_penult_y_median': ('spatial',
        'Mid-stroke drift  (median)',
        'Typical 2nd-vs-penultimate vertical drift — robust baseline-stability measure.'),
    'diff_2nd_penult_y_mean': ('spatial',
        'Mid-stroke drift  (mean)',
        'Average 2nd-vs-penultimate vertical drift.'),
    'diff_2nd_penult_y_max': ('spatial',
        'Mid-stroke drift  (max)',
        'Largest 2nd-vs-penultimate vertical drift.'),
    'var_seg_y_min': ('spatial',
        'Stroke-low-point variability',
        'How much the lowest point of each stroke varies. Big variation = inconsistent descender depth.'),
    'var_seg_y_max': ('spatial',
        'Stroke-high-point variability',
        'How much the highest point of each stroke varies. Big variation = inconsistent ascender height.'),
    'var_seg_y_median': ('spatial',
        'Stroke-midpoint variability',
        'How much the median vertical position of each stroke varies.'),
    'var_seg_y_mean': ('spatial',
        'Stroke-centre variability',
        'How much the average vertical position of each stroke varies — captures baseline drift over the page.'),
}


# Match longest family prefix first so e.g. "jerk_x_std" finds family
# "jerk_x" before "jerk".
_FAMILY_DESC_ORDERED = sorted(_FAMILY_DESC.items(),
                              key=lambda kv: -len(kv[0]))


def feature_meta(key):
    """Return (category, short_label, description) for a Drotár feature key.

    Pure helper — no Qt deps, safe to unit test.
    """
    if key in _SCALAR_META:
        return _SCALAR_META[key]
    # Split fam_stat for vector/segment features
    for fam, (cat, fam_lbl) in _FAMILY_DESC_ORDERED:
        if key.startswith(fam + '_'):
            stat = key[len(fam) + 1:]
            stat_lbl = _STAT_DESC.get(stat, stat)
            short = f"{fam_lbl.capitalize()} — {stat_lbl}"
            desc  = f"{stat_lbl.capitalize()} of {fam_lbl} across the writing sample."
            return cat, short, desc
    return 'kinematic', key.replace('_', ' '), 'Unrecognised feature.'


def feature_mask_excluding(excluded_categories):
    """Return a boolean numpy mask of length N_FEATURES (= 119) where True
    means 'keep this feature'. `excluded_categories` is an iterable of
    category keys from _FEAT_CATEGORY (e.g. {'spatial', 'pressure'}).
    Empty set / None = all features kept."""
    excluded = set(excluded_categories or ())
    return np.array(
        [feature_meta(k)[0] not in excluded for k in DROTAR_FEATURE_KEYS],
        dtype=bool,
    )


class FeatureImportanceTab(QWidget):
    """Top-N Random-Forest feature importances, with a category-colour
    legend and a plain-English explanation table — designed so a reader
    who has never seen the codebase can understand what each feature
    measures and why it matters for dysgraphia screening."""

    DEFAULT_TOP_N = 15

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # Header row
        top = QHBoxLayout()
        top.addWidget(QLabel("Show top:"))
        self.selector = QComboBox()
        for n in (10, 15, 20, 30, 50):
            self.selector.addItem(str(n), n)
        self.selector.setCurrentText(str(self.DEFAULT_TOP_N))
        self.selector.setStyleSheet(
            "QComboBox{border:1px solid #c8d3e8;border-radius:5px;"
            "padding:4px 8px;background:white;font-size:11px;}")
        self.selector.currentIndexChanged.connect(self._redraw)
        top.addWidget(self.selector)
        top.addWidget(QLabel("features"))
        top.addStretch()
        # Category legend
        for cat_key, (color, name, _) in _FEAT_CATEGORY.items():
            chip = QLabel(f" {name} ")
            chip.setStyleSheet(
                f"background:{color};color:white;padding:2px 8px;"
                f"border-radius:8px;font-size:10px;font-weight:600;"
                f"margin-left:4px;")
            top.addWidget(chip)
        layout.addLayout(top)

        # Splitter: chart on top, explanation table on bottom
        splitter = QSplitter(Qt.Orientation.Vertical)

        chart_w = QWidget(); chart_v = QVBoxLayout(chart_w)
        chart_v.setContentsMargins(0, 0, 0, 0)
        self.fig    = Figure(figsize=(8, 4), facecolor='white')
        self.canvas = FigureCanvasQTAgg(self.fig)
        chart_v.addWidget(self.canvas)
        splitter.addWidget(chart_w)

        self.explainer = QTextBrowser()
        self.explainer.setOpenExternalLinks(False)
        self.explainer.setStyleSheet(
            "QTextBrowser{background:white;border:1px solid #e2e8f0;"
            "border-radius:6px;}")
        splitter.addWidget(self.explainer)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        self._models = None
        self._placeholder()

    def _placeholder(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5,
                'Train a model to see which handwriting features\n'
                'drove the Random-Forest decisions.',
                ha='center', va='center', transform=ax.transAxes,
                color='#94a3b8', fontsize=11, multialignment='center')
        ax.axis('off')
        self.canvas.draw()
        self.explainer.setHtml(
            "<div style='font-family:Segoe UI;padding:20px;color:#64748b;"
            "font-size:12px;line-height:1.55;'>"
            "<b style='color:#1e3a8a;'>What goes here.</b> Once you train, "
            "each row below explains one of the top features — what it "
            "actually measures about the handwriting, and which broader "
            "category it belongs to. The chart bars above are colour-coded "
            "by the same categories shown in the legend."
            "</div>")

    def update_plot(self, models_dict):
        self._models = models_dict
        self._redraw()

    def _redraw(self):
        if self._models is None:
            return
        imp = self._models.get('importances')
        if imp is None or len(imp) == 0:
            self._placeholder()
            return

        top_n = self.selector.currentData() or self.DEFAULT_TOP_N
        # Prefer the kept-keys tuple saved with the model — falls back to
        # the full 119-key list, then to anonymous fN labels.
        kept_keys = self._models.get('feature_keys_used')
        if kept_keys and len(kept_keys) == len(imp):
            keys = list(kept_keys)
        elif len(DROTAR_FEATURE_KEYS) == len(imp):
            keys = list(DROTAR_FEATURE_KEYS)
        else:
            keys = [f'f{i}' for i in range(len(imp))]
        order     = np.argsort(imp)[::-1][:top_n]
        top_imp   = imp[order]
        top_keys  = [keys[i] for i in order]
        top_meta  = [feature_meta(k) for k in top_keys]
        # (cat, short, desc) per feature
        top_short = [meta[1] for meta in top_meta]
        top_cats  = [meta[0] for meta in top_meta]
        top_color = [_FEAT_CATEGORY[c][0] for c in top_cats]

        # ── chart ────────────────────────────────────────────────────────
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.set_facecolor('#fafbff')
        y_pos = np.arange(len(top_imp))[::-1]
        bars  = ax.barh(y_pos, top_imp, color=top_color,
                        edgecolor='white', linewidth=0.6)
        for bar, v in zip(bars, top_imp):
            ax.text(bar.get_width() + max(top_imp) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f'{v:.3f}', va='center', ha='left',
                    fontsize=8, color='#1e293b')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_short, fontsize=8.5)
        ax.set_xlabel('Random-Forest Gini importance (higher = more useful for the model)',
                      fontsize=9, color='#64748b')
        ax.set_title(f'Top {len(top_imp)} handwriting features',
                     fontsize=11, fontweight='bold',
                     color='#1e3a8a', pad=10)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
        ax.grid(True, axis='x', alpha=0.2)
        # Give long labels enough room
        self.fig.subplots_adjust(left=0.32, right=0.96, top=0.92, bottom=0.14)
        self.canvas.draw()

        # ── explanation table ────────────────────────────────────────────
        head_style = ("background:#1e3a8a;color:white;font-weight:600;"
                      "padding:7px 9px;text-align:left;font-size:11px;")
        cell_style = ("padding:7px 9px;border-bottom:1px solid #e2e8f0;"
                      "font-size:11px;color:#1e293b;vertical-align:top;")
        rank_style = cell_style + "font-family:Consolas;color:#64748b;text-align:center;width:36px;"

        rows = [
            "<div style='font-family:Segoe UI;'>",
            "<div style='font-size:12px;color:#1e3a8a;font-weight:700;"
            "margin-bottom:6px;'>What each feature actually measures</div>",
            "<table style='width:100%;border-collapse:collapse;background:white;"
            "border:1px solid #e2e8f0;border-radius:6px;overflow:hidden;'>",
            "<tr>",
            f"<th style='{head_style}width:36px;text-align:center;'>#</th>",
            f"<th style='{head_style}'>Feature</th>",
            f"<th style='{head_style}width:110px;'>Category</th>",
            f"<th style='{head_style}'>What it measures</th>",
            f"<th style='{head_style}width:80px;text-align:right;'>Importance</th>",
            "</tr>",
        ]
        for rank, (key, short, (cat, _s, desc), val) in enumerate(
                zip(top_keys, top_short, top_meta, top_imp), start=1):
            color, cat_name, _tag = _FEAT_CATEGORY[cat]
            chip = (f"<span style='background:{color};color:white;"
                    f"padding:2px 8px;border-radius:8px;font-size:10px;"
                    f"font-weight:600;'>{cat_name}</span>")
            rows.append("<tr>")
            rows.append(f"<td style='{rank_style}'>{rank}</td>")
            rows.append(f"<td style='{cell_style}'><b>{short}</b><br>"
                        f"<span style='color:#94a3b8;font-size:9.5px;"
                        f"font-family:Consolas;'>{key}</span></td>")
            rows.append(f"<td style='{cell_style}'>{chip}</td>")
            rows.append(f"<td style='{cell_style}color:#475569;'>{desc}</td>")
            rows.append(f"<td style='{cell_style}text-align:right;"
                        f"font-family:Consolas;'>{val:.3f}</td>")
            rows.append("</tr>")
        rows.append("</table>")

        # Category legend / glossary
        rows.append(
            "<div style='margin-top:14px;font-size:10.5px;color:#475569;"
            "line-height:1.55;'>"
            "<b style='color:#1e3a8a;'>Categories.</b><ul style='margin:6px 0 0 18px;padding:0;'>")
        for _, (color, name, tag) in _FEAT_CATEGORY.items():
            rows.append(
                f"<li><span style='background:{color};color:white;"
                f"padding:1px 7px;border-radius:7px;font-size:9.5px;"
                f"font-weight:600;'>{name}</span> &nbsp; {tag}.</li>")
        rows.append("</ul></div>")

        rows.append(
            "<div style='margin-top:10px;font-size:10.5px;color:#64748b;"
            "line-height:1.5;'>"
            "<b>Reading the importance score.</b> Random-Forest Gini "
            "importance measures how much each feature reduces "
            "class-impurity across all decision splits in the forest. "
            "Importances sum to 1.0 across all 119 features, so the "
            "individual numbers look small — what matters is the "
            "<i>relative ranking</i>."
            "</div></div>")

        self.explainer.setHtml("".join(rows))


# =============================================================================
# ML Trainer Window
# =============================================================================

class MLTrainerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dysgraphia Detective  —  ML Trainer")
        self.resize(1300, 760)
        self._analysis = None
        self._models   = _try_load_model()
        self._worker   = None
        self._build_ui()
        if self._models:
            self._on_model_loaded(self._models, announce=False)

    def _build_ui(self):
        root   = QWidget(); self.setCentralWidget(root)
        root_v = QVBoxLayout(root)
        root_v.setContentsMargins(12, 10, 12, 10)
        root_v.setSpacing(10)

        # Title bar
        top = QHBoxLayout()
        title = QLabel("ML Trainer  —  Draw, Save & Train")
        title.setFont(QFont("Segoe UI", 15, QFont.Weight.Bold))
        title.setStyleSheet("color:#1e3a8a;")
        top.addWidget(title); top.addStretch()
        self.model_badge = QLabel("No model loaded")
        self.model_badge.setStyleSheet(
            "padding:4px 12px; background:#f1f5f9; color:#64748b;"
            "border-radius:12px; font-size:10px; font-weight:600;")
        top.addWidget(self.model_badge)
        root_v.addLayout(top)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── Left: canvas + save buttons ──────────────────────────────────────
        left_card = QFrame(); left_card.setObjectName("card")
        left_v    = QVBoxLayout(left_card)
        left_v.setContentsMargins(14, 14, 14, 14)
        left_v.setSpacing(8)

        lbl = QLabel("Handwriting Canvas")
        lbl.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        lbl.setStyleSheet("color:#1e3a8a;")
        left_v.addWidget(lbl)

        self.canvas = DrawingCanvas()
        left_v.addWidget(self.canvas)

        # Clear + Analyze
        r1 = QHBoxLayout(); r1.setSpacing(8)
        self.clear_btn   = QPushButton("Clear")
        self.analyze_btn = QPushButton("Analyze")
        self.clear_btn.setStyleSheet(_S_CLEAR); self.analyze_btn.setStyleSheet(_S_ANALYZE)
        for b in (self.clear_btn, self.analyze_btn):
            b.setMinimumHeight(36); b.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.clear_btn.clicked.connect(self._on_clear)
        self.analyze_btn.clicked.connect(self._on_analyze)
        r1.addWidget(self.clear_btn, 1); r1.addWidget(self.analyze_btn, 2)
        left_v.addLayout(r1)

        # Save as Fluent / Dysgraphic
        r2 = QHBoxLayout(); r2.setSpacing(8)
        self.fluent_btn = QPushButton("Save as Fluent")
        self.dysg_btn   = QPushButton("Save as Dysgraphic")
        self.fluent_btn.setStyleSheet(_S_FLUENT); self.dysg_btn.setStyleSheet(_S_DYSG)
        for b in (self.fluent_btn, self.dysg_btn):
            b.setMinimumHeight(36); b.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            b.setEnabled(False)
        self.fluent_btn.clicked.connect(lambda: self._on_save('fluent'))
        self.dysg_btn.clicked.connect(lambda: self._on_save('dysgraphic'))
        r2.addWidget(self.fluent_btn, 1); r2.addWidget(self.dysg_btn, 1)
        left_v.addLayout(r2)

        # Save as Test Sample
        r3 = QHBoxLayout()
        self.test_btn = QPushButton("Save as Test Sample")
        self.test_btn.setStyleSheet(_S_TEST)
        self.test_btn.setMinimumHeight(32)
        self.test_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.test_btn.setEnabled(False)
        self.test_btn.setToolTip(
            "Save to dataset/test/fluent/ or dataset/test/dysgraphic/ for labelled evaluation.")
        self.test_btn.clicked.connect(self._on_save_test)
        r3.addWidget(self.test_btn)
        left_v.addLayout(r3)

        # Verdict box
        self.verdict_lbl = QLabel("Draw on the canvas above and click  Analyze")
        self.verdict_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict_lbl.setWordWrap(True)
        self.verdict_lbl.setFont(QFont("Segoe UI", 11))
        self.verdict_lbl.setStyleSheet(
            "padding:12px; border:1px solid #dde3ee; border-radius:8px;"
            "background:#f8fafc; color:#64748b;")
        left_v.addWidget(self.verdict_lbl)
        left_v.addStretch()

        # ── Right: training controls + 3 ML tabs ─────────────────────────────
        right_w = QWidget()
        right_v = QVBoxLayout(right_w)
        right_v.setContentsMargins(6, 0, 0, 0)
        right_v.setSpacing(8)

        # Feature-category exclude row — lets the user retrain on a subset
        # of the 119 Drotár features. Default: Spatial excluded (so the
        # model decision can't depend on letter-sizing / baseline drift).
        excl_row = QHBoxLayout(); excl_row.setSpacing(6)
        excl_lbl = QLabel("Exclude features:")
        excl_lbl.setStyleSheet("font-size:10px;color:#64748b;font-weight:600;")
        excl_row.addWidget(excl_lbl)
        self.exclude_boxes = {}
        for cat_key, (color, name, _tag) in _FEAT_CATEGORY.items():
            cb = QCheckBox(name)
            cb.setStyleSheet(
                f"QCheckBox{{font-size:10px;color:#1e293b;padding:0 6px;}}"
                f"QCheckBox::indicator{{width:13px;height:13px;}}"
                f"QCheckBox::indicator:checked{{background:{color};"
                f"border:1px solid {color};border-radius:3px;}}"
                f"QCheckBox::indicator:unchecked{{background:white;"
                f"border:1px solid #cbd5e1;border-radius:3px;}}")
            if cat_key == 'spatial':
                cb.setChecked(True)
            cb.stateChanged.connect(self._refresh_feature_subset_label)
            self.exclude_boxes[cat_key] = cb
            excl_row.addWidget(cb)
        excl_row.addStretch()
        self.feature_subset_lbl = QLabel("")
        self.feature_subset_lbl.setStyleSheet(
            "font-size:10px;color:#64748b;font-style:italic;")
        excl_row.addWidget(self.feature_subset_lbl)
        right_v.addLayout(excl_row)
        self._refresh_feature_subset_label()

        # Training button: real on-surface stroke data only
        btn_row_modes = QHBoxLayout(); btn_row_modes.setSpacing(8)
        self.online_btn = QPushButton("Train on Real Data")
        self.online_btn.setStyleSheet(_S_RETRAIN)
        self.online_btn.setMinimumHeight(34)
        self.online_btn.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.online_btn.clicked.connect(self._on_retrain)
        btn_row_modes.addWidget(self.online_btn)
        right_v.addLayout(btn_row_modes)

        # Dataset / test buttons row
        btn_row2 = QHBoxLayout(); btn_row2.setSpacing(8)
        self.check_btn    = QPushButton("Check Dataset")
        self.test_run_btn = QPushButton("Run Test Samples")
        self.check_btn.setStyleSheet(_S_CHECK); self.test_run_btn.setStyleSheet(_S_TEST)
        for b in (self.check_btn, self.test_run_btn):
            b.setMinimumHeight(30); b.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
        self.check_btn.clicked.connect(self._on_check)
        self.test_run_btn.clicked.connect(self._on_test_run)
        btn_row2.addWidget(self.check_btn); btn_row2.addWidget(self.test_run_btn)
        right_v.addLayout(btn_row2)

        # Counter label
        self.counter_lbl = QLabel("")
        self.counter_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.counter_lbl.setStyleSheet("color:#64748b; font-size:10px;")
        right_v.addWidget(self.counter_lbl)
        self._refresh_counter()

        if not _SKLEARN_OK:
            self.online_btn.setEnabled(False)
            self.online_btn.setText("scikit-learn not installed")

        # ML result tabs
        tabs = QTabWidget()
        self.acc_tab     = AccuracyTab()
        self.metrics_tab = MetricsTab()
        self.cm_tab      = ConfusionMatrixTab()
        self.fi_tab      = FeatureImportanceTab()
        self.drotar_tab  = DrotarTestTab()
        tabs.addTab(self.acc_tab,     "Accuracy")
        tabs.addTab(self.metrics_tab, "Metrics")
        tabs.addTab(self.cm_tab,      "Confusion Matrix")
        tabs.addTab(self.fi_tab,      "Feature Importance")
        tabs.addTab(self.drotar_tab,  "Drotár Dataset")
        right_v.addWidget(tabs, 1)

        splitter.addWidget(left_card); splitter.addWidget(right_w)
        splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1)
        splitter.setSizes([640, 640])
        root_v.addWidget(splitter, 1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _refresh_counter(self):
        # JSON counts
        nfj = len(list((DATASET_DIR/'fluent'/'json').glob('*.json')))     if (DATASET_DIR/'fluent'/'json').exists()     else 0
        ndj = len(list((DATASET_DIR/'dysgraphic'/'json').glob('*.json'))) if (DATASET_DIR/'dysgraphic'/'json').exists() else 0
        # Image counts (any extension)
        def _count_imgs(p):
            n = 0
            if p.exists():
                for ext in ('*.png','*.jpg','*.jpeg','*.bmp','*.tif','*.tiff'):
                    n += len(list(p.glob(ext)))
            return n
        nfi = _count_imgs(DATASET_DIR/'fluent'/'images')
        ndi = _count_imgs(DATASET_DIR/'dysgraphic'/'images')
        tf  = DATASET_DIR / 'test'
        tnf  = len(list((tf/'fluent'/'json').glob('*.json')))     if (tf/'fluent'/'json').exists()     else 0
        tnd  = len(list((tf/'dysgraphic'/'json').glob('*.json'))) if (tf/'dysgraphic'/'json').exists() else 0
        tnfi = _count_imgs(tf/'fluent'/'images')
        tndi = _count_imgs(tf/'dysgraphic'/'images')
        ok = (nfj >= 2 and ndj >= 2) or (nfi >= 2 and ndi >= 2)
        self.counter_lbl.setText(
            f"Fluent:  {nfj} JSON / {nfi} images   |   "
            f"Dysgraphic:  {ndj} JSON / {ndi} images"
            + ("   — ready" if ok else "   — need 2+ of each to train")
            + f"     |   Test:  {tnf}J+{tnfi}I  F  /  {tnd}J+{tndi}I  D")
        self.counter_lbl.setStyleSheet(
            f"color:{'#166534' if ok else '#92400e'}; font-size:10px; font-weight:600;")

    def _excluded_categories(self):
        return tuple(sorted(
            cat for cat, cb in self.exclude_boxes.items() if cb.isChecked()))

    def _refresh_feature_subset_label(self):
        excl = self._excluded_categories()
        if not excl:
            self.feature_subset_lbl.setText(
                f"all {N_FEATURES} features")
            return
        mask    = feature_mask_excluding(excl)
        n_keep  = int(np.sum(mask))
        n_drop  = N_FEATURES - n_keep
        names   = ", ".join(_FEAT_CATEGORY[c][1] for c in excl)
        self.feature_subset_lbl.setText(
            f"using {n_keep} / {N_FEATURES} features  "
            f"(excluding {n_drop}: {names})")

    def _update_model_badge(self):
        m = self._models
        if m is None:
            self.model_badge.setText("No model loaded")
            self.model_badge.setStyleSheet(
                "padding:4px 12px; background:#f1f5f9; color:#64748b;"
                "border-radius:12px; font-size:10px; font-weight:600;")
        else:
            nf = m.get('n_fluent', '?'); nd = m.get('n_dysg', '?')
            excl = m.get('excluded_categories') or ()
            tail = ""
            if excl:
                tail = "  ·  no " + "/".join(
                    _FEAT_CATEGORY[c][1].lower() for c in excl)
            txt = f"Model: real  ({nf}F + {nd}D){tail}"
            bg, fg = '#f0fdf4', '#166534'
            self.model_badge.setText(txt)
            self.model_badge.setStyleSheet(
                f"padding:4px 12px; background:{bg}; color:{fg};"
                f"border-radius:12px; font-size:10px; font-weight:600;")

    def _on_model_loaded(self, m, announce=True):
        self._models = m
        self.acc_tab.update_plot(m)
        self.metrics_tab.update_plot(m)
        self.cm_tab.update_plot(m)
        self.fi_tab.update_plot(m)
        self.drotar_tab.set_models(m)
        self._update_model_badge()
        if self._analysis:
            vec = _extract_feature_vector(
                self._analysis.get('kinematic_features'),
                self._analysis.get('pressure_features'),
                self._analysis.get('spatial_features'),
                drotar=self._analysis.get('drotar_features'))
            if vec is not None:
                self._show_ml_verdict(vec)

    # ── Button handlers ───────────────────────────────────────────────────────

    def _on_clear(self):
        self.canvas.clear()
        self.fluent_btn.setEnabled(False); self.dysg_btn.setEnabled(False)
        self.test_btn.setEnabled(False)
        self.verdict_lbl.setText("Draw on the canvas above and click  Analyze")
        self.verdict_lbl.setStyleSheet(
            "padding:12px; border:1px solid #dde3ee; border-radius:8px;"
            "background:#f8fafc; color:#64748b;")
        self._analysis = None

    def _on_analyze(self):
        if not self.canvas.strokes:
            QMessageBox.warning(self, "Nothing drawn", "Please write on the canvas first.")
            return
        if sum(len(s) for s in self.canvas.strokes) < 20:
            QMessageBox.warning(self, "Too short", "Please write more before analyzing.")
            return

        kin     = compute_kinematic_features(self.canvas.strokes)
        pres    = compute_pressure_features(self.canvas.strokes)
        spatial = compute_spatial_features(self.canvas.strokes, DrawingCanvas.LINE_SPACING)
        if kin is None:
            QMessageBox.warning(self, "Error", "Could not compute features."); return

        t_spd, speed = compute_speed_profile(self.canvas.strokes)
        if t_spd is not None:
            trend_spd = fit_trend(t_spd, speed)
            rv = abs(trend_spd['slope']) / max(kin['v_mean'], 1)
            cv = kin['v_std'] / max(kin['v_mean'], 1)
            # Trend claim requires a real fit (R^2 >= 0.25) AND a meaningful
            # slope magnitude (|slope|/v_mean >= 0.10).
            if trend_spd['r2'] < 0.25 or rv < 0.10:
                type_spd = 'neutral'
            elif trend_spd['slope'] < 0:
                type_spd = 'bad'
            else:
                type_spd = 'good'
            # High variability overrides everything
            if cv > 0.8:
                type_spd = 'bad'
        else:
            type_spd = 'neutral'

        type_prs = ('bad' if pres and pres['p_std'] > 0.15 else
                    'neutral' if pres and pres['p_std'] >= 0.05 else 'good')

        # Rule-based verdict — each indicator counts separately
        indicators = [type_spd == 'bad', type_prs == 'bad', kin['in_air_ratio'] > 0.4]
        if spatial:
            _, _, s_itype = slant_type(spatial.get('theta_mean', 0.0), spatial.get('theta_std', 0.0))
            indicators.append(spatial['delta'] > 12)
            indicators.append(spatial['h_std'] > 25)
            indicators.append(s_itype == 'bad')
        n_bad = sum(indicators)

        if n_bad == 0:
            rule_txt, rfg, rbg = "Rule-based: Fluent writing pattern", "#166534", "#f0fdf4"
        elif n_bad == 1:
            rule_txt, rfg, rbg = "Rule-based: Borderline (1 indicator)", "#92400e", "#fffbeb"
        else:
            rule_txt, rfg, rbg = "Rule-based: Dysgraphic indicators detected", "#991b1b", "#fef2f2"

        drotar_feats = compute_drotar_features(self.canvas.strokes)

        self._analysis = {
            'timestamp': datetime.now().isoformat(),
            'n_strokes': len(self.canvas.strokes),
            'n_points':  sum(len(s) for s in self.canvas.strokes),
            'kinematic_features': kin,
            'pressure_features':  pres,
            'spatial_features':   spatial,
            'drotar_features':    drotar_feats,
        }

        ml_txt = ""
        vec = _kin_only_vec(kin, pres, spatial, drotar=drotar_feats)
        if vec is not None:
            vec = np.array(vec, dtype=float)
        if vec is not None and self._models:
            pred, prob = _ensemble_predict(self._models, vec)
            plabel = "Fluent" if pred == 0 else "Dysgraphic"
            ml_txt = f"     |     ML: {plabel} ({prob*100:.0f}%)"

        self.verdict_lbl.setText(rule_txt + ml_txt)
        self.verdict_lbl.setStyleSheet(
            f"padding:12px; border:1px solid {rfg}; border-radius:8px;"
            f"background:{rbg}; color:{rfg}; font-weight:bold;")

        self.fluent_btn.setEnabled(True); self.dysg_btn.setEnabled(True)
        self.test_btn.setEnabled(True)

    def _show_ml_verdict(self, vec):
        if self._models is None:
            return
        pred, prob = _ensemble_predict(self._models, vec)
        plabel = "Fluent" if pred == 0 else "Dysgraphic"
        fg = "#166534" if pred == 0 else "#991b1b"
        bg = "#f0fdf4"  if pred == 0 else "#fef2f2"
        self.verdict_lbl.setText(f"ML Ensemble: {plabel}  ({prob*100:.1f}%)")
        self.verdict_lbl.setStyleSheet(
            f"padding:12px; border:1px solid {fg}; border-radius:8px;"
            f"background:{bg}; color:{fg}; font-weight:bold;")

    def _on_save(self, label: str):
        if self._analysis is None:
            QMessageBox.warning(self, "Nothing to save", "Run Analyze first."); return
        name     = _next_sample_name(label)
        img_dir  = DATASET_DIR / label / 'images'
        json_dir = DATASET_DIR / label / 'json'
        img_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        self.canvas.capture_image(img_dir / f"{name}.png")
        payload = dict(self._analysis)
        payload['label']  = label
        payload['sample'] = name
        payload['strokes'] = [[[float(v) for v in pt] for pt in s] for s in self.canvas.strokes]
        with open(json_dir / f"{name}.json", 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        self._refresh_counter()
        QMessageBox.information(
            self, "Saved",
            f"Saved as:  {name}\n"
            f"Image:  {img_dir / (name + '.png')}\n"
            f"JSON:   {json_dir / (name + '.json')}")

    def _on_save_test(self):
        if self._analysis is None:
            QMessageBox.warning(self, "Nothing to save", "Run Analyze first."); return
        dlg = _TestLabelDialog(self)
        dlg.exec()
        if dlg.chosen is None:
            return
        sublabel = dlg.chosen
        name     = _next_test_sample_name(sublabel)
        img_dir  = DATASET_DIR / 'test' / sublabel / 'images'
        json_dir = DATASET_DIR / 'test' / sublabel / 'json'
        img_dir.mkdir(parents=True, exist_ok=True)
        json_dir.mkdir(parents=True, exist_ok=True)
        self.canvas.capture_image(img_dir / f"{name}.png")
        payload = dict(self._analysis)
        payload['label']     = 'test'
        payload['sublabel']  = sublabel
        payload['sample']    = name
        payload['strokes']   = [[[float(v) for v in pt] for pt in s] for s in self.canvas.strokes]
        with open(json_dir / f"{name}.json", 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2)
        self._refresh_counter()
        QMessageBox.information(
            self, "Saved",
            f"Test sample saved as:  {name}\n"
            f"Category:  {sublabel}\n"
            f"JSON:  {json_dir / (name + '.json')}")

    def _on_retrain(self):
        self._refresh_counter()
        self._set_busy(True)
        self._worker = RetrainWorker(
            excluded_categories=self._excluded_categories())
        self._worker.finished.connect(self._on_retrain_done)
        self._worker.error.connect(self._on_retrain_error)
        self._worker.start()

    def _on_retrain_done(self, m):
        self._on_model_loaded(m)
        self._refresh_counter()
        self._set_busy(False)
        nf = m.get('n_fluent', '?'); nd = m.get('n_dysg', '?')
        excl = m.get('excluded_categories') or ()
        mask = m.get('feature_mask')
        n_kept = int(np.sum(mask)) if mask is not None else N_FEATURES
        excl_line = (
            f"Excluded categories: {', '.join(_FEAT_CATEGORY[c][1] for c in excl)}\n"
            if excl else "")
        QMessageBox.information(self, "Training complete",
            f"Trained on {nf} fluent + {nd} dysgraphic sessions.\n"
            f"{excl_line}"
            f"Using {n_kept} / {N_FEATURES} features.\n\n"
            f"Saved to:\n  {MODEL_ONLINE_PATH}\n  {MODEL_PATH}  (current active)")

    def _on_retrain_error(self, msg):
        QMessageBox.warning(self, "Training failed", msg)
        self._set_busy(False)

    def _on_check(self):
        stats, warns = scan_dataset(DATASET_DIR)
        f     = stats.get('fluent', {})
        d     = stats.get('dysgraphic', {})
        nf_pair  = f.get('matched', 0)
        nd_pair  = d.get('matched', 0)
        nf_img   = f.get('img_total', 0)
        nd_img   = d.get('img_total', 0)
        nf_json  = f.get('json_total', 0)
        nd_json  = d.get('json_total', 0)
        nf_imgonly = len(f.get('img_only', []))
        nd_imgonly = len(d.get('img_only', []))
        t     = stats.get('test', {})
        tnf   = t.get('fluent', 0)
        tnd   = t.get('dysgraphic', 0)
        tnfi  = t.get('fluent_img', 0)
        tndi  = t.get('dysgraphic_img', 0)

        msg = (
            f"Dataset at:  {DATASET_DIR}\n\n"
            f"FLUENT:\n"
            f"  JSON files:        {nf_json}\n"
            f"  Image files:       {nf_img}    ({nf_imgonly} image-only)\n"
            f"  Matched pairs:     {nf_pair}\n\n"
            f"DYSGRAPHIC:\n"
            f"  JSON files:        {nd_json}\n"
            f"  Image files:       {nd_img}    ({nd_imgonly} image-only)\n"
            f"  Matched pairs:     {nd_pair}\n\n"
            f"USABLE TRAINING SETS:\n"
            f"  Online  (JSON only):           "
            f"{nf_json} fluent + {nd_json} dysgraphic\n"
            f"  Offline (images, any source):  "
            f"{nf_img} fluent + {nd_img} dysgraphic\n"
            f"  Both    (matched pairs):       "
            f"{nf_pair} fluent + {nd_pair} dysgraphic\n\n"
            f"TEST FOLDER:\n"
            f"  Fluent:      {tnf} JSON  /  {tnfi} images\n"
            f"  Dysgraphic:  {tnd} JSON  /  {tndi} images\n"
        )
        if warns:
            msg += "\nWarnings:\n" + "\n".join(f"  • {w}" for w in warns)
        QMessageBox.information(self, "Dataset Report", msg)
        self._refresh_counter()

    def _on_test_run(self):
        if self._models is None:
            QMessageBox.warning(self, "No Model",
                "No model loaded.\nClick 'Train on Real Data' first.")
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            self._do_test_run()
        finally:
            QApplication.restoreOverrideCursor()

    def _do_test_run(self):
        tf      = DATASET_DIR / 'test'
        results, y_true, y_pred = [], [], []

        def _run_one(name, lbl_int, sublabel, kin, pres, spatial,
                     drotar=None, strokes=None):
            try:
                if not kin and not pres and drotar is None and not strokes:
                    return
                vec = _extract_feature_vector(kin, pres, spatial,
                                              drotar=drotar, strokes=strokes)
                if vec is None:
                    return
                # Hard guard against feature-width mismatch (avoids C-level
                # crash inside scikit-learn for a single corrupt sample).
                # _ensemble_predict applies the model's feature_mask
                # internally, so we just need the full 119-vector here.
                if vec.shape[0] != N_FEATURES:
                    return
                pred, prob = _ensemble_predict(self._models, vec)
                results.append({
                    'name':       name,
                    'true_label': sublabel,
                    'prediction': 'DYSGRAPHIC' if pred == 1 else 'FLUENT',
                    'confidence': prob,
                    'correct':    (pred == lbl_int),
                })
                y_true.append(lbl_int); y_pred.append(pred)
            except Exception:
                return

        for lbl_int, sublabel in ((0, 'fluent'), (1, 'dysgraphic')):
            jdir = tf / sublabel / 'json'
            if not jdir.exists():
                continue
            for jf in sorted(jdir.glob('*.json')):
                try:
                    data    = json.loads(jf.read_text(encoding='utf-8'))
                    kin     = data.get('kinematic_features')
                    pres    = data.get('pressure_features')
                    spatial = data.get('spatial_features')
                    drotar  = data.get('drotar_features')
                    strokes = data.get('strokes')
                    _run_one(jf.stem, lbl_int, sublabel,
                             kin, pres, spatial,
                             drotar=drotar, strokes=strokes)
                except Exception:
                    continue

        if not results:
            QMessageBox.information(self, "No Test Samples",
                "No test samples readable.\n\n"
                "Need JSON files in dataset/test/<label>/json/.")
            return

        acc = accuracy_score(y_true, y_pred) * 100 if y_true else 0.0
        try:
            dlg = TestResultsDialog(results, 'online', acc, parent=self)
            dlg.exec()
        except Exception as e:
            QMessageBox.information(
                self, "Test Results",
                f"Tested {len(results)} samples — Accuracy: {acc:.1f}%\n"
                f"(GUI table failed to render: {e})")

    def _set_busy(self, busy):
        self.online_btn.setEnabled(not busy)
        self.online_btn.setText("Training..." if busy else "Train on Real Data")
        QApplication.processEvents()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(APP_STYLE)
    win = MLTrainerWindow()
    win.show()
    sys.exit(app.exec())

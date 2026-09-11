"""
dysgraphia_detective.py  —  v3.0
The Dysgraphia Detective | Bachelor Thesis SS26
Supervisor: Dr. Yomna Hassan

Main analysis application — Analyze only.
For training and saving samples use:  dysgraphia_ml.py
"""

import sys
import time
import json
import pickle
import subprocess
import numpy as np
from pathlib import Path
from scipy import stats
from collections import defaultdict

try:
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSplitter, QMessageBox, QTabWidget,
    QTextBrowser, QFrame, QScrollArea,
)
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QPainter, QPen, QColor, QImage, QFont

import matplotlib
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure


# =============================================================================
# Constants
# =============================================================================

def _find_dataset_dir():
    seen = set()
    candidates = []
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


DATASET_DIR = _find_dataset_dir() or (Path.cwd() / 'dataset').resolve()
MODEL_PATH        = DATASET_DIR / 'model.pkl'
MODEL_ONLINE_PATH = DATASET_DIR / 'model_online.pkl'

N_FEATURES = 119  # Drotár-style on-surface feature set (no altitude/azimuth)

# =============================================================================
# Stylesheet
# =============================================================================

APP_STYLE = """
QMainWindow, QWidget { background: #f0f4f8; font-family: 'Segoe UI', Arial, sans-serif; color: #1e293b; }
QFrame#card { background: white; border: 1px solid #dde3ee; border-radius: 12px; }
QTabWidget::pane { border: 1px solid #dde3ee; background: white; border-radius: 0 10px 10px 10px; }
QTabBar::tab {
    background: #dde3ee; color: #374151; padding: 8px 18px;
    border: 1px solid #c8d3e8; border-bottom: none;
    border-top-left-radius: 7px; border-top-right-radius: 7px;
    font-weight: 700; font-size: 10px; margin-right: 2px;
}
QTabBar::tab:selected { background: white; color: #1d4ed8; border-bottom: 2px solid #2563eb; }
QTabBar::tab:hover:!selected { background: #c8d3e8; color: #1e3a8a; }
QTextBrowser { border: none; background: white; color: #1e293b; }
QPushButton { border-radius: 7px; font-weight: 700; font-size: 11px; padding: 8px 14px; }
QSplitter::handle { background: #dde3ee; width: 1px; }
QScrollArea { border: none; background: transparent; }
QMessageBox { background: white; color: #1e293b; }
QMessageBox QLabel { color: #1e293b; background: transparent; font-size: 13px; min-width: 280px; }
QMessageBox QPushButton {
    background: #2563eb; color: white; padding: 6px 20px;
    border-radius: 6px; min-width: 80px; font-size: 12px; font-weight: 700;
}
QMessageBox QPushButton:hover { background: #1d4ed8; }
"""

_S_CLEAR   = "QPushButton{background:#eef1f7;color:#374151;border:1px solid #cbd5e1;border-radius:7px;}QPushButton:hover{background:#dde3ee;}"
_S_ANALYZE = "QPushButton{background:#2563eb;color:white;border:none;border-radius:7px;}QPushButton:hover{background:#1d4ed8;}QPushButton:disabled{background:#bfdbfe;color:#1d4ed8;border:1px solid #93c5fd;}"
_S_ML_BTN  = "QPushButton{background:#7c3aed;color:white;border:none;border-radius:7px;}QPushButton:hover{background:#6d28d9;}"
_S_RELOAD  = "QPushButton{background:#0891b2;color:white;border:none;border-radius:7px;}QPushButton:hover{background:#0e7490;}"

COLORS = {
    'good':    ('#166534', '#dcfce7'),
    'bad':     ('#991b1b', '#fef2f2'),
    'neutral': ('#374151', '#f9fafb'),
}


# =============================================================================
# Drawing Canvas
# =============================================================================

class DrawingCanvas(QWidget):
    LINE_SPACING = 70

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(680, 380)
        self._img     = QImage(680, 380, QImage.Format.Format_RGB32)
        self._img.fill(QColor(255, 255, 255))
        self.strokes  = []
        self._stroke  = []
        self._drawing = False
        self.setAttribute(Qt.WidgetAttribute.WA_TabletTracking)
        self._draw_guidelines()

    def _draw_guidelines(self):
        p = QPainter(self._img)
        p.setPen(QPen(QColor(196, 216, 240), 1))
        for y in range(self.LINE_SPACING, 380, self.LINE_SPACING):
            p.drawLine(20, y, 660, y)
        p.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.MouseButton.LeftButton:
            self._drawing = True; self._stroke = []
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


def compute_pressure_profile(strokes):
    all_t, all_p = [], []
    t_offset = None
    for stroke in strokes:
        if not stroke:
            continue
        arr = np.array(stroke)
        ts, pressure = arr[:, 2], arr[:, 3]
        if t_offset is None:
            t_offset = ts[0]
        all_t.extend((ts - t_offset).tolist())
        all_p.extend(pressure.tolist())
    if not all_t:
        return None, None
    return np.array(all_t), np.array(all_p)


def compute_kinematic_features(strokes):
    if not strokes:
        return None
    all_v        = []
    t_on_surface = 0.0
    all_pts = [pt for stroke in strokes for pt in stroke]
    if len(all_pts) < 2:
        return None
    t_first = all_pts[0][2]; t_last = all_pts[-1][2]
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
    a_arr = np.diff(v_arr); j_arr = np.diff(a_arr)
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
    p = np.array(all_p)
    p_active = p[p > 0.05]
    if len(p_active) == 0:
        p_active = p
    return {
        'p_mean':  float(np.mean(p_active)),
        'p_std':   float(np.std(p_active)),
        'p_max':   float(np.max(p_active)),
        'p_range': float(np.max(p_active) - np.min(p_active)),
    }


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
        mean_raw_slope = 0.0; baseline_slope_deg = 0.0

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


def interpret_speed(slope, r2, v_mean, j_mean=0, v_std=0):
    """Interpret the linear trend of the speed profile.

    A claim of 'motor fatigue' / 'warm-up' requires BOTH:
      * R^2 >= 0.25       (the linear model actually fits the data), AND
      * |slope|/v_mean    >= 0.10   (the slope is large enough to matter).

    Anything weaker is reported as 'no clear trend' — the high-variability or
    high-jerk signals below provide a more reliable channel for that case.
    """
    rv = abs(slope) / max(v_mean, 1)
    cv = v_std / max(v_mean, 1)

    if r2 < 0.25 or rv < 0.10:
        cat, txt = 'neutral', 'No clear linear speed trend.'
    elif slope < 0:
        cat, txt = 'bad',     'Speed decreases over time — motor fatigue pattern.'
    else:
        cat, txt = 'good',    'Speed increases over time — warm-up pattern.'

    # Variability override (more reliable than trend on a noisy profile)
    if cv > 0.8:
        cat = 'bad'
        txt = txt.rstrip('.') + '.  High speed variability — inconsistent pacing.'
    elif cv > 0.5 and cat == 'neutral':
        txt += '  Speed varies considerably (CV = {:.2f}).'.format(cv)

    # Jerk override — strongest motor-control marker
    if j_mean / max(v_mean, 1) > 1.5:
        cat = 'bad'
        txt += '  High jerk — irregular motor control.'

    # If neither trend nor variability nor jerk fired anything, the writing is
    # actually steady — say so.
    if cat == 'neutral' and cv <= 0.5 and r2 < 0.25:
        cat, txt = 'good', 'Stable, consistent pacing throughout the session.'

    return txt, cat


def interpret_pressure(p_std, p_max=0, p_range=0):
    if p_std < 0.05:
        cat, txt = 'good',    'Steady grip — pressure is consistent throughout.'
    elif p_std < 0.15:
        cat, txt = 'neutral', 'Moderate pressure variation.'
    else:
        cat, txt = 'bad',     'Irregular grip — pressure varies significantly.'
    if p_range > 0.5 and cat == 'good':
        cat, txt = 'neutral', txt + '  Wide pressure range detected.'
    return txt, cat


def slant_type(theta_mean, theta_std):
    if theta_std > 20:
        return ('Variable / Mixed',
                f'Slant changes direction stroke-to-stroke (σ={theta_std:.0f}°). Inconsistency is a dysgraphia marker. ⚠', 'bad')
    if theta_mean > 45:
        return ('Extreme Right', f'Letters lean far forward ({theta_mean:.0f}°). ⚠', 'bad')
    if theta_mean < -45:
        return ('Extreme Left', f'Letters lean far backward ({theta_mean:.0f}°). ⚠', 'bad')
    if theta_mean > 15:
        return ('Right / Forward', f'Letters lean right ({theta_mean:.0f}°). ✓', 'good')
    if theta_mean < -15:
        return ('Left / Backward', f'Letters lean left ({theta_mean:.0f}°). ✓', 'good')
    return ('Upright / Vertical', f'Letters stand straight ({theta_mean:.0f}°). ✓', 'good')


def _extract_kin_vector(kin, pres, spatial=None, drotar=None, strokes=None):
    # `kin`, `pres`, `spatial` are still computed for the Speed / Pressure /
    # Spatial tabs and the clinical report but are NOT used as model features.
    # The model trains on the 119-element Drotár-style vector built from the
    # raw on-surface stroke samples (no altitude/azimuth — Deco 01 limitation).
    from dysgraphia_ml import compute_drotar_features, DROTAR_FEATURE_KEYS
    if drotar is None and strokes is not None:
        drotar = compute_drotar_features(strokes)
    if drotar is None:
        return None
    return [float(drotar.get(k, 0.0)) for k in DROTAR_FEATURE_KEYS]


def _extract_feature_vector(kin, pres, spatial, drotar=None, strokes=None):
    """Build the 119-feature Drotár vector. Returns None if neither a cached
    `drotar` dict nor raw `strokes` are available."""
    kin_vec = _extract_kin_vector(kin, pres, spatial, drotar=drotar, strokes=strokes)
    return np.array(kin_vec, dtype=float) if kin_vec is not None else None


def _try_load_model():
    """Load the on-surface kinematic model. Tries the mode-specific copy
    first, then the generic model.pkl."""
    for path in (MODEL_ONLINE_PATH, MODEL_PATH):
        try:
            if not path.exists():
                continue
            with open(path, 'rb') as f:
                m = pickle.load(f)
            if m['scaler'].n_features_in_ != N_FEATURES:
                continue
            m.setdefault('mode', 'online')
            return m
        except Exception:
            continue
    return None


def _ensemble_predict(models, vec):
    v   = vec.reshape(1, -1)
    v_s = models['scaler'].transform(v)
    votes = [models['rf'].predict(v)[0],
             models['svm'].predict(v_s)[0],
             models['lr'].predict(v_s)[0]]
    if 'gb' in models:
        votes.append(models['gb'].predict(v_s)[0])
    if 'ab' in models:
        votes.append(models['ab'].predict(v_s)[0])
    pred  = max(set(votes), key=votes.count)
    probs = [models['rf'].predict_proba(v)[0][pred],
             models['svm'].predict_proba(v_s)[0][pred],
             models['lr'].predict_proba(v_s)[0][pred]]
    if 'gb' in models:
        probs.append(models['gb'].predict_proba(v_s)[0][pred])
    if 'ab' in models:
        probs.append(models['ab'].predict_proba(v_s)[0][pred])
    return int(pred), float(np.mean(probs))


# =============================================================================
# Profile Panel (Speed / Pressure)
# =============================================================================

class ProfilePanel(QWidget):
    def __init__(self, title, ylabel, parent=None):
        super().__init__(parent)
        self.title = title; self.ylabel = ylabel
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.fig = Figure(figsize=(6, 3.2), dpi=100)
        self.fig.patch.set_facecolor('white')
        self.ax  = self.fig.add_subplot(111)
        self.mpl_canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.mpl_canvas)
        self.interp_label = QLabel("Draw something, then click  Analyze")
        self.interp_label.setWordWrap(True)
        self.interp_label.setFont(QFont("Segoe UI", 10))
        self.interp_label.setStyleSheet(
            "padding:10px 12px; background:#f0f4ff; border-radius:6px;"
            "border-left:4px solid #2563eb; color:#1e3a8a;")
        layout.addWidget(self.interp_label)
        self.reset()

    def plot(self, t, values, trend, interp_text, interp_type):
        self.ax.clear()
        self.ax.set_facecolor('#fafbff')
        self.ax.plot(t, values, color='#3b82f6', linewidth=1.4, alpha=0.85, label='Measured')
        self.ax.plot(t, trend['line'], color='#ef4444', linewidth=2,
                     linestyle='--', label=f"Trend  R²={trend['r2']:.2f}")
        self.ax.set_title(self.title, fontsize=11, fontweight='bold', color='#1e3a8a', pad=8)
        self.ax.set_xlabel("Time (s)", fontsize=9, color='#64748b')
        self.ax.set_ylabel(self.ylabel, fontsize=9, color='#64748b')
        self.ax.legend(fontsize=8, framealpha=0.7)
        self.ax.grid(True, alpha=0.2, color='#94a3b8')
        self.ax.spines[['top', 'right']].set_visible(False)
        self.fig.tight_layout(pad=1.0)
        self.mpl_canvas.draw()
        fg, bg = COLORS.get(interp_type, COLORS['neutral'])
        self.interp_label.setText(f"Analysis:  {interp_text}")
        self.interp_label.setStyleSheet(
            f"padding:10px 12px; background:{bg}; border-radius:6px;"
            f"border-left:4px solid {fg}; color:{fg};")

    def reset(self):
        self.ax.clear()
        self.ax.set_facecolor('#fafbff')
        self.ax.text(0.5, 0.5, 'Draw something, then click  Analyze',
                     ha='center', va='center', transform=self.ax.transAxes,
                     color='#94a3b8', fontsize=10)
        self.ax.axis('off')
        self.mpl_canvas.draw()
        self.interp_label.setText("Draw something, then click  Analyze")
        self.interp_label.setStyleSheet(
            "padding:10px 12px; background:#f0f4ff; border-radius:6px;"
            "border-left:4px solid #2563eb; color:#1e3a8a;")


# =============================================================================
# Spatial Panel
# =============================================================================

class SpatialPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        self.fig = Figure(figsize=(6, 4.8), dpi=100)
        self.fig.patch.set_facecolor('white')
        self.mpl_canvas = FigureCanvasQTAgg(self.fig)
        layout.addWidget(self.mpl_canvas)
        self.info_label = QLabel("Draw something, then click  Analyze")
        self.info_label.setWordWrap(True)
        self.info_label.setFont(QFont("Segoe UI", 10))
        self.info_label.setStyleSheet(
            "padding:10px 12px; background:#f0f4ff; border-radius:6px;"
            "border-left:4px solid #2563eb; color:#1e3a8a;")
        layout.addWidget(self.info_label)
        self.reset()

    def reset(self):
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, 'Draw something, then click  Analyze',
                ha='center', va='center', transform=ax.transAxes,
                color='#94a3b8', fontsize=10)
        ax.axis('off')
        self.mpl_canvas.draw()
        self.info_label.setText("Draw something, then click  Analyze")
        self.info_label.setStyleSheet(
            "padding:10px 12px; background:#f0f4ff; border-radius:6px;"
            "border-left:4px solid #2563eb; color:#1e3a8a;")

    def plot(self, strokes, spatial):
        from matplotlib.lines import Line2D
        self.fig.clear()
        gs   = self.fig.add_gridspec(2, 2, height_ratios=[3.5, 1],
                                     hspace=0.55, wspace=0.38,
                                     left=0.09, right=0.97, top=0.93, bottom=0.09)
        ax   = self.fig.add_subplot(gs[0, :])
        ax_h = self.fig.add_subplot(gs[1, 0])
        ax_s = self.fig.add_subplot(gs[1, 1])

        guidelines   = spatial.get('guidelines',   list(range(70, 400, 70)))
        touch_thresh = float(spatial.get('touch_thresh', 28.0))

        def nearest_guide(y):
            return min(guidelines, key=lambda g: abs(g - y))

        stroke_heights, stroke_slants, line_groups = [], [], defaultdict(list)

        for cluster in _group_letters(strokes):
            all_pts = [pt for s in cluster for pt in s]
            if len(all_pts) < 3:
                continue
            arr      = np.array(all_pts)
            y_top    = float(np.min(arr[:, 1])); y_bottom = float(np.max(arr[:, 1]))
            guide    = nearest_guide(y_bottom)
            if abs(y_bottom - guide) <= touch_thresh:
                h = float(guide - y_top)
                if h > 0:
                    stroke_heights.append(h)
                line_groups[guide].append(cluster)
            for s in cluster:
                if len(s) < 3:
                    continue
                arr_s = np.array(s)
                if float(np.max(arr_s[:, 1]) - np.min(arr_s[:, 1])) > 10:
                    dx = float(arr_s[-1, 0] - arr_s[0, 0])
                    dy = float(arr_s[-1, 1] - arr_s[0, 1])
                    if abs(dy) > 5:
                        stroke_slants.append(float(np.degrees(np.arctan2(dx, dy))))

        ax.set_facecolor('#fafbff')
        ax.set_title("Spatial Layout Overlay", fontsize=10, fontweight='bold', color='#1e3a8a')
        for guide in guidelines:
            ax.axhline(y=guide, color='#bfdbfe', linewidth=1.0, linestyle='--', alpha=0.9, zorder=1)
        for stroke in strokes:
            if len(stroke) < 2:
                continue
            arr = np.array(stroke)
            ax.plot(arr[:, 0], arr[:, 1], color='#3b82f6', linewidth=1.4, alpha=0.75, zorder=2)
        for cluster in _group_letters(strokes):
            all_pts = [pt for s in cluster for pt in s]
            if len(all_pts) < 3:
                continue
            arr    = np.array(all_pts)
            y_top  = float(np.min(arr[:, 1])); y_bottom = float(np.max(arr[:, 1]))
            guide  = nearest_guide(y_bottom); cx = float(np.mean(arr[:, 0]))
            if abs(y_bottom - guide) <= touch_thresh:
                ax.plot([cx, cx], [y_top, guide], color='#f97316', linewidth=1.6, alpha=0.7, zorder=3)
                for yy in (y_top, guide):
                    ax.plot([cx - 5, cx + 5], [yy, yy], color='#f97316', linewidth=1.6, alpha=0.7, zorder=3)
                if abs(y_bottom - guide) > 2:
                    ax.plot([cx, cx], [y_bottom, guide], color='#dc2626', linewidth=1.0, linestyle=':', alpha=0.6, zorder=3)
        for guide, group in line_groups.items():
            if len(group) < 2:
                continue
            sorted_g = sorted(group, key=_cluster_x_min)
            line_gaps = [_cluster_x_min(sorted_g[i + 1]) - _cluster_x_max(sorted_g[i])
                         for i in range(len(sorted_g) - 1)]
            pos_gaps  = [g for g in line_gaps if g > 0]
            thresh    = (float(np.mean(pos_gaps) + np.std(pos_gaps)) if len(pos_gaps) > 1 else float('inf'))
            y_arrow   = guide + 13
            for i, gap in enumerate(line_gaps):
                if gap <= 2:
                    continue
                x0 = _cluster_x_max(sorted_g[i]); x1 = _cluster_x_min(sorted_g[i + 1])
                color = '#dc2626' if gap > thresh else '#16a34a'
                mid   = (x0 + x1) / 2
                ax.plot([x0, x1], [y_arrow, y_arrow], color=color, linewidth=1.8, solid_capstyle='butt', zorder=4)
                for xc in (x0, x1):
                    ax.plot([xc, xc], [y_arrow - 4, y_arrow + 4], color=color, linewidth=1.8, zorder=4)
                ax.text(mid, y_arrow - 6, f'{gap:.0f}px', ha='center', va='bottom', fontsize=6, color=color)
        ax.set_xlim(-5, 690); ax.set_ylim(390, -5)
        ax.set_xlabel("x (px)", fontsize=8, color='#64748b'); ax.set_ylabel("y (px)", fontsize=8, color='#64748b')
        ax.tick_params(labelsize=7); ax.spines[['top', 'right']].set_visible(False)
        legend_handles = [
            Line2D([0], [0], color='#3b82f6', lw=1.4, label='Stroke'),
            Line2D([0], [0], color='#bfdbfe', lw=1.0, linestyle='--', label='Guideline'),
            Line2D([0], [0], color='#f97316', lw=1.6, label='Height bracket'),
            Line2D([0], [0], color='#dc2626', lw=1.0, linestyle=':', label='Baseline deviation'),
            Line2D([0], [0], color='#16a34a', lw=1.4, label='Letter gap'),
            Line2D([0], [0], color='#dc2626', lw=1.4, label='Word gap'),
        ]
        ax.legend(handles=legend_handles, fontsize=7, loc='upper right', framealpha=0.85)

        ax_h.set_facecolor('#fafbff')
        if stroke_heights:
            mean_h = float(np.mean(stroke_heights)); sd_h = float(np.std(stroke_heights))
            bc = ['#16a34a' if abs(h - mean_h) <= sd_h else
                  '#f97316' if abs(h - mean_h) <= 2 * sd_h else '#dc2626' for h in stroke_heights]
            ax_h.bar(range(len(stroke_heights)), stroke_heights, color=bc, alpha=0.85, zorder=2)
            ax_h.axhline(mean_h, color='#2563eb', linewidth=1.5, linestyle='--',
                         label=f'Mean={mean_h:.0f}px', zorder=3)
            ax_h.set_title("Letter Heights", fontsize=9, fontweight='bold', color='#1e3a8a')
            ax_h.set_xlabel("Stroke #", fontsize=7, color='#64748b')
            ax_h.set_ylabel("px", fontsize=7, color='#64748b')
            ax_h.tick_params(labelsize=7); ax_h.legend(fontsize=7)
            ax_h.spines[['top', 'right']].set_visible(False)
            ax_h.grid(True, alpha=0.15, axis='y')
        else:
            ax_h.text(0.5, 0.5, 'No data', ha='center', va='center',
                      transform=ax_h.transAxes, color='#94a3b8', fontsize=9); ax_h.axis('off')

        ax_s.set_facecolor('#fafbff')
        if stroke_slants:
            bc_s = ['#16a34a' if abs(a) < 15 else
                    ('#3b82f6' if a < 45 else '#dc2626') if a > 0 else
                    ('#f97316' if a > -45 else '#dc2626') for a in stroke_slants]
            ax_s.bar(range(len(stroke_slants)), stroke_slants, color=bc_s, alpha=0.85, zorder=2)
            ax_s.axhline(0,  color='#64748b', linewidth=0.9, zorder=1)
            ax_s.axhline( 15, color='#3b82f6', linewidth=0.7, linestyle=':', alpha=0.5)
            ax_s.axhline(-15, color='#f97316', linewidth=0.7, linestyle=':', alpha=0.5)
            mean_sl = float(np.mean(stroke_slants)); std_sl = float(np.std(stroke_slants))
            s_label, _, _ = slant_type(mean_sl, std_sl)
            ax_s.axhline(mean_sl, color='#2563eb', linewidth=1.5, linestyle='--',
                         label=f'Mean {mean_sl:+.0f}°', zorder=3)
            ax_s.set_title(f"Slant  —  {s_label}", fontsize=9, fontweight='bold', color='#1e3a8a')
            ax_s.set_xlabel("Stroke #", fontsize=7, color='#64748b')
            ax_s.set_ylabel("Angle (°)", fontsize=7, color='#64748b')
            ax_s.tick_params(labelsize=7); ax_s.legend(fontsize=7)
            ax_s.spines[['top', 'right']].set_visible(False)
            ax_s.grid(True, alpha=0.15, axis='y')
        else:
            ax_s.text(0.5, 0.5, 'No slant data', ha='center', va='center',
                      transform=ax_s.transAxes, color='#94a3b8', fontsize=9); ax_s.axis('off')

        self.mpl_canvas.draw()

        sw_str  = f"{spatial['sw_mean']:.0f} px" if spatial['sw_mean'] > 0 else 'n/a'
        tm      = spatial.get('theta_mean', 0.0); ts = spatial.get('theta_std', 0.0)
        s_label, _, s_itype = slant_type(tm, ts)
        h_icon = '✓' if 20 <= spatial['h_mean'] <= 80 else '⚠'
        d_icon = '✓' if spatial['delta'] < 6 else '⚠'
        t_icon = '✓' if s_itype == 'good' else '⚠'
        self.info_label.setText(
            f"Height = {spatial['h_mean']:.0f} px {h_icon}  ·  "
            f"Height std = {spatial['h_std']:.0f} px  ·  "
            f"Letter gap = {spatial['sl_mean']:.0f} px  ·  "
            f"Word gap = {sw_str}  ·  "
            f"Baseline dev = {spatial['delta']:.0f} px {d_icon}  ·  "
            f"Slant = {s_label} ({tm:+.0f}°, σ={ts:.0f}°) {t_icon}")
        n_bad = sum([spatial['h_mean'] < 10 or spatial['h_mean'] > 120,
                     spatial['delta'] > 12, spatial['h_std'] > 25, s_itype == 'bad'])
        fg, bg, border = (('#166534','#f0fdf4','#16a34a') if n_bad == 0 else
                          ('#92400e','#fffbeb','#f59e0b') if n_bad == 1 else
                          ('#991b1b','#fef2f2','#dc2626'))
        self.info_label.setStyleSheet(
            f"padding:10px 12px; background:{bg}; border-radius:6px;"
            f"border-left:4px solid {border}; color:{fg};")


# =============================================================================
# Feature HTML table
# =============================================================================

def _build_features_html(kin, pres, spatial=None):
    def classify(key, v):
        if key == 'v_mean':       return 'ok' if v > 100 else ('bad' if v < 60 else 'warn')
        if key == 'v_std':
            cv = v / max(kin['v_mean'], 1)
            return 'ok' if cv < 0.60 else ('bad' if cv > 0.80 else 'warn')
        if key == 'a_mean':       a = abs(v); return 'ok' if a < 30 else ('bad' if a > 80 else 'warn')
        if key == 'j_mean':       r = v / max(kin['v_mean'], 1); return 'ok' if r < 1.0 else ('bad' if r > 1.5 else 'warn')
        if key == 'in_air_ratio': return 'ok' if v < 0.30 else ('bad' if v > 0.40 else 'warn')
        if key == 'n_lifts':      return 'ok'
        if key == 'p_mean':       return 'ok' if 0.30 <= v <= 0.75 else ('bad' if (v < 0.20 or v > 0.80) else 'warn')
        if key == 'p_std':        return 'ok' if v < 0.05 else ('bad' if v > 0.15 else 'warn')
        if key == 'p_max':        return 'ok' if v <= 0.85 else ('bad' if v > 0.90 else 'warn')
        if key == 'p_range':      return 'ok' if v < 0.30 else ('bad' if v > 0.50 else 'warn')
        if key == 'h_mean':       return 'ok' if 20 <= v <= 80 else ('bad' if v < 10 or v > 120 else 'warn')
        if key == 'h_std':        return 'ok' if v < 15 else ('bad' if v > 30 else 'warn')
        if key == 'sl_mean':      return 'ok' if 5 <= v <= 40 else ('bad' if v < 2 or v > 60 else 'warn')
        if key == 'sw_mean':      return ('warn' if v == 0 else 'ok' if 30 <= v <= 100 else ('bad' if v < 15 or v > 150 else 'warn'))
        if key == 'delta':        return 'ok' if v < 6 else ('bad' if v > 12 else 'warn')
        if key == 'theta':        return 'ok' if v <= 45 else 'bad'
        if key == 'theta_mean':   return 'ok' if abs(v) <= 45 else 'bad'
        if key == 'theta_std':    return 'ok' if v < 15 else ('bad' if v > 20 else 'warn')
        return 'ok'

    C = {
        'ok':   {'row':'#f0fdf4','text':'#166534','label':'Normal',     'icon':'&#10003;'},
        'warn': {'row':'#fffbeb','text':'#92400e','label':'Borderline', 'icon':'&#126;'},
        'bad':  {'row':'#fef2f2','text':'#991b1b','label':'Dysgraphic', 'icon':'&#9888;'},
    }
    def badge(s): c = C[s]; return f'<b style="color:{c["text"]};">{c["icon"]} {c["label"]}</b>'
    td  = 'padding:7px 10px; border-bottom:1px solid #f1f5f9; font-size:11px;'
    th  = 'background:#1e3a8a; color:white; padding:9px 10px; font-size:10px; font-weight:bold; text-align:left;'
    sec = 'background:#2563eb; color:white; font-weight:bold; font-size:11px; padding:9px 14px;'

    def row(sym, name, meaning, val_str, unit, normal, dysgraphic, key, raw):
        s = classify(key, raw); cc = C[s]
        return (f'<tr style="background:{cc["row"]};">'
                f'<td style="{td} font-weight:bold; font-size:14px; color:#1e3a8a;">{sym}</td>'
                f'<td style="{td}">{name}<br><span style="font-size:9px;color:#94a3b8;font-style:italic;">{meaning}</span></td>'
                f'<td style="{td} font-weight:bold; color:{cc["text"]}; text-align:right; font-size:12px;">{val_str}</td>'
                f'<td style="{td} color:#94a3b8;">{unit}</td>'
                f'<td style="{td} color:#166534;">{normal}</td>'
                f'<td style="{td} color:#991b1b;">{dysgraphic}</td>'
                f'<td style="{td}">{badge(s)}</td></tr>')

    html = (f'<html><body style="font-family:Segoe UI,Arial,sans-serif;margin:0;padding:10px;background:white;">'
            f'<table width="100%" cellspacing="0" cellpadding="0" style="border-collapse:collapse;">'
            f'<tr><th style="{th}">Symbol</th><th style="{th}">Feature</th>'
            f'<th style="{th}" align="right">Measured</th><th style="{th}">Unit</th>'
            f'<th style="{th}">&#10003; Normal</th><th style="{th}">&#9888; Dysgraphic</th>'
            f'<th style="{th}">Status</th></tr>'
            f'<tr><td colspan="7" style="{sec}">Kinematic Features</td></tr>'
            + row('v&#772;','Mean velocity','Average pen speed',f"{kin['v_mean']:.1f}",'px/s','&gt;100','&lt;60','v_mean',kin['v_mean'])
            + row('&sigma;v','Velocity std','How consistent speed is',f"{kin['v_std']/max(kin['v_mean'],1):.2f}",'CV','CV&lt;0.60','CV&gt;0.80','v_std',kin['v_std'])
            + row('a&#772;','Mean acceleration','Rate of speed change',f"{kin['a_mean']:+.1f}",'px/s&sup2;','|a|&lt;30','|a|&gt;80','a_mean',kin['a_mean'])
            + row('j&#772;','Mean jerk','Smoothness of movement',f"{kin['j_mean']/max(kin['v_mean'],1):.2f}",'j/v','&lt;1.0','&gt;1.5','j_mean',kin['j_mean'])
            + row('r<sub>a</sub>','In-air ratio','Pen lift fraction',f"{kin['in_air_ratio']:.3f}",'',' &lt;0.30','&gt;0.40','in_air_ratio',kin['in_air_ratio'])
            + row('N','Pen lifts','Stroke count',str(kin['n_lifts']),'count','&mdash;','&mdash;','n_lifts',kin['n_lifts'])
            + f'<tr><td colspan="7" style="{sec}">Pressure Features</td></tr>'
            + row('p&#772;','Mean pressure','Typical pen force',f"{pres['p_mean']:.3f}",'0-1','0.30-0.75','&lt;0.20 or &gt;0.80','p_mean',pres['p_mean'])
            + row('&sigma;p','Pressure std','Grip variability',f"{pres['p_std']:.3f}",'0-1','&lt;0.05','&gt;0.15','p_std',pres['p_std'])
            + row('p<sub>max</sub>','Peak pressure','Heaviest force',f"{pres['p_max']:.3f}",'0-1','&le;0.85','&gt;0.90','p_max',pres['p_max'])
            + row('&Delta;p','Pressure range','Max minus min',f"{pres['p_range']:.3f}",'0-1','&lt;0.30','&gt;0.50','p_range',pres['p_range']))

    if spatial:
        sw_str = f"{spatial['sw_mean']:.1f}" if spatial['sw_mean'] > 0 else "n/a"
        tm = spatial.get('theta_mean', 0.0); ts = spatial.get('theta_std', 0.0)
        s_label, _, _ = slant_type(tm, ts)
        html += (f'<tr><td colspan="7" style="{sec}">Spatial Features</td></tr>'
                 + row('H&#772;','Letter height','Avg height above baseline',f"{spatial['h_mean']:.1f}",'px','20-80','&lt;10 or &gt;120','h_mean',spatial['h_mean'])
                 + row('&sigma;<sub>H</sub>','Height std','Consistency of letter sizes',f"{spatial['h_std']:.1f}",'px','&lt;15','&gt;30','h_std',spatial['h_std'])
                 + row('S&#772;<sub>l</sub>','Letter gap','Space between letters',f"{spatial['sl_mean']:.1f}",'px','5-40','&lt;2 or &gt;60','sl_mean',spatial['sl_mean'])
                 + row('S&#772;<sub>w</sub>','Word gap','Space between words',sw_str,'px','30-100','&lt;15 or &gt;150','sw_mean',spatial['sw_mean'])
                 + row('&delta;','Baseline dev','How closely writing follows guideline',f"{spatial['delta']:.1f}",'px','&lt;8','&gt;20','delta',spatial['delta'])
                 + row('&theta;','Slant direction','Which way letters lean',f"{tm:+.1f}&deg; ({s_label})",'','|&theta;|&le;45&deg;','|&theta;|&gt;45&deg;','theta_mean',tm)
                 + row('&sigma;<sub>&theta;</sub>','Slant consistency','How uniform the lean is',f"{ts:.1f}",'&deg;','&lt;15&deg;','&gt;20&deg;','theta_std',ts))

    html += ('</table><p style="color:#94a3b8;font-size:9px;margin-top:8px;padding:6px;background:#f8fafc;border-left:3px solid #cbd5e1;">'
             'Thresholds from digitised handwriting literature. Mouse input reports constant pressure (0.70).</p>'
             '</body></html>')
    return html


# =============================================================================
# Main Window
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("The Dysgraphia Detective  —  Thesis SS26")
        self.resize(1380, 780)
        self._analysis = None
        self._model    = _try_load_model()
        self._build_ui()
        self._update_model_badge()

    def _build_ui(self):
        root   = QWidget(); self.setCentralWidget(root)
        root_v = QVBoxLayout(root)
        root_v.setContentsMargins(12, 10, 12, 10)
        root_v.setSpacing(10)

        # Title bar
        top = QHBoxLayout()
        title = QLabel("The Dysgraphia Detective")
        title.setFont(QFont("Segoe UI", 16, QFont.Weight.Bold))
        title.setStyleSheet("color:#1e3a8a;")
        top.addWidget(title); top.addStretch()

        reload_btn = QPushButton("Reload Model")
        reload_btn.setStyleSheet(_S_RELOAD)
        reload_btn.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        reload_btn.setMinimumHeight(28)
        reload_btn.setToolTip("Reload model.pkl after training in the ML Trainer app.")
        reload_btn.clicked.connect(self._on_reload_model)
        top.addWidget(reload_btn)

        ml_btn = QPushButton("Open ML Trainer")
        ml_btn.setStyleSheet(_S_ML_BTN)
        ml_btn.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        ml_btn.setMinimumHeight(28)
        ml_btn.clicked.connect(self._on_open_ml)
        top.addWidget(ml_btn)

        self.model_badge = QLabel("No model loaded")
        self.model_badge.setStyleSheet(
            "padding:4px 12px; background:#f1f5f9; color:#64748b;"
            "border-radius:12px; font-size:10px; font-weight:600;")
        top.addWidget(self.model_badge)
        root_v.addLayout(top)

        # Body
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left card
        left_card = QFrame(); left_card.setObjectName("card")
        left_v    = QVBoxLayout(left_card)
        left_v.setContentsMargins(14, 14, 14, 14)
        left_v.setSpacing(10)

        lbl_canvas = QLabel("Handwriting Canvas")
        lbl_canvas.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        lbl_canvas.setStyleSheet("color:#1e3a8a;")
        left_v.addWidget(lbl_canvas)

        self.canvas = DrawingCanvas()
        left_v.addWidget(self.canvas)

        # Clear + Analyze (only two buttons)
        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        self.clear_btn   = QPushButton("Clear")
        self.analyze_btn = QPushButton("Analyze")
        self.clear_btn.setStyleSheet(_S_CLEAR)
        self.analyze_btn.setStyleSheet(_S_ANALYZE)
        for b in (self.clear_btn, self.analyze_btn):
            b.setMinimumHeight(42); b.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        self.clear_btn.clicked.connect(self._on_clear)
        self.analyze_btn.clicked.connect(self._on_analyze)
        btn_row.addWidget(self.clear_btn, 1); btn_row.addWidget(self.analyze_btn, 2)
        left_v.addLayout(btn_row)

        # Verdict box
        self.verdict_lbl = QLabel("Draw on the canvas above and click  Analyze")
        self.verdict_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.verdict_lbl.setWordWrap(True)
        self.verdict_lbl.setFont(QFont("Segoe UI", 11))
        self.verdict_lbl.setStyleSheet(
            "padding:14px; border:1px solid #dde3ee; border-radius:8px;"
            "background:#f8fafc; color:#64748b;")
        left_v.addWidget(self.verdict_lbl)

        hint = QLabel("To save samples and train the model → Open ML Trainer")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setStyleSheet("color:#94a3b8; font-size:9px; font-style:italic;")
        left_v.addWidget(hint)
        left_v.addStretch()

        # Right tabs
        tabs = QTabWidget()
        self.speed_panel    = ProfilePanel("Speed Profile", "Speed (px/s)")
        self.pressure_panel = ProfilePanel("Pressure Profile", "Pressure (0-1)")
        self.spatial_panel  = SpatialPanel()
        self.feat_browser   = QTextBrowser()
        self.feat_browser.setHtml(
            "<p style='font-family:Segoe UI;color:#94a3b8;padding:20px;font-size:12px;'>"
            "Run Analyze to see extracted features.</p>")
        tabs.addTab(self.speed_panel,    "Speed")
        tabs.addTab(self.pressure_panel, "Pressure")
        tabs.addTab(self.spatial_panel,  "Spatial")
        tabs.addTab(self.feat_browser,   "Features")

        splitter.addWidget(left_card); splitter.addWidget(tabs)
        splitter.setStretchFactor(0, 0); splitter.setStretchFactor(1, 1)
        splitter.setSizes([730, 630])
        root_v.addWidget(splitter, 1)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _update_model_badge(self):
        m = self._model
        if m is None:
            self.model_badge.setText(
                "No model — open ML Trainer and click ‘Train on Real Data’")
            self.model_badge.setStyleSheet(
                "padding:4px 12px; background:#fef3c7; color:#92400e;"
                "border-radius:12px; font-size:10px; font-weight:600;")
        else:
            nf = m.get('n_fluent', '?'); nd = m.get('n_dysg', '?')
            txt = f"Model: real  ({nf}F + {nd}D)"
            bg, fg = '#dcfce7', '#15803d'
            self.model_badge.setText(txt)
            self.model_badge.setStyleSheet(
                f"padding:4px 12px; background:{bg}; color:{fg};"
                f"border-radius:12px; font-size:10px; font-weight:600;")

    def _on_reload_model(self):
        m = _try_load_model()
        if m:
            self._model = m
            self._update_model_badge()
            nf  = m.get('n_fluent', '?'); nd = m.get('n_dysg', '?')
            QMessageBox.information(self, "Model Reloaded",
                f"Model loaded successfully.\n"
                f"Sessions: {nf} fluent + {nd} dysgraphic")
        else:
            QMessageBox.warning(self, "No Model Found",
                f"No trained model found at:\n{MODEL_PATH}\n\n"
                "Open the ML Trainer and train a model first.")

    def _on_open_ml(self):
        try:
            ml_path = Path(__file__).resolve().parent / 'dysgraphia_ml.py'
            subprocess.Popen([sys.executable, str(ml_path)],
                             creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0)
        except Exception as e:
            QMessageBox.warning(self, "Could not open ML Trainer", str(e))

    # ── Button handlers ───────────────────────────────────────────────────────

    def _on_clear(self):
        self.canvas.clear()
        self.speed_panel.reset(); self.pressure_panel.reset(); self.spatial_panel.reset()
        self.feat_browser.setHtml(
            "<p style='font-family:Segoe UI;color:#94a3b8;padding:20px;font-size:12px;'>"
            "Canvas cleared. Draw and run Analyze again.</p>")
        self.verdict_lbl.setText("Draw on the canvas above and click  Analyze")
        self.verdict_lbl.setStyleSheet(
            "padding:14px; border:1px solid #dde3ee; border-radius:8px;"
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
        if t_spd is None:
            QMessageBox.warning(self, "Error", "Could not compute speed profile."); return

        trend_spd            = fit_trend(t_spd, speed)
        interp_spd, type_spd = interpret_speed(trend_spd['slope'], trend_spd['r2'],
                                               kin['v_mean'], kin['j_mean'], kin['v_std'])
        self.speed_panel.plot(t_spd, speed, trend_spd, interp_spd, type_spd)

        t_prs, pressure      = compute_pressure_profile(self.canvas.strokes)
        trend_prs            = fit_trend(t_prs, pressure)
        interp_prs, type_prs = interpret_pressure(pres['p_std'], pres['p_max'], pres['p_range'])
        self.pressure_panel.plot(t_prs, pressure, trend_prs, interp_prs, type_prs)

        self.feat_browser.setHtml(_build_features_html(kin, pres, spatial))
        if spatial:
            self.spatial_panel.plot(self.canvas.strokes, spatial)

        # Rule-based verdict — each indicator counts separately
        indicators = [type_spd == 'bad', type_prs == 'bad', kin['in_air_ratio'] > 0.4]
        if spatial:
            _, _, s_itype = slant_type(spatial.get('theta_mean', 0.0), spatial.get('theta_std', 0.0))
            indicators.append(spatial['delta'] > 12)       # baseline drift (17% of line height)
            indicators.append(spatial['h_std'] > 25)       # letter height inconsistency
            indicators.append(s_itype == 'bad')            # slant counted separately
        n_bad = sum(indicators)

        if n_bad == 0:
            rule_txt, rfg, rbg = "Rule-based: Fluent writing pattern", "#166534", "#f0fdf4"
        elif n_bad == 1:
            rule_txt, rfg, rbg = "Rule-based: Borderline (1 indicator)", "#92400e", "#fffbeb"
        else:
            rule_txt, rfg, rbg = "Rule-based: Dysgraphic indicators detected", "#991b1b", "#fef2f2"

        # ML prediction from loaded model
        ml_txt = ""
        if self._model:
            vec = _extract_feature_vector(kin, pres, spatial,
                                          strokes=self.canvas.strokes)
            if vec is not None:
                try:
                    pred, prob = _ensemble_predict(self._model, vec)
                    plabel = "Fluent" if pred == 0 else "Dysgraphic"
                    n_models = len([x for x in ['rf','svm','lr','gb','ab']
                                    if x in self._model])
                    ml_txt = (f"     |     ML ({n_models} models): "
                              f"{plabel} ({prob*100:.0f}%)")
                except Exception:
                    pass

        self.verdict_lbl.setText(rule_txt + ml_txt)
        self.verdict_lbl.setStyleSheet(
            f"padding:14px; border:1px solid {rfg}; border-radius:8px;"
            f"background:{rbg}; color:{rfg}; font-weight:bold;")

        self._analysis = {
            'kinematic_features': kin,
            'pressure_features':  pres,
            'spatial_features':   spatial,
        }


# =============================================================================
# Entry point
# =============================================================================

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(APP_STYLE)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())

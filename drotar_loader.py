"""
drotar_loader.py - Load the Drotár & Dobeš (2020) public dysgraphia dataset
(120 subjects, 89 labelled, Wacom Intuos Pro Large) into the same
(stroke-list, features) format used elsewhere in this project.

Drotár .svc file format (one row per pen sample, space-separated):
    x  y  timestamp_ms  pen_status  azimuth_deg_x10  altitude_deg_x10  pressure_raw

Unit conversions applied to bring Drotár into the XCV pipeline's units:
    * x, y       -> XCV-equivalent pixels   (scaling factor _PX_PER_TABLET_UNIT)
    * y          -> flipped (Wacom y is bottom-up, screen y is top-down)
    * timestamp  -> seconds                 (divide by 1000)
    * pressure   -> [0, 1]                  (divide by 1024)
    * azimuth    -> degrees                 (divide by 10)
    * altitude   -> degrees                 (divide by 10)

Pen-status segmentation: each contiguous run of pen_status == 1 is one stroke
(matches the XCV canvas semantics). Runs of pen_status == 0 are in-air gaps.
"""

from pathlib import Path
import numpy as np
import openpyxl

# Wacom Intuos Pro Large: 5080 lines per inch. A typical letter on the XCV
# canvas is ~30 px tall and represents ~5 mm = 0.197 in = 1000 tablet units.
# So 30 px per 1000 tablet units -> 0.03 px per tablet unit.
_PX_PER_TABLET_UNIT = 0.030

# Pressure: Wacom .svc files use 10-bit (0..1023) raw pressure.
_PRESSURE_MAX = 1024.0

DROTAR_DIR = Path(__file__).resolve().parent / 'drotar_dataset'


# =============================================================================
# Label loader
# =============================================================================

def load_labels():
    """Return dict {subject_id: 'DYSGR' | 'NORMAL'} for every labelled subject."""
    wb = openpyxl.load_workbook(DROTAR_DIR / 'labels.xlsx', data_only=True)
    ws = wb.active
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        sid, diag = row[0], row[1]
        if sid is None or diag is None:
            continue
        sid = str(sid).strip().zfill(5)
        diag = str(diag).strip()
        if diag.upper().startswith('DYS'):
            out[sid] = 'DYSGR'
        elif diag in ('0', 'NORMAL', 'CTRL'):
            out[sid] = 'NORMAL'
    return out


# =============================================================================
# .svc parser  -- returns a stroke list in XCV format
# =============================================================================

def parse_svc(svc_path):
    """Parse one .svc file into (strokes, all_samples).

    strokes      -> list of strokes; each stroke is list of (x, y, t, p) tuples
                    in XCV units (pixels, seconds, pressure 0..1).
    all_samples  -> numpy array of every sample, shape (N, 7), in raw tablet
                    units (used by tilt-feature extractor below).
    """
    lines = svc_path.read_text().splitlines()
    n = int(lines[0])
    arr = np.array([list(map(int, ln.split())) for ln in lines[1:1 + n]],
                   dtype=np.int64)
    # Columns: x y t pen_status azimuth altitude pressure

    # Convert to XCV units (vectorised)
    t0 = arr[0, 2]
    x_px  = arr[:, 0].astype(float) * _PX_PER_TABLET_UNIT
    # Flip y: Wacom y is bottom-up, screen y is top-down.
    y_max = arr[:, 1].max()
    y_px  = (y_max - arr[:, 1].astype(float)) * _PX_PER_TABLET_UNIT
    t_s   = (arr[:, 2].astype(float) - t0) / 1000.0
    p_01  = arr[:, 6].astype(float) / _PRESSURE_MAX

    pen_status = arr[:, 3]  # 1 = on surface, 0 = in air

    # Segment into strokes by pen-status transitions
    strokes = []
    current = []
    for i in range(n):
        if pen_status[i] == 1:
            current.append((float(x_px[i]), float(y_px[i]),
                            float(t_s[i]), float(p_01[i])))
        else:
            if current:
                strokes.append(current)
                current = []
    if current:
        strokes.append(current)

    return strokes, arr


# =============================================================================
# Tilt features (azimuth, altitude) — the 3 extras we agreed to add
# =============================================================================

def compute_tilt_features(raw_arr):
    """Compute 3 tilt features from the raw .svc array. On-surface samples only.

    Returns [tilt_mean, tilt_std, azimuth_consistency] in degrees.
    tilt = 90° - altitude. Higher tilt = pen leaning further from vertical.
    azimuth_consistency = 1 - circular_std(azimuth)/180 (1.0 = perfectly consistent).
    """
    on_surface = raw_arr[raw_arr[:, 3] == 1]
    if len(on_surface) < 10:
        return [0.0, 0.0, 0.0]
    altitude_deg = on_surface[:, 5].astype(float) / 10.0
    azimuth_deg  = on_surface[:, 4].astype(float) / 10.0
    tilt = 90.0 - altitude_deg
    tilt_mean = float(np.mean(tilt))
    tilt_std  = float(np.std(tilt))
    # Circular standard deviation for azimuth (handles 0°/360° wrap-around)
    rad = np.deg2rad(azimuth_deg)
    R = np.sqrt(np.mean(np.cos(rad))**2 + np.mean(np.sin(rad))**2)
    circ_std_deg = np.rad2deg(np.sqrt(-2.0 * np.log(max(R, 1e-9))))
    azimuth_consistency = float(max(0.0, 1.0 - circ_std_deg / 180.0))
    return [tilt_mean, tilt_std, azimuth_consistency]


# =============================================================================
# Convenience: load ALL labelled subjects into (stroke list, label) pairs
# =============================================================================

def iter_subjects():
    """Yield (subject_id, strokes, raw_arr, label) for every labelled subject
    whose .svc file is on disk. label is 0 = NORMAL, 1 = DYSGR."""
    labels = load_labels()
    for sid in sorted(labels.keys()):
        user_dir = DROTAR_DIR / f'user{sid}'
        svc_files = list(user_dir.rglob('*.svc'))
        if not svc_files:
            continue
        try:
            strokes, raw = parse_svc(svc_files[0])
        except Exception as e:
            print(f"[drotar] WARNING: {sid} parse failed: {e}")
            continue
        if not strokes:
            continue
        label = 1 if labels[sid] == 'DYSGR' else 0
        yield sid, strokes, raw, label


if __name__ == '__main__':
    # Sanity check
    labels = load_labels()
    print(f"Labels loaded: {len(labels)}")
    from collections import Counter
    print(f"  Distribution: {Counter(labels.values())}")
    n_ok = 0
    sample_seen = False
    for sid, strokes, raw, lbl in iter_subjects():
        n_ok += 1
        if not sample_seen:
            print(f"\nFirst subject {sid}:  label={lbl}  "
                  f"strokes={len(strokes)}  total samples={len(raw)}")
            xs = [pt[0] for s in strokes for pt in s]
            ys = [pt[1] for s in strokes for pt in s]
            ts = [pt[2] for s in strokes for pt in s]
            ps = [pt[3] for s in strokes for pt in s]
            print(f"  After unit conversion:")
            print(f"    x range:        {min(xs):.1f} .. {max(xs):.1f}  px")
            print(f"    y range:        {min(ys):.1f} .. {max(ys):.1f}  px")
            print(f"    duration:       {max(ts):.2f}  s")
            print(f"    pressure range: {min(ps):.3f} .. {max(ps):.3f}")
            tilt = compute_tilt_features(raw)
            print(f"  Tilt features:   {tilt}")
            sample_seen = True
    print(f"\nTotal subjects successfully loaded: {n_ok}")

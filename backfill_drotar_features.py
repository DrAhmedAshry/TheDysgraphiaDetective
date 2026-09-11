"""Re-extract the 119 Drotár-style features for every existing session in
`dataset/{fluent,dysgraphic}/json/` and `dataset/test/{fluent,dysgraphic}/json/`,
write the resulting dict into the JSON under the `drotar_features` key.

Safe to re-run — overwrites the key in place; leaves all other JSON fields
untouched.
"""
import json
from pathlib import Path

from dysgraphia_ml import compute_drotar_features, DROTAR_FEATURE_KEYS

DATASET_DIR = Path(__file__).resolve().parent / 'dataset'

SOURCES = [
    DATASET_DIR / 'fluent'    / 'json',
    DATASET_DIR / 'dysgraphic'/ 'json',
    DATASET_DIR / 'test' / 'fluent'    / 'json',
    DATASET_DIR / 'test' / 'dysgraphic'/ 'json',
]


def process(jdir: Path) -> tuple[int, int, int]:
    if not jdir.exists():
        return 0, 0, 0
    n_ok = n_noraw = n_fail = 0
    for jf in sorted(jdir.glob('*.json')):
        try:
            data = json.loads(jf.read_text(encoding='utf-8'))
        except Exception as e:
            print(f"  [skip] {jf.name}: cannot parse ({e})")
            n_fail += 1
            continue
        strokes = data.get('strokes')
        if not strokes:
            n_noraw += 1
            continue
        feats = compute_drotar_features(strokes)
        if feats is None:
            n_fail += 1
            continue
        # Sanity: make sure every canonical key is present
        for k in DROTAR_FEATURE_KEYS:
            feats.setdefault(k, 0.0)
        data['drotar_features'] = feats
        jf.write_text(json.dumps(data, indent=2), encoding='utf-8')
        n_ok += 1
    return n_ok, n_noraw, n_fail


if __name__ == '__main__':
    total_ok = total_noraw = total_fail = 0
    for src in SOURCES:
        ok, noraw, fail = process(src)
        total_ok    += ok
        total_noraw += noraw
        total_fail  += fail
        rel = src.relative_to(DATASET_DIR)
        print(f"  {str(rel):<35}  ok={ok:4d}  no_strokes={noraw:3d}  fail={fail:3d}")
    print(f"\nTOTAL                                ok={total_ok}  "
          f"no_strokes={total_noraw}  fail={total_fail}")
    print(f"Each updated JSON now carries {len(DROTAR_FEATURE_KEYS)} "
          "features under data['drotar_features'].")

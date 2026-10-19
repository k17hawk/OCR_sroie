"""Rescale a DevaLabel v2 JSON to the size of the image your canvas actually loads.

usage:
  python3 rescale.py in.json out.json --image path/to/canvas.png
  python3 rescale.py in.json out.json --size 2409 3436
  python3 rescale.py in.json out.json --size 2409 3436 --src-size 2480 3508
  python3 rescale.py in.json out.json --size 2409 3436 --auto-src

Scales every bbox, quad, p0/p1, font_size and ruling-line width, plus
width/height. Text and structure are untouched.

All coordinates are emitted as integers (the DevaLabel server rejects floats
with HTTP 422 "Input should be a valid integer, got a number with a
fractional part").

Order of precedence for the source canvas size (W x H that the coordinates
were drawn in):
  1. --src-size W H           (explicit)
  2. --auto-src               (inferred from max coordinate extent)
  3. d["width"], d["height"]  (whatever the JSON declares)

If the declared size is smaller than the maximum coordinate found, a warning
is printed and, unless --src-size or --auto-src is given, the inferred size
is used automatically -- this is the usual cause of "rescale did nothing".
"""
import argparse
import json
import sys


def max_extent(doc):
    """Return (max_x, max_y) across every bbox / quad / p0 / p1 in the doc."""
    mx = my = 0.0

    def walk(o):
        nonlocal mx, my
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "bbox" and isinstance(v, list) and len(v) == 4:
                    mx = max(mx, v[2])
                    my = max(my, v[3])
                elif k in ("p0", "p1") and isinstance(v, list) and len(v) == 2:
                    mx = max(mx, v[0])
                    my = max(my, v[1])
                elif k == "quad" and isinstance(v, list) and v and isinstance(v[0], list):
                    for p in v:
                        mx = max(mx, p[0])
                        my = max(my, p[1])
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(doc)
    return mx, my


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--image", help="canvas image to read W/H from")
    ap.add_argument("--size", nargs=2, type=int, metavar=("W", "H"),
                    help="target canvas size in pixels")
    ap.add_argument("--src-size", nargs=2, type=int, metavar=("W", "H"),
                    help="size the annotations were drawn in (overrides JSON)")
    ap.add_argument("--auto-src", action="store_true",
                    help="infer source size from max coordinate extent")
    a = ap.parse_args()

    # ---- target size ----
    if a.image:
        from PIL import Image
        W, H = Image.open(a.image).size
    elif a.size:
        W, H = a.size
    else:
        ap.error("give --image or --size")

    d = json.load(open(a.src, encoding="utf-8"))
    declared_w, declared_h = d["width"], d["height"]
    ext_w, ext_h = max_extent(d)

    # ---- source size ----
    if a.src_size:
        SW, SH = a.src_size
        src_mode = "cli"
    elif a.auto_src:
        SW, SH = ext_w, ext_h
        src_mode = "auto"
    else:
        SW, SH = declared_w, declared_h
        src_mode = "json"
        if ext_w > declared_w + 1 or ext_h > declared_h + 1:
            print(
                f"[warn] declared size {declared_w}x{declared_h} is smaller than "
                f"max coordinate extent {ext_w:.0f}x{ext_h:.0f}",
                file=sys.stderr,
            )
            print(
                "[warn] using inferred source size "
                f"{ext_w:.0f}x{ext_h:.0f} (pass --src-size to override)",
                file=sys.stderr,
            )
            SW, SH = ext_w, ext_h
            src_mode = "auto"

    sx, sy = W / SW, H / SH
    s_avg = (sx + sy) / 2

    if abs(sx - 1.0) < 1e-9 and abs(sy - 1.0) < 1e-9:
        print(
            f"[warn] scale is 1.0 ({W}x{H} == src {SW:.0f}x{SH:.0f}) -- "
            "nothing will change. Use --src-size or --auto-src.",
            file=sys.stderr,
        )

    # ---- integer-safe helpers ----
    def clamp(v, hi):
        v = int(round(v))
        if v < 0:
            return 0
        if v > hi:
            return int(hi)
        return v

    def box(b):
        return [clamp(b[0] * sx, W),
                clamp(b[1] * sy, H),
                clamp(b[2] * sx, W),
                clamp(b[3] * sy, H)]

    def pt(p):
        return [clamp(p[0] * sx, W),
                clamp(p[1] * sy, H)]

    def walk(o, key=None):
        if isinstance(o, dict):
            return {k: walk(v, k) for k, v in o.items()}
        if isinstance(o, list):
            if key == "bbox" and len(o) == 4 and all(isinstance(v, (int, float)) for v in o):
                return box(o)
            if key in ("stamp_bboxes", "signature_bboxes"):
                return [box(b) for b in o]
            if key == "quad":
                return [pt(p) for p in o]
            if key in ("p0", "p1"):
                return pt(o)
            return [walk(v) for v in o]
        if key == "font_size" and isinstance(o, (int, float)):
            return int(round(o * s_avg))
        if key == "width" and isinstance(o, (int, float)):
            # ruling_lines[].width (stroke thickness)
            return max(1, int(round(o * s_avg)))
        return o

    d = walk(d)
    d["width"], d["height"] = int(W), int(H)

    json.dump(d, open(a.dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

    print(
        f"src={SW:.0f}x{SH:.0f} ({src_mode})  "
        f"->  dst={W}x{H}  scale x{sx:.4f} / y{sy:.4f}  (int output)"
    )


if __name__ == "__main__":
    main()

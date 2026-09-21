"""Generate the Laya logo: SVG sources plus PNG renders.

The mark is one ring in two states. Half of it is an unbroken stroke; the other half breaks into
points that shrink and fade, and the smallest point sits next to where the stroke begins again --
so the eye closes the loop on its own.

`laya` (लय) is Sanskrit for dissolution. The mark reads as that cycle: becoming and dissolving,
neither one the end. It carries a second meaning specific to this project -- a continuous stroke
resolving into discrete points is what the model does, turning unstructured state into a typed
decision.

  python3 notebooks/make_logo.py   ->  assets/logo-mark.svg  logo-mark.png
                                       assets/logo-lockup.svg  logo-lockup.png
                                       assets/logo-mark-mono.svg
"""
import math
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(REPO, "assets")
os.makedirs(ASSETS, exist_ok=True)

INK = "#111111"
BLUE = "#2a78d6"
PAPER = "#eceef1"      # wordmark ink for dark backgrounds

SIZE = 64          # viewBox
CX = CY = 32.0
R = 21.0
STROKE = 5.0

# The stroke covers this sweep; the points cover the rest.
ARC_FROM, ARC_TO = 118.0, 300.0      # degrees, measured counter-clockwise from +x
N_DOTS = 10
DOT_MAX, DOT_MIN = 3.1, 0.85
OP_MAX, OP_MIN = 0.95, 0.42


def pt(deg, r=R):
    a = math.radians(deg)
    return CX + r * math.cos(a), CY - r * math.sin(a)      # SVG y grows downward


def arc_path():
    x0, y0 = pt(ARC_FROM)
    x1, y1 = pt(ARC_TO)
    sweep = (ARC_TO - ARC_FROM) % 360
    large = 1 if sweep > 180 else 0
    # sweep-flag 0 draws counter-clockwise in SVG's y-down space
    return "M %.3f %.3f A %.1f %.1f 0 %d 0 %.3f %.3f" % (x0, y0, R, R, large, x1, y1)


def dots(color):
    """Points leaving the stroke and spiralling inward, shrinking toward a single point.

    Inward rather than around: `laya` is absorption, and a ring that dissolves into its own
    centre says that where a ring of even dots would just read as a loading spinner. It is also
    what the model does -- a spread of options collapsing to one decision.
    """
    span = 268.0                     # how far the spiral travels before reaching the centre
    out = []
    for i in range(N_DOTS):
        f = (i + 1) / float(N_DOTS)
        deg = ARC_TO + span * f
        r = R * (1.0 - 0.93 * (f ** 1.08))          # ease inward, ending near the centre
        x, y = pt(deg, r)
        rr = DOT_MAX + (DOT_MIN - DOT_MAX) * (f ** 0.85)
        op = OP_MAX + (OP_MIN - OP_MAX) * (f ** 1.4)
        out.append('    <circle cx="%.3f" cy="%.3f" r="%.3f" fill="%s" opacity="%.3f"/>'
                   % (x, y, rr, color, op))
    out.append('    <circle cx="%.2f" cy="%.2f" r="2.35" fill="%s"/>' % (CX, CY, color))
    return "\n".join(out)


def mark_svg(color, use_current=False):
    c = "currentColor" if use_current else color
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d"
     width="%d" height="%d" role="img" aria-label="Laya">
  <title>Laya</title>
  <desc>An open stroke whose end breaks into points that spiral inward and shrink to a single
  point: becoming, then dissolving into one.</desc>
  <g fill="none" stroke="none">
    <path d="%s" stroke="%s" stroke-width="%.1f" stroke-linecap="round" opacity="0.95"/>
%s
  </g>
</svg>
''' % (SIZE, SIZE, SIZE, SIZE, arc_path(), c, STROKE, dots(c))


def lockup_svg(color, text=INK):
    """Mark plus wordmark, left aligned, for README headers.

    `text` is the wordmark ink -- dark for light backgrounds, near-white for dark ones, so the
    README header stays legible under either GitHub theme.
    """
    w, h = 252, 72
    s = 0.95                                   # mark scale inside the lockup
    tx, ty = 2, (h - SIZE * s) / 2
    return '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d"
     width="%d" height="%d" role="img" aria-label="Laya">
  <title>Laya</title>
  <g transform="translate(%.2f,%.2f) scale(%.3f)">
    <path d="%s" fill="none" stroke="%s" stroke-width="%.1f" stroke-linecap="round" opacity="0.95"/>
%s
  </g>
  <text x="70" y="%d" font-family="DejaVu Sans, Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="34" font-weight="600" letter-spacing="0.5" fill="%s">laya</text>
  <text x="71.5" y="%d" font-family="DejaVu Sans, Helvetica Neue, Helvetica, Arial, sans-serif"
        font-size="10.5" letter-spacing="2.6" fill="%s" opacity="0.62">DECISIONS, NOT TEXT</text>
</svg>
''' % (w, h, w, h, tx, ty, s, arc_path(), color, STROKE, dots(color), 42, text, 57, text)


def render(svg_path, png_path, width):
    subprocess.run(["rsvg-convert", "-w", str(width), "-a",
                    "-o", png_path, svg_path], check=True)
    return os.path.getsize(png_path)


def main():
    files = [
        ("logo-mark.svg", mark_svg(BLUE), "logo-mark.png", 512),
        ("logo-mark-ink.svg", mark_svg(INK), "logo-mark-ink.png", 512),
        ("logo-mark-mono.svg", mark_svg(None, use_current=True), None, None),
        ("logo-lockup.svg", lockup_svg(BLUE), "logo-lockup.png", 1040),
        ("logo-lockup-dark.svg", lockup_svg(BLUE, PAPER), "logo-lockup-dark.png", 1040),
    ]
    for name, svg, png, width in files:
        sp = os.path.join(ASSETS, name)
        with open(sp, "w") as f:
            f.write(svg)
        line = "  %-22s %5d B" % (name, os.path.getsize(sp))
        if png:
            size = render(sp, os.path.join(ASSETS, png), width)
            line += "   -> %-20s %6.1f KB" % (png, size / 1024)
        print(line)


if __name__ == "__main__":
    main()

"""Generate teaching diagrams using only the standard library."""
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def svg(body, width=900, height=400):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#fff"/>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#334155"/></marker></defs>
<g font-family="Arial, sans-serif" font-size="18" fill="#172033">{body}</g></svg>'''


def label(x, y, text, size=18, color="#172033"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}">{escape(text)}</text>'


def line(x1, y1, x2, y2, color="#334155", arrow=False):
    marker = ' marker-end="url(#arrow)"' if arrow else ''
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2"{marker}/>'


def residual():
    body = label(30, 35, "Residual: y = x + αF(x)", 24)
    body += '<rect x="270" y="145" width="170" height="65" rx="10" fill="#dbeafe" stroke="#2563eb"/>'
    body += label(310, 184, "αF(x)", 24)
    body += label(60, 184, "x", 24) + line(90, 177, 270, 177, arrow=True)
    body += line(440, 177, 620, 177, arrow=True)
    body += '<circle cx="645" cy="177" r="24" fill="#dcfce7" stroke="#15803d"/>'
    body += label(637, 185, "+", 26) + line(669, 177, 800, 177, arrow=True)
    body += label(815, 184, "y", 24)
    body += '<path d="M160 177 V85 H645 V150" fill="none" stroke="#334155" stroke-width="2" marker-end="url(#arrow)"/>'
    body += label(300, 75, "shortcut: identity")
    body += label(90, 275, "Backward: gₓ = gᵧ + α Jꜰᵀ gᵧ", 23)
    body += label(90, 315, "Два шляхи → сума внесків у градієнт входу.")
    body += label(90, 350, "Нульова гілка F(x) = 0 → вихід y = x.")
    return svg(body)


def depth():
    # Logarithms are computed analytically: no underflow in the drawing.
    from math import log10
    left, right, top, bottom = 100, 850, 75, 370
    def point(n, log_gradient):
        return left + n / 1000 * (right - left), bottom - (log_gradient + 320) / 520 * (bottom - top)
    body = label(30, 30, "Градієнт за входом: логарифмічна шкала", 24)
    for power in [-300, -200, -100, 0, 100, 200]:
        _, y = point(0, power)
        body += line(left, y, right, y, "#e2e8f0")
        body += label(20, y + 5, f"10^{power}", 14)
    for n in [0, 250, 500, 750, 1000]:
        x, _ = point(n, 0)
        body += label(x - 15, 395, str(n), 14)
    body += line(left, top, left, bottom) + line(left, bottom, right, bottom)
    for slope, color in [(log10(.5), "#2563eb"), (log10(1.5), "#dc2626"), (0, "#15803d")]:
        x0, y0 = point(0, 0)
        x1, y1 = point(1000, slope * 1000)
        body += line(x0, y0, x1, y1, color)
    body += label(120, 440, "Синій: plain w=.5 АБО residual w=−.5", 17, "#2563eb")
    body += label(120, 468, "Червоний: residual w=.5; зелений: residual w=0", 17)
    body += label(610, 395, "N (кількість блоків)", 16)
    return svg(body, height=500)


if __name__ == "__main__":
    (ROOT / "figures").mkdir(exist_ok=True)
    (ROOT / "figures/residual.svg").write_text(residual(), encoding="utf-8")
    (ROOT / "figures/depth.svg").write_text(depth(), encoding="utf-8")

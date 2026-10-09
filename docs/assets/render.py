"""Render the kernellib README assets: icon, logos and the two diagrams.

Every output is a plain SVG written next to this file. The icon shares the
gaussx and pyrox tile (a 120-unit rounded square, r = 26, white glyph) in the
kernellib teal gradient. Its glyph is a stationary kernel's Gram matrix:
bright on the diagonal, fading with distance from it.

Diagrams come in a light and a dark variant for GitHub's
``<picture>`` / ``prefers-color-scheme`` switch.

Run from the repo root:

    uv run --no-project python docs/assets/render.py
"""

from pathlib import Path


OUT = Path(__file__).parent

DOT = " \N{MIDDLE DOT} "
SANS = "ui-sans-serif, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace"

# Teal: the kernellib gradient and its wordmark accents.
C1, C2 = "#14B8A6", "#0284C7"
ACCENT = {"light": "#0F766E", "dark": "#5EEAD4"}

THEME = {
    "light": {
        "text": "#1F2937",
        "muted": "#6B7280",
        "card": "#F9FAFB",
        "line": "#D1D5DB",
        "arrow": "#9CA3AF",
        "hi_fill": "#F0FDFA",
        "hi_line": "#0D9488",
        "site": "#0F766E",
        "band": "#F3F4F6",
    },
    "dark": {
        "text": "#E5E7EB",
        "muted": "#9CA3AF",
        "card": "#111827",
        "line": "#374151",
        "arrow": "#6B7280",
        "hi_fill": "#06231F",
        "hi_line": "#14B8A6",
        "site": "#5EEAD4",
        "band": "#1F2937",
    },
}

W = "#FFFFFF"


# ---------------------------------------------------------------- glyph


def glyph_gram(gid: str) -> str:
    """A Gram matrix K_ij = k(x_i, x_j): a bright diagonal band that fades."""
    bands = [(26, 0.22), (16, 0.45), (8, 0.95)]  # half-width, opacity
    paths = "".join(
        f'<path d="M24 24 L{24 + w} 24 L104 {104 - w} L104 104 L{104 - w} 104 '
        f'L24 {24 + w} Z" fill="{W}" fill-opacity="{a}"/>'
        for w, a in bands
    )
    return (
        f'<defs><clipPath id="{gid}-clip">'
        '<rect x="24" y="24" width="80" height="80" rx="8"/></clipPath></defs>'
        f'<g clip-path="url(#{gid}-clip)">'
        f'<rect x="24" y="24" width="80" height="80" fill="{W}" fill-opacity="0.1"/>'
        f"{paths}</g>"
        '<rect x="24" y="24" width="80" height="80" rx="8" fill="none" '
        f'stroke="{W}" stroke-opacity="0.6" stroke-width="2.5"/>'
    )


def tile(gid: str, x: float = 0, y: float = 0, size: float = 128) -> str:
    """The gradient tile with the Gram glyph, placed at (x, y), ``size`` wide."""
    s = size / 128
    return (
        f'<defs><linearGradient id="{gid}" x1="0" y1="0" x2="1" y2="1">'
        f'<stop offset="0" stop-color="{C1}"/><stop offset="1" stop-color="{C2}"/>'
        "</linearGradient></defs>"
        f'<g transform="translate({x} {y}) scale({s:g})">'
        f'<rect x="4" y="4" width="120" height="120" rx="26" fill="url(#{gid})"/>'
        f"{glyph_gram(gid)}</g>"
    )


def svg(width: int, height: int, label: str, title: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{label}">\n'
        f"<title>{title}</title>\n{body}\n</svg>\n"
    )


def text(
    x: float,
    y: float,
    s: str,
    *,
    size: float = 14,
    weight: int = 400,
    fill: str,
    family: str = SANS,
    anchor: str = "start",
) -> str:
    # xml:space keeps leading spaces, so indented code lines stay indented.
    return (
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="{family}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" '
        f'xml:space="preserve">{s}</text>'
    )


def arrow_defs(mid: str, colour: str) -> str:
    return (
        f'<defs><marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0 0 L10 5 L0 10 z" fill="{colour}"/></marker></defs>'
    )


def line(points: list[tuple[float, float]], colour: str, head: str | None) -> str:
    d = " ".join(f"{'M' if i == 0 else 'L'}{x} {y}" for i, (x, y) in enumerate(points))
    end = f' marker-end="url(#{head})"' if head else ""
    return (
        f'<path d="{d}" fill="none" stroke="{colour}" stroke-width="1.6" '
        f'stroke-linejoin="round"{end}/>'
    )


def card(
    x: float,
    y: float,
    w: float,
    h: float,
    t: dict[str, str],
    *,
    highlight: bool = False,
) -> str:
    fill, stroke, sw = (
        (t["hi_fill"], t["hi_line"], 2) if highlight else (t["card"], t["line"], 1)
    )
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="{sw}"/>'
    )


# ---------------------------------------------------------------- icon and logo


def write_icon() -> None:
    (OUT / "icon.svg").write_text(svg(128, 128, "kernellib", "kernellib", tile("ic")))


def write_logos() -> None:
    for mode, t in THEME.items():
        body = tile(f"logo-{mode}", x=6, y=6) + (
            f'<text x="154" y="92" font-family="{SANS}" font-size="72" '
            f'font-weight="700" letter-spacing="-1.5" fill="{t["text"]}">'
            f'kernel<tspan fill="{ACCENT[mode]}">lib</tspan></text>'
        )
        (OUT / f"logo-{mode}.svg").write_text(
            svg(470, 140, "kernellib", "kernellib", body)
        )


# ---------------------------------------------------------------- hero


USES = [
    ("k(X1, X2)", "dense Gram, under jit" + DOT + "vmap" + DOT + "grad"),
    ("kl.to_operator(k, X)", "a gaussx operator, dense or implicit"),
    ("RandomFourierFeatures" + DOT + "NystromFeatures", "feature maps"),
    ("KRR" + DOT + "Falkon" + DOT + "EigenPro", "regression at scale"),
    ("hsic" + DOT + "cka" + DOT + "mmd_squared" + DOT + "KernelPCA", "dependence"),
]


def write_hero() -> None:
    """One kernel object fanning out to the five ways the package uses it."""
    width, height = 960, 356
    for mode, t in THEME.items():
        head = f"hero-{mode}-head"
        parts = [arrow_defs(head, t["arrow"])]

        mx, my, mw, mh = 20, 98, 330, 160
        parts.append(card(mx, my, mw, mh, t, highlight=True))
        parts.append(tile(f"hero-{mode}-icon", mx + 18, my + 16, 36))
        parts.append(
            text(
                mx + 64,
                my + 40,
                "k: kl.AbstractKernel",
                size=14.5,
                weight=700,
                fill=t["text"],
                family=MONO,
            )
        )
        code = [
            "k = kl.Matern(lengthscale=20.0, nu=1.5)",
            "k = k + kl.Periodic() * kl.Linear()",
            "dk = kl.Derivative(k, dx=0)",
        ]
        for i, s in enumerate(code):
            parts.append(
                text(mx + 18, my + 78 + 20 * i, s, size=12, fill=t["text"], family=MONO)
            )
        parts.append(
            text(
                mx + 18,
                my + 142,
                "composes" + DOT + "differentiates" + DOT + "a PyTree",
                size=12,
                weight=600,
                fill=t["site"],
                family=MONO,
            )
        )

        cy = my + mh / 2
        trunk_x, ex, ew, eh, gap = 400, 460, 480, 52, 14
        centres = [20 + eh / 2 + i * (eh + gap) for i in range(len(USES))]
        parts.append(line([(mx + mw, cy), (trunk_x, cy)], t["arrow"], None))
        parts.append(
            line([(trunk_x, centres[0]), (trunk_x, centres[-1])], t["arrow"], None)
        )
        for c in centres:
            parts.append(line([(trunk_x, c), (ex - 3, c)], t["arrow"], head))

        for (name, what), c in zip(USES, centres, strict=True):
            parts.append(card(ex, c - eh / 2, ew, eh, t))
            parts.append(
                text(
                    ex + 18,
                    c + 5,
                    name,
                    size=13.5,
                    weight=700,
                    fill=t["text"],
                    family=MONO,
                )
            )
            parts.append(
                text(ex + ew - 18, c + 5, what, size=13, fill=t["muted"], anchor="end")
            )

        label = (
            "One kernel object evaluates a Gram matrix, becomes a gaussx operator, "
            "approximates itself with feature maps, and feeds the regression and "
            "dependence methods."
        )
        (OUT / f"hero-{mode}.svg").write_text(
            svg(
                width,
                height,
                label,
                "One kernel, five uses",
                "\n".join(parts),
            )
        )


# ---------------------------------------------------------------- layers


PACKAGES = {
    # name: (x, y, w, description)
    "geonnax": (30, 20, 380, "basis functions" + DOT + "random-feature primitives"),
    "gaussx": (
        550,
        20,
        380,
        "structured operators" + DOT + "solvers" + DOT + "Gaussians",
    ),
    "kernellib": (240, 168, 480, ""),
    "pyrox-gp": (30, 330, 380, "GP priors and guides on kernellib kernels"),
    "pyrox-lgm": (550, 330, 380, "GMRF structure from kernellib graphs"),
}


def write_layers() -> None:
    """Where kernellib sits; arrows point at what each package imports."""
    width, height, h = 960, 484, 80
    for mode, t in THEME.items():
        head = f"layers-{mode}-head"
        a = t["arrow"]
        parts = [arrow_defs(head, a)]

        # kernellib → geonnax, kernellib → gaussx; pyrox-gp, pyrox-lgm → kernellib.
        parts.append(line([(330, 168), (330, 103)], a, head))
        parts.append(line([(630, 168), (630, 103)], a, head))
        parts.append(line([(330, 330), (330, 265)], a, head))
        parts.append(line([(630, 330), (630, 265)], a, head))
        for x, y, s in [
            (342, 140, "bases" + DOT + "random features"),
            (642, 140, "operators" + DOT + "solvers"),
            (342, 302, "kernels" + DOT + "spectral bases"),
            (642, 302, "graphs" + DOT + "structure matrices"),
        ]:
            parts.append(text(x, y, s, size=12, fill=t["muted"]))

        for name, (x, y, w, desc) in PACKAGES.items():
            core = name == "kernellib"
            hh = 94 if core else h
            parts.append(card(x, y, w, hh, t, highlight=core))
            if core:
                parts.append(tile(f"layers-{mode}-icon", x + 16, y + 23, 48))
                parts.append(
                    text(
                        x + 80,
                        y + 32,
                        name,
                        size=16,
                        weight=700,
                        fill=t["site"],
                        family=MONO,
                    )
                )
                for i, s in enumerate(
                    [
                        "kernels"
                        + DOT
                        + "operators"
                        + DOT
                        + "feature maps"
                        + DOT
                        + "KRR at scale",
                        "dependence (HSIC, CKA, MMD)"
                        + DOT
                        + "graphs"
                        + DOT
                        + "eigenmaps",
                    ]
                ):
                    parts.append(
                        text(x + 80, y + 56 + 20 * i, s, size=13, fill=t["text"])
                    )
            else:
                parts.append(
                    text(
                        x + 20,
                        y + 32,
                        name,
                        size=16,
                        weight=700,
                        fill=t["text"],
                        family=MONO,
                    )
                )
                parts.append(text(x + 20, y + 58, desc, size=13, fill=t["text"]))

        parts.append(
            f'<rect x="20" y="430" width="920" height="40" rx="10" fill="{t["band"]}"/>'
        )
        for x, head_word, names in [
            (44, "Foundations", ["jax", "equinox", "lineax", "jaxtyping", "einx"]),
            (500, "Optional", ["scikit-learn [sklearn]", "pynndescent [neighbors]"]),
        ]:
            parts.append(
                f'<text x="{x}" y="455" font-family="{SANS}" font-size="13" '
                f'fill="{t["muted"]}"><tspan font-weight="700" fill="{t["text"]}">'
                f"{head_word}</tspan>  {DOT.join(names)}</text>"
            )

        label = (
            "kernellib imports geonnax and gaussx; pyrox-gp and pyrox-lgm import "
            "kernellib. gaussx never imports kernellib, and kernellib never "
            "imports NumPyro."
        )
        (OUT / f"layers-{mode}.svg").write_text(
            svg(width, height, label, "Where kernellib sits", "\n".join(parts))
        )


if __name__ == "__main__":
    write_icon()
    write_logos()
    write_hero()
    write_layers()

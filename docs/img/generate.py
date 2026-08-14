"""Emit the light and dark turn-scope diagrams from one definition.

    python docs/img/generate.py docs/img /path/to/Inter-Regular.ttf

Two decisions that are not obvious from the output:

**Two files, not one with a media query.** GitHub sanitises `<style>` out of SVGs
embedded in markdown, so `@media (prefers-color-scheme: dark)` never runs. The supported
route is a `<picture>` with a `prefers-color-scheme` `<source>`, which needs two files.
Generating both from one definition is what stops them drifting.

**Inter is converted to outlines.** A sanitised SVG cannot load a webfont, so
`font-family="Inter"` renders as Inter only for a viewer who happens to have it
installed, and as something else for everyone else. Outlining bakes the shapes in. Code
literals stay live text in the system monospace stack, which is what the design system
itself uses for them, and which resolves everywhere without embedding.

Colours are design-system tokens. There is no dark palette and no warning token, so the
dark set is derived to hold the warm cast, and the two findings are separated using the
existing primary/secondary pair rather than by inventing a hue.
"""

import pathlib
import sys

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.ttLib import TTFont

MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"

LIGHT = dict(
    bg="#fef8f3",  # surface-bright
    card="#ffffff",  # surface-container-lowest
    outline="#d9d0c7",  # outline
    ink="#2C1810",  # on-surface
    muted="#58413e",  # on-surface-variant
    fault="#8b2219",  # primary
    note="#74584e",  # secondary
    ghost="#ded9d4",  # surface-dim
)
DARK = dict(
    bg="#1a120e",
    card="#241a15",
    outline="#4a3c34",
    ink="#f5ece5",
    muted="#b8a89e",
    fault="#ffb4a9",
    note="#c9b0a4",
    ghost="#3a2c25",
)

# x, micro-label, body, body-is-a-code-literal, emphasis
#   "note"  the refusal: the fact the reply contradicts, coloured throughout
#   "fault" the claim: border and label carry it while the body stays ink, so the card
#           reads as an ordinary reply that happens to be false, which is the point
STEPS = [
    (40, "GENERATION", "plans", False, None),
    (232, "TOOL", "search_orders", True, None),
    (424, "TOOL", '{"sent": false}', True, "note"),
    # Typographic quotes on purpose: this is rendered display copy, not code, and
    # Inter draws them properly. RUF001 flags them as ambiguous, which is the right
    # default for source and the wrong one for a string that becomes a picture.
    (620, "GENERATION", "“I’ve let them know.”", False, "fault"),  # noqa: RUF001
]


class Outliner:
    """Text to glyph outlines. Only what this diagram needs: no kerning, no shaping."""

    def __init__(self, ttf: str):
        font = TTFont(ttf)
        self.upem = font["head"].unitsPerEm
        self.cmap = font.getBestCmap()
        self.glyphs = font.getGlyphSet()
        self.hmtx = font["hmtx"]

    def _advance(self, ch: str) -> float:
        name = self.cmap.get(ord(ch))
        return self.hmtx[name][0] if name else self.upem / 2

    def width(self, text: str, size: float, tracking: float = 0.0) -> float:
        units = sum(self._advance(c) for c in text)
        units += tracking * self.upem * max(len(text) - 1, 0)
        return units * size / self.upem

    def path(self, text, size, x, y, fill, anchor="middle", tracking=0.0) -> str:
        scale = size / self.upem
        if anchor == "middle":
            x -= self.width(text, size, tracking) / 2
        parts, cursor = [], 0.0
        for ch in text:
            name = self.cmap.get(ord(ch))
            if name:
                pen = SVGPathPen(self.glyphs)
                self.glyphs[name].draw(pen)
                if d := pen.getCommands():
                    parts.append(
                        f'<path transform="translate({cursor:.0f} 0)" d="{d}"/>'
                    )
            cursor += self._advance(ch) + tracking * self.upem
        # scale(s, -s) flips the font's y-up outlines into the SVG's y-down space, which
        # puts the baseline exactly on y.
        return (
            f'<g transform="translate({x:.2f} {y:.2f}) scale({scale:.6f} {-scale:.6f})"'
            f' fill="{fill}">{"".join(parts)}</g>'
        )


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render(c: dict, ol: Outliner) -> str:
    o = [
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 820 300" width="820"'
        ' height="300" role="img">',
        "<title>One turn of four steps. Each step passes when scored on its own; the"
        " failure is the relationship between the tool that declined and the reply that"
        " claims it succeeded.</title>",
        f'<rect x="0" y="0" width="820" height="300" fill="{c["bg"]}"/>',
        # The bracket is the whole point: the finding spans two steps.
        f'<path d="M 424 74 L 424 58 L 780 58 L 780 74" fill="none"'
        f' stroke="{c["fault"]}" stroke-width="1.25"/>',
        f'<text x="602" y="44" font-family="{MONO}" font-size="11.5"'
        f' fill="{c["fault"]}" text-anchor="middle" letter-spacing="0.06em">'
        "UNVERIFIED_CLAIM</text>",
    ]
    for x, kind, body, literal, emphasis in STEPS:
        stroke = c[emphasis] if emphasis else c["outline"]
        o.append(
            f'<rect x="{x}" y="106" width="160" height="72" rx="6" fill="{c["card"]}"'
            f' stroke="{stroke}" stroke-width="{1.25 if emphasis else 0.75}"/>'
        )
        # Uppercase micro-labels are Inter with tracking, per the type rules.
        o.append(
            ol.path(
                kind,
                9.5,
                x + 80,
                134,
                c[emphasis] if emphasis else c["muted"],
                tracking=0.09,
            )
        )
        fill = c["note"] if emphasis == "note" else c["ink"]
        if literal:
            o.append(
                f'<text x="{x + 80}" y="157" font-family="{MONO}" font-size="12"'
                f' fill="{fill}" text-anchor="middle">{esc(body)}</text>'
            )
        else:
            o.append(ol.path(body, 12, x + 80, 157, fill))
    for x in (204, 396, 588):
        o.append(
            f'<path d="M {x} 142 L {x + 20} 142" stroke="{c["outline"]}"'
            ' stroke-width="1"/>'
        )
    for x, *_ in STEPS:
        o.append(
            f'<rect x="{x}" y="210" width="160" height="38" rx="6" fill="none"'
            f' stroke="{c["ghost"]}" stroke-width="1" stroke-dasharray="3 4"/>'
        )
        # Drawn rather than set: Inter has no U+2713, so a glyph would silently vanish.
        o.append(
            f'<path d="M {x + 74} 229 l 3.5 3.5 l 7 -8" fill="none"'
            f' stroke="{c["muted"]}" stroke-width="1.4" stroke-linecap="round"'
            ' stroke-linejoin="round"/>'
        )
    o.append(
        ol.path(
            "Scored one observation at a time, every step passes.",
            12.5,
            410,
            278,
            c["muted"],
        )
    )
    o.append("</svg>")
    return "\n".join(o) + "\n"


def main() -> None:
    out = pathlib.Path(sys.argv[1])
    ol = Outliner(sys.argv[2])
    (out / "turn-scope-light.svg").write_text(render(LIGHT, ol))
    (out / "turn-scope-dark.svg").write_text(render(DARK, ol))
    print("wrote turn-scope-light.svg and turn-scope-dark.svg")


if __name__ == "__main__":
    main()

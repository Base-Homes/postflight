# Diagram sources

`turn-scope-light.svg` and `turn-scope-dark.svg` are **generated**. Their text is
converted to outlines, so they are tens of kilobytes of path data and cannot be usefully
hand-edited. Change `generate.py` and re-run it instead.

```bash
pip install -e ".[docs]"
curl -sL -o Inter.ttf \
  "https://github.com/google/fonts/raw/main/ofl/inter/Inter%5Bopsz%2Cwght%5D.ttf"
python docs/img/generate.py docs/img Inter.ttf
```

That is the variable Inter. `generate.py` pins its axes, because a variable font's
default master does not necessarily match the static release of the same weight. With
the pin, this reproduces the committed files byte for byte.

Both files come from one definition, which is what keeps the light and dark versions
from drifting apart. `generate.py`'s docstring explains why there are two files and why
the text is outlined; both are consequences of how GitHub renders SVGs in markdown.

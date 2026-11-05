# QR artwork for KiCad

From the repository root, using Python 3.9 or newer:

```sh
python -m pip install -r circuit/requirements.txt
python circuit/qr_code.py "https://example.com" --size-mm 20 -o circuit/qr.svg
```

Replace the URL with your own text or link. `--size-mm` sets the total square
size, **including** the four-module quiet zone. Error correction defaults to M;
use `--error-correction L`, `M`, `Q`, or `H` to change it. Longer payloads and
higher error correction can require more modules at the same physical size.
The command reports the resulting module width; check it against your board
manufacturer's minimum silkscreen feature size.

The two steps can also be used from Python (with `circuit` on the import path):

```python
from pathlib import Path
from qr_code import generate_qr, matrix_to_svg

matrix = generate_qr("https://example.com")
Path("qr.svg").write_text(matrix_to_svg(matrix, size_mm=20), encoding="utf-8")
```

This converts the generated QR matrix, not an existing PNG or JPEG. There is no
image tracing or Pillow dependency. SVGs contain only filled, unstroked
rectangles, with adjacent dark modules merged horizontally.

## Import into KiCad

1. In the PCB Editor, select **File > Import > Graphics** and choose the SVG.
2. Select your intended layer (usually **F.Silkscreen**) and keep the import
   scale at **1.0**. Check the physical size after placement.
3. Keep the entire blank margin clear of text, pads, tracks, and other artwork.
   The SVG has **no white background**: blank modules and the quiet zone do not
   create any objects or enforce a KiCad keepout.
4. Inspect the Gerber output and test scanning a sample at actual size, ideally
   on the finished board. Avoid tiny modules; shorten the URL or enlarge the
   QR code if needed.

SVG black means "draw on the selected layer", not black ink. White silkscreen
on a dark board reverses the intended QR contrast and may not scan reliably
with all readers. Prefer a dark-on-light process where possible and test the
actual finish. Copper and solder-mask layers have different fabrication
effects; importing this file does not create pads or mask openings for you.

See the [KiCad PCB Editor manual](https://docs.kicad.org/9.0/en/pcbnew/pcbnew.html)
for graphics import options.

## Tests

```sh
python -m unittest discover -s circuit -p 'test_qr_code.py'
```

#!/usr/bin/env python3
"""Generate a QR matrix and export filled SVG geometry for KiCad."""

import argparse
import math
from pathlib import Path
from xml.etree import ElementTree as ET

import qrcode
from qrcode.exceptions import DataOverflowError


ERROR_CORRECTION = {
    "L": qrcode.constants.ERROR_CORRECT_L,
    "M": qrcode.constants.ERROR_CORRECT_M,
    "Q": qrcode.constants.ERROR_CORRECT_Q,
    "H": qrcode.constants.ERROR_CORRECT_H,
}


def generate_qr(data: str, error_correction: str = "M") -> list[list[bool]]:
    """Return a QR matrix, including a four-module blank border."""
    if not data:
        raise ValueError("QR data must not be empty")
    if error_correction not in ERROR_CORRECTION:
        raise ValueError("Error correction must be L, M, Q, or H")
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECTION[error_correction],
        box_size=1,
        border=4,
    )
    qr.add_data(data)
    qr.make(fit=True)
    return [[bool(cell) for cell in row] for row in qr.get_matrix()]


def matrix_to_svg(matrix: list[list[bool]], size_mm: float = 20.0) -> str:
    """Render a square QR matrix; size includes any border in the matrix."""
    if not math.isfinite(size_mm) or size_mm <= 0:
        raise ValueError("SVG size must be a finite, positive number")
    count = len(matrix)
    if not count or any(len(row) != count for row in matrix):
        raise ValueError("QR matrix must be nonempty and square")
    if any(not isinstance(cell, bool) for row in matrix for cell in row):
        raise ValueError("QR matrix cells must be booleans")

    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "width": f"{size_mm}mm",
            "height": f"{size_mm}mm",
            "viewBox": f"0 0 {count} {count}",
        },
    )
    for y, row in enumerate(matrix):
        x = 0
        while x < count:
            if not row[x]:
                x += 1
                continue
            start = x
            while x < count and row[x]:
                x += 1
            ET.SubElement(
                svg,
                "rect",
                {
                    "x": str(start),
                    "y": str(y),
                    "width": str(x - start),
                    "height": "1",
                    "fill": "#000000",
                    "stroke": "none",
                },
            )
    return ET.tostring(svg, encoding="unicode", xml_declaration=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", help="Text or URL to encode")
    parser.add_argument("-o", "--output", type=Path, default=Path("qr.svg"))
    parser.add_argument(
        "--size-mm", type=float, default=20.0,
        help="Total SVG width and height in mm, including quiet zone (default: 20)",
    )
    parser.add_argument(
        "--error-correction", choices=ERROR_CORRECTION, default="M",
        help="QR error correction level (default: M)",
    )
    args = parser.parse_args()
    try:
        matrix = generate_qr(args.data, args.error_correction)
        svg = matrix_to_svg(matrix, args.size_mm)
        args.output.write_text(svg + "\n", encoding="utf-8")
    except (ValueError, DataOverflowError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(
        f"Saved {args.output}: {args.size_mm:g} mm square, "
        f"{args.size_mm / len(matrix):.3f} mm per module (including quiet zone)."
    )


if __name__ == "__main__":
    main()

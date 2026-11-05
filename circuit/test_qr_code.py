import contextlib
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

from qrcode.exceptions import DataOverflowError

from qr_code import generate_qr, main, matrix_to_svg


class QRCodeTests(unittest.TestCase):
    def test_quiet_zone_and_finder_pattern(self):
        matrix = generate_qr("https://example.com")
        count = len(matrix)
        self.assertEqual((count - 8 - 21) % 4, 0)
        for i in range(4):
            self.assertFalse(any(matrix[i]))
            self.assertFalse(any(matrix[-1 - i]))
            self.assertFalse(any(row[i] or row[-1 - i] for row in matrix))
        self.assertTrue(all(matrix[4][4:11]))
        self.assertEqual(matrix[5][4:11], [True, False, False, False, False, False, True])

    def test_error_correction_levels_and_unicode(self):
        for level in ("L", "M", "Q", "H"):
            with self.subTest(level=level):
                matrix = generate_qr("Hello \u4e16\u754c", level)
                self.assertTrue(all(len(row) == len(matrix) for row in matrix))

    def test_invalid_payload(self):
        with self.assertRaises(ValueError):
            generate_qr("")
        with self.assertRaises(ValueError):
            generate_qr("hello", "invalid")
        with self.assertRaises((ValueError, DataOverflowError)):
            generate_qr("x" * 10000)

    def test_svg_exactly_matches_matrix(self):
        matrix = generate_qr("https://example.com")
        svg = ET.fromstring(matrix_to_svg(matrix, 25.0))
        self.assertEqual(svg.attrib["width"], "25.0mm")
        self.assertEqual(svg.attrib["height"], "25.0mm")
        self.assertEqual(svg.attrib["viewBox"], f"0 0 {len(matrix)} {len(matrix)}")
        rendered = [[False] * len(matrix) for _ in matrix]
        for rect in svg:
            self.assertEqual(rect.tag, "{http://www.w3.org/2000/svg}rect")
            self.assertEqual(rect.attrib["fill"], "#000000")
            self.assertEqual(rect.attrib["stroke"], "none")
            self.assertEqual(rect.attrib["height"], "1")
            x, y = int(rect.attrib["x"]), int(rect.attrib["y"])
            for dx in range(int(rect.attrib["width"])):
                self.assertFalse(rendered[y][x + dx])
                rendered[y][x + dx] = True
        self.assertEqual(rendered, matrix)

    def test_horizontal_runs_are_merged(self):
        svg = ET.fromstring(matrix_to_svg([[True, True], [False, True]]))
        self.assertEqual(len(svg), 2)
        self.assertEqual(svg[0].attrib["width"], "2")

    def test_invalid_size(self):
        for size in (0, -1, float("inf"), float("-inf"), float("nan")):
            with self.subTest(size=size), self.assertRaises(ValueError):
                matrix_to_svg([[True]], size)

    def test_invalid_matrix(self):
        for matrix in ([], [[]], [[True, False]], [[1]]):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                matrix_to_svg(matrix)

    def test_cli_writes_svg(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "qr.svg"
            argv = ["qr_code.py", "https://example.com", "-o", str(output), "--size-mm", "30"]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertEqual(ET.parse(output).getroot().attrib["width"], "30.0mm")

    def test_cli_reports_errors_without_success(self):
        with tempfile.TemporaryDirectory() as directory:
            for data, output, size in (
                ("", Path(directory) / "empty.svg", "20"),
                ("x" * 10000, Path(directory) / "oversized.svg", "20"),
                ("hello", Path(directory) / "bad-size.svg", "nan"),
                ("hello", Path(directory) / "missing" / "qr.svg", "20"),
            ):
                stdout, stderr = io.StringIO(), io.StringIO()
                argv = ["qr_code.py", data, "-o", str(output), "--size-mm", size]
                with (
                    self.subTest(data=data, size=size),
                    patch.object(sys, "argv", argv),
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                    self.assertRaises(SystemExit) as result,
                ):
                    main()
                self.assertEqual(result.exception.code, 2)
                self.assertIn("error:", stderr.getvalue())
                self.assertEqual(stdout.getvalue(), "")
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

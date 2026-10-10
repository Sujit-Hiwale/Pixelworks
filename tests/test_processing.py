import sys
import types
import unittest
from io import BytesIO
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image
from PIL.TiffImagePlugin import IFDRational
from pypdf import PdfReader

from image_toolkit.downloads import make_zip
from image_toolkit.pdf import images_to_pdf
from image_toolkit.metadata import get_image_dpi, set_dpi_metadata
from image_toolkit.processing import (
    MAX_BATCH, MAX_BATCH_PIXELS, MAX_PIXELS, compress_to_target, crop_image, encode_image,
    load_image, make_contact_sheet, process_batch, resize_image, rotate_custom, rotate_image,
    safe_filename, unique_filenames,
)


class ProcessingTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (120, 80), "#336699")

    def test_encode_roundtrip_and_alpha_transparency(self):
        for fmt in ("JPEG", "PNG", "WEBP"):
            with self.subTest(fmt=fmt):
                payload = encode_image(self.image, fmt, 80)
                decoded = load_image(payload)
                self.assertEqual(decoded.size, self.image.size)
                self.assertEqual(decoded.format, fmt)
        transparent = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
        decoded = load_image(encode_image(transparent, "PNG"))
        self.assertEqual(decoded.mode, "RGBA")
        self.assertEqual(decoded.getpixel((0, 0))[3], 0)
        palette_transparent = Image.new("P", (8, 8), 0)
        palette_transparent.putpalette([0,0,0, 255,0,0] + [0,0,0]*254)
        palette_transparent.info["transparency"] = 0
        decoded_palette = load_image(encode_image(palette_transparent, "PNG"))
        self.assertEqual(decoded_palette.getpixel((0,0))[3], 0)
        self.assertEqual(load_image(encode_image(palette_transparent, "JPEG")).getpixel((0,0)), (255,255,255))

    def test_resize_crop_bounds_and_rotation(self):
        self.assertEqual(resize_image(self.image, 60, 40).size, (60, 40))
        self.assertEqual(crop_image(self.image, 2, 1, 90, 70).size, (88, 69))
        with self.assertRaises(ValueError): crop_image(self.image, -1, 1, 4, 5)
        with self.assertRaises(ValueError): crop_image(self.image, 9, 2, 3, 4)
        self.assertEqual(rotate_image(self.image, 90).size, (80, 120))
        self.assertEqual(rotate_custom(self.image, -90, True).size, (80, 120))
        self.assertEqual(rotate_custom(self.image, 90, False).size, self.image.size)
        with self.assertRaises(ValueError): rotate_custom(self.image, 181)
        class LargeImage:
            width = height = 4000
            def rotate(self, *args, **kwargs): raise AssertionError("limit should be checked before allocation")
        with self.assertRaisesRegex(ValueError, "30 megapixel"):
            rotate_custom(LargeImage(), 45)

    def test_target_compression_measures_actual_size_and_fallback(self):
        result = compress_to_target(self.image, "JPEG", 100_000)
        self.assertTrue(result["met_target"])
        self.assertLessEqual(len(result["data"]), 100_000)
        impossible = compress_to_target(self.image, "JPEG", 1)
        self.assertFalse(impossible["met_target"])
        self.assertGreater(len(impossible["data"]), 1)
        png = compress_to_target(self.image, "PNG", 1)
        self.assertIsNone(png["quality"])
        self.assertFalse(png["met_target"])

    def test_target_search_measures_nearby_sizes_and_preserves_alpha_formats(self):
        transparent = Image.new("RGBA", (120, 80), (20, 160, 220, 0))
        transparent.putalpha(128)
        jpeg = compress_to_target(transparent, "JPEG", 20_000)
        webp = compress_to_target(transparent, "WEBP", 20_000)
        for result, expected_mode in ((jpeg, "RGB"), (webp, "RGBA")):
            with Image.open(BytesIO(result["data"])) as decoded:
                self.assertEqual(decoded.mode, expected_mode)
                self.assertEqual(decoded.size, result["dimensions"])
            self.assertEqual(result["actual_bytes"], len(result["data"]))
            self.assertEqual(result["difference_bytes"], len(result["data"]) - result["target_bytes"])
        png = compress_to_target(transparent, "PNG", 2_000)
        with Image.open(BytesIO(png["data"])) as decoded:
            self.assertEqual(decoded.mode, "RGBA")
            self.assertEqual(decoded.size, png["dimensions"])

    def test_900kb_jpeg_targets_use_measured_quality_and_downsampling(self):
        source = Image.fromarray(np.random.default_rng(21).integers(0, 256, (900, 1400, 3), dtype=np.uint8))
        original = encode_image(source, "JPEG", 85)
        self.assertGreater(len(original), 850_000)
        for target_kb in (500, 100, 50, 20, 10, 5):
            target = target_kb * 1024
            result = compress_to_target(source, "JPEG", target)
            with self.subTest(target_kb=target_kb):
                self.assertEqual(len(result["data"]), result["actual_bytes"])
                self.assertEqual(result["difference_bytes"], len(result["data"]) - target)
                self.assertEqual(result["format"], "JPEG")
                with Image.open(BytesIO(result["data"])) as decoded:
                    self.assertEqual(decoded.size, result["dimensions"])
                    self.assertAlmostEqual(decoded.width / decoded.height, source.width / source.height, delta=.01)
                self.assertEqual(result["met_target"], len(result["data"]) <= target)
                self.assertLess(abs(result["difference_bytes"]), max(80, int(target * .03)))

    def test_batch_continues_after_failure(self):
        def operation(image):
            if image.width == 7: raise ValueError("bad one")
            return image.copy()
        items = [("ok", self.image), ("bad", Image.new("RGB", (7, 8))), ("ok2", self.image)]
        success, failures = process_batch(items, operation)
        self.assertEqual([name for name, _ in success], ["ok", "ok2"])
        self.assertEqual(failures[0][0], "bad")

    def test_contact_sheet_settings_control_size_and_labels(self):
        with_names = make_contact_sheet([("one.jpg", self.image)], 1, include_filenames=True, font_size=18, border_thickness=4, spacing=9)
        without_names = make_contact_sheet([("one.jpg", self.image)], 1, include_filenames=False, border_thickness=0, spacing=0)
        self.assertGreater(with_names.height, without_names.height)
        self.assertEqual(without_names.size, (240, 200))
        self.assertEqual(without_names.getpixel((0,0)), (240,241,247))

    def test_zip_integrity_and_filename_collisions(self):
        payload = make_zip([("../same.png", b"one"), ("same.png", b"two")])
        with ZipFile(BytesIO(payload)) as archive:
            self.assertTrue(archive.testzip() is None)
            self.assertEqual(archive.namelist(), ["same.png", "same_2.png"])
            self.assertEqual(archive.read("same_2.png"), b"two")
        self.assertEqual(safe_filename("../../same image.png"), "same_image.png")
        self.assertEqual(unique_filenames(["same.png", "SAME.PNG"]), ["same.png", "SAME_2.PNG"])
        self.assertEqual(unique_filenames(["x.png", "x.png", "x_2.png"]), ["x.png", "x_2.png", "x_2_2.png"])

    def test_corrupted_image_and_pixel_limit(self):
        with self.assertRaises(ValueError): load_image(b"not an image")
        too_large = Image.new("RGB", (7000, 6000), "white")
        data = BytesIO(); too_large.save(data, format="PNG")
        with self.assertRaises(ValueError): load_image(data.getvalue())
        self.assertEqual(MAX_BATCH, 10)
        self.assertEqual(MAX_PIXELS, 30_000_000)
        self.assertEqual(MAX_BATCH_PIXELS, 60_000_000)

    def test_pdf_original_pages_keep_individual_dimensions_and_transparency(self):
        first = Image.new("P", (120, 80), 0)
        first.putpalette([0,0,0, 255,0,0] + [0,0,0]*254)
        first.info["transparency"] = 0
        second = Image.new("RGB", (40, 160), "#ff0000")
        payload = images_to_pdf([first, second], page_size="Original")
        reader = PdfReader(BytesIO(payload))
        self.assertEqual(len(reader.pages), 2)
        self.assertGreater(float(reader.pages[0].mediabox.width), float(reader.pages[1].mediabox.width))
        self.assertAlmostEqual(float(reader.pages[0].mediabox.width), 120 * 72 / 96)
        self.assertAlmostEqual(float(reader.pages[0].mediabox.height), 80 * 72 / 96)
        self.assertAlmostEqual(float(reader.pages[1].mediabox.width), 40 * 72 / 96)
        self.assertAlmostEqual(float(reader.pages[1].mediabox.height), 160 * 72 / 96)
        for page, expected in zip(reader.pages, [(120, 80), (40, 160)]):
            image = page.images[0].image
            self.assertEqual(image.size, expected)

    def test_pdf_original_uses_dpi_metadata_for_physical_page_size(self):
        image = Image.new("RGB", (300, 200), "red")
        image.info["dpi"] = (300, 300)
        page = PdfReader(BytesIO(images_to_pdf([image], page_size="Original"))).pages[0]
        self.assertAlmostEqual(float(page.mediabox.width), 300 * 72 / 300)
        self.assertAlmostEqual(float(page.mediabox.height), 200 * 72 / 300)

    def test_dpi_metadata_presets_keep_pixel_dimensions_and_content(self):
        original = Image.new("RGB", (31, 19), "#2458a0")
        for fmt, extension in [("JPEG", "jpg"), ("PNG", "png"), ("WEBP", "webp"),
                               ("BMP", "bmp"), ("TIFF", "tif")]:
            for dpi in (72, 150, 300, 600, 137):
                with self.subTest(fmt=fmt, dpi=dpi):
                    source = BytesIO()
                    original.save(source, format=fmt)
                    changed = set_dpi_metadata(source.getvalue(), dpi)
                    with Image.open(BytesIO(source.getvalue())) as before, Image.open(BytesIO(changed)) as after:
                        self.assertEqual(before.size, after.size)
                        self.assertTrue(np.array_equal(np.asarray(before), np.asarray(after)))
                        read_dpi = get_image_dpi(after)
                        self.assertIsNotNone(read_dpi)
                        self.assertAlmostEqual(read_dpi[0], dpi, delta=.03)
                        self.assertAlmostEqual(read_dpi[1], dpi, delta=.03)
        gif = BytesIO(); original.save(gif, format="GIF")
        with self.assertRaisesRegex(ValueError, "GIF"):
            set_dpi_metadata(gif.getvalue(), 300)
        exif = Image.Exif(); exif[282] = IFDRational(72, 1); exif[283] = IFDRational(96, 1); exif[296] = 2
        jpeg_with_exif = BytesIO(); original.save(jpeg_with_exif, format="JPEG", exif=exif)
        changed = set_dpi_metadata(jpeg_with_exif.getvalue(), 600)
        with Image.open(BytesIO(changed)) as decoded:
            self.assertAlmostEqual(float(decoded.getexif()[282]), 600)
            self.assertAlmostEqual(float(decoded.getexif()[283]), 600)

    def test_ocr_missing_package_has_actionable_error(self):
        from image_toolkit.ocr import extract_text
        with patch.dict(sys.modules, {"pytesseract": None}):
            with self.assertRaisesRegex(RuntimeError, "pytesseract"):
                extract_text(self.image)

    def test_ocr_missing_engine_is_reported(self):
        from image_toolkit.ocr import extract_text
        class TesseractNotFoundError(Exception): pass
        fake = types.ModuleType("pytesseract")
        fake.TesseractNotFoundError = TesseractNotFoundError
        fake.Output = types.SimpleNamespace(DICT="dict")
        fake.image_to_data = lambda *a, **k: (_ for _ in ()).throw(TesseractNotFoundError("missing"))
        with patch.dict(sys.modules, {"pytesseract": fake}):
            with self.assertRaisesRegex(RuntimeError, "system tesseract-ocr"):
                extract_text(self.image)

    def test_ocr_missing_language_data_is_reported(self):
        from image_toolkit.ocr import extract_text
        fake = types.ModuleType("pytesseract")
        fake.TesseractNotFoundError = type("TesseractNotFoundError", (Exception,), {})
        fake.Output = types.SimpleNamespace(DICT="dict")
        fake.image_to_data = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("Error opening data file /usr/share/tessdata/fra.traineddata"))
        with patch.dict(sys.modules, {"pytesseract": fake}):
            with self.assertRaisesRegex(RuntimeError, "language data for 'fra'"):
                extract_text(self.image, language="fra")

    def test_perspective_transform_rejects_missing_opencv(self):
        from image_toolkit.scanner import perspective_transform, preview_to_source
        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.INTER_CUBIC = 2
        fake_cv2.BORDER_REPLICATE = 1
        fake_cv2.getPerspectiveTransform = lambda points, target: np.eye(3)
        fake_cv2.warpPerspective = lambda array, matrix, size, **kwargs: np.zeros((size[1], size[0], 3), dtype=np.uint8)
        with patch.dict(sys.modules, {"cv2": fake_cv2}):
            output = perspective_transform(self.image, [[0,0],[119,0],[119,79],[0,79]])
        self.assertEqual(output.size, self.image.size)
        self.assertEqual(preview_to_source((425, 283.5), (850, 567), (1200, 800)), [600.0, 400.0])
        with self.assertRaises(ValueError):
            perspective_transform(self.image, [[-1,0],[119,0],[119,79],[0,79]])

    def test_scanner_corner_validation_and_order(self):
        from image_toolkit.scanner import order_points, validate_corners, default_corners
        expected = [[10, 10], [110, 10], [110, 70], [10, 70]]
        self.assertEqual(order_points([[110,70],[10,10],[10,70],[110,10]]).tolist(), expected)
        self.assertEqual(validate_corners(expected, self.image.size).shape, (4, 2))
        self.assertEqual(len(default_corners(self.image.size)), 4)
        with self.assertRaisesRegex(ValueError, "inside"):
            validate_corners([[-1,10],[110,10],[110,70],[10,70]], self.image.size)
        with self.assertRaisesRegex(ValueError, "cross"):
            validate_corners([[10,10],[110,70],[110,10],[10,70]], self.image.size)
        with self.assertRaisesRegex(ValueError, "different"):
            validate_corners([[10,10],[110,10],[110,10],[10,70]], self.image.size)

if __name__ == "__main__":
    unittest.main()

# Pixelworks — All-in-One Image Toolkit

Pixelworks is a free, responsive Streamlit app for common image tasks. Pillow handles image processing in the app process; the app does not send images to external APIs. The Color Palette Extractor was removed from the project.

## Tools

- **Convert & compress:** batch conversion between JPEG, PNG and WebP; automatic compression; measured target-size optimization that searches actual encoded bytes and downsizes proportionally only when needed.
- **Resize:** batch percentage or target-dimension resizing, with optional aspect-ratio preservation.
- **Crop, rotate & flip:** draggable crop rectangle with aspect presets when `streamlit-cropper` is installed, precise pixel bounds, shared batch crop, clockwise and anticlockwise quarter turns, custom angles, and horizontal/vertical flips. Positive custom angles are clockwise. Rotation expands the canvas by default; turn that option off to crop to the original canvas.
- **Adjust image:** batch brightness, contrast, saturation and sharpness adjustments.
- **Image details:** dimensions, format, file size, non-location EXIF, and metadata-free downloads.
- **DPI Setter:** update horizontal and vertical resolution metadata without changing pixel dimensions; JPEG, PNG, WebP, BMP and TIFF are supported where metadata can be stored.
- **Contact sheet:** configurable columns, filenames/font size, thumbnail border, spacing and background; PNG export.
- **Document scanner:** staged OpenCV candidate generation from contrast edges, adaptive regions and Hough line groups, with a seeded GrabCut fallback for foreground regions; confidence/uncertainty is shown and crops are never applied automatically. A lightweight local Streamlit component provides four genuinely draggable corners, live perspective preview, reset/automatic-candidate controls and PNG/JPEG download. Review images one at a time and collect corrected outputs into a ZIP.
- **Image to PDF:** selected image order, one page per image, Original/A4/Letter page sizes, and optional aspect-preserving fit for fixed page sizes. Original is the default and preserves each image's own raster size; its PDF physical size uses stored DPI or 96 DPI when missing.
- **OCR text extractor:** local Tesseract recognition for batches, editable text, TXT downloads/ZIP, detected word regions, grayscale, threshold, scale and optional deskew.

Previews can be clicked to open a larger view and closed without leaving the tool. Individual outputs are downloadable; batch operations provide ZIP downloads containing successful results. A batch continues when one image fails.

## Local installation and startup

Python 3.10–3.12 is recommended. OpenCV, Pillow and Streamlit are pinned in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL Streamlit prints. The default `packages.txt` also requests the Tesseract executable and English language data on Streamlit Community Cloud. For a local system:

```bash
# Debian/Ubuntu
sudo apt-get install tesseract-ocr tesseract-ocr-eng
# macOS
brew install tesseract
```

For other OCR languages, install the matching Tesseract language data (for example, `tesseract-ocr-fra`) and enter its language code such as `fra` in the OCR tool. The Python wrapper is pinned in `requirements.txt`; Tesseract and language data are system packages, not pip packages.

## Streamlit Community Cloud deployment

1. Push the project to a GitHub repository.
2. Create an app in [Streamlit Community Cloud](https://share.streamlit.io/), select the repository and branch, and set `app.py` as the entry point.
3. The deploy reads `requirements.txt`, `.streamlit/config.toml`, and `packages.txt`.

OCR requires the OS executable and selected language files. `packages.txt` includes Tesseract plus English. Community Cloud's available system packages and resource limits may change; OCR displays a direct error if the executable or language data is unavailable. Document scanning depends on CPU OpenCV. Large images and high concurrency may exceed a free instance's memory budget.

## Supported formats and limits

- Uploads: JPEG/JPG, PNG, WebP, BMP, TIFF and GIF (first frame only).
- Conversion outputs: JPEG, PNG and WebP. PNG is lossless and does not expose a quality control. JPEG cannot represent transparency; the interface lets users choose a flattening background.
- Upload caps: 20 MB per image, 80 MB total per batch, 10 images per batch, 30 megapixels per image and 60 megapixels total across a batch. The Streamlit server's configured request cap is also 20 MB per file.
- Target-size compression measures each final encoding. JPEG/WebP binary-search compression quality and, only if needed, search proportional LANCZOS downscales. PNG uses lossless encoding and downsizes only when necessary. The displayed status compares actual bytes with the target; an output larger than the target is labeled as the closest measured result.
- DPI Setter changes metadata only: it does not create pixels. JPEG density and PNG physical-pixel chunks are patched without re-encoding pixel data; BMP resolution fields are patched. WebP uses lossless encoding with EXIF resolution tags, and TIFF is reserialized without resampling. GIF has no standard DPI metadata and is not supported by this tool.
- Image to PDF Original mode creates an individually sized page per image, using its own DPI metadata. Images without DPI use the documented 96 DPI default. Fixed A4/Letter pages remain optional.
- Document detection is heuristic and CPU-only. It combines contrast/edge maps, adaptive threshold regions, contour/rotated-region candidates and Hough line groups. Confidence is a geometric and edge-evidence score, not semantic document classification. A page indistinguishable from its background cannot be identified reliably; verify the overlay and drag the handles or use Manual mode. The manual editor maps displayed points back to source pixels and validates crossed, degenerate and out-of-bounds selections. The four-point transform uses OpenCV homography.
- OCR recognition quality depends on image clarity, language packs, and the installed Tesseract version. “Deskew” uses OpenCV's estimated text-line angle; it may not help every page.

## Privacy and resource handling

- Uploaded files and generated outputs are handled in memory. Pixelworks does not intentionally write uploads or results to disk.
- Streamlit retains widget data and some generated results in the active session while needed for interaction/downloads. Ending the session releases session state; this app has no durable user-file storage.
- Files are decoded as images, never executed. Pillow decoding validates the format and checks pixel count; EXIF orientation is applied. Malformed files are rejected, output names are sanitized, and ZIP name collisions are made unique.
- Metadata is omitted from encoded images. GPS is not shown in image details. No external API receives image content.

## Tests

Run unit tests and Python compilation checks:

```bash
python -m unittest discover -s tests -v
python -m compileall -q app.py image_toolkit tests
```

The tests cover conversions and alpha, measured target compression, DPI metadata, resize/crop/rotation, batch failure isolation, size limits and corrupted input, ZIP integrity and filename collisions, contact-sheet settings, PDF page sizing and raster dimensions, OCR dependency errors, mocked perspective transforms and scanner corner validation.

## Project layout

```text
app.py                              Streamlit navigation, tools and shared UI helpers
image_toolkit/processing.py         Pillow operations, limits, compression and batch helpers
image_toolkit/scanner.py            OpenCV document detection and perspective correction
image_toolkit/metadata.py           DPI metadata reading and writing
components/four_point_editor/       Lightweight draggable four-corner Streamlit component
image_toolkit/ocr.py                Local Tesseract wrapper and image preparation
image_toolkit/pdf.py                In-memory PDF generation
image_toolkit/downloads.py          Collision-safe in-memory ZIP generation
requirements.txt                    Pinned core dependencies
packages.txt                        Streamlit Cloud system package for English OCR
tests/                              Processing and archive checks
```

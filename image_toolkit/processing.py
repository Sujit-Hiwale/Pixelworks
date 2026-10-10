"""In-memory image operations built on Pillow."""

from __future__ import annotations

from io import BytesIO
import math
import re
from typing import Iterable

from PIL import Image, ImageEnhance, ImageOps, UnidentifiedImageError

FORMAT_MAP = {"JPEG": "JPEG", "JPG": "JPEG", "PNG": "PNG", "WEBP": "WEBP"}
MIME_MAP = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}
MAX_BATCH = 10
MAX_PIXELS = 30_000_000
MAX_BATCH_PIXELS = 60_000_000
MAX_BATCH_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def load_image(data: bytes) -> Image.Image:
    """Decode a supported raster image safely and return an oriented copy."""
    try:
        with Image.open(BytesIO(data)) as source:
            if source.format not in {"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF"}:
                raise ValueError("This image format is not supported.")
            if source.width * source.height > MAX_PIXELS:
                raise ValueError("This image is too large to process (maximum 30 megapixels).")
            source.seek(0)
            original_format = source.format
            image = ImageOps.exif_transpose(source)
            image.load()
            result = image.copy()
            result.format = original_format
            return result
    except (UnidentifiedImageError, OSError, EOFError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


def peek_image_size(data: bytes) -> tuple[int, int]:
    """Read header dimensions before decoding pixels into memory."""
    try:
        with Image.open(BytesIO(data)) as image:
            if image.format not in {"JPEG", "PNG", "WEBP", "GIF", "BMP", "TIFF"}:
                raise ValueError("This image format is not supported.")
            if image.width * image.height > MAX_PIXELS:
                raise ValueError("This image is too large to process (maximum 30 megapixels).")
            return image.size
    except (UnidentifiedImageError, OSError, EOFError, Image.DecompressionBombError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc


def has_transparency(image: Image.Image) -> bool:
    return "A" in image.getbands() or "transparency" in image.info


def encode_image(image: Image.Image, fmt: str, quality: int = 85) -> bytes:
    fmt = FORMAT_MAP.get(fmt.upper())
    if fmt is None:
        raise ValueError("Choose JPEG, PNG, or WebP as the output format.")
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("The output is larger than the 30 megapixel limit.")
    # Preserve palette transparency as pixel alpha before clearing ancillary info.
    if "transparency" in image.info and "A" not in image.getbands():
        image = image.convert("RGBA")
    # Never carry source metadata into an export. The UI exposes a dedicated
    # metadata-clean action, and routine exports are clean by default as well.
    image = image.copy()
    image.info.clear()
    try:
        image.getexif().clear()
    except (AttributeError, OSError):
        pass
    if fmt == "JPEG":
        if "A" in image.getbands():
            rgba = image.convert("RGBA")
            background = Image.new("RGB", image.size, "white")
            background.paste(rgba, mask=rgba.getchannel("A"))
            image = background
        else:
            image = image.convert("RGB")
    elif fmt == "PNG":
        if image.mode not in ("RGB", "RGBA", "L", "LA", "P"):
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
    elif fmt == "WEBP":
        if image.mode not in ("RGB", "RGBA"):
            image = image.convert("RGBA" if "A" in image.getbands() else "RGB")

    output = BytesIO()
    options = {"format": fmt}
    if fmt in {"JPEG", "WEBP"}:
        options["quality"] = int(quality)
        options["optimize"] = True
    elif fmt == "PNG":
        options["optimize"] = True
    # Saving a clean image copy intentionally drops EXIF and other metadata.
    image.save(output, **options)
    return output.getvalue()


def resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
    if not 1 <= width <= 12000 or not 1 <= height <= 12000:
        raise ValueError("Width and height must be between 1 and 12,000 pixels.")
    if width * height > MAX_PIXELS:
        raise ValueError("The output is larger than the 30 megapixel limit.")
    return image.resize((width, height), Image.Resampling.LANCZOS)


def rotate_image(image: Image.Image, degrees: int) -> Image.Image:
    return image.rotate(-degrees, expand=True, resample=Image.Resampling.BICUBIC)


def rotate_custom(image: Image.Image, degrees: float, expand: bool = True) -> Image.Image:
    """Rotate clockwise for positive angles; expand=True preserves all pixels."""
    if not -180 <= degrees <= 180:
        raise ValueError("Rotation angle must be between −180° and 180°.")
    if expand:
        radians = math.radians(degrees)
        cosine, sine = abs(round(math.cos(radians), 12)), abs(round(math.sin(radians), 12))
        out_width = math.ceil(image.width * cosine + image.height * sine)
        out_height = math.ceil(image.height * cosine + image.width * sine)
        if out_width * out_height > MAX_PIXELS:
            raise ValueError("Expanded rotation would exceed the 30 megapixel limit.")
    return image.rotate(-degrees, expand=expand, resample=Image.Resampling.BICUBIC)


def crop_image(image: Image.Image, left: int, top: int, right: int, bottom: int) -> Image.Image:
    if left < 0 or top < 0 or right > image.width or bottom > image.height or right <= left or bottom <= top:
        raise ValueError("Crop bounds must describe a non-empty area inside the image.")
    return image.crop((left, top, right, bottom))


def compress_to_target(image: Image.Image, fmt: str, target_bytes: int, *, max_quality: int = 95,
                       min_quality: int = 1) -> dict:
    """Find the measured encoding closest to target, resizing only if quality alone misses.

    Quality is binary-searched at each size. If full-size encoding cannot hit the
    target closely, a second binary search reduces dimensions proportionally with LANCZOS.
    The nearest actual encoded byte count is returned; a too-large output is never
    described as meeting the target.
    """
    fmt = FORMAT_MAP.get(fmt.upper())
    if target_bytes <= 0:
        raise ValueError("Target size must be greater than zero.")
    if fmt not in {"JPEG", "WEBP", "PNG"}:
        raise ValueError("Target compression supports JPEG, PNG, and WebP.")
    if min_quality < 1 or max_quality > 95 or min_quality > max_quality:
        raise ValueError("Quality search bounds must be within 1–95.")
    source = image.copy()
    if fmt == "JPEG" and has_transparency(source):
        rgba = source.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, "white")
        flattened.paste(rgba, mask=rgba.getchannel("A"))
        source = flattened

    def search_at_size(candidate: Image.Image):
        measured: list[tuple[bytes, int | None]] = []
        if fmt == "PNG":
            measured.append((encode_image(candidate, fmt), None))
        else:
            low, high = min_quality, max_quality
            while low <= high:
                q = (low + high) // 2
                payload = encode_image(candidate, fmt, q)
                measured.append((payload, q))
                if len(payload) > target_bytes:
                    high = q - 1
                else:
                    low = q + 1
            # Include valid extrema; encoder size can be locally non-monotonic.
            for q in {min_quality, max_quality, max(min_quality, high), min(max_quality, low)}:
                payload = encode_image(candidate, fmt, q)
                measured.append((payload, q))
        under = [item for item in measured if len(item[0]) <= target_bytes]
        chosen = min(measured, key=lambda item: (abs(len(item[0]) - target_bytes), len(item[0]) > target_bytes))
        return chosen, bool(under)

    original_best, original_has_under = search_at_size(source)
    # Prefer a full-resolution result whenever one satisfies the target.
    # A full-resolution result within one percent is already practically on target;
    # retain it rather than resampling for a negligible byte difference.
    if original_has_under and abs(len(original_best[0]) - target_bytes) <= target_bytes * .01:
        chosen = original_best
        resized = False
        chosen_dimensions = source.size
    else:
        chosen = original_best
        chosen_dimensions = source.size
        chosen_area = source.width * source.height
        low_scale = 1 / min(source.width, source.height)
        high_scale = 1.0
        seen_sizes = {(source.width, source.height)}
        # Search scales, not guessed file sizes. Every candidate is encoded and measured.
        for _ in range(16):
            scale = (low_scale + high_scale) / 2
            dimensions = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
            if dimensions in seen_sizes:
                if high_scale - low_scale < 1 / max(source.size):
                    break
                # Probe a slightly smaller scale if rounding repeats a size.
                high_scale = scale
                continue
            seen_sizes.add(dimensions)
            candidate = source.resize(dimensions, Image.Resampling.LANCZOS) if dimensions != source.size else source
            measured, has_under = search_at_size(candidate)
            area = dimensions[0] * dimensions[1]
            if (abs(len(measured[0]) - target_bytes), -area) < (abs(len(chosen[0]) - target_bytes), -chosen_area):
                chosen, chosen_area, chosen_dimensions = measured, area, dimensions
            if has_under:
                low_scale = scale
            else:
                high_scale = scale
        # Always test the smallest aspect-preserving raster as a final feasible bound.
        minimum_scale = 1 / min(source.width, source.height)
        minimum_size = (max(1, round(source.width * minimum_scale)), max(1, round(source.height * minimum_scale)))
        if minimum_size not in seen_sizes:
            smallest = source.resize(minimum_size, Image.Resampling.LANCZOS)
            measured, _ = search_at_size(smallest)
            if (abs(len(measured[0]) - target_bytes), -minimum_size[0] * minimum_size[1]) < (abs(len(chosen[0]) - target_bytes), -chosen_area):
                chosen, chosen_area, chosen_dimensions = measured, minimum_size[0] * minimum_size[1], minimum_size
        # Refine integer rounding around the best scale; these neighboring rasters
        # can differ by only a few encoded bytes near very small targets.
        long_axis = 0 if source.width >= source.height else 1
        original_long = source.size[long_axis]
        best_long = chosen_dimensions[long_axis]
        radius = max(2, round(original_long * .008))
        for length in range(max(1, best_long - radius), min(original_long, best_long + radius) + 1):
            scale = length / original_long
            dimensions = (max(1, round(source.width * scale)), max(1, round(source.height * scale)))
            if dimensions in seen_sizes:
                continue
            seen_sizes.add(dimensions)
            candidate = source.resize(dimensions, Image.Resampling.LANCZOS)
            if fmt == "PNG":
                local = [(encode_image(candidate, fmt), None)]
            else:
                q0 = chosen[1] or min_quality
                qualities = {max(min_quality, q0 - 1), q0, min(max_quality, q0 + 1)}
                local = [(encode_image(candidate, fmt, q), q) for q in qualities]
            measured = min(local, key=lambda item: (abs(len(item[0]) - target_bytes), len(item[0]) > target_bytes))
            area = dimensions[0] * dimensions[1]
            if (abs(len(measured[0]) - target_bytes), -area) < (abs(len(chosen[0]) - target_bytes), -chosen_area):
                chosen, chosen_area, chosen_dimensions = measured, area, dimensions
        resized = chosen_area < source.width * source.height

    actual = len(chosen[0])
    reached = actual <= target_bytes
    if fmt == "PNG":
        note = "Target reached with lossless PNG encoding." if reached else "PNG is lossless; dimensions were reduced and this was the closest measured result."
    elif reached:
        note = "Target reached using the highest-quality measured result that fits."
    else:
        note = "Closest measured result is still above the requested target; container overhead and image detail set a practical minimum."
    return {"data": chosen[0], "quality": chosen[1], "met_target": reached,
            "actual_bytes": actual, "target_bytes": target_bytes, "difference_bytes": actual - target_bytes,
            "dimensions": chosen_dimensions,
            "resized": resized, "format": fmt, "note": note}


def safe_filename(filename: str, fallback: str = "image") -> str:
    """Keep only a safe basename and normalize characters for downloadable outputs."""
    name = filename.replace("\\", "/").split("/")[-1]
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-")
    return name or fallback


def unique_filenames(names: Iterable[str]) -> list[str]:
    result = []
    used: set[str] = set()
    for raw in names:
        name = safe_filename(raw)
        stem, dot, ext = name.rpartition(".")
        if not dot:
            stem, ext = name, ""
        candidate = name
        index = 2
        while candidate.casefold() in used:
            candidate = f"{stem}_{index}{'.' + ext if ext else ''}"
            index += 1
        name = candidate
        used.add(name.casefold())
        result.append(name)
    return result


def process_batch(items: Iterable[tuple[str, Image.Image]], operation) -> tuple[list[tuple[str, Image.Image]], list[tuple[str, str]]]:
    """Run operation independently; return successes and per-file failures."""
    outputs, failures = [], []
    for name, image in items:
        try:
            outputs.append((name, operation(image)))
        except Exception as exc:
            failures.append((name, str(exc)))
    return outputs, failures


def apply_adjustments(image: Image.Image, brightness: float = 1, contrast: float = 1,
                      saturation: float = 1, sharpness: float = 1) -> Image.Image:
    for enhancer, factor in ((ImageEnhance.Brightness, brightness), (ImageEnhance.Contrast, contrast),
                             (ImageEnhance.Color, saturation), (ImageEnhance.Sharpness, sharpness)):
        image = enhancer(image).enhance(factor)
    return image


def image_info(image: Image.Image, original_format: str, file_size: int) -> dict:
    safe_exif = {}
    try:
        exif = image.getexif()
        safe_exif = {str(k): str(v)[:180] for k, v in exif.items() if k not in {34853}}
    except (AttributeError, OSError):
        pass
    return {"width": image.width, "height": image.height, "format": original_format or image.format or "Unknown",
            "mode": image.mode, "file_size": file_size, "exif": safe_exif}


def make_contact_sheet(images: Iterable[tuple[str, Image.Image]], columns: int = 3,
                       cell_size: tuple[int, int] = (240, 200), *, include_filenames: bool = True,
                       font_size: int = 14, border_thickness: int = 2, border_color: str = "#FFFFFF",
                       spacing: int = 12, background: str = "#F0F1F7") -> Image.Image:
    entries = list(images)
    if not entries or not 1 <= columns <= 12:
        raise ValueError("Provide at least one image and between 1 and 12 columns.")
    if spacing < 0 or border_thickness < 0 or font_size < 6:
        raise ValueError("Spacing, border thickness, and font size must be non-negative and readable.")
    label_height = font_size + 14 if include_filenames else 0
    cell_width = cell_size[0] + spacing
    cell_height = cell_size[1] + label_height + spacing
    rows = (len(entries) + columns - 1) // columns
    sheet = Image.new("RGB", (spacing + columns * cell_width, spacing + rows * cell_height), background)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", font_size)
    except OSError:
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()
    for index, (name, original) in enumerate(entries):
        tile = original.copy()
        tile.thumbnail((cell_size[0] - 2 * border_thickness, cell_size[1] - 2 * border_thickness))
        x0 = spacing + (index % columns) * cell_width
        y0 = spacing + (index // columns) * cell_height
        x = x0 + (cell_size[0] - tile.width) // 2
        y = y0 + (cell_size[1] - tile.height) // 2
        if "A" in tile.getbands() or "transparency" in tile.info:
            rgba = tile.convert("RGBA")
            sheet.paste(rgba, (x, y), rgba.getchannel("A"))
        else:
            sheet.paste(tile, (x, y))
        if border_thickness:
            draw.rectangle((x, y, x + tile.width - 1, y + tile.height - 1), outline=border_color,
                           width=border_thickness)
        if include_filenames:
            label = safe_filename(name)
            max_width = cell_size[0]
            while label and draw.textbbox((0, 0), label, font=font)[2] > max_width:
                label = label[:-2] + "…"
            draw.text((x0, y0 + cell_size[1] + 6), label, fill="#202238", font=font)
    return sheet

"""Image resolution metadata helpers; pixel data is never resized."""

from __future__ import annotations

from io import BytesIO
import math
import struct
import zlib

from PIL import Image
from PIL.TiffImagePlugin import IFDRational


def get_image_dpi(image: Image.Image) -> tuple[float, float] | None:
    value = image.info.get("dpi")
    if value and len(value) >= 2:
        try:
            x, y = float(value[0]), float(value[1])
            if math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0:
                return x, y
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    try:
        exif = image.getexif()
        unit = int(exif.get(296, 2))
        x, y = float(exif[282]), float(exif[283])
        if unit == 3:  # pixels per centimetre
            x, y = x * 2.54, y * 2.54
        if unit in (2, 3) and x > 0 and y > 0:
            return x, y
    except (KeyError, TypeError, ValueError, ZeroDivisionError, AttributeError):
        pass
    return None


def _png_with_dpi(data: bytes, dpi: int) -> bytes:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("This PNG file is malformed.")
    ppm = round(dpi / 0.0254)
    phys_data = struct.pack(">IIB", ppm, ppm, 1)
    phys = struct.pack(">I", len(phys_data)) + b"pHYs" + phys_data
    phys += struct.pack(">I", zlib.crc32(b"pHYs" + phys_data) & 0xffffffff)
    chunks, offset, inserted = [], 8, False
    while offset + 12 <= len(data):
        length = struct.unpack_from(">I", data, offset)[0]
        end = offset + 12 + length
        if end > len(data):
            raise ValueError("This PNG file is malformed.")
        kind = data[offset + 4:offset + 8]
        if kind != b"pHYs":
            chunks.append(data[offset:end])
        if kind == b"IHDR":
            chunks.append(phys)
            inserted = True
        offset = end
        if kind == b"IEND":
            break
    if not inserted:
        raise ValueError("This PNG file has no valid image header.")
    return data[:8] + b"".join(chunks)


def _jpeg_with_dpi(data: bytes, dpi: int) -> bytes:
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("This JPEG file is malformed.")
    payload_start = None
    exif_segments = []
    position = 2
    while position + 4 <= len(data) and data[position] == 0xff:
        marker_start = position
        while position < len(data) and data[position] == 0xff:
            position += 1
        if position >= len(data):
            break
        marker = data[position]
        position += 1
        if marker in (0xd8, 0xd9) or 0xd0 <= marker <= 0xd7:
            continue
        if position + 2 > len(data):
            break
        segment_length = struct.unpack_from(">H", data, position)[0]
        segment_end = position + segment_length
        if segment_length < 2 or segment_end > len(data):
            break
        if marker == 0xe0 and data[position + 2:position + 7] == b"JFIF\x00":
            if payload_start is None:
                payload_start = marker_start
        if marker == 0xe1 and data[position + 2:position + 8] == b"Exif\x00\x00":
            exif_segments.append((marker_start, segment_end))
        if marker == 0xda:  # start of compressed image data
            break
        position = segment_end
    mutable = bytearray(data)
    for start, end in exif_segments:
        tiff = start + 10
        byteorder = data[tiff:tiff+2]
        if byteorder == b"II":
            endian = "<"
        elif byteorder == b"MM":
            endian = ">"
        else:
            continue
        try:
            ifd_offset = struct.unpack_from(endian + "I", data, tiff + 4)[0]
            ifd = tiff + ifd_offset
            count = struct.unpack_from(endian + "H", data, ifd)[0]
            for index in range(count):
                entry = ifd + 2 + index * 12
                if entry + 12 > end:
                    break
                tag, kind, values = struct.unpack_from(endian + "HHI", data, entry)
                if tag in (282, 283) and kind == 5 and values >= 1:
                    value_offset = struct.unpack_from(endian + "I", data, entry + 8)[0]
                    value = tiff + value_offset
                    if value + 8 <= end:
                        struct.pack_into(endian + "II", mutable, value, dpi, 1)
                elif tag == 296 and kind == 3 and values >= 1:
                    struct.pack_into(endian + "H", mutable, entry + 8, 2)
        except (struct.error, IndexError):
            continue
    if payload_start is None:
        segment = b"\xff\xe0\x00\x10JFIF\x00\x01\x01\x01" + struct.pack(">HH", dpi, dpi) + b"\x00\x00"
        return bytes(mutable[:2] + segment + mutable[2:])
    mutable[payload_start + 11] = 1  # JFIF density unit: dots per inch
    struct.pack_into(">HH", mutable, payload_start + 12, dpi, dpi)
    return bytes(mutable)


def set_dpi_metadata(data: bytes, dpi: int) -> bytes:
    """Write DPI metadata to supported image formats without changing raster dimensions.

    JPEG JFIF density, PNG pHYs and BMP pixels-per-meter fields are patched in
    place at the container level. TIFF is losslessly reserialized by Pillow.
    WebP stores resolution in EXIF and is re-encoded in lossless mode. GIF has
    no standard DPI metadata and is deliberately not offered.
    """
    if not isinstance(dpi, (int, float)) or not math.isfinite(float(dpi)) or not 1 <= dpi <= 65535:
        raise ValueError("DPI must be a whole number between 1 and 65,535.")
    dpi = int(round(dpi))
    with Image.open(BytesIO(data)) as source:
        fmt = source.format
        source.load()
        if fmt == "JPEG":
            return _jpeg_with_dpi(data, dpi)
        if fmt == "PNG":
            return _png_with_dpi(data, dpi)
        if fmt == "BMP":
            if len(data) < 46 or struct.unpack_from("<I", data, 14)[0] < 40:
                raise ValueError("This BMP header does not support DPI metadata.")
            mutable = bytearray(data)
            pixels_per_meter = round(dpi / 0.0254)
            struct.pack_into("<ii", mutable, 38, pixels_per_meter, pixels_per_meter)
            return bytes(mutable)
        output = BytesIO()
        if fmt == "TIFF":
            source.save(output, format="TIFF", dpi=(dpi, dpi))
        elif fmt == "WEBP":
            exif = source.getexif()
            exif[282] = IFDRational(dpi, 1)
            exif[283] = IFDRational(dpi, 1)
            exif[296] = 2
            source.save(output, format="WEBP", lossless=True, exif=exif.tobytes())
        else:
            raise ValueError("DPI metadata can be written for JPEG, PNG, WebP, BMP and TIFF. GIF does not support standard DPI metadata.")
        return output.getvalue()

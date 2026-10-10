"""In-memory PDF assembly with per-image page sizes for Original mode."""

from io import BytesIO

from PIL import Image
from reportlab.lib.pagesizes import A4, letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from image_toolkit.metadata import get_image_dpi

PAGE_SIZES = {"A4": A4, "Letter": letter}
DEFAULT_DPI = 96


def images_to_pdf(images: list[Image.Image], *, page_size: str = "Original", fit: bool = True) -> bytes:
    if not images:
        raise ValueError("Add at least one image to create a PDF.")
    if page_size != "Original" and page_size not in PAGE_SIZES:
        raise ValueError("Choose A4, Letter, or Original page size.")
    output = BytesIO()
    pdf = canvas.Canvas(output)
    for source in images:
        if page_size == "Original":
            dpi = get_image_dpi(source) or (DEFAULT_DPI, DEFAULT_DPI)
            page_w = source.width * 72 / dpi[0]
            page_h = source.height * 72 / dpi[1]
        else:
            page_w, page_h = PAGE_SIZES[page_size]
        if page_w <= 0 or page_h <= 0:
            raise ValueError("Image DPI metadata produced invalid physical page dimensions.")
        pdf.setPageSize((page_w, page_h))

        # PNG embedding preserves the source raster's dimensions and alpha. In Original
        # mode it is placed 1:1 in physical units derived from this image's own DPI.
        if "transparency" in source.info and "A" not in source.getbands():
            image = source.convert("RGBA")
        else:
            image = source.copy()
        if page_size == "Original":
            draw_w, draw_h = page_w, page_h
        elif fit:
            scale = min(page_w / image.width, page_h / image.height)
            draw_w, draw_h = image.width * scale, image.height * scale
        else:
            draw_w, draw_h = float(image.width), float(image.height)
        x, y = (page_w - draw_w) / 2, (page_h - draw_h) / 2
        image_buffer = BytesIO()
        image.save(image_buffer, format="PNG")
        image_buffer.seek(0)
        pdf.drawImage(ImageReader(image_buffer), x, y, width=draw_w, height=draw_h,
                      preserveAspectRatio=False, mask="auto")
        pdf.showPage()
    pdf.save()
    return output.getvalue()

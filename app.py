from __future__ import annotations

import base64
import hashlib
import html
from pathlib import Path
from io import BytesIO

import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageDraw

from image_toolkit.downloads import make_zip
from image_toolkit.processing import (
    MAX_BATCH, MAX_BATCH_PIXELS, MAX_BATCH_UPLOAD_BYTES, MAX_UPLOAD_BYTES, MIME_MAP,
    apply_adjustments, compress_to_target, crop_image, encode_image, has_transparency, image_info, load_image, make_contact_sheet,
    peek_image_size, resize_image, rotate_custom, safe_filename, unique_filenames,
)
from image_toolkit.metadata import get_image_dpi, set_dpi_metadata

st.set_page_config(page_title="Pixelworks — Image Toolkit", page_icon="✦", layout="wide",
                   initial_sidebar_state="expanded")
four_point_editor = components.declare_component(
    "pixelworks_four_point_editor", path=str(Path(__file__).parent / "components" / "four_point_editor")
)
st.markdown("""
<style>
html,body,[class*="css"]{font-family:'Aptos','Segoe UI',sans-serif}.block-container{max-width:1220px;padding-top:2rem;padding-bottom:4rem}
h1,h2,h3{font-family:'Aptos Display','Segoe UI',sans-serif!important;letter-spacing:-.035em}
.hero{padding:2.35rem 2.6rem;border-radius:24px;background:linear-gradient(120deg,#25264B,#4549B9 58%,#7779F7);color:white;margin-bottom:1.5rem}
.hero h1{color:white;font-size:2.65rem;margin:.5rem 0}.hero p{color:#E1E2FF;max-width:620px;font-size:1.06rem;margin:0}
.eyebrow{color:#BFC0FF;font-weight:700;font-size:.76rem;letter-spacing:.14em;text-transform:uppercase}
.card{background:white;border:1px solid #E9EAF2;border-radius:18px;padding:1.2rem 1.3rem;min-height:140px}
.card h3{margin:.2rem 0 .4rem;font-size:1.05rem}.card p{color:#6F7187;font-size:.9rem;margin:0}
section[data-testid="stSidebar"]{border-right:1px solid #E9EAF2}div[data-testid="stMetric"]{background:white;border:1px solid #E9EAF2;padding:.8rem 1rem;border-radius:14px}
</style>""", unsafe_allow_html=True)

PAGES = ["Welcome", "Convert & compress", "Resize", "Crop, rotate & flip", "Adjust image",
         "Image details", "DPI Setter", "Contact sheet", "Document scanner", "Image to PDF", "OCR text extractor"]
with st.sidebar:
    st.markdown("### ✦ Pixelworks")
    st.caption("Your everyday image toolkit")
    selected = st.radio("Workspace", PAGES, label_visibility="collapsed")
    st.divider()
    st.caption("🔒 Images stay in this app process; no image API calls are made.")
    st.caption("JPEG · PNG · WebP · BMP · TIFF · GIF")

if selected != "OCR text extractor":
    st.session_state.pop("ocr_results", None)
    for state_key in [key for key in st.session_state if key.startswith("ocr_text_")]:
        st.session_state.pop(state_key, None)
if selected != "Image to PDF":
    st.session_state.pop("pdf_result", None)
    st.session_state.pop("pdf_page_count", None)


def upload_images(label="Drop images here or browse files", multiple=True):
    uploads = st.file_uploader(label, type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif"],
                               accept_multiple_files=multiple, help="20 MB per file · maximum 10 files")
    if not uploads:
        return []
    uploads = uploads if isinstance(uploads, list) else [uploads]
    if len(uploads) > MAX_BATCH:
        st.error(f"The batch limit is {MAX_BATCH} images.")
        return []
    names = unique_filenames(item.name for item in uploads)
    valid = []
    pixel_sum = 0
    byte_sum = 0
    for item, safe_name in zip(uploads, names):
        data = item.getvalue()
        if len(data) > MAX_UPLOAD_BYTES:
            st.error(f"{safe_name}: exceeds the 20 MB upload limit.")
            continue
        try:
            width, height = peek_image_size(data)
            if byte_sum + len(data) > MAX_BATCH_UPLOAD_BYTES:
                st.error(f"{safe_name}: batch total exceeds the 80 MB upload limit.")
                continue
            if pixel_sum + width * height > MAX_BATCH_PIXELS:
                st.error(f"{safe_name}: batch total exceeds the 60 megapixel processing limit.")
                continue
            valid.append((safe_name, data, load_image(data)))
            pixel_sum += width * height
            byte_sum += len(data)
        except (ValueError, OSError) as exc:
            st.error(f"{safe_name}: {exc}")
    return valid


def extension(fmt):
    return "jpg" if fmt == "JPEG" else fmt.lower()


def make_name(name, fmt, suffix=""):
    stem = safe_filename(name).rsplit(".", 1)[0]
    return f"{stem}{suffix}.{extension(fmt)}"


def image_bytes_uri(image: Image.Image):
    buffer = BytesIO()
    preview = image.copy()
    preview.thumbnail((2400, 2400), Image.Resampling.LANCZOS)
    if "A" in preview.getbands():
        preview.save(buffer, format="PNG", optimize=True)
        mime = "image/png"
    else:
        preview.convert("RGB").save(buffer, format="JPEG", quality=88, optimize=True)
        mime = "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def clickable_preview(image: Image.Image, label: str, key: str):
    """Click opens a larger browser view; closing it returns without rerunning settings."""
    uri = image_bytes_uri(image)
    safe_label = html.escape(label)
    document = f"""<!doctype html><meta name='viewport' content='width=device-width,initial-scale=1'><body data-preview-id='{html.escape(key)}'>
    <style>*{{box-sizing:border-box}}body{{margin:0;font:14px sans-serif;color:#38394d}}.thumb{{width:100%;height:330px;object-fit:contain;background:repeating-conic-gradient(#eee 0% 25%,white 0% 50%) 50%/20px 20px;border:1px solid #ddd;border-radius:12px;cursor:zoom-in}}.modal{{display:none;position:fixed;z-index:99;inset:0;background:#111e;align-items:center;justify-content:center;padding:24px}}.modal img{{max-width:95vw;max-height:82vh;object-fit:contain;background:white;border-radius:8px}}button{{position:absolute;right:24px;top:20px;border:0;border-radius:20px;padding:10px 16px;font-weight:700;cursor:pointer}}.hint{{margin:.35rem 0;color:#777}}</style>
    <img class='thumb' src='{uri}' alt='{safe_label}' title='Click to enlarge'>
    <div class='hint'>{safe_label} · click to open a larger view; close the tab or use the close control</div><div class='modal' role='dialog'><button aria-label='Close'>Close ×</button><img alt='{safe_label}'></div>
    <script>const m=document.querySelector('.modal'),thumb=document.querySelector('.thumb'),large=m.querySelector('img');thumb.onclick=()=>{{try{{const parts=thumb.src.split(',');const raw=atob(parts[1]);const bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);const type=parts[0].match(/data:(.*);base64/)[1];const url=URL.createObjectURL(new Blob([bytes],{{type}}));const tab=window.open(url,'_blank');if(!tab){{large.src=thumb.src;m.style.display='flex'}}}}catch(e){{large.src=thumb.src;m.style.display='flex'}}}};document.querySelector('button').onclick=()=>m.style.display='none';m.onclick=e=>{{if(e.target===m)m.style.display='none'}};document.addEventListener('keydown',e=>{{if(e.key==='Escape')m.style.display='none'}})</script>"""
    components.html(document, height=375, scrolling=False)


def compare_previews(original, processed, key):
    left, right = st.columns(2)
    with left:
        st.caption("Original · click to enlarge")
        clickable_preview(original, "Original", f"{key}_original")
    with right:
        st.caption("Processed · click to enlarge")
        clickable_preview(processed, "Processed", f"{key}_processed")


def download_results(results, title, *, zip_name="pixelworks-results.zip"):
    """Render successes with independent downloads and an archive; failures stay isolated."""
    successes, failures = results
    for name, message in failures:
        st.warning(f"{name}: {message}")
    if not successes:
        return
    archive = []
    for index, result in enumerate(successes):
        # (filename, bytes, original-image, processed-image, detail)
        name, data, original, preview, detail = result
        archive.append((name, data))
        with st.container(border=True):
            st.markdown(f"**{name}**")
            if detail:
                st.caption(detail)
            l, r = st.columns([1, 1])
            with l:
                compare_previews(original, preview, f"{title}_{index}")
            with r:
                st.download_button("Download", data, file_name=name, mime=mime_for(name), key=f"{title}_dl_{index}", type="primary")
    if archive:
        st.download_button("Download all successful results as ZIP", make_zip(archive), zip_name, "application/zip", type="primary")


def mime_for(name):
    suffix = name.rsplit(".", 1)[-1].lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp",
            "pdf": "application/pdf", "txt": "text/plain"}.get(suffix, "application/octet-stream")


def render_home():
    st.markdown('<div class="hero"><span class="eyebrow">A little more room to create</span><h1>All your image tools.<br>One simple workspace.</h1><p>Convert, compress, scan and fine-tune images with free tools processed locally in this app.</p></div>', unsafe_allow_html=True)
    st.markdown("### Pick a tool")
    cards = [("↗", "Convert & compress", "Convert formats, target a measured size, or compress a batch."),
             ("⤢", "Resize", "Resize by percentage or exact dimensions."),
             ("◩", "Crop, rotate & flip", "Interactive crops, rotations and flips applied across a batch."),
             ("☼", "Adjust image", "Tune brightness, contrast, saturation and sharpness."),
             ("⌕", "Image details", "Inspect dimensions and metadata, then export a clean copy."),
             ("▦", "Contact sheet", "Arrange labeled thumbnails into a downloadable sheet."),
             ("◉", "DPI Setter", "Change print-resolution metadata without changing pixel dimensions."),
             ("▤", "Document scanner", "Detect or manually select corners and correct perspective."),
             ("PDF", "Image to PDF", "Reorder images and export one image per PDF page."),
             ("T", "OCR text extractor", "Recognize text locally with Tesseract.")]
    for start in range(0, len(cards), 3):
        cols = st.columns(3)
        for col, (icon, title, desc) in zip(cols, cards[start:start+3]):
            with col:
                st.markdown(f'<div class="card"><div style="font-size:1.4rem;color:#5B5FEF">{icon}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)
        st.write("")
    st.info("Core image tools need no paid APIs or GPU. OCR uses the local Tesseract installation; see the README for setup details.")


def render_convert():
    st.title("Convert & compress")
    st.caption("Convert a batch with automatic compression, or ask for a target size measured from the final encoded bytes.")
    files = upload_images()
    if not files: return
    c1, c2 = st.columns(2)
    fmt = c1.selectbox("Output format", ["JPEG", "PNG", "WEBP"])
    use_target = c2.checkbox("Target file size")
    target_bytes = None
    if use_target:
        a, b = st.columns([2, 1])
        target = a.number_input("Target size", min_value=1.0, value=250.0, step=10.0)
        unit = b.selectbox("Unit", ["KB", "MB"])
        target_bytes = int(target * (1024 if unit == "KB" else 1024 * 1024))
        if fmt == "PNG": st.info("PNG uses lossless encoding; if needed, the target optimizer reduces dimensions proportionally.")
    st.caption("JPEG flattens transparent pixels onto the selected background.")
    jpeg_bg = st.color_picker("JPEG transparency background", "#FFFFFF") if fmt == "JPEG" else "#FFFFFF"
    successes, failures = [], []
    for name, raw, image in files:
        try:
            work = image
            if fmt == "JPEG" and has_transparency(work):
                rgba = work.convert("RGBA")
                bg = Image.new("RGB", work.size, jpeg_bg)
                bg.paste(rgba, mask=rgba.getchannel("A")); work = bg
            if target_bytes:
                compressed = compress_to_target(work, fmt, target_bytes)
                data = compressed["data"]
                met = compressed["met_target"]
                divisor = 1024 if unit == "KB" else 1024*1024
                actual = len(data) / divisor
                difference = compressed["difference_bytes"] / divisor
                sign = "+" if difference > 0 else ""
                status = "Target reached" if met else "Closest achievable result"
                out_w, out_h = compressed["dimensions"]
                detail = (f"Input {len(raw)/divisor:.2f} {unit} · target {target:g} {unit} · actual {actual:.2f} {unit} · "
                          f"difference {sign}{difference:.2f} {unit} · output {out_w} × {out_h} px · {fmt} · {status} — {compressed['note']}")
                preview = load_image(data)
            else:
                data = encode_image(work, fmt, 85)
                detail = f"Original {len(raw)/1024:.1f} KB · output {len(data)/1024:.1f} KB · reduction {(1-len(data)/len(raw))*100:.1f}% · automatic compression"
                preview = load_image(data)
            successes.append((make_name(name, fmt), data, image, preview, detail))
        except Exception as exc:
            failures.append((name, str(exc)))
    download_results((successes, failures), "convert", zip_name="pixelworks-converted.zip")


def render_resize():
    st.title("Resize")
    files = upload_images()
    if not files: return
    mode = st.radio("Resize method", ["Percentage", "Dimensions"], horizontal=True)
    keep = st.checkbox("Keep aspect ratio", True)
    if mode == "Percentage":
        percent = st.slider("Scale", 10, 200, 50)
        dimensions = None
    else:
        c1, c2 = st.columns(2)
        dimensions = (c1.number_input("Width (px)", 1, 12000, files[0][2].width),
                      c2.number_input("Height (px)", 1, 12000, files[0][2].height))
        percent = None
    successes, failures = [], []
    for name, _, image in files:
        try:
            if mode == "Percentage": width, height = round(image.width*percent/100), round(image.height*percent/100)
            elif keep:
                ratio = image.width / image.height
                width, height = dimensions
                if width != files[0][2].width: height = round(width / ratio)
                else: width = round(height * ratio)
            else: width, height = dimensions
            result = resize_image(image, int(width), int(height))
            fmt = original_format(image)
            successes.append((make_name(name, fmt, "-resized"), encode_image(result, fmt), image, result, f"{image.width} × {image.height} → {result.width} × {result.height}"))
        except Exception as exc: failures.append((name, str(exc)))
    download_results((successes, failures), "resize", zip_name="pixelworks-resized.zip")


def original_format(image):
    value = (image.format or "PNG").upper()
    if value == "JPG": value = "JPEG"
    return value if value in MIME_MAP else "PNG"


def render_crop():
    st.title("Crop, rotate & flip")
    st.caption("Crop with a draggable rectangle, or enter precise pixel coordinates. Positive custom angles rotate clockwise; the canvas expands to preserve the full image.")
    files = upload_images()
    if not files: return
    try:
        from streamlit_cropper import st_cropper
    except ImportError:
        st_cropper = None
    aspect_label = st.selectbox("Crop aspect ratio", ["Freeform", "1:1", "4:3", "16:9", "9:16"])
    aspect = None if aspect_label == "Freeform" else tuple(map(float, aspect_label.split(":")))
    crop_by_name = {}
    first_name, _, base = files[0]
    base_left, base_top, base_right, base_bottom = 0, 0, base.width, base.height
    if st_cropper:
        st.caption("Drag and resize a shared crop rectangle. The same proportional crop is applied to each image.")
        clickable_preview(base, first_name, "crop_original_0")
        box = st_cropper(base, realtime_update=True, box_color="#5B5FEF", aspect_ratio=aspect,
                         return_type="box", key=f"crop_box_{first_name}_{base.width}x{base.height}_{len(files[0][1])}")
    else:
        st.warning("Interactive crop UI unavailable. Install streamlit-cropper; precise coordinates remain available below.")
    if isinstance(box if st_cropper else None, dict):
        base_left = int(box.get("left", 0)); base_top = int(box.get("top", 0))
        base_right = base_left + int(box.get("width", base.width))
        base_bottom = base_top + int(box.get("height", base.height))
    with st.expander("Precise crop coordinates", expanded=not bool(st_cropper)):
        cols = st.columns(4)
        left = cols[0].number_input("Left", 0, base.width-1, base_left)
        top = cols[1].number_input("Top", 0, base.height-1, base_top)
        right = cols[2].number_input("Right", 1, base.width, base_right)
        bottom = cols[3].number_input("Bottom", 1, base.height, base_bottom)
    base_left, base_top, base_right, base_bottom = int(left), int(top), int(right), int(bottom)
    if st_cropper:
        try:
            crop_by_name[first_name] = crop_image(base, base_left, base_top, base_right, base_bottom)
            clickable_preview(crop_by_name[first_name], "Crop preview", "crop_preview_shared")
        except ValueError as exc:
            st.error(str(exc)); return
    st.session_state.setdefault("rotation_angle", 0.0)
    cols = st.columns(3)
    cols[0].button("↻ Clockwise 90°", use_container_width=True, on_click=lambda: st.session_state.update(rotation_angle=90.0))
    cols[1].button("↺ Anticlockwise 90°", use_container_width=True, on_click=lambda: st.session_state.update(rotation_angle=-90.0))
    custom = cols[2].number_input("Custom angle (°)", min_value=-180.0, max_value=180.0, step=1.0, key="rotation_angle")
    expand = st.checkbox("Expand rotated canvas", True)
    flip_h = st.checkbox("Flip horizontally")
    flip_v = st.checkbox("Flip vertically")
    successes, failures = [], []
    for name, _, image in files:
        try:
            if st_cropper:
                if name not in crop_by_name:
                    sx, sy = image.width/base.width, image.height/base.height
                    bounds = (round(base_left*sx), round(base_top*sy), round(base_right*sx), round(base_bottom*sy))
                    crop_by_name[name] = crop_image(image, *bounds)
                result = crop_by_name[name]
            else:
                sx, sy = image.width/base.width, image.height/base.height
                x1, y1, x2, y2 = round(left*sx), round(top*sy), round(right*sx), round(bottom*sy)
                result = crop_image(image, x1, y1, x2, y2)
            angle = custom
            if angle: result = rotate_custom(result, angle, expand=expand)
            if flip_h: result = result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if flip_v: result = result.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            fmt = original_format(image)
            crop_detail = (f"Interactive crop {base_left},{base_top}–{base_right},{base_bottom}" if st_cropper
                           else f"Crop {x1},{y1}–{x2},{y2}")
            successes.append((make_name(name, fmt, "-edited"), encode_image(result, fmt), image, result, f"{crop_detail}; {angle}° clockwise convention; canvas {'expanded' if expand else 'cropped'}"))
        except Exception as exc: failures.append((name, str(exc)))
    download_results((successes, failures), "crop", zip_name="pixelworks-edited.zip")


def render_adjust():
    st.title("Adjust image")
    files = upload_images()
    if not files: return
    cols = st.columns(4)
    factors = [col.slider(label, 0.0, 2.0, 1.0, .05) for col, label in zip(cols, ["Brightness", "Contrast", "Saturation", "Sharpness"])]
    outputs, failures = [], []
    for name, _, image in files:
        try:
            result = apply_adjustments(image, *factors); fmt = original_format(image)
            outputs.append((make_name(name, fmt, "-adjusted"), encode_image(result, fmt), image, result, "Shared settings applied to this image."))
        except Exception as exc: failures.append((name, str(exc)))
    download_results((outputs, failures), "adjust", zip_name="pixelworks-adjusted.zip")


def render_details():
    st.title("Image details")
    st.caption("GPS fields are excluded. Clean exports remove embedded metadata.")
    files = upload_images()
    if not files: return
    archive = []
    for i, (name, raw, image) in enumerate(files):
        info = image_info(image, original_format(image), len(raw))
        st.subheader(name)
        c = st.columns(4); c[0].metric("Dimensions", f"{info['width']} × {info['height']}"); c[1].metric("Format", info['format']); c[2].metric("File size", f"{len(raw)/1024:.1f} KB"); c[3].metric("Mode", info['mode'])
        if info["exif"]:
            with st.expander("EXIF metadata"): st.json(info["exif"])
        fmt = original_format(image); clean = encode_image(image, fmt)
        archive.append((make_name(name, fmt, "-clean"), clean))
        c1, c2 = st.columns([1, 2])
        with c1: st.download_button("Download without metadata", clean, make_name(name, fmt, "-clean"), MIME_MAP[fmt], key=f"clean_{i}")
        with c2: clickable_preview(image, "Original", f"details_{i}")
    if len(files) > 1:
        st.download_button("Download clean images as ZIP", make_zip(archive), "pixelworks-clean-images.zip", "application/zip", type="primary")


def render_dpi_setter():
    st.title("DPI Setter")
    st.caption("Change resolution metadata without resizing or resampling the image. DPI metadata does not increase the image's pixel count.")
    files = upload_images()
    if not files: return
    preset = st.selectbox("New DPI", [72, 96, 150, 200, 300, 600, "Custom"])
    dpi = st.number_input("Custom DPI", min_value=1, max_value=65535, value=300, step=1) if preset == "Custom" else int(preset)
    st.info(f"Selected output resolution: {int(dpi)} DPI horizontally and vertically. Pixel dimensions will remain unchanged.")
    outputs, failures, archive = [], [], []
    for name, raw, image in files:
        source_dpi = get_image_dpi(image)
        source_text = f"{source_dpi[0]:g} × {source_dpi[1]:g} DPI" if source_dpi else "Not recorded"
        raw_dimensions = peek_image_size(raw)
        st.subheader(name)
        st.caption(f"Original: {raw_dimensions[0]} × {raw_dimensions[1]} px · DPI: {source_text} · New: {int(dpi)} DPI")
        try:
            output = set_dpi_metadata(raw, int(dpi))
            with Image.open(BytesIO(output)) as check:
                if check.size != raw_dimensions:
                    raise RuntimeError("DPI metadata update unexpectedly changed the pixel dimensions.")
                fmt = check.format
                output_dpi = get_image_dpi(check)
            output_text = f"{output_dpi[0]:.2f} × {output_dpi[1]:.2f} DPI" if output_dpi else f"{int(dpi)} DPI"
            detail = f"{raw_dimensions[0]} × {raw_dimensions[1]} px unchanged · output metadata {output_text}"
            output_name = make_name(name, fmt)
            outputs.append((output_name, output, image, image, detail, fmt))
            archive.append((output_name, output))
        except Exception as exc:
            failures.append((name, str(exc)))
    for name, error in failures: st.error(f"{name}: {error}")
    for index, (name, data, _, _, detail, fmt) in enumerate(outputs):
        st.caption(detail)
        mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp",
                "BMP": "image/bmp", "TIFF": "image/tiff"}.get(fmt, "application/octet-stream")
        st.download_button("Download with new DPI metadata", data, name, mime, key=f"dpi_download_{index}")
    if len(archive) > 1:
        st.download_button("Download all updated images as ZIP", make_zip(archive), "pixelworks-dpi-updated.zip", "application/zip")


def render_contact():
    st.title("Contact sheet")
    files = upload_images()
    if not files: return
    c1, c2 = st.columns(2)
    columns = c1.slider("Columns", 1, min(8, len(files)), min(3, len(files)))
    include = c2.toggle("Include filenames", True)
    c1, c2, c3 = st.columns(3)
    font_size = c1.slider("Filename size", 8, 28, 14, disabled=not include)
    border = c2.slider("Thumbnail border", 0, 12, 2)
    spacing = c3.slider("Spacing", 0, 40, 12)
    c1, c2 = st.columns(2)
    border_color = c1.color_picker("Border color", "#FFFFFF")
    background = c2.color_picker("Sheet background", "#F0F1F7")
    try:
        sheet = make_contact_sheet([(n, im) for n, _, im in files], columns, include_filenames=include,
                                   font_size=font_size, border_thickness=border, border_color=border_color,
                                   spacing=spacing, background=background)
        clickable_preview(sheet, "Contact sheet", "contact_sheet")
        data = encode_image(sheet, "PNG")
        st.download_button("Download contact sheet (PNG)", data, "pixelworks-contact-sheet.png", "image/png", type="primary")
    except Exception as exc: st.error(str(exc))


def render_scanner():
    st.title("Document scanner")
    st.caption("Handles are numbered top-left, top-right, bottom-right, bottom-left. Drag them to review the selection, then preview and apply a true perspective correction.")
    files = upload_images()
    if not files: return
    batch_fingerprint = tuple((name, hashlib.sha256(raw).hexdigest()) for name, raw, _ in files)
    if st.session_state.get("scanner_batch_fingerprint") != batch_fingerprint:
        st.session_state["scanner_batch_fingerprint"] = batch_fingerprint
        st.session_state["scanner_batch_results"] = {}
    mode = st.radio("Scanner mode", ["Automatic", "Manual"], horizontal=True, key="scanner_mode")
    effect = st.selectbox("Output enhancement", ["Color", "Grayscale", "Black & white"])
    from image_toolkit.scanner import detect_document, default_corners, enhance_scan, perspective_transform, validate_corners
    names = [n for n, _, _ in files]
    selected_name = st.selectbox("Image to scan", names)
    _, raw, image = next(row for row in files if row[0] == selected_name)
    token = f"{selected_name}:{hashlib.sha256(raw).hexdigest()}:{image.width}x{image.height}"
    state_key = f"scanner_editor_{token}"
    state = st.session_state.setdefault(state_key, {"corners": None, "detection": None, "reset": 0, "result": None})
    if mode == "Automatic" and state["detection"] is None:
        with st.spinner("Checking image contrast, edges and candidate regions…"):
            try:
                state["detection"] = detect_document(image)
            except Exception as exc:
                state["detection"] = type("DetectionFailure", (), {"corners": None, "confidence": 0.0,
                    "method": "error", "message": str(exc)})()
    if state["corners"] is None:
        detected = state["detection"] if mode == "Automatic" else None
        state["corners"] = detected.corners if detected and detected.corners else default_corners(image.size)
    if mode == "Automatic":
        detection = state["detection"]
        if detection and detection.corners:
            st.caption(f"Candidate: {detection.method} · confidence {detection.confidence:.0%}")
            if detection.confidence < .62: st.warning(detection.message)
            else: st.info(detection.message)
        else:
            st.warning("Automatic detection is uncertain. Drag the default handles to the document corners, or switch to Manual.")
    try:
        preview_bytes = encode_image(image, "JPEG", 88)
        result = four_point_editor(image=base64.b64encode(preview_bytes).decode("ascii"), corners=state["corners"],
                                   resetToken=state["reset"], maxWidth=880, key=f"four_corner_{token}", default=None)
        if isinstance(result, dict) and result.get("corners"):
            updated = result["corners"]
            if any(abs(float(a)-float(b)) > .25 for old, new in zip(state["corners"], updated) for a, b in zip(old, new)):
                state["result"] = None
                state["result_signature"] = None
            state["corners"] = updated
    except Exception as exc:
        st.error(f"The interactive corner editor could not load: {exc}")
        return
    c1, c2, c3 = st.columns(3)
    if c1.button("Reset corners", key=f"reset_{token}"):
        state["corners"] = default_corners(image.size); state["result"] = None
        state["result_signature"] = None; state["reset"] += 1; st.rerun()
    def return_to_automatic():
        st.session_state["scanner_mode"] = "Automatic"
        if state["detection"] is None:
            try: state["detection"] = detect_document(image)
            except Exception: pass
        if state["detection"] and state["detection"].corners:
            state["corners"] = state["detection"].corners
        else:
            state["corners"] = default_corners(image.size)
        state["result"] = None
        state["result_signature"] = None
        state["reset"] += 1
    c2.button("Use automatic detection", key=f"automatic_{token}", on_click=return_to_automatic)
    if c3.button("Clear corrected result", key=f"clear_result_{token}"):
        state["result"] = None
    fmt = st.radio("Download format", ["PNG", "JPEG"], horizontal=True, key=f"scan_fmt_{token}")
    result_signature = (tuple(tuple(round(float(value), 2) for value in point) for point in state["corners"]), effect)
    try:
        valid = validate_corners(state["corners"], image.size)
        corrected = enhance_scan(perspective_transform(image, valid), effect)
        left, right = st.columns(2)
        with left: st.image(image, caption="Original", use_container_width=True)
        with right: st.image(corrected, caption="Perspective-corrected preview", use_container_width=True)
        if st.button("Apply correction", type="primary", key=f"apply_{token}"):
            state["result"] = corrected.copy()
            state["result_signature"] = result_signature
            st.session_state["scanner_batch_results"][selected_name] = corrected.copy()
        if state["result"] is not None and state.get("result_signature") == result_signature:
            data = encode_image(state["result"], fmt, quality=92)
            st.download_button(f"Download corrected {fmt}", data, make_name(selected_name, fmt, "-scan"),
                               "image/png" if fmt == "PNG" else "image/jpeg", type="primary")
    except (ValueError, RuntimeError) as exc:
        st.warning(f"The current corner selection is invalid: {exc}")
    batch_results = st.session_state.get("scanner_batch_results", {})
    if batch_results:
        archive_data = make_zip([(make_name(name, fmt, "-scan"), encode_image(result, fmt, quality=92))
                                 for name, result in batch_results.items()])
        st.caption(f"{len(batch_results)} image(s) ready in the reviewed scan batch.")
        st.download_button("Download reviewed scans as ZIP", archive_data, "pixelworks-scans.zip",
                           "application/zip", key="scanner_batch_zip")


def render_pdf():
    st.title("Image to PDF")
    st.caption("One page per image. Select and arrange images in the order you want them in the PDF.")
    files = upload_images()
    if not files: return
    selected_names = st.multiselect("Images in PDF order", [f[0] for f in files], default=[f[0] for f in files])
    order_valid = True
    if selected_names:
        order_key = abs(hash(tuple(selected_names)))
        with st.expander("Set page order", expanded=False):
            ordered_names = [st.selectbox(f"PDF page {slot+1}", selected_names, index=slot,
                                          key=f"pdf_order_{order_key}_{slot}")
                             for slot in range(len(selected_names))]
        order_valid = len(set(ordered_names)) == len(selected_names)
        if order_valid:
            selected_names = ordered_names
        else:
            st.warning("Choose each selected image exactly once in the page-order controls.")
    size = st.selectbox("Page size", ["Original", "A4", "Letter"])
    fit = st.checkbox("Fit image to page while preserving aspect ratio", True, disabled=(size == "Original"))
    st.caption("Original uses each image's own pixel dimensions and DPI metadata. If DPI is missing, 96 DPI is assumed; image pixels are embedded without resizing.")
    pdf_signature = (tuple(selected_names), tuple((name, len(raw)) for name, raw, _ in files), size, fit)
    if st.session_state.get("pdf_signature") != pdf_signature:
        st.session_state.pop("pdf_result", None)
    if st.button("Create PDF", type="primary", disabled=not selected_names or not order_valid):
        try:
            from image_toolkit.pdf import images_to_pdf
            if size == "Original":
                lookup = {name: raw for name, raw, _ in files}
                ordered = []
                for name in selected_names:
                    with Image.open(BytesIO(lookup[name])) as source:
                        source.seek(0); source.load()
                        original_raster = source.copy()
                        original_raster.format = source.format
                        ordered.append(original_raster)
            else:
                lookup = {name: image for name, _, image in files}
                ordered = [lookup[name] for name in selected_names]
            pdf_bytes = images_to_pdf(ordered, page_size=size, fit=fit)
            st.session_state["pdf_result"] = pdf_bytes
            st.session_state["pdf_page_count"] = len(ordered)
            st.session_state["pdf_signature"] = pdf_signature
        except Exception as exc: st.error(f"PDF creation failed: {exc}")
    if st.session_state.get("pdf_result"):
        st.success(f"Created a {st.session_state['pdf_page_count']}-page PDF.")
        st.download_button("Download PDF", st.session_state["pdf_result"], "pixelworks-images.pdf", "application/pdf", type="primary")


def render_ocr():
    st.title("OCR text extractor")
    st.caption("Text recognition runs locally through Tesseract. Results are editable before download.")
    files = upload_images()
    if not files: return
    c1,c2,c3 = st.columns(3)
    language = c1.text_input("Tesseract language", "eng", help="Install matching system language data, e.g. tesseract-ocr-fra.")
    gray = c2.checkbox("Grayscale and contrast", True)
    threshold = c3.checkbox("Black & white threshold", False)
    c1,c2 = st.columns(2)
    resize_factor = c1.select_slider("OCR image scale", options=[.5,.75,1.,1.5,2.,3.], value=1.)
    deskew = c2.checkbox("Deskew enhancement", False)
    ocr_signature = (tuple((name, len(raw)) for name, raw, _ in files), language, gray, threshold, resize_factor, deskew)
    if st.session_state.get("ocr_signature") != ocr_signature:
        st.session_state.pop("ocr_results", None)
    if st.button("Extract text", type="primary"):
        from image_toolkit.ocr import extract_text
        success_texts, failures, zip_items = [], [], []
        for i, (name, _, _) in enumerate(files):
            st.session_state.pop(f"ocr_text_{i}", None)
        for index, (name, _, image) in enumerate(files):
            try:
                text, boxes = extract_text(image, language=language, grayscale=gray, threshold=threshold,
                                           resize_factor=resize_factor, deskew=deskew)
                if not text.strip(): st.info(f"{name}: no recognizable text was found.")
                success_texts.append({"name": name, "text": text, "boxes": boxes, "image": image})
            except Exception as exc:
                failures.append((name, str(exc)))
        for name, error in failures: st.error(f"{name}: {error}")
        st.session_state["ocr_results"] = (success_texts, failures)
        st.session_state["ocr_signature"] = ocr_signature
    if st.session_state.get("ocr_results"):
        success_texts, failures = st.session_state["ocr_results"]
        for name, error in failures:
            st.error(f"{name}: {error}")
        zip_items = []
        for index, result in enumerate(success_texts):
            name, image, boxes = result["name"], result["image"], result["boxes"]
            with st.expander(f"{name} — recognized text", expanded=True):
                clickable_preview(image, "Original OCR image", f"ocr_original_{index}")
                if not result["text"].strip():
                    st.info("No recognizable text was found in this image.")
                edited = st.text_area("Extracted text", result["text"], height=180, key=f"ocr_text_{index}")
                text_data = edited.encode("utf-8")
                st.download_button("Download TXT", text_data, make_name(name,"TXT","-text"), "text/plain", key=f"ocr_dl_{index}")
                if boxes:
                    preview = image.convert("RGB").copy(); draw = ImageDraw.Draw(preview)
                    for box in boxes:
                        draw.rectangle((box["left"],box["top"],box["left"]+box["width"],box["top"]+box["height"]), outline="#F14C75", width=2)
                    clickable_preview(preview, f"{len(boxes)} recognized word regions", f"ocr_preview_{index}")
            zip_items.append((make_name(name,"TXT","-text"), text_data))
        if len(files) > 1 and zip_items:
            st.download_button("Download all TXT files as ZIP", make_zip(zip_items), "pixelworks-ocr-text.zip", "application/zip", type="primary")


RENDERERS = {"Welcome": render_home, "Convert & compress": render_convert, "Resize": render_resize,
             "Crop, rotate & flip": render_crop, "Adjust image": render_adjust, "Image details": render_details,
             "DPI Setter": render_dpi_setter, "Contact sheet": render_contact,
             "Document scanner": render_scanner, "Image to PDF": render_pdf, "OCR text extractor": render_ocr}

try:
    RENDERERS[selected]()
except Exception as exc:
    st.error("This action could not be completed. Try a smaller or different image.")
    st.caption(f"{type(exc).__name__}: {exc}")

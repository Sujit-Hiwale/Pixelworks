"""Download helpers that keep generated artifacts in memory."""

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile
from image_toolkit.processing import unique_filenames


def make_zip(files: list[tuple[str, bytes]]) -> bytes:
    output = BytesIO()
    names = unique_filenames(name for name, _ in files)
    with ZipFile(output, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, (_, data) in zip(names, files):
            archive.writestr(name, data)
    return output.getvalue()

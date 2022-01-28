# stdlib
import hashlib
from pathlib import Path
import shutil


def file_copy(src: Path, dst: Path) -> None:
    shutil.copy(src, dst)


def file_md5(fname: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

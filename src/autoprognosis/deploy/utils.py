# stdlib
from contextlib import closing
import hashlib
from pathlib import Path
import shutil
import socket

# autoprognosis absolute
from autoprognosis.utils.pip import install

for retry in range(2):
    try:
        # third party
        import psutil

        break
    except ImportError:
        depends = ["psutil"]
        install(depends)


def get_ports(pid: int) -> list:
    ports = []
    p = psutil.Process(pid)
    for conn in p.connections():
        if conn.status != "LISTEN":
            continue
        ports.append(conn.laddr.port)

    return ports


def is_local_port_open(port: int) -> bool:
    host = "127.0.0.1"
    is_open = False
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        if sock.connect_ex((host, port)) == 0:
            is_open = True
    return is_open


def file_copy(src: Path, dst: Path) -> None:
    shutil.copy(src, dst)


def file_md5(fname: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

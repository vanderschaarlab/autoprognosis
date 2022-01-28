# stdlib
import multiprocessing
from multiprocessing import Process
from multiprocessing.process import BaseProcess
import os
from pathlib import Path
import time
from typing import Tuple

# adjutorium absolute
from adjutorium.apps.classification.classification_template import (
    classification_dashboard,
)
from adjutorium.apps.survival_analysis.survival_analysis_template import (
    survival_analysis_dashboard,
)
from adjutorium.deploy.utils import get_ports, is_local_port_open
import adjutorium.logger as log
from adjutorium.utils.serialization import load_model_from_file

BASELINE_PORT = 9000


def run_server(app_path: Path, port: int) -> None:
    app_params = load_model_from_file(app_path)

    if app_params["type"] == "risk_estimation":
        app = survival_analysis_dashboard(
            app_params["title"],
            app_params["banner_title"],
            app_params["models"],
            app_params["column_types"],
            app_params["encoders"],
            app_params["menu_components"],
            app_params["time_horizons"],
            app_params["plot_alternatives"],
        )
    elif app_params["type"] == "classification":
        app = classification_dashboard(
            app_params["title"],
            app_params["banner_title"],
            app_params["models"],
            app_params["column_types"],
            app_params["encoders"],
            app_params["menu_components"],
            app_params["plot_alternatives"],
        )
    else:
        raise RuntimeError(f"unsupported task {app.type}")

    app.run_server(
        debug=False,
        host="0.0.0.0",
        port=port,
    )


def get_app_name(app_path: Path) -> str:
    app_path = Path(app_path)
    return app_path.parts[-2]


def is_app_server_running(app_path: Path) -> bool:
    process_name = get_app_name(app_path)
    current_pid = os.getpid()
    current_ports = get_ports(current_pid)

    for p in multiprocessing.active_children():
        if p.name == process_name:
            if p.pid is None:
                continue
            ports = get_ports(p.pid)
            ports = [p for p in ports if p not in current_ports]
            return len(ports) > 0

    return False


def start_app_server(app_path: Path, daemon: bool = False) -> Tuple[BaseProcess, str]:
    output_url = "http://127.0.0.1:{}"
    process_name = get_app_name(app_path)
    current_pid = os.getpid()
    current_ports = get_ports(current_pid)

    log.info(f"ports current pid: {current_ports}")
    for p in multiprocessing.active_children():
        if p.name == process_name:
            if p.pid is None:
                continue
            ports = get_ports(p.pid)
            ports = [p for p in ports if p not in current_ports]
            if len(ports) == 0:
                p.kill()
            else:
                log.info(f"app already running {app_path}, {p.pid}, {get_ports(p.pid)}")

                return p, output_url.format(ports[0])

    port = BASELINE_PORT
    while is_local_port_open(port):
        port += 1

    log.info(f"starting app name {process_name}, {port}")
    p = Process(
        target=run_server, daemon=daemon, args=(app_path, port), name=process_name
    )

    p.start()

    for retry in range(5):
        assert p.pid is not None
        ports = get_ports(p.pid)
        if port in ports:
            break
        time.sleep(0.5)
    return p, output_url.format(port)


def stop_app_server(app_path: Path) -> None:
    process_name = get_app_name(app_path)
    for p in multiprocessing.active_children():
        if p.name == process_name:
            try:
                p.kill()
            except BaseException as e:
                log.error(f"failed to stop process {p.pid}, {e}")

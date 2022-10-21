# stdlib
import multiprocessing
import os
from pathlib import Path
import re

# autoprognosis absolute
from autoprognosis.deploy.utils import get_ports
import autoprognosis.logger as log
from autoprognosis.plugins import Plugins
from autoprognosis.utils.serialization import load_model_from_file

try:
    port = os.getenv("PORT")
    BASELINE_PORT = 9000
    if port:
        BASELINE_PORT = int(port)
except BaseException:
    BASELINE_PORT = 9000

chars = r"A-Za-z0-9/\-:.,_$%'()[\]<> "
shortest_run = 4

regexp = "[%s]{%d,}" % (chars, shortest_run)
regexp_b = regexp.encode()
pattern = re.compile(regexp_b)


def load_depends(app_path: Path) -> None:
    seen = set()
    with open(app_path, "rb") as f:
        data = f.read()
        strings = pattern.findall(data)
        for string in strings:
            decoded = string.decode()
            if "plugin_" in decoded and "autoprognosis" in decoded:
                path = Path(decoded)
                plugin = path.stem.split("plugin_")[1]

                if plugin in seen:
                    continue
                seen.add(plugin)

                Plugins().get_any_type(plugin)


def run_server_streamlit(app_path: Path, port: int = 9000) -> None:
    load_depends(app_path)

    app_params = load_model_from_file(app_path)

    if app_params["type"] == "risk_estimation":
        # autoprognosis absolute
        from autoprognosis.apps.survival_analysis.survival_analysis_template_streamlit import (
            survival_analysis_dashboard,
        )

        app = survival_analysis_dashboard(
            app_params["title"],
            app_params["banner_title"],
            app_params["models"],
            app_params["column_types"],
            app_params["encoders"],
            app_params["menu_components"],
            app_params["time_horizons"],
            app_params["plot_alternatives"],
            app_params["extras_cbk"],
            app_params["auth"],
        )
    elif app_params["type"] == "classification":
        # autoprognosis absolute
        from autoprognosis.apps.classification.classification_template_streamlit import (
            classification_dashboard,
        )

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


def get_app_name(app_path: Path) -> str:
    app_path = Path(app_path).resolve()
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


def start_app_server(app_path: Path, daemon: bool = False) -> None:
    return run_server_streamlit(app_path)


def stop_app_server(app_path: Path) -> None:
    process_name = get_app_name(app_path)
    for p in multiprocessing.active_children():
        if p.name == process_name:
            try:
                p.kill()
            except BaseException as e:
                log.error(f"failed to stop process {p.pid}, {e}")

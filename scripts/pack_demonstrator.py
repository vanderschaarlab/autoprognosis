# stdlib
from pathlib import Path
import shutil
import subprocess

# third party
import click


def build_wheel() -> Path:
    out = Path("dist")
    try:
        shutil.rmtree(out)
    except BaseException:
        pass

    subprocess.run("python setup.py bdist_wheel", shell=True, check=True)

    out_wheel = None
    for fn in out.glob("*"):
        if fn.suffix == ".whl":
            out_wheel = fn

    assert out_wheel is not None, "Invalid wheel"

    return fn


@click.command()
@click.option("--output", type=str, default="image_bin")
@click.option("--app", type=str)
def run(output: Path, app: str) -> None:
    output = Path(output)
    output_data = output / "third_party"
    try:
        shutil.rmtree(output)
    except BaseException:
        pass
    output.mkdir(parents=True, exist_ok=True)
    output_data.mkdir(parents=True, exist_ok=True)

    # Copy Adjutorium wheel
    local_wheel = build_wheel()
    shutil.copy(local_wheel, output_data / local_wheel.name)
    for fn in Path("third_party").glob("*"):
        if fn.suffix == ".whl":
            shutil.copy(fn, output_data / fn.name)

    # Copy server template
    for fn in Path("third_party/image_template").glob("*"):
        shutil.copy(fn, output / fn.name)

    # Copy server runner
    shutil.copy("scripts/run_demonstrator.py", output / "run_demonstrator.py")

    # Copy app
    shutil.copy(app, output / "app.p")

    # Update requirements txt
    with open(output / "requirements.txt", "a") as f:
        f.write(str(local_wheel.resolve()))
        f.close()


if __name__ == "__main__":
    run()

"""Guard denoising checkpoint reuse independently of image conversion."""
import json
from pathlib import Path


def check_resume(record, config):
    path = Path(record)
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("Denoising options changed; use a new output directory.")
    elif config["method"] != "mppca":
        raise ValueError("Cannot resume adaptive denoising without denoising.json; use a new output directory.")

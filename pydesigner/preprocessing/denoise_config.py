"""Guard denoising checkpoint reuse independently of image conversion."""
import json
from pathlib import Path


def check_resume(record, config, legacy_method="mppca"):
    path = Path(record)
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError("Denoising options changed; use a new output directory.")
    elif config["method"] != legacy_method:
        raise ValueError("Cannot resume the selected correction without its parameter record; use a new output directory.")

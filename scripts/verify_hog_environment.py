from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy
import PIL
import sklearn
import skimage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    output_path = Path(args.output_json).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "PASS",
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "is_virtual_environment": sys.prefix != sys.base_prefix,
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "numpy": numpy.__version__,
        "pillow": PIL.__version__,
        "scikit_image": skimage.__version__,
        "scikit_learn": sklearn.__version__,
    }

    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
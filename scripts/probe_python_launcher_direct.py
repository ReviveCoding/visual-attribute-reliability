from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--marker-json", required=True)
    parser.add_argument("--probe-name", required=True)
    args = parser.parse_args()

    marker_path = Path(args.marker_json).resolve()
    marker_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "status": "PASS",
        "probe_name": args.probe_name,
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "sys_prefix": sys.prefix,
        "sys_base_prefix": sys.base_prefix,
        "is_virtual_environment": sys.prefix != sys.base_prefix,
        "cwd": os.getcwd(),
    }

    marker_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
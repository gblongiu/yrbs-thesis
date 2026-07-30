import argparse
import platform
import sys

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import LOGS_DIR, RAW_FILE_2023
from src.utils.logging import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the local Python environment and required raw workbook presence.")
    parser.parse_args()

    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "raw_file_exists": RAW_FILE_2023.exists(),
    }
    write_json(LOGS_DIR / "environment_check.json", info)
    print("Wrote outputs/logs/environment_check.json")


if __name__ == "__main__":
    main()

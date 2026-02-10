import platform
import sys

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RAW_FILE_2023, COMBINED_FILE, LOGS_DIR
from src.utils.logging import write_json


def main() -> None:
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "raw_file_exists": RAW_FILE_2023.exists(),
        "combined_file_exists": COMBINED_FILE.exists(),
    }
    write_json(LOGS_DIR / "environment_check.json", info)
    print("Wrote outputs/logs/environment_check.json")


if __name__ == "__main__":
    main()

"""Run one agent campaign with a skill call per block. See src/loop/main.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.loop.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

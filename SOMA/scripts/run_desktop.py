"""Launch the SOMA desktop UI.

Usage (from ``SOMA/``)::

    python -m scripts.run_desktop [--synthetic] [--config configs/dev.yaml]
                                  [--data-dir data] [--checkpoint path] [--smoke]

Also works as ``python scripts/run_desktop.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soma.ui.desktop.app import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

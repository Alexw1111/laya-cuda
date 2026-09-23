"""Allow `python -m laya_cuda`."""
import sys

from .cli import main

sys.exit(main())

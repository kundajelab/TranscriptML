#!/usr/bin/env python3
from __future__ import annotations

import sys

from hparam_sweep_utils import main


if __name__ == "__main__":
    main(["summarize-sweep", *sys.argv[1:]])

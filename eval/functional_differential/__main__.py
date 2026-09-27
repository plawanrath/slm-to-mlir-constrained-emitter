"""Run the gold-differential functional evaluation end to end.

    python -m eval.functional_differential [--out-dir DIR] [--dialects arith,linalg]

Step 1 runs the Day-56 driver (gold gate + randomized differential trials);
step 2 builds the 180-row per-prompt table and checks it against the summary.
Requires the pinned LLVM 19.1.7 container (scripts/env/docker-compose.yml, or any
container named by SLM_MLIR_CONTAINER that passes the same checks).
"""
from __future__ import annotations

import sys

from scripts import day56_e12c_differential_frozen as driver
from scripts import day61_functional_per_prompt as per_prompt

if __name__ == "__main__":
    driver.main()
    sys.exit(per_prompt.main())

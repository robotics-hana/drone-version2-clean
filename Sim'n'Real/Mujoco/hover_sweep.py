"""Find an MPPI configuration that holds station for 20 s.

The controller holds position for ~4 s and diverges around t=8 s, with the arm
frozen. Every FSM phase that ever reported ARRIVED was <=4 s, so the whole
pipeline has been running inside the window before divergence -- which is why
arm slew rate looked like the culprit for so long. Horizon 8 at dt=0.05 is only
0.4 s of lookahead, which is short for a position loop on a drone.
"""
import io
import contextlib
import numpy as np
from hover_endurance import run

CONFIGS = [
    (25, 8), (25, 16), (25, 25), (25, 40),
    (50, 25), (50, 40),
]

if __name__ == "__main__":
    print(f"{'samples':>8} {'horizon':>8} {'lookahead':>10}  result")
    for ns, H in CONFIGS:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok = run(20.0, ns, H, move_arm=False, verbose=False)
        tail = [l for l in buf.getvalue().splitlines() if l.strip().startswith("=>")]
        line = tail[-1].strip() if tail else "(no summary)"
        print(f"{ns:>8} {H:>8} {H*0.05:>9.2f}s  {line[3:]}")

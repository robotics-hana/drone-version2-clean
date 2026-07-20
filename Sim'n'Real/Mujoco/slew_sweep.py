"""At horizon 16, how fast can the arm slew before the drone loses attitude?

Horizon 16 (0.80 s lookahead) holds station for 20 s with a frozen arm, where
horizon 8 diverged at ~2 s. A faster arm slew shortens EXTEND, which shortens
the whole episode, which matters because endurance is finite: at slew 0.05 the
episode is ~43 s but the drone only held ~26 s.
"""
import io
import contextlib
from hover_endurance import run

if __name__ == "__main__":
    print(f"{'slew':>6} {'swing time':>11}  result")
    for slew in (0.05, 0.15, 0.30, 0.50):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            run(30.0, 25, 16, move_arm=True, verbose=False, slew=slew)
        tail = [l for l in buf.getvalue().splitlines() if l.strip().startswith("=>")]
        line = tail[-1].strip()[3:] if tail else "(none)"
        print(f"{slew:>6.2f} {0.7467/slew:>10.1f}s  {line}")

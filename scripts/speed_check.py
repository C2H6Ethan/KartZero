"""Check live kart data and measure emulator speed, alone and in parallel.

Holding the accelerator from a standing start must raise the speed and move the
kart. If the memory addresses were wrong for our game, the numbers would not
behave that way. Run make_start_state.py first.

Usage: python scripts/speed_check.py [process counts, default 1 2 4 8 12 16]
"""

import multiprocessing as mp
import sys
import time
from multiprocessing.synchronize import Barrier, Event

from kartzero.game import Button, Game

FRAMES = 1800


def worker(ready: Barrier, go: Event, results: "mp.Queue[float]") -> None:
    with Game() as game:
        game.load_snapshot()
        game.hold(Button.A)
        ready.wait()
        go.wait()
        start = time.perf_counter()
        for _ in range(FRAMES):
            game.advance()
            game.kart.speed()
            game.kart.position()
        results.put(FRAMES / (time.perf_counter() - start))


def measure_parallel(count: int) -> list[float]:
    ready, go = mp.Barrier(count + 1), mp.Event()
    results: "mp.Queue[float]" = mp.Queue()
    procs = [mp.Process(target=worker, args=(ready, go, results)) for _ in range(count)]
    for proc in procs:
        proc.start()
    ready.wait()
    go.set()
    rates = [results.get() for _ in procs]
    for proc in procs:
        proc.join()
    return rates


def main() -> None:
    with Game() as game:
        game.load_snapshot()
        game.hold(Button.A)
        print("speed and position while holding the accelerator:")
        for second in range(12):
            game.advance(60)
            x, _, z = game.kart.position()
            print(f"  t={second + 1:2d}s speed={game.kart.speed():7.3f} x={x:9.1f} z={z:9.1f}")

    counts = [int(arg) for arg in sys.argv[1:]] or [1, 2, 4, 8, 12, 16]
    print("processes  each (frames/s)  total (frames/s)  total decisions/hour at 1 per 4 frames")
    for count in counts:
        rates = measure_parallel(count)
        total = sum(rates)
        print(f"{count:9d}  {total / count:15.1f}  {total:16.1f}  {total * 3600 / 4:12,.0f}", flush=True)


if __name__ == "__main__":
    main()

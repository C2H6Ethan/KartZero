"""Check live kart data and measure emulator speed, alone and in parallel.

Holding the accelerator from a standing start must raise the speed and move the
kart. If the memory addresses were wrong for our cartridge, the numbers would not
behave that way. Run make_start_state.py first.

Usage: python scripts/speed_check.py [process counts, default 1 2 4 8 12 16]
"""

import multiprocessing as mp
import sys
import time
from pathlib import Path

from desmume.controls import Keys, keymask
from desmume.emulator import DeSmuME
from desmume.mkds.mkds import RACER_PTR_ADDR, driver_t

ROOT = Path(__file__).resolve().parent.parent
ROM = ROOT / "game" / "mkds.nds"
STATE = ROOT / "savestates" / "f8c_time_trial.dst"
A = keymask(Keys.KEY_A)

SPEED_OFFSET = driver_t.speed.offset
POSITION_OFFSET = driver_t.position.offset
FX32_ONE = 4096  # game numbers are fixed point: 4096 means 1.0
FRAMES = 1800


def start_emulator():
    emu = DeSmuME()
    emu.open(str(ROM))
    emu.volume_set(0)
    emu.savestate.load_file(str(STATE))
    emu.input.keypad_update(A)
    return emu


def read_kart(emu):
    mem = emu.memory.signed
    racer = emu.memory.unsigned.read_long(RACER_PTR_ADDR)
    speed = mem.read_long(racer + SPEED_OFFSET) / FX32_ONE
    x = mem.read_long(racer + POSITION_OFFSET) / FX32_ONE
    z = mem.read_long(racer + POSITION_OFFSET + 8) / FX32_ONE
    return speed, x, z


def worker(ready, go, results):
    emu = start_emulator()
    ready.wait()
    go.wait()
    start = time.perf_counter()
    for _ in range(FRAMES):
        emu.skip_next_frame()
        emu.cycle(with_joystick=False)
        read_kart(emu)
    results.put(FRAMES / (time.perf_counter() - start))
    emu.destroy()


def measure_parallel(count):
    ready, go = mp.Barrier(count + 1), mp.Event()
    results = mp.Queue()
    procs = [mp.Process(target=worker, args=(ready, go, results)) for _ in range(count)]
    for proc in procs:
        proc.start()
    ready.wait()
    go.set()
    rates = [results.get() for _ in procs]
    for proc in procs:
        proc.join()
    return rates


def main():
    emu = start_emulator()
    print("speed and position while holding the accelerator:")
    for second in range(12):
        for _ in range(60):
            emu.cycle(with_joystick=False)
        speed, x, z = read_kart(emu)
        print(f"  t={second + 1:2d}s speed={speed:7.3f} x={x:9.1f} z={z:9.1f}")
    emu.destroy()

    counts = [int(arg) for arg in sys.argv[1:]] or [1, 2, 4, 8, 12, 16]
    print("processes  each (frames/s)  total (frames/s)  total decisions/hour at 1 per 4 frames")
    for count in counts:
        rates = measure_parallel(count)
        total = sum(rates)
        print(f"{count:9d}  {total / count:15.1f}  {total:16.1f}  {total * 3600 / 4:12,.0f}", flush=True)


if __name__ == "__main__":
    main()

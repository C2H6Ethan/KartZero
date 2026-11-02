# KartZero project notes

Where the project stands, and everything we've measured and decided. Last updated: 2 November 2026.

"Measured" means we ran it on this PC. Anything else is marked as a source or a guess.

## Where we are now

**Stage: setup finished, no training yet.** Nothing has learned to drive.

Working:
- The game runs without a window, and scripts can press its buttons and read its memory.
- A snapshot starts a Figure-8 Time Trial with Dry Bones in the ROB-BLS and no ghost.
- Replays are identical, and the GPU works with PyTorch.

Next steps, in order:
1. Confirm the checkpoint data in memory, because the reward needs it. Also check the clock and collision data.
2. Run the short speed tests below.
3. Write failing tests for the reward, then I write the reward. Observations and the training step follow the same way.
4. Run a first PPO training baseline.

## Goal and targets

Beat **1:27.832** on Figure-8 Circuit self-taught.

| Order | Target | Time | Source |
|---:|---|---:|---|
| 1 | Staff ghost | 1:36.481 | Shown in the game |
| 2 | **MarI/O Kart, the main goal** | 1:27.832 | Its video: laps 29.717, 29.107, 29.008, with 73 mini-turbos |
| 3 | My best time | 1:23.013, best lap 27.158 | My save. I don't know if it used the mushroom |
| 4 | Human record, stretch only | 1:11.387 | mkwrs, Taiga, 19 April 2024. Other sites list 1:11.510 and 1:11.625, so recheck before quoting it |

MarI/O Kart was rewarded for drifting and snaking. We reward only progress and time.

## Setup

- **PC:** Ryzen 7 5800X (8 cores, 16 threads), 32 GB RAM, RX 9070 XT, Windows 11.
- **Software:** Python 3.12 in `.venv`, because AMD's PyTorch needs it, and `requirements.txt` pins the packages. `py-desmume-mkds` 0.0.20 runs DeSmuME 0.9.12.
- **Local files git ignores:** the game goes in `game/mkds.nds`, and a copy of my save goes in `game/mkds.sav`. My real save is backed up on my DS, so the copy can be changed freely.

Build the snapshot:

```bash
python scripts/clear_ghosts.py
```

```bash
python scripts/make_start_state.py
```

Re-measure speed:

```bash
python scripts/speed_check.py
```

| Script | What it does |
|---|---|
| `scripts/clear_ghosts.py` | Removes personal ghosts from the save copy, leaving only the staff ghost |
| `scripts/make_start_state.py` | Goes through the menus, checks the driver and kart in memory, and saves the snapshot |
| `scripts/speed_check.py` | Drives with the accelerator held and measures speed with 1 or more emulators |
| `scripts/boot_check.py` | Takes screenshots while stepping through menus, for working out new menu paths |

## Measurements

**Game:** US version, game code `AMCE`.

**Memory on `AMCE`:**
- Confirmed: racer pointer and position, plus speed, which is in distance per frame. Speed × 60 matches the distance the kart covers per second.
- Also confirmed: at the race config pointer + `0x68`, the driver is at offset 0 and the kart at offset 4. Dry Bones is 8, R.O.B. is 11, ROB-BLS is 34, Streamliner is 23.
- Not confirmed: checkpoints, clock and collision data.
- Wrong: `COURSE_ID_ADDR` read 20 on every screen.

**Emulator speed, with drawing off and no AI running.** At one decision per 4 frames, decisions per hour = frames per second × 3,600 ÷ 4.

| Emulators | Each (frames/s) | Total (frames/s) | Decisions per hour |
|---:|---:|---:|---:|
| 1 | 271 | 271 | 244,000 |
| 8 | 193 | 1,542 | 1.39M |

An earlier Mario and B Dasher snapshot gave 992 frames/s with 4 emulators, and 2,108 with 16. Speed drops per emulator because they share 8 physical cores, though I didn't measure how much each cause contributes. Training will be slower than this because the AI needs time too.

**Other results:**
- Snapshot reload: about 119 ms, measured on an earlier snapshot.
- Replays: the same snapshot and buttons gave the same position to the last digit, in one process and across separate runs.
- ROB-BLS top speed: 7.72, compared with about 7.46 for the B Dasher.
- The starting mushroom: this setup starts with one in the item box. Pressing L used it, speed reached 10.04, and the box was empty afterwards.
- GPU: PyTorch 2.9.1 with ROCm 7.2.1 works with driver 32.0.32015.7029. A training step on a small network with a batch of 4096 took about 1.2 ms.

**The driver matters.** R.O.B. and Dry Bones in the Streamliner, picked through the real menus, with identical buttons:
- Accelerator only: exactly the same.
- Long mixed runs: 2 of 6 matched, and the other 4 split apart partway through.
- Top speed: the same or nearly the same.

The wiki points to driver weight and a hidden off-road penalty, but the cause isn't confirmed. Never swap the driver and assume results carry over.

## Decisions

- **Dry Bones in the ROB-BLS.** MarI/O Kart's video shows this pair, which keeps the comparison fair. Its code doesn't name them. A fan wiki lists the record holder using it too. The save has all 36 karts for every driver, and ROB-BLS is kart 7 for Dry Bones. MarI/O Kart's code loads a complete save called `100percent.sav`, which is likely how it got the pair. We won't download someone else's save.
- **No ghost in the race.** The menu is set to Ghost Data: None, and my personal ghosts are removed from our copy.
- **Items off until we beat 1:27.832.** MarI/O Kart's controls have no item button, so a free boost would make our win unfair. Random presses would also waste the mushroom at the start of most runs, which is reasoning, not a test. Build the button as a setting, try it after we win, and report times with and without items separately.
- **Figure-8 only.** The best racing results usually come from track-specific training, such as [GT Sophy](https://www.gtplanet.net/gran-turismo-7s-sophy-artificial-intelligence-driving-system-revealed/) and [Fuchs et al.](https://arxiv.org/abs/2008.07971). Transfer results are mixed, and I found no direct comparison on this game. Don't give the AI a track name, so its senses stay usable elsewhere. Testing other tracks is an idea for after the main goal.
- **Emulator only, no rewrite.** At this speed we don't need a custom simulator.
- **Training approach.** Keep the environment independent of the algorithm. Start with PPO because it's simpler to learn, then measure a replay-based method such as IQN, because emulator frames are expensive.
- **Decisions every 4 frames to start,** with `gamma` set to match. This is a starting guess we can test. Let the kart recover from crashes, within a time limit.
- **Hide the emulator behind our own small interface,** and use the CPU if the GPU ever fails, because a small network may not need it.

## Emulator traps

- The library's `MarioKart` class crashes on Windows because `libdesmume.dll` doesn't export `MMU`. Use the plain `DeSmuME` class with `emu.memory`, and write our own kart data reader.
- Run one emulator per process. A second one in the same process crashes.
- The emulator saves to `mkds.dsv` beside the Python install, not in the project, and all emulators share that path. An old save changes the menus, so `make_start_state.py` deletes it first. A `desmume.ini` didn't move it in either place tested.
- Menus depend on the save. For example, this save adds a Ghost Data step and a fresh save adds first-run prompts. If the save changes, re-check the menu path.
- The racer pointer can hold other values while a race loads. Trust it only once the race screen is up.
- Track names highlight by themselves as a preview, so a highlight doesn't mean the cursor is there.
- Changing the driver through memory during loading works, but it starts the race one frame late, so that method is not for comparisons. The final snapshot doesn't use it.
- AMD's downloads can fail partway with an SSL error. Download with resume, using `curl -C -`, then install the files.

## Speed tests to run before training

None of these has been tested yet. Today's speed is good enough, so they don't block training.

1. **JIT mode.** The log says `CPU mode: Interpreter`. The library has no setting for it, so this probably needs a rebuilt DLL.
2. **3D rendering off.** Each emulator starts 16 rendering threads. Check whether they keep working while drawing is skipped.
3. **Pinning emulators to CPU cores.** Easy to test, but it probably won't raise the total much.

Keep a change only if replays stay identical and the physics don't change.

## How long training may take

These are guesses until the first real run. They assume about 1M decisions per hour. Only the first row has outside evidence (Linesight on Trackmania, a different game).

| Milestone | Decisions | Time |
|---|---:|---:|
| Finishes races regularly | About 1M | About 1 hour |
| Beats the staff ghost | 3–10M | 3–10 hours |
| Beats MarI/O Kart | 10–50M | 10–50 hours |

## Bugs to test for

Reading MarI/O Kart's code suggested these bugs. We use its ideas, not its training code.

- The first observation after a reset must come from after the snapshot loads.
- Reward terms must not change state that other terms read, and reversing must never pay.
- A cooldown must actually apply, and reward terms must not pay every frame by accident.
- The save slot we write must be the slot we load, and every slot must be reachable.

## Ghost slots in the save

Ghosts start at `0xAF00` in the save, at `0xE00` bytes each, in 32 course slots (layout from MarI/O Kart's code). An empty slot starts with `0xFF 0xFF`. Clearing slots this way removed the ghosts, and the game accepted the save (tested).

## Still unknown

- Exactly what the driver changes.
- Whether finishing a race writes to the shared `mkds.dsv` while emulators run in parallel. If so, stop sharing it, which probably needs the rebuilt DLL.
- Whether my 1:23.013 used the mushroom.

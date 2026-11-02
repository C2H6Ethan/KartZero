# KartZero project notes

Where the project stands, and everything we've measured and decided. Last updated: 2 November 2026.

"Measured" means we ran it on this PC. Anything else is marked as a source or a guess.

## Where we are now

**Stage: second training run, without reverse.** The first run learned to turn around and drive the track in reverse. In a replay it was reversing 99.2% of the time at speed -3, with no forward driving above speed 5. That earned points because the kart still moved the right way along the track, and it explains why progress stopped rising at about 80%: 200 seconds in reverse only covers about 79% of a race. So the reverse button was removed and training restarted from nothing.

**Official times come from test laps.** Each saved version gets one race with no exploration, always the likeliest button, in `runs/<run>/versions/test_laps.csv`. `watch.py --list` shows them. A separate process runs them, so training doesn't wait, and any untested versions get tested on the next `--resume`. Practice can mislead: the first AI reached 80% with dice rolls but just held reverse on top picks. The live viewer rolls the dice, so it shows practice, not test results.

Working:
- The `kartzero` package runs the game without a window, presses buttons, and reads kart, race and track data through simple methods.
- A snapshot starts a Figure-8 Time Trial with Dry Bones in the ROB-BLS and no ghost.
- Race progress is confirmed good enough for the reward.
- Replays are identical, and the GPU works with PyTorch.

The reward, observations and learning maths are finished: I wrote `kartzero/reward.py`, `kartzero/observation.py` and `kartzero/training.py`, and their tests pass.

Next steps, in order:
1. Watch the first PPO training run, and compare it with my prediction: it learns from corners going badly and stops picking the moves that led there.
2. Watch whether it discovers mini-turbos by itself. A replay at about 14.5 minutes had 0 boosts in 5 races, with the mini-turbo count never above 1. Boosts aren't logged yet.
3. Use the test laps and training logs in `runs/` to decide what to adjust.
4. Measure a replay-based method such as IQN against PPO.

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

Install the packages in `requirements.txt`, then install our own package:

```bash
pip install -e .
```

Run the tests, which don't need the game:

```bash
python -m pytest
```

Check the types:

```bash
python -m mypy
```

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

## Code

| Class | Use it for | Examples |
|---|---|---|
| `Game` in `kartzero/game.py` | Running the game | `game.start_race()`, `game.hold(Button.A)`, `game.advance(4)`, `game.open_window()` |
| `Kart` in `kartzero/kart.py`, as `game.kart` | Kart and race data | `speed()`, `position()`, `heading()`, `drift_angle()`, `turning_rate()`, `mini_turbo_count()`, `boost_time_left()`, `is_off_road()`, `is_on_ground()`, `race_progress()`, `has_finished()`, `lap()`, `checkpoint()` |
| `Track` in `kartzero/track.py`, as `game.track` | The track layout | `checkpoint_count()`, `checkpoint_center(index)`, `collision_triangles()` |
| `RaceEnvironment` in `kartzero/environment.py` | One race, one decision at a time, and the 12 actions | `reset()`, `step(action)` |
| `DriverNetwork` in `kartzero/network.py` | Button odds and the guess of future points | `network(observations)` |
| `Trainer` in `kartzero/learner.py` | Driving the emulators, then learning from each chunk | `run_training(settings, minutes, viewer)` |
| `run_lap_tester` in `kartzero/test_lap.py` | Driving an official test lap for each saved version, in its own process | `drive_test_lap(network, environment)` |
| `run_viewer` in `kartzero/viewer.py` | Showing the AI live in a resizable window, rolling the dice like training. It finishes each race before loading the newest AI, so it never switches mid-race | `run_viewer()` |
| `GameWindow` in `kartzero/window.py` | The window itself. The emulator's own window can't stretch its picture | `show(image)` |
| `TrackSensors` in `kartzero/sensors.py` | Distances from the kart | `wall_distances(position, heading)`, `road_edge_distances(position, heading)` |

Only one `Game` can run per process. Tests in `tests/` use fake memory.

| Script | What it does |
|---|---|
| `scripts/clear_ghosts.py` | Removes personal ghosts from the save copy, leaving only the staff ghost |
| `scripts/make_start_state.py` | Goes through the menus, checks the driver and kart in memory, and saves the snapshot |
| `scripts/speed_check.py` | Drives with the accelerator held and measures speed with 1 or more emulators |
| `scripts/replay_check.py` | Checks that the same buttons give the same result |
| `scripts/progress_check.py` | Checks race progress by driving forward, reversing and completing laps with a test driver |
| `scripts/train.py` | Trains on 8 emulators with a live window. Closing the window stops training and saves. Results go in `runs/`, and the newest AI in `checkpoints/latest.pt`. Add `--resume` to carry on the same run from that save |
| `scripts/watch.py` | Shows the newest saved AI racing in the window, with no training. `--list` shows old versions, and `--hours 30` plays the one nearest 30 hours of training |
| `scripts/observation_check.py` | Builds the AI's inputs through a real race, and shows the range of each one |
| `scripts/sensor_check.py` | Checks the kart readings and sensors on the real game, and draws a map to check by eye |
| `scripts/watch_test_driver.py` | Opens a window to watch the test driver race all 3 laps at normal speed |
| `scripts/boot_check.py` | Takes screenshots while stepping through menus, for working out new menu paths |

## Measurements

**Game:** US version, game code `AMCE`.

**Memory on `AMCE`:**
- Confirmed: racer pointer and position, plus speed, which is in distance per frame. Speed × 60 matches the distance the kart covers per second.
- Also confirmed: at the race config pointer + `0x68`, the driver is at offset 0 and the kart at offset 4. Dry Bones is 8, R.O.B. is 11, ROB-BLS is 34, Streamliner is 23.
- Also confirmed: race progress, lap progress, checkpoint and lap, plus the checkpoint positions on the track (see below).
- Also confirmed by driving: `driftRotY` (about ±60° in a full drift), `driftLeftRightCount` (mini-turbo ready at 4), `boostTimer` (55 frames from a mini-turbo), `yRotSpeed` (read it as signed), `floorColType` (0 road and 3 off-road on Figure-8), and `floorDriverColType` (-1 in the air).
- Not useful for sliding: `direction`, `drivingDirection` and `velocity` barely differ in a drift, so we use `driftRotY`.
- Not confirmed: the clock.

**Sensors, tested on the game:**
- Wall sensor: 4 crashes all read 10.5 to 10.8 on the nearest wall, which is the kart's size. It never read a wall touching the kart in a race with no crashes. The thin wall pieces above the road don't block it.
- Road edge sensor: paints the road on a top-down grid once, then walks along each ray until the surface changes. Off the road it measures the way back. It agreed with the game's floor type at 97.5% of decisions in a full race, and it's accurate to about two grid squares (8 units). My first version, using triangle sides, gave false edges inside the road and was replaced.
- Cost: both sensors every decision lower speed from 495 to 479 frames/s (about 3%).
- The track has 902 collision triangles, and reading them is fast.
- Observations on the real game: all 36 inputs arrive in a full race with no invalid values, at 475 frames/s for one emulator. Distances to the second and third checkpoints reach 1.58 and 2.42 on long straights, above the usual -1 to 1. I kept that for now and can scale it if training suffers.
- AI runs never receive images. `scripts/sensor_check.py` draws a map only for checking the sensors by eye.
- Wrong: `COURSE_ID_ADDR` read 20 on every screen.

**Race progress, tested by driving:**
- It runs from 0 at the start line and rises about a third per lap: 0.3333 at the start of lap 2 and 0.6665 at the start of lap 3.
- Over two laps with the test driver, it rose 1,301 times out of 1,301 samples or held steady, with no drops. The biggest rise was only twice the typical one, so it pays no surprise jumps.
- Reversing makes it fall. Reversing over the start line makes it negative, smoothly, and driving back over the line brings it up again, also smoothly.
- **Use race progress for the reward, not lap progress.** Lap progress jumps between 0.99 and 0 at the line.
- The lap counter reads 0 behind the start line, so it changes from 0 to 1 just after the race starts.
- Figure-8 has 26 checkpoints, and the test driver passed all of them.
- The test driver finished all 3 laps in about 2:08, steering only, with no drifting. It isn't a target, but it gives us a rough idea of plain driving. Lap changes came at the same times with and without the window.
- **The finish:** race progress passes 1.0 (1.0002) and the lap counter goes to 4. **The game then drives the kart by itself and progress keeps rising** (1.1096 ten seconds later), and it used the mushroom. So a run must end the moment progress reaches 1.0, or the AI collects points for driving it didn't do.

**Emulator speed, with drawing off and no AI running.** At one decision per 4 frames, decisions per hour = frames per second × 3,600 ÷ 4.

| Emulators | Each (frames/s) | Total (frames/s) | Decisions per hour |
|---:|---:|---:|---:|
| 1 | 451 | 451 | 406,000 |
| 8 | 297 | 2,379 | 2.14M |
| 12 | 218 | 2,610 | 2.35M |
| 16 | 188 | 3,005 | 2.70M |

Speed drops per emulator because they share 8 physical cores, though I didn't measure how much each cause contributes. Training will be slower than this because the AI needs time too.

**During real training, with the AI running on 8 emulators:** about 1.29M decisions per hour, measured over 2 minutes with no viewer. That's slower than the 2.14M with no AI, so the time guesses below may need adjusting.

**Speed tests, all finished:**
- **JIT mode: kept.** About 1.8 times faster per emulator (about 253 to 452 frames/s), and 8 emulators went from about 1,543 to about 2,335 frames/s in the comparison test. `Game` turns it on, with a check that stops it writing to the wrong place if the library version changes. Compared with the interpreter, 6 random runs of 1,500 decisions matched at every decision, and the test driver's lap timings over a full race were identical.
- **Larger JIT block size: not used.** Sizes 50 and 100 gave only about 3% more speed. Replays still matched at size 100, so it isn't worth changing from the default of 12.
- **Turning off graphics: no gain.** With drawing skipped, CPU time matches real time (1.01), so the rendering threads are idle. Turning graphics layers off changed speed by under 1%. Textures can't be skipped safely, because the game's own code handles them and changing that could change the game.
- **Pinning emulators to CPU cores: no gain.** 8 emulators gave 2,314 frames/s pinned, compared with 2,335 unpinned.
- **A live viewer: about 2% cost.** 8 emulators plus one extra emulator drawing at 60 frames/s gave 2,296 frames/s, compared with 2,335 without it. This drew frames but opened no window, and it played no AI, so re-measure it with the real viewer.

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

- **No reverse button.** The first AI got stuck driving in reverse. MarI/O Kart's controls had no reverse either, so the comparison stays fair, and the AI can still slow down by letting go of A. This changes the buttons, not the reward, so it's still self-taught. That leaves 9 actions and 35 inputs.
- **Old versions are kept.** Each is saved 3% later than the last, and at least 25,000 decisions apart, so about 200 by 100 million decisions, in `runs/<run>/versions/`. Each records its total training time, which only counts time spent training and adds up across resumes. Each is about 0.6 MB.
- **First list of what the AI senses (`kartzero/observation.py`).** Speed, drift angle, turning rate, mini-turbo count, boost time left, off-road, on the ground, the buttons held, 9 wall distances, 9 road edge distances, and the distance and angle to the next 3 checkpoints. That's 36 numbers, all relative to the kart, with no map position and no track name. Only add something if training shows a need, because changing the inputs usually means restarting training.
- **Reward numbers: 2000 points per unit of race progress, minus 0.25 per decision.** I chose these: the per-decision values are small, stuck is -0.25, reversing -0.77, test driver pace +0.79, and MarI/O Kart pace +1.27. They're a first choice to test in training. Only the ratio changes which runs rank higher.
- **Dry Bones in the ROB-BLS.** MarI/O Kart's video shows this pair, which keeps the comparison fair. Its code doesn't name them. A fan wiki lists the record holder using it too. The save has all 36 karts for every driver, and ROB-BLS is kart 7 for Dry Bones. MarI/O Kart's code loads a complete save called `100percent.sav`, which is likely how it got the pair. We won't download someone else's save.
- **No ghost in the race.** The menu is set to Ghost Data: None, and my personal ghosts are removed from our copy.
- **Items off until we beat 1:27.832.** MarI/O Kart's controls have no item button, so a free boost would make our win unfair. Random presses would also waste the mushroom at the start of most runs, which is reasoning, not a test. Build the button as a setting, try it after we win, and report times with and without items separately.
- **Figure-8 only.** The best racing results usually come from track-specific training, such as [GT Sophy](https://www.gtplanet.net/gran-turismo-7s-sophy-artificial-intelligence-driving-system-revealed/) and [Fuchs et al.](https://arxiv.org/abs/2008.07971). Transfer results are mixed, and I found no direct comparison on this game. Don't give the AI a track name, so its senses stay usable elsewhere. Testing other tracks is an idea for after the main goal.
- **Emulator only, no rewrite.** At this speed we don't need a custom simulator.
- **Training approach.** Keep the environment independent of the algorithm. Start with PPO because it's simpler to learn, then measure a replay-based method such as IQN, because emulator frames are expensive. Evolution ("breeding") is the other main option: it's left out because it mostly scores whole runs, which gives it less to learn from per race. It's a possible later experiment, not something we've ruled out by testing.
- **Decisions every 4 frames to start,** with `gamma` set to match. This is a starting guess we can test. Let the kart recover from crashes, within a time limit.
- **Use the CPU if the GPU ever fails,** because a small network may not need it.

## Emulator traps

- The library's `MarioKart` class crashes on Windows because `libdesmume.dll` doesn't export `MMU`. Our `Game` and `Kart` classes use the plain `DeSmuME` class instead.
- Run one emulator per process. A second one in the same process crashes.
- JIT is switched on through a memory location specific to this exact DLL. If `py-desmume-mkds` changes version, `Game` stops with an error and the location must be found again.
- The emulator saves to `mkds.dsv` beside the Python install, not in the project, and all emulators share that path. An old save changes the menus, so `make_start_state.py` deletes it first. A `desmume.ini` didn't move it in either place tested.
- Menus depend on the save. For example, this save adds a Ghost Data step and a fresh save adds first-run prompts. If the save changes, re-check the menu path.
- The racer pointer can hold other values while a race loads. Trust it only once the race screen is up.
- Track names highlight by themselves as a preview, so a highlight doesn't mean the cursor is there.
- Changing the driver through memory during loading works, but it starts the race one frame late, so that method is not for comparisons. The final snapshot doesn't use it.
- AMD's downloads can fail partway with an SSL error. Download with resume, using `curl -C -`, then install the files.

## How long training may take

These are guesses until the first real run. They assume about 1.5M decisions per hour, since the AI will slow things below the 2.14M maximum for 8 emulators. Only the first row has outside evidence (Linesight on Trackmania, a different game).

| Milestone | Decisions | Time |
|---|---:|---:|
| Finishes races regularly | About 1M | About 40 minutes |
| Beats the staff ghost | 3–10M | 2–7 hours |
| Beats MarI/O Kart | 10–50M | 7–33 hours |

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

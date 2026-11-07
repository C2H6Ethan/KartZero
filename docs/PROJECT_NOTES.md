# KartZero project notes

Where the project stands, and everything we've measured and decided. Last updated: 4 November 2026.

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

**The second run (`runs/20261102-213453`) stalled at about 1:42 in game time.** After 21.7 hours and 31M decisions, test laps sat between 1:39.1 and 1:44.3 on our old clock, and the best, 1:39.133 (1:41.478 in game time), came at 14.9 hours. The 21.7-hour version is 1:42.967 in game time (1:40.60 on the old clock). See "The race clock" below. Practice races averaged about 105 seconds from 2 hours onward, so almost all the learning happened in the first 2 hours. A test race with the 21.7-hour version, measured:
- It's at 7.5 speed or above for 94% of the race, against a top speed of 7.72, and never goes off-road.
- Its drift angle is above 5° 29% of the time, but it fires 0 mini-turbos, and the count never goes above 1. At 2 hours the angle stayed near 0 and it drove the same time. Later checks (below) showed most of that angle comes from hops, not real drifts.
- It's still fairly random: the top button averages 66% likely, and the randomness rose from 0.50 at 2 hours to 0.97 (the most possible is 2.20).

So about 1:42 looks like the limit of driving at top speed without boosts, and the gap to 1:27.832 is mini-turbos. Rough estimate, not measured: one mini-turbo adds about 127 units of distance, which is about 5.5 points, and 73 of them would save about 20 seconds before the cost of charging them.

**It hops instead of drifting (measured 3 November, sticky 0.5 run at about 1 hour, top picks).** Its drift angle is above 5° for 76% of the race, but reading its buttons decision by decision shows a repeated pattern: press R with no direction (a hop), hold it with no steering while the angle shrinks, let go of R and steer, then hop again. A real drift needs R pressed with a direction, then R kept held. It sometimes starts one (count 1, after R with LEFT), but lets go of R on the very next decision. So the count stays at 0 or 1, and the mini-turbo rhythm never starts. The other two runs drive the same way. My first summary counted left and right presses on either side of a hop as "switches", which made it look closer to snaking than it is. The made-up drift model assumed real drifts, so its numbers don't apply to this.

**Sticky practice comparison: no winner (3 November).** In 2 hours each (about 0.91M decisions), sticky 0, 0.25 and 0.5 had 0, 0 and 1 boosts in practice. Average practice reward didn't rise in any of them. The final AIs drove 1:43.28, 1:44.24 and 1:43.83 in game time on top picks, none better than the 1:42.967 they started from. Test laps in game time, over the 34 versions after the start: best 1:41.99, 1:42.90 and 1:42.06, average 1:43.67, 1:44.93 and 1:43.37. So all three got slightly worse on average, and none beat the old run's best of 1:41.48. All three still hop and let go of R straight away, and their drift habits shift around from version to version. So plain stickiness, at least at these chances and this length, doesn't unlock mini-turbos.

The setup: three 2-hour runs started together, with sticky chances of 0, 0.25 and 0.5. All start from the 21.7-hour version, use 2 emulators each, and use 1,024 decisions per chunk, so each update still learns from 2,048 decisions. Each gets its own `checkpoints/compare-sticky<P>.pt` and a run folder ending in `-sticky<P>`. They run at about 0.40M decisions per hour each. The training log now has `boosts_per_minute` and `picks_ignored`, and `settings.json` marks uncommitted code with `+uncommitted`.

**GAE and exploration comparison: GAE slightly ahead, still no boosts (4 November).** Three 2-hour runs with the same setup as the sticky comparison and no stickiness: `gae` (advantage blend 0.95), `explore` (exploration weight 0.03, up from 0.01, a guess) and `both`. The baseline is the `sticky0` run. About 1.07M decisions each, at about 0.54M per hour each.

| Run | Boosts in practice | Practice reward, first → last 10% | Best test lap | Average test lap | Average of last 10 |
|---|---:|---:|---:|---:|---:|
| baseline (`sticky0`) | 0 | 1.045 → 1.025 | 1:41.99 | 1:43.67 | 1:43.32 |
| `gae` | 0 | 1.043 → 1.055 | **1:41.40** | **1:42.73** | **1:42.49** |
| `explore` | 0 | 1.042 → 0.969 | 1:42.56 | 1:45.01 | 1:44.63 |
| `both` | 0 | 1.027 → 1.011 | 1:42.91 | 1:44.29 | 1:43.81 |

Averages are over the versions after the start (34 for the baseline, 39 or 40 for the others). `gae` is the only run whose practice reward rose, and its average test lap is about 0.9 seconds better than the baseline. Its best, 1:41.40 at 1.3 hours, beats the old run's 1:41.48, but only by 0.08 seconds. That's one run each with one seed, so the gap could partly be luck. More exploration made things worse, alone and with GAE. None of the four runs fired a single boost in practice, so the mini-turbo problem is unsolved.

### Mini-turbo experiments (started 7 November)

The goal: find a general change, with no reward for technique, that lets the AI discover mini-turbos. Each attempt is listed here with what it showed.

**1. How the game counts a mini-turbo (measured, frame by frame).** The hop lasts about 13 frames. The drift starts on landing, at count 1. A steer only counts if it comes at least 10 frames after the last counted one, which is 3 decisions. Steers 4 to 8 frames apart never got past count 2. Steers 10 to 32 frames apart all reached 4, and no upper limit showed up. Four counted steers (right, left, right, left in a left drift), then letting go of R, give a 55-frame boost. Letting go at count 2 gives nothing. A plain R press with no direction between steers doesn't break the count. The fields `driftLeftCount`, `driftRightCount` and `driftLeftRightTimeout` (set to 10 by each counted steer, then counting down) show this.

**2. What one mini-turbo is worth inside the AI's own race (measured).** I paused the `gae` run's 2-hour version's test race every 20 decisions and compared, over the next 120 decisions: keep driving, one scripted mini-turbo then keep driving, or the same drift with no boost.
- The better of the two mini-turbo directions gained **+2.3 points on average** (median +2.4, better at 53 of 69 points). That's about 0.1 seconds.
- The wrong direction often lost 20 to 120 points.
- A drift with no boost lost **15 points on average**, even in the better direction, and was better at only 4 of 69 points.

So half a mini-turbo is punished, and the payoff only arrives after the whole move, about 17 decisions. The AI's habit of letting go of R straight away is the right lesson from what it has seen. This is an exploration trap, not a missing reward.

**3. Chains of scripted mini-turbos (measured, every 60 decisions).** One gained +1.9 on average. Two in a row lost 16.8, three lost 19.4 and four lost 25.7, because the script steers blindly and wrecks the line. So this doesn't measure the full prize: a mini-turbo only pays if the drift's steering follows the track. Our evidence of the full prize is still MarI/O Kart's 1:27.8 with 73 of them.

**4. How often random play fires a mini-turbo (measured).** 30,000 decisions (33 minutes of game time) each, restarting every 60 seconds, with 7 kinds of dice: a new random button every decision, sticky 0.5, sticky 0.8, εz-greedy (holds of heavy-tailed length, from Dabney et al. 2020), and separate dice per button changing with chance 0.1, 0.2 or 0.3. **All 7 gave 0 boosts.** Count 2 showed up in at most 0.06% of decisions and count 3 never. So rolling dice differently can't find this alone. Exploring needs to carry on from rare moments, not just press buttons differently.

**5. Research (7 November).** I found no kart AI that learned mini-turbos without being rewarded for them. A Mario Kart Wii agent (Beyond The Rainbow, 2024) had drift buttons but a shaped reward. Useful general ideas: temporally extended exploration (Dabney et al.), Go-Explore's archive of rare states (Ecoffet et al.), and off-policy replay like Linesight's IQN.

**6. Two new options, both general (built and tested, 7 November).**
- `--sticky-per-button`: the 9 actions are 3 pedal choices (nothing, A, A + R) × 3 steering choices. With this on, the pedal and the steering roll their own sticky dice, with the same chance. When only one half of a pick was used, PPO updates only that half's odds. The earlier idea of separate dice for R and for direction was set aside because it hinted at drifting. This version treats both halves the same way, but it's still a judgement call.
- `--archive-start-chance P` (`kartzero/archive.py`), in the style of Go-Explore: each emulator counts how often it sees each "cell" and saves game states for the rarest ones. A cell is the checkpoint plus the 11 kart inputs (speed, drift angle, turning, count, boost, off-road, ground and buttons), each rounded to a quarter of its range, the same rule for all. A practice race then starts from a saved state with chance P, picked with weight 1/√visits, and stops after 300 decisions. Saves are rationed to one per 500 decisions per emulator, because one takes 0.22 s (a load takes 0.12 s, and each is 2.7 MB). Those races don't count towards finishes or best times, and the states are deleted when training stops. The reward is unchanged.
- The log now has `share_at_count_2` and `share_at_count_3`, only for watching.

**7. First archive run: a design mistake (stopped at 23 minutes).** Two runs from scratch, 4 emulators each, GAE 0.95, sticky 0.75 per button, the same seed: `perbutton`, and `archive-noexplore` with archive chance 0.5. At 23 minutes both drove about 1:48 in test laps, and neither reached count 2 in recent practice, even after 128 archive races. The mistake: Go-Explore returns to a rare moment and then explores *randomly*, but mine handed control straight to the AI, which had already learned to let go of R. So it could never get past count 1. I stopped it. The folder is kept as `20261107-124558-archive-noexplore`.

**8. Archive with random exploring after the return (running).** After an archive start, the first 0 to 30 decisions (a random length) are random buttons. They're marked as ignored picks, so PPO doesn't push the odds with them, but the value guess learns from them. Each moment reached that way can be archived too, so the hope is count 1 → 2 → 3 → 4 a step at a time. Once the AI itself takes over at count 4, simply letting go of R fires a boost, which it can learn from directly, and then work backwards from there. Restarted as `archive` with `--archive-explore-decisions 30`, alongside the still-running `perbutton`.

**Result of 7 and 8 (stopped at 77 and 52 minutes):** no boosts at all in either. `perbutton` reached 1.0M decisions, with count 2 in at most 0.05% of practice and best test lap 1:46.15. `archive` reached 0.58M decisions and 279 archive races, with count 2 in about 0.01% of practice and best test lap 1:46.98. Returning to rare moments didn't help, because the rare moments it kept were mostly unrelated (slow, off-road, odd turning), and random buttons from them still rarely held R long enough.

**10. Pure Go-Explore, with no learning (measured).** It seeded the archive with the trained AI's own race, then kept returning to rare moments and pressing random buttons for 40 decisions, saving every new cell. Each save costs 0.22 s, so it ran only about 20 decisions per second. After 40 minutes (60,880 decisions, 8,823 cells) it had reached count 2, but never 4 and never a boost. The library can only save states to files. Its numbered slots are files too, and loading one writes backup states into the Python install folder, so don't use them.

**11. Letting go at count 4 always works (measured).** At delays of 0 to 32 frames after reaching 4, and with nothing, A, A + LEFT, A + RIGHT or LEFT held when letting go, every try gave the 55-frame boost. The count also jumps straight from 2 to 4 on the last steer, so there's never a count of 3. That means the log's `share_at_count_3` really counts moments at 4.

**12. Heavy-tailed holds per half (measured).** Random play where the pedal and the steering each draw a hold length from the same heavy-tailed distribution reached count 4 for the first time (0.03% of decisions, with μ = 1.6), but no boosts in 30,000 decisions. With the trained AI driving, overriding one half with chance 2% per decision reached count 4 in 0.065% of decisions, still with 0 boosts in 20,000, and races slowed to 135 s on average. The bigger problem is that overridden picks can't teach PPO anything, because the AI didn't make them.

**13. The AI picks its own hold lengths (`--hold-lengths`, running).** From FiGAR (Sharma et al. 2017). Each decision, the AI picks a pedal and a steering, each with a hold length from 1, 2, 4, 8, 16 or 32 decisions. A half only takes a new pick when its hold has run out. That's 18 × 18 = 324 actions, and 2 extra inputs say how long each hold has left (`kartzero/holds.py`). A long A + R hold over several short steering holds is then the AI's own choice, so PPO can learn from it. Both halves get the same lengths, and the reward is unchanged. Untrained random picks reached count 1 in about 1% of decisions but never count 2 in 60,000, though random driving is slow and constantly in walls. Started from scratch on 8 emulators with GAE 0.95, as `holds`.

**14. What starts a drift (measured).** A drift only starts if a direction is held at the moment R is pressed, at speed, with A held. Pressing LEFT only for the first 4 frames of the hop was enough.
- A hop with no direction never becomes a drift, even with steering afterwards. That's what the trained AIs do.
- At speed 2 there was no drift, and without A there was none either.
- Once drifting, 40 frames of R with no steering didn't end it, and the counting steers still worked afterwards.

**15. Hold lengths learn to drive much faster (measured, `holds` at 15 minutes).** Best test lap 1:43.30 after 14 minutes and 0.35M decisions, against 1:46 for `perbutton` after 77 minutes, and about 2 hours for the old runs to reach 1:42 to 1:43. Practice reward was 1.03 by 15 minutes. Its share of picks where neither half decided fell from 0.82 to 0.57 as it learned shorter holds. But count 2 never happened in any round, and there were no boosts.

**16. Curiosity (`--curiosity-weight`, running).** Random Network Distillation (Burda et al. 2018): a fixed random network describes each set of inputs, and a second network learns to predict that description. Its error is high for moments it hasn't seen much, and that error, centred on 0 within each chunk and capped, is added to the reward with a weight. Every input counts the same, so it knows nothing about drifting. It's a reward that isn't progress or time, so I asked whether it counts as self-taught. Decided 7 November: yes, it does (see Decisions). Running from scratch as `curious-EXPERIMENT`: hold lengths, weight 0.3 (a guess), GAE 0.95, 4 emulators.

**9. Self-imitation (`--self-imitation-weight`, built, not yet run).** From Oh et al. 2018. It keeps the last 200,000 decisions with the real points that followed. Each round, it takes 4 samples of 512 and pushes the odds towards the action really held, wherever the points beat the guess. A worry, from our own numbers: a mini-turbo is worth about 2 to 5 points, while the guess is usually off by about 5 to 8 (guess error 0.002 to 0.007 in hundreds of points), so it may copy noise as much as mini-turbos.

Next steps, in order:
1. Train `gae` (advantage blend 0.95, exploration 0.01) for longer on all 8 emulators, from the `gae` run's newest version, and check whether boosts appear and test laps keep improving. Not started.
2. If that stalls too, measure a replay-based method such as IQN.
3. Make test laps count boosts as well as time. Wait until the comparison ends, because its lap testers are writing the current file format.
4. If stickiness is ever used again and mini-turbos appear, turn it down in steps such as 0.5 → 0.25 → 0, checking test laps and boosts after each step. That's a guess, not tested.

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
| `Kart` in `kartzero/kart.py`, as `game.kart` | Kart and race data | `speed()`, `position()`, `heading()`, `drift_angle()`, `turning_rate()`, `mini_turbo_count()`, `boost_time_left()`, `is_off_road()`, `is_on_ground()`, `race_progress()`, `has_finished()`, `lap()`, `checkpoint()`, `race_clock()`, `finish_time()` |
| `Track` in `kartzero/track.py`, as `game.track` | The track layout | `checkpoint_count()`, `checkpoint_center(index)`, `collision_triangles()` |
| `RaceEnvironment` in `kartzero/environment.py` | One race, one decision at a time, and the 9 actions. With a sticky chance, practice sometimes repeats the last button, and the step says so (`pick_ignored`). It also reports new boosts (`boost_started`). With `per_button`, the pedal and steering roll separate sticky dice. With an archive, some races start from a saved rare moment (`from_archive`) | `reset()`, `step(action)` |
| `StateArchive` in `kartzero/archive.py` | Counting how often each coarse kart state is seen, and keeping saved game states of the rarest, Go-Explore style | `visit(cell)`, `file_for(cell, action)`, `pick()` |
| `DriverNetwork` in `kartzero/network.py` | Button odds and the guess of future points | `network(observations)` |
| `Trainer` in `kartzero/learner.py` | Driving the emulators, then learning from each chunk | `run_training(settings, minutes, viewer)` |
| `run_lap_tester` in `kartzero/test_lap.py` | Driving an official test lap for each saved version, in its own process | `drive_test_lap(network, environment)` |
| `run_viewer` in `kartzero/viewer.py` | Showing the AI live in a resizable window, rolling the dice like training. It finishes each race before loading the newest AI, so it never switches mid-race | `run_viewer()` |
| `GameWindow` in `kartzero/window.py` | The window itself. The emulator's own window can't stretch its picture | `show(image)` |
| `TrainingRun` in `kartzero/runs.py` | One folder in `runs/`, without loading PyTorch | `all_runs()`, `find_run("sticky0.5")`, `sticky_chance()`, `started_from()`, `best_test_lap()` |
| `TrackSensors` in `kartzero/sensors.py` | Distances from the kart | `wall_distances(position, heading)`, `road_edge_distances(position, heading)` |

Only one `Game` can run per process. Tests in `tests/` use fake memory.

| Script | What it does |
|---|---|
| `scripts/clear_ghosts.py` | Removes personal ghosts from the save copy, leaving only the staff ghost |
| `scripts/make_start_state.py` | Goes through the menus, checks the driver and kart in memory, and saves the snapshot |
| `scripts/speed_check.py` | Drives with the accelerator held and measures speed with 1 or more emulators |
| `scripts/replay_check.py` | Checks that the same buttons give the same result |
| `scripts/progress_check.py` | Checks race progress by driving forward, reversing and completing laps with a test driver |
| `scripts/train.py` | Trains on 8 emulators with a live window. Closing the window stops training and saves. Results go in `runs/`, and the newest AI in `checkpoints/latest.pt`. Add `--resume` to carry on the same run from that save. `--sticky-chance`, `--sticky-per-button`, `--archive-start-chance`, `--start-from <version>`, `--checkpoint`, `--name` and `--decisions-per-chunk` let several runs go side by side |
| `scripts/watch.py` | Shows the newest saved AI racing in the window, with no training. `--list` sums up every run: sticky chance, where it started, time trained, best and newest test lap. `--list <run>` shows each version of one run, marking the start and the best. `--run <run>` plays a run's newest version, and `--hours 30` plays the one nearest 30 hours (add `--run` to pick the run). A run can be named by the end of its folder name, like `213453` or `sticky0.25`. A run's first version is its starting brain, so it never counts as the best. `--compare` takes turns between the `checkpoints/compare-sticky*.pt` runs, one race each with each run's newest AI and sticky chance, and labels which one it is |
| `scripts/retime_test_laps.py` | Re-drives every version of a run to record test laps in official game time. Only for runs that have stopped training |
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
- **Mini-turbo timing, tested with a scripted driver at one decision per 4 frames:** holding R and switching left and right every 3 or 4 decisions (12 or 16 frames) reached a count of 4 and gave a 52-frame boost. So did every 5 decisions. Switching every 1 or 2 decisions only reached a count of 2, and every 6, 8 or 12 lost the count before the boost, possibly from hitting a wall (not checked). So the window is holds of about 3 to 5 decisions (12 to 20 frames). So a mini-turbo takes about 15 decisions of holding R with steady direction changes, which random button picks almost never do. Two of the right-first tries lost the count, probably by hitting a wall near the start, but that isn't checked.
- Not useful for sliding: `direction`, `drivingDirection` and `velocity` barely differ in a drift, so we use `driftRotY`.
- Also confirmed: the race clock (`race_status.time.lapTime`, shown on screen as TIME) and the official finish time (`totalMilliseconds` in our driver status). See "The race clock" below.

**The race clock, measured.** Times were reported as decisions ÷ 15, which was too fast:
- The game's clock starts in the countdown. When our snapshot lets the kart move, the screen already shows TIME 0:02:395, and the clock has run 145 frames (2.42 s). The kart moved at that same moment whether A was held through the countdown or pressed at any point from the clock's start, so it isn't a stall from pressing too early.
- Five test laps from different versions: the official time was 2.344 to 2.367 s more than ours, 2.356 s on average. That's the 2.42 s countdown, less our finish coming up to 4 frames late at the end of a decision.
- **So all times now come from the game:** its official finish time at the line, or its clock if the race didn't finish. They match the times in the targets table, which come from the game's own screen. The reward is unchanged, because a fixed offset doesn't change which driving is faster.
- `scripts/retime_test_laps.py` re-drives every version of a run to record official times, and keeps the old file as `test_laps_decision_clock.csv`.
- Training logs written before this fix (`best_seconds` in `log.csv`) are still on the old clock.

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
- The test driver finished all 3 laps in about 2:08 on our old clock (about 2:10 in game time), steering only, with no drifting. It isn't a target, but it gives us a rough idea of plain driving. Lap changes came at the same times with and without the window.
- **The finish:** race progress passes 1.0 (1.0002) and the lap counter goes to 4. **The game then drives the kart by itself and progress keeps rising** (1.1096 ten seconds later), and it used the mushroom. So a run must end the moment progress reaches 1.0, or the AI collects points for driving it didn't do.

**Emulator speed, with drawing off and no AI running.** At one decision per 4 frames, decisions per hour = frames per second × 3,600 ÷ 4.

| Emulators | Each (frames/s) | Total (frames/s) | Decisions per hour |
|---:|---:|---:|---:|
| 1 | 451 | 451 | 406,000 |
| 8 | 297 | 2,379 | 2.14M |
| 12 | 218 | 2,610 | 2.35M |
| 16 | 188 | 3,005 | 2.70M |

Speed drops per emulator because they share 8 physical cores, though I didn't measure how much each cause contributes. Training will be slower than this because the AI needs time too.

**During real training, with the AI running on 8 emulators:** about 1.43M decisions per hour, with no viewer or test laps running. Driving takes 4.86 seconds per round and learning 0.30 seconds. A profile shows 94.5% of a decision is the emulator itself, and our Python code is about 5%, so tuning Python further wouldn't help much.

**Two fixes raised this from 1.19M (about 20%), with test laps unchanged to the last digit:**
- The value guess is only requested when a race stops mid-way, not on every emulator at every decision.
- The snapshot is saved one frame before the kart can move, so a restart takes 157 ms, not 953 ms. All 8 emulators run in step, so one restart used to hold up the others.

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

- **A curiosity bonus is allowed (7 November).** Curiosity pays for reaching moments the AI has rarely seen, judged over all its inputs alike. It knows nothing about this game or any technique and would work the same on any game, so it counts as self-taught exploration, not a technique reward. It still pulls the AI away from pure speed while it's on, so plan to turn it down or off once it has done its job, and report which runs used it.
- **No help aimed at certain buttons, for now.** An idea was to roll separate sticky dice for R and for the direction, so drifts last longer. It isn't a reward, but it hints towards drifting, so it's set aside. First try every option that treats all buttons the same (stickiness, GAE, longer runs, IQN). Only consider button-specific help if all of those fail.
- **GAE is a setting, `advantage_blend`.** At 1.0, the default, it gives exactly the old advantages (real points minus the guess), and a test checks that. Lower values lean more on the value guess for later decisions, which spreads credit more cleanly but trusts the guess more. 0.95 is the usual PPO choice. The value guess then learns towards the blended points too.
- **Settings live in `kartzero/settings.py`,** so tools like `watch.py --list` can read them without loading PyTorch. `--list` shows each run's settings that differ from the defaults.
- **Sticky practice, with ignored picks left out.** When the sticky dice repeat the last button, that decision is left out of the button-odds update and kept for the value guess. It's left out even when the AI picked the same button anyway: leaving out only the ones that differed would push the AI towards repeating. Test laps never use stickiness. The viewer doesn't either, so it shows slightly less sticky driving than practice.
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

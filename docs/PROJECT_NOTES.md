# KartZero project notes

Where the project stands, and everything we've measured and decided. Last updated: 8 November 2026.

"Measured" means we ran it on this PC. Anything else is marked as a source or a guess.

## Where we are now

**Stage: getting the AI to discover mini-turbos.** Curiosity (a bonus for reaching rarely seen moments) made it discover mini-turbos, and overnight on 7 to 8 November a curiosity-trained AI **beat the staff ghost**, and on 8 November its best test lap reached **1:27.96 with 58 mini-turbos** (a slower-learning polish run with no curiosity), 0.13 seconds off MarI/O Kart. See "Mini-turbo experiments" below. Earlier: **second training run, without reverse.** The first run learned to turn around and drive the track in reverse. In a replay it was reversing 99.2% of the time at speed -3, with no forward driving above speed 5. That earned points because the kart still moved the right way along the track, and it explains why progress stopped rising at about 80%: 200 seconds in reverse only covers about 79% of a race. So the reverse button was removed and training restarted from nothing.

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

### Mini-turbo experiments (7 November)

The goal: find a general change, with no reward for any technique, that lets the AI discover mini-turbos. Every attempt is listed here in order, with what it showed. Three of them were removed from the code afterwards (per-button sticky dice, the Go-Explore archive, self-imitation). They're kept in git history at commit `1981d89` on the `mini-turbo-experiments` branch. The run folders and scratch scripts for the measurements below aren't in the repo.

**Summary so far.** The game makes mini-turbos an exploration trap: half a drift costs points, and the reward only comes after the whole 17-decision move. Every way of adding randomness (dice schemes, Go-Explore, letting the AI pick hold lengths) failed to find a single boost. **Curiosity worked:** the AI started drifting for real, fired boosts in practice, and its 43-minute brain fired the first mini-turbo ever on a test lap. Hold lengths make the AI learn to drive about 8 times faster, but on their own they don't find drifting.

**1. How the game counts a mini-turbo (measured, frame by frame).**
- **Starting a drift:** R must be pressed while a direction is held, at speed, with A held. Holding LEFT for only the first 4 frames was enough. A hop with no direction never becomes a drift, whatever the steering does afterwards, and that's what all our earlier AIs did. At speed 2 there was no drift, and without A there was none either.
- **The hop and the count:** the hop lasts about 13 frames, and the drift starts on landing at count 1.
- **Counting steers:** a steer only counts if it comes at least 10 frames (3 decisions) after the last counted one. Steers 4 to 8 frames apart never got past count 2. Steers 10 to 32 frames apart all reached 4, and no upper limit showed up. The count goes 1, 2, then straight to 4 on the fourth counted steer (right, left, right, left in a left drift), so it's never 3.
- **Gaps are fine:** R held with no direction for 40 frames between steers didn't end the drift or the count.
- **Firing the boost:** letting go of R at count 4 always gave a 55-frame boost, at delays of 0 to 32 frames and whatever else was held. Letting go at count 2 gave nothing.
- The fields `driftLeftCount`, `driftRightCount` and `driftLeftRightTimeout` (set to 10 by each counted steer, then counting down) show all this.

**2. What one mini-turbo is worth inside the AI's own race (measured).** I paused the `gae` run's 2-hour test race every 20 decisions and compared the next 120 decisions three ways: keep driving, one scripted mini-turbo then keep driving, or the same drift with no boost.
- The better of the two mini-turbo directions gained **+2.3 points on average** (median +2.4, better at 53 of 69 points). That's about 0.1 seconds.
- The wrong direction often lost 20 to 120 points.
- A drift with no boost lost **15 points on average**, even in the better direction, and was better at only 4 of 69 points.

So the AI's habit of avoiding drifts is the right lesson from what it had seen. Chains of 2 to 4 scripted mini-turbos lost 17 to 26 points, because the script steers blindly. So this doesn't measure the full prize: a mini-turbo only pays if the drift's steering follows the track. Our evidence of the full prize is still MarI/O Kart's 1:27.8 with 73 of them.

**3. Random play with different dice (measured, no boosts).** 30,000 decisions each, restarting every 60 seconds. A new random button every decision, sticky 0.5, sticky 0.8, εz-greedy (heavy-tailed holds, Dabney et al. 2020), and separate dice per half changing with chance 0.1, 0.2 or 0.3 all gave 0 boosts, and count 2 in at most 0.06% of decisions. Heavy-tailed holds drawn separately for the pedal and the steering reached count 4 in 0.03% of decisions, still with no boost. With the trained AI driving and one half overridden by such a hold with chance 2% per decision: count 4 in 0.065% of decisions, 0 boosts in 20,000, and races slowed to 135 s. Overridden picks also can't teach PPO anything, because the AI didn't make them.

**4. Research.** I found no kart AI that learned mini-turbos without being rewarded for them. A Mario Kart Wii agent (Beyond The Rainbow, 2024) had drift buttons but a shaped reward. The general ideas tried here: temporally extended exploration (Dabney et al.), Go-Explore (Ecoffet et al.), self-imitation (Oh et al. 2018), FiGAR hold lengths (Sharma et al. 2017) and curiosity by Random Network Distillation (Burda et al. 2018).

**5. Per-button sticky dice (removed).** The pedal and the steering rolled their own sticky dice at the same chance, 0.75. In a 77-minute run from scratch (`perbutton`, 1.0M decisions) there were no boosts, count 2 in at most 0.05% of practice, and best test lap 1:46.15.

**6. Go-Explore archive (removed).** Each emulator saved game states of its rarest moments (by checkpoint and the 11 kart inputs, each rounded the same way), and half of practice races started from one. After returning, it pressed random buttons for 0 to 30 decisions. The first version forgot that random exploring step, which was my mistake. In 52 minutes (`archive`, 0.58M decisions, 279 archive races) there were no boosts and count 2 in about 0.01% of practice. The rare moments it kept were mostly unrelated (slow, off-road, odd turning). A pure Go-Explore loop with no learning ran 60,880 decisions over 8,823 cells and reached count 2 at best. Saving a state takes 0.22 s and loading one 0.12 s (2.7 MB each), only to files. The library's numbered save slots write backup states into the Python install folder, so don't use them.

**7. Self-imitation (removed, never run).** It would re-learn from past decisions that beat the value guess. Our numbers suggested it would copy noise: a mini-turbo is worth about 2 to 5 points, while the guess is usually off by 5 to 8.

**8. Hold lengths (`--hold-lengths`, kept, `kartzero/holds.py`).** Each decision, the AI picks a pedal (nothing, A, A + R) and a steering (straight, LEFT, RIGHT), each with a hold length of 1, 2, 4, 8, 16 or 32 decisions. A half only picks again when its hold runs out, so "keep A + R held while the steering changes" is one choice the AI can make, and PPO learns from it. That's 324 actions and 2 more inputs (how long each hold has left). When only one half picked, PPO updates only that half's odds. Results from scratch on 8 emulators (`holds`):
- best test lap 1:43.30 after 14 minutes, against about 2 hours for the old runs to reach 1:42 to 1:43
- then a plateau: 1:43.22 at 55 minutes, while `gae` without hold lengths reached 1:41.40
- count 2 almost never, and no boosts

So it speeds up learning but doesn't raise the ceiling or find drifting alone.

**9. Curiosity (`--curiosity-weight`, kept, `kartzero/curiosity.py`).** A fixed random network describes each set of inputs, and a second network learns to predict that description. Its error is high for moments it has rarely seen, and that error, centred on 0 within each chunk and capped, is added to the reward with a weight. Every input counts the same. Allowed as self-taught (see Decisions). From scratch with hold lengths, weight 0.3 (a guess), GAE 0.95 and 4 emulators (`curious-EXPERIMENT`):
- **Practice:** count 2 or more in about 3% of practice decisions by 20 minutes, count 4 reached, about 21 boosts in the first 37 minutes.
- **Test laps** (no exploration, no bonus, counted with `scripts/lap_boosts.py`): the AI really drifts.

| Training time | Lap | Boosts | Highest count | Drifting |
|---|---:|---:|---:|---:|
| 14 min | 1:52.86 | 0 | 1 | 19% of decisions |
| 29 min | 1:52.63 | 0 | 2 | 22% |
| 43 min | 1:58.60 | **1** | 4 | 37% |

- Driving is slower while the bonus is on: best test lap 1:45.96, against 1:43.22 for `holds` alone.

**10. Handing over from curiosity to speed alone (built 7 November).** The aim is an AI that does mini-turbos because they're fast, not because they're new. Signs that it's ready:
- test laps (no bonus) show boosts reliably, several per lap over 3 or 4 versions in a row
- `drift_newness` in the log falls towards 1, meaning drift moments (count 2 or more) look no newer than average, so the bonus is no longer what keeps it drifting
- practice boosts hold steady while that happens

The tools for it:
- `--curiosity-fade-decisions N` lowers the weight in a straight line to 0 over the first N decisions of a run.
- The log shows `curiosity_weight` and `drift_newness`.
- Checkpoints now save the curiosity network, so `--resume` carries on smoothly.
- A run started from a saved brain gets a fresh curiosity network, which first learns for 20 chunks (about 2 minutes on 8 emulators) without paying any bonus, because a new one finds everything equally strange.

Tested on the real game: the weight fell from 0.292 to 0.146 over half of a 20,000-decision fade, the warm-up paid nothing, and a resumed run picked up the saved state.

**11. The curiosity AI dropped mini-turbos by itself, with the bonus still on (measured).** `curious-EXPERIMENT` (stopped at 101 minutes, 1.20M decisions; I first wrote 87 minutes by mistake), averaged over stretches of about 8 minutes:

| Training time | Practice at count 2+ | Practice at count 4 | Boosts in practice |
|---|---:|---:|---:|
| 0–9 min | 0.5% | 0.02% | 3 |
| 29–38 min | 1.8% | 0.15% | 11 |
| 38–46 min | **4.0%** | 0.33% | **23** |
| 47–55 min | 2.9% | 0.25% | 18 |
| 63–70 min | 2.0% | 0.05% | 4 |
| 78–86 min | 1.1% | 0.04% | 4 |
| 86–94 min | 1.5% | 0.06% | 7 |
| 94–102 min | **3.3%** | 0.19% | **20** |

Test laps, every version from 35 to 58 minutes (`lap_boosts.py curious-EXPERIMENT --between 35 60`):
- the 43- and 45-minute brains fired 1 and 2 boosts, drifting 37% and 36% of the time
- from 48 minutes on, no boosts (except reaching count 4 once at 54 minutes)
- brains from 56 to 98 minutes drift only 4 to 16% of the time, with 0 boosts

The likely reason: once drifting became familiar, its bonus faded, and the plain points took over (about +2 for a good mini-turbo, about −15 for a failed drift). Its attempts weren't reliable enough, so it dropped them. That's the prediction made before the fade, happening early. **Then a second wave:** practice drifting rose again from about 86 minutes (20 boosts in the last 8 minutes), though test laps hadn't caught up by 98 minutes. The likely reason: while the AI rarely drifted, the student network forgot drift moments, so they became new again. So with a constant bonus, drifting seems to come in waves rather than once.

**12. Hold lengths are needed for curiosity to work (measured).** Over the first 56 minutes, same settings except hold lengths:

| Run | Decisions | Practice at count 2+ | At count 4 | Boosts in practice |
|---|---:|---:|---:|---:|
| with hold lengths (`curious-EXPERIMENT`) | 0.63M | **2.0%** | 0.13% | **63** |
| without (`curious-noholds`, stopped at 56 min) | 0.71M | 0.4% | 0.01% | 9 |

So hold lengths stay.

**13. Making the curious phase last longer (running).** Item 11 suggests the AI quits before its mini-turbos get reliable, because drifting stops being new after about 15 minutes. Curiosity is a bonus for any unusual moment (a wall it normally avoids, off-road, an odd crash angle, or a drift), and slowing the student network keeps all of them new for longer, the distracting ones included. `--curiosity-learning-rate` sets how fast the student learns (default 0.0001). `curious-slow` uses 0.000025, 4 times slower (a guess), with the weight at a constant 0.3, from the 45-minute peak brain. The idea, suggested on 7 November: build a starting point that boosts a lot, and only then fade.

**14. Test-lap boosts so far (measured about 16:40).** Every saved version, counted with `lap_boosts.py`. Both peak-start runs begin from the same brain, which had 2 boosts, so that one isn't counted for either.

| Run | Versions after the start | Versions with a boost | Most in one lap |
|---|---:|---:|---|
| `curious-slow` (slow student, from the peak) | 7, up to 17 min | 2 | **3**, at 11.8 min (1:56.11, drifting 31.5%) |
| `curious-fade` (fading, from the peak) | 12, up to 28 min | 1 | 1, at 6 min |
| `curious-slow-new` (slow student, from new) | 4, up to 10 min | 0 | still learning to drive |

3 mini-turbos in one test lap is the most so far.

**15. The slow student from new: 13 mini-turbos in one test lap (measured, about 17:20).** `curious-slow-new` test laps, each version:

| Training time | Lap | Mini-turbos | Drifting |
|---|---:|---:|---:|
| 28 min | 1:53.80 | 0 | 27% |
| 33 min | 1:53.14 | 2 | 38% |
| 36 min | 1:51.04 | 1 | 24% |
| 39 min | 1:51.10 | 1 | 26% |
| 42 min | 1:51.26 | 0 | 17% |
| 45 min | 1:56.30 | 1 | 22% |
| 47 min | 1:52.46 | 2 | 35% |
| 50 min | 1:53.29 | **13** | 58% |
| 53 min | 1:59.45 | **10** | 56% |

Practice: 317 boosts in the first 53 minutes, 132 of them in minutes 48 to 53. It drifts much more than the peak-start runs did, and it learned to drive at about the same pace as the normal student (item 13). Laps are still slower than the 1:43 of plain driving, so its mini-turbos don't pay for its drifting yet.

**16. How the two peak-start runs ended (stopped about 17:35).**
- `curious-fade` (87 min, 0.88M decisions): once the weight reached 0, drifting vanished. Its newest four brains drift 0 to 3% of the time with **0 mini-turbos**, laps 1:44 to 1:47, best test lap 1:43.55. The forgetting prediction was right: started from a brain whose mini-turbos weren't reliable, it dropped them all.
- `curious-slow` (76 min, 0.74M decisions): caught up late, with about 1,110 practice boosts (1.9 a minute at the end), count 2+ in 8.4% of practice, and 3 and 5 mini-turbos on the 71- and 74-minute test laps. Below `curious-slow-new` but rising. Its checkpoint `checkpoints/exp-curious-slow.pt` includes the curiosity state, so it can be resumed.

**17. Is the winning recipe luck? Seeds 2 and 3 (running).** Training uses random numbers for the brain's starting settings, the practice dice and the learning order, and a seed fixes them all. Every run so far used seed 1, and early luck can snowball, so part of 13 against 3 could be luck. `train.py` has a `--seed` option now. Running alongside seed 1, with the same settings (slow student, from new, hold lengths, curiosity 0.3, GAE 0.95, 4 emulators, 150 minutes): `curious-slow-new-seed2` and `curious-slow-new-seed3`. If both also reach many test-lap mini-turbos by about 50 minutes, the recipe works; if not, seed 1 was lucky.

**17b. Seed results (measured, about 19:00).** Seed 1 ran 150 minutes. Seeds 2 and 3 were stopped at about 78 minutes (0.72M decisions). Test laps, counted with `lap_boosts.py`, now saved per run in `versions/mini_turbo_counts.csv` and shown by `watch.py --list`:

| Seed | First lap with 10+ mini-turbos | Steadiest stretch | Laps there |
|---:|---|---|---|
| 1 | 0.48M decisions (50 min) | 1h10m–1h22m: 17, 14, 19, 16, 25 | 1:46.85 – 1:48.80 |
| 2 | 0.61M decisions (67 min) | 1h07m–1h18m: 11, 5, 10, 11, 11 | 1:47.42 – 1:58.77 |
| 3 | not by 0.72M | at most 1 per lap | — |

So the recipe worked for 2 seeds out of 3, at different times. Seed 1 was the luckiest, not a fluke. Seed 1 also had waves: laps of 2:00 to 2:04 from 1h52m to 2h07m, then its best lap of all at 2h14m (1:45.66, 20 mini-turbos), whose neighbours were 1:55 and a lap that didn't finish.

**The pick (step 1 of item 18):** seed 1's 1h13m brain (`0001h13m_000000692224_decisions.pt`), 1:46.85 with 14 mini-turbos. It's the fastest lap in the steadiest stretch, and both neighbours are strong. The 2h14m brain is faster, but its neighbours aren't.

**Step 2, running from about 19:00:** `best-curiosity0` and `best-curiosity0.05`, 75 minutes each, 4 emulators each, from that brain, otherwise the same recipe (slow student, hold lengths, GAE 0.95).

**18. Tonight's plan (agreed 7 November, about 18:20).** Rules written down before the results:
1. When seed 1 finishes (about 18:50), test every recent version of all three seeds. Pick the best brain by test-lap mini-turbos and time, preferring one whose neighbouring versions also did well, because one lucky lap out of about 60 isn't trustworthy. Then stop seeds 2 and 3.
2. From that brain, two runs of about 75 minutes on 4 emulators each, the same recipe except curiosity **0** against **0.05**, a small leftover (both keep the slow student). `boost_newness` is in the log for these, alongside `drift_newness`.
3. The winner has the fastest average of its last 5 test laps, as long as it still fires mini-turbos (at least 3 per lap on average over those 5). Plain driving at about 1:42 can never reach 1:27.8, so a run that has dropped them can't win. If neither keeps them, 0.05 wins, because it can still explore.
4. Overnight: the winning setting, from the winner's best brain, on all 8 emulators, for 10 hours.
5. Allowed to switch to something new overnight, but only if the run is clearly failing. That means, at least 2 hours in, both: no new best test lap in the last hour with the last-5 average no better than the starting brain's, **and** test-lap mini-turbos trending to about 0 (or practice reward falling for over an hour). In order of what to try: raise curiosity again from the best brain, then a different student speed, then a replay-based method (IQN). Every switch and its reason goes in these notes.

**19. What happened to tonight's plan (about 19:40).** Step 2's two runs from seed 1's 1h13m brain (1:46.85, 14 mini-turbos):

| Run | Mini-turbos per test lap over time | Laps at the end |
|---|---|---|
| curiosity 0 (stopped at 37 min) | 17, 19, then 11–14, 3–7, then 0 for the last five | 1:44.55 – 1:45.70 |
| curiosity 0.05 (left to finish) | 10–14, then 2–5, then 0, 0, 4, 2 | 1:43.92 – 1:47.18 |

Laps got **faster as mini-turbos went away** (1:46.85 with 14, to 1:44.55 with 0). So the way this AI does mini-turbos still costs more than it saves, and with speed as the only judge, dropping them is right for it. Even 0.05 didn't keep them. By rule 3 of item 18, 0.05 would have run overnight, but on this evidence it would just become a plain driver at about 1:44, below our old no-drift best of 1:41.40. **Changed with the owner's OK:** overnight instead resumes seed 1 (`curious-slow-new`) with curiosity 0.3, the only setting that has kept mini-turbos going, for 10 hours on 8 emulators (256 decisions per chunk, still 2,048 per update). Its checkpoint carries the curiosity network, and at resume `boost_newness` was 1.7 to 2.2, so curiosity still pulls towards boosting. Risks: it swings in waves, so judge the night by its best saved versions, not its last; and it may just plateau. Every version is kept with its test lap and mini-turbo count. Resuming needed `TrainingLog`, which lines up log columns when newer code adds some.

**The night, split (from about 20:20).** At the owner's suggestion, the night is split between two runs instead of one, for two chances at a good brain at about the same total training: seed 1 (`curious-slow-new`, resumed at 2.05M decisions) and seed 2 (`curious-slow-new-seed2`, resumed at 0.74M decisions), 4 emulators each, 512 decisions per chunk, curiosity 0.3, slow student, hold lengths, GAE 0.95, 600 minutes. `best-curiosity0.05` finished first and ended with 0 or 1 mini-turbos on most laps, at 1:44 to 1:49.

Night checks:
- 20:32: seed 1 at 2.17M decisions: best 1:44.49 with 24 mini-turbos (at 2h56m), 14 to 28 on every lap since 2h39m. That beats the curiosity-0 brains' 1:44.55 with 0, so its mini-turbos now pay for themselves. Seed 2: 3 to 11 per lap, 1:53 to 1:58.
- 21:22: seed 1 at 2.67M: **1:42.92 with 28 mini-turbos** (at 4h04m), laps 1:45 to 1:52 with 23 to 28 since 3h25m. Seed 2 at 1.44M: 8 to 18 per lap, best 1:48.22.
- 22:13: seed 1 at 3.28M: **1:38.83 with 37 mini-turbos** (at 4h54m), **the fastest test lap of any AI so far**, 2.6 seconds better than the no-drift record of 1:41.40, and 2.35 seconds off the staff ghost (1:36.48). Also 1:40.41 with 33 and 1:41.35 with 35. Seed 2 at 2.05M: best 1:43.25 with 28, and 16 to 28 per lap.
- 23:03: seed 1 at 3.92M: a dip (1:45 to 1:51 from 5h02m to 5h26m), then 1:41.14 with 35 and 1:40.45 with **40**, the most mini-turbos yet. Best still 1:38.83. Seed 2 at 2.67M: best 1:43.20 with 25.
- 23:54: seed 1 at 4.54M: 33 to 44 mini-turbos on every lap since 5h34m, laps 1:40.39 to 1:46.94, best still 1:38.83. Seed 2 at 3.28M caught up: **1:38.86 with 36** (at 4h34m), and 1:39.79 with 27. Two seeds of the same recipe now drive about 1:39, so it isn't one lucky run.
- 00:49 (8 November): **seed 1 beat the staff ghost: 1:34.47 with 49 mini-turbos** (at 7h19m, 5.11M decisions, `0007h19m_000005113856_decisions.pt`), 2.0 seconds under 1:36.48. Next version: 1:37.06 with 43. Since 6h44m every lap has 43 to 49 mini-turbos. Seed 2 at 4.04M: best 1:38.16 with 42.
- 01:45: seed 1 at 5.93M: 1:35.68 with 48 (at 7h56m), also under the staff ghost, and 1:37.06 to 1:40.79 with 43 to 48 since. Best still 1:34.47. Seed 2 at 4.68M: 1:40.19 with 44, best still 1:38.16.
- 02:40: seed 1 at 6.67M: 1:36.72, 1:36.96 and 1:37.23 with **50 to 51** mini-turbos, a steady run near the staff ghost's time. Best still 1:34.47. Seed 2 at 5.43M: **1:37.49 with 47**, a new best, and 1:38.07 with 38 (one lap at 6h50m didn't finish).
- 03:36: seed 1 at 7.51M: **new best 1:33.83 with 51** (at 10h23m, `0010h23m_000007510016_decisions.pt`), after 1:36.61 with 49, and up to **54** mini-turbos in one lap (1:37.20). Seed 2 at 6.11M: 43 to 51 per lap, 1:39 to 1:41, with one slip to 1:51.71, best still 1:37.49.
- 04:31: seed 1 at 8.20M: 1:35.54 with 51 and 1:34.02 with **55**, best still 1:33.83. Seed 2 at 6.87M: **1:34.24 with 50**, also under the staff ghost, and 1:36.86 with 48. Both seeds now beat the staff ghost.
- 05:26: seed 1 at 8.70M dipped: 1:47.14 with 55 and 1:46.28 with **60**, the most mini-turbos yet but slow laps, so another wave down. Best still 1:33.83. Seed 2 at 7.51M: 1:35.01 with 57 and 1:36.17 with 51, best still 1:34.24.

**Result of the night (both ended about 06:20 on 8 November).**

| Run | Training at the end | Best test laps (mini-turbos) | Last 5 test laps |
|---|---|---|---|
| Seed 1 (`curious-slow-new`) | 13.0 h, 9.51M decisions | **1:32.65 (50)** at 12h36m, 1:33.83 (51), 1:34.02 (55), 1:34.47 (49) | 1:47.14, 1:46.28, 1:44.27, 1:32.65, 1:35.99 |
| Seed 2 (`curious-slow-new-seed2`) | 11.2 h, 8.20M decisions | **1:32.86 (51)** at 11h09m, 1:33.15 (49), 1:33.30 (48), 1:34.24 (50) | 1:37.33, 1:36.17, 1:33.15, 1:33.30, 1:32.86 |

- **Best test lap: 1:32.65 with 50 mini-turbos** (`runs/20261107-162144-curious-slow-new/versions/0012h36m_000009234432_decisions.pt`). That's 3.8 seconds under the staff ghost, 8.75 seconds better than the no-drift record (1:41.40), and **4.8 seconds off MarI/O Kart's 1:27.83**.
- Both seeds ended near 1:33, so the recipe is repeatable with long training.
- Seed 2 ended steadiest, with its last three laps at 1:33.15, 1:33.30 and 1:32.86. Seed 1 still swings in waves (1:46 to 1:47 then 1:32.65).
- Neither had levelled off by the end, so more training is the obvious next step.
- Mini-turbos per lap rose from about 14 (evening) to about 50 to 60. MarI/O Kart used 73.

Next steps, in order:
1. Keep training both seeds with `--resume` (same settings: 4 emulators, 512 decisions per chunk, curiosity 0.3, student rate 0.000025, hold lengths, GAE 0.95). Neither had levelled off. **Running from 07:02 on 8 November, for up to 10 hours:** seed 1 from 9.62M decisions, seed 2 from 8.35M.
   - 08:00: seed 1 at 10.09M: 1:43.52 with 47, then 1:34.35 with 57. Seed 2 at 8.97M: 1:40.14 with 42, then 1:33.85 with 49. No new bests yet. Versions now come about every 20 minutes per run (the 3% rule at this size).
   - 08:59: **seed 2 new best: 1:31.85 with 52** (at 12h54m, 9.51M decisions, `0012h54m_000009512960_decisions.pt`), after 1:32.21 with 51. Seed 1 at 11.03M is the steadiest it has been: 1:33.43 to 1:34.35 with 52 to 58 on its last four laps.
   - 09:58: seed 1 at 11.70M looks level: its last six laps are all 1:33.43 to 1:34.35, with 49 to 58 mini-turbos, and no new best since 1:32.65 at 12h36m (about 3 hours of training). Seed 2 at 10.39M slipped back to 1:36.05 to 1:37.48 after its 1:31.85.
   - 10:03: seed 1 stopped at about 11.9M decisions, because it had levelled off. Seed 2 keeps going.
2. **Polish run (`polish`, started 10:03 on 8 November).** From seed 1's best brain (`0012h36m_000009234432_decisions.pt`, 1:32.65 with 50), curiosity fades in a straight line from 0.3 to 0 over 1M decisions (about 1.7 hours on 4 emulators), then about 3 hours at 0. Otherwise the same recipe. The idea: the deliberate odd moves go away and only speed decides. Last time a fade wiped out mini-turbos, but they cost time back then; now they pay for themselves (50+ per lap, faster than any lap without them), so this time they may stay. The question: do test laps get faster and steadier, and do mini-turbos stay?
   - 10:57 (53 minutes, 0.66M decisions, weight down to 0.10): **new best 1:30.95 with 50 mini-turbos** (`runs/20261108-100320-polish/versions/0000h49m_000000612352_decisions.pt`), also 1:31.15 with 58, 1:31.32 with 54 and 1:31.59 with 52. All 24 test laps since the start brain have 44 to 59 mini-turbos, and 18 of them are under 1:35. Seed 2 (curiosity still 0.3) at 11.03M: 1:34.87 to 1:40.13, best still 1:31.85.
   - 11:56 (1h52m, 1.35M decisions, curiosity now 0): **new best 1:29.90 with 58 mini-turbos** (at 1h04m, weight about 0.06, `runs/20261108-100320-polish/versions/0001h04m_000000798720_decisions.pt`), **2.07 seconds off MarI/O Kart**. Since the weight reached 0 (at 1M decisions), all 10 test laps keep 44 to 52 mini-turbos, mostly 1:31 to 1:33 (one 1:37.79). So the mini-turbos survived the fade to zero this time, and the laps are the steadiest yet. Seed 2 at 11.70M: 1:33.14 to 1:40.13, best still 1:31.85.
   - 12:54 (about 2h50m, curiosity 0 for about 2 hours): best still 1:29.90. Recent laps 1:30.29 to 1:33.29 with 45 to 54 mini-turbos, including 1:30.29 and 1:30.68, so it's holding steady near its best with no bonus. Seed 2: 1:33.14 to 1:40.13, best still 1:31.85.
   - 13:53: **new best 1:29.87 with 54 mini-turbos** (at 2h51m, 2.05M decisions, `runs/20261108-100320-polish/versions/0002h51m_000002045952_decisions.pt`), trained with **no curiosity at all** for the last 1M decisions, so it's pure progress-and-time learning. Also 1:29.98 with 57. Last six laps 1:29.98 to 1:32.26. Seed 2: 1:33.14 to 1:38.36, best still 1:31.85.
   - **Result (ended 15:03, 5.0 hours, 3.21M decisions).** Best still **1:29.87 with 54**, from 2h51m. No new best in the last 2 hours. Its 40 test laps after curiosity reached 0 all finished, with **44 to 57 mini-turbos each**, and 17 of them were under 1:32 (4 laps in the whole run under 1:30: 1:29.87, 1:29.90, 1:29.94, 1:29.98). Averages of 10 laps in a row: 1:32.01, then 1:31.49, then 1:32.31 for the last 10. So it **levelled off at about 1:31 to 1:32**, with its best laps near 1:30. The fade kept mini-turbos for good this time, but plain speed learning alone didn't keep improving after about 2 hours at 0.
   - Seed 2 (curiosity 0.3, still running) at 13.56M decisions: 1:32.95 to 1:38.36 lately, best still 1:31.85.
3. **Slow polish (`polish-slow`, started 15:06 on 8 November, 5 hours).** From the 1:29.87 brain, with the learning rate at 0.0001 instead of 0.0003 (`--learning-rate`, new), curiosity 0 from the start, otherwise the same recipe on 4 emulators. The idea, a guess: the polish brain swings by 1 to 2 seconds from version to version, which suggests each update is too big for fine details, so smaller steps may keep what works and climb past the plateau. It's judged by its average test lap over 10 versions against the polish run's 1:31.5 to 1:32.3, and by whether it beats 1:29.87.
   - 15:59 (51 min, about 0.4M decisions): **new best 1:28.32 with 59 mini-turbos** (at 34 min, `runs/20261108-150614-polish-slow/versions/0000h34m_000000266240_decisions.pt`), **0.49 seconds off MarI/O Kart**. Also 1:28.53 with 62 and 1:28.83 with 57. The average of its last 10 test laps is **1:29.81**, about 2 seconds better than the first polish run's 10-lap averages (1:31.49 to 1:32.31), and 5 of those 10 are under 1:30. So far the smaller steps are clearly better. One run with one seed, so that part isn't proven. Seed 2 at 13.97M: 1:31.84 with 61, a hair under its old best of 1:31.85.
   - 16:31 (1h24m, about 0.68M decisions): **new best 1:27.98 with 57 mini-turbos** (at 1h10m, `runs/20261108-150614-polish-slow/versions/0001h10m_000000638976_decisions.pt`), **0.15 seconds off MarI/O Kart**. Also 1:28.59 and 1:28.89. Its last 10 test laps average 1:30.24, a little worse than at 15:59 because of two slow laps (1:34.57 and 1:33.20). Seed 2 at its 20h23m version: 1:40.61, best still 1:31.84.
   - Seed 2 ended about 17:00 (21.0 hours, 14.82M decisions, curiosity 0.3 throughout): its last version set its best, **1:31.36 with 50** (`0021h00m_000014819328_decisions.pt`), but its last 5 laps swing from 1:31.36 to 1:40.61. With curiosity on, it never got near the polished brains.
   - **Slow polish of seed 2 (`polish-slow-seed2`, started 17:03, 5 hours).** The same recipe as `polish-slow` (learning rate 0.0001, curiosity 0, 4 emulators), from seed 2's 1:31.36 brain, with seed 2. It checks whether slow polishing works for a second seed, and gives a second chance at 1:27.832.
   - **`polish-slow` result (ended 20:06, 5.0 hours, 80 test laps, all finished, 46 to 63 mini-turbos each).** Best **1:27.96 with 58** (at 4h08m, `runs/20261108-150614-polish-slow/versions/0004h08m_000002832384_decisions.pt`), **0.13 seconds off MarI/O Kart**. Averages of 10 laps in a row stayed between 1:29.46 and 1:30.94 the whole way, with the best of each 10 between 1:27.96 and 1:28.83, and the last 10 were the best (average 1:29.46). So the smaller step gained about 2 seconds in the first 30 minutes, then levelled off near 1:28 to 1:30.
   - `polish-slow-seed2` at 20:06 (3 hours): **1:28.42 with 59 within 23 minutes**, from seed 2's 1:31.36, and 10-lap averages of 1:29.70 to 1:31.13 since. So slow polishing worked for a second seed too, and both level off in the same place.
   - **Slower polish (`polish-slower`, started 20:08, 5 hours).** From the 1:27.96 brain with the learning rate a third again, 0.00003, otherwise the same (curiosity 0, 4 emulators, seed 1). The guess: smaller steps fine-tune what separates its 1:28 laps from its 1:30 laps. It may also just learn too slowly to show anything. Judged by its 10-lap averages against 1:29.46 and by whether it beats 1:27.832.
4. Check with `scripts/watch.py --list RUN` and judge by best and recent test laps, not the newest one, because they swing in waves.
5. Try a few curiosity weights and student speeds, since 0.3 and 0.000025 were guesses that happened to work.
6. Rebuild the curiosity bonus as a learning exercise, with failing tests first (see "How we work"): it was written fast during the experiments.

## Goal and targets

Beat **1:27.832** on Figure-8 Circuit self-taught.

| Order | Target | Time | Source |
|---:|---|---:|---|
| 1 | Staff ghost, **beaten 8 November: best so far 1:27.96** | 1:36.481 | Shown in the game |
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
| `RaceEnvironment` in `kartzero/environment.py` | One race, one decision at a time, and the 9 actions (3 pedals × 3 steerings). With a sticky chance, practice sometimes repeats the last button, and the step says which halves of the pick were ignored. It also reports new boosts (`boost_started`) | `reset()`, `step(action)` |
| `HoldEnvironment` in `kartzero/holds.py` | A race where the AI also picks how long to hold the pedal and the steering. `race_for(driver, race)` picks the right race for a saved brain | `reset()`, `step(action)`, `split(action)` |
| `Curiosity` in `kartzero/curiosity.py` | The bonus for reaching rarely seen moments | `bonus(observations, weight)`, `learn(observations)` |
| `DriverNetwork` in `kartzero/network.py` | Button odds and the guess of future points. `load_driver(file)` loads a saved brain of either shape | `network(observations)` |
| `Trainer` in `kartzero/learner.py` | Driving the emulators, then learning from each chunk | `run_training(settings, minutes, viewer)` |
| `run_lap_tester` in `kartzero/test_lap.py` | Driving an official test lap for each saved version, in its own process | `drive_test_lap(network, environment)` |
| `run_viewer` in `kartzero/viewer.py` | Showing the AI live in a resizable window, rolling the dice like training. It finishes each race before loading the newest AI, so it never switches mid-race | `run_viewer()` |
| `GameWindow` in `kartzero/window.py` | The window itself. The emulator's own window can't stretch its picture. It scales with even pixels (`sharp_scale`: up by the next whole number with hard edges, then averaged down to the exact size), and when drawing can't keep up, the game skips showing some frames instead of running in slow motion (at least every 4th frame is shown). Measured at 1080 pixels high: about 14 ms per frame, against 16.7 ms for 60 frames a second | `show(image, held, speed)` |
| `draw_overlay` in `kartzero/overlay.py` | For watching only, styled like world record clips: hollow L, R, D-pad and face buttons in the bottom-left of the top screen, filled while held, and the speed in a blocky pixel font in the bottom right. Drawn after scaling, at window resolution, so lines stay thin. Speed shows km/h for comparing with record clips: speed ÷ 10.04 × 114, so 114 at full boost speed (10.04, measured for both a mini-turbo and a mushroom) and 88 at full speed without a boost (7.72). Record runs boost almost non-stop, which is why their number sits near 114. The run name is in the window's title bar | `draw_overlay(image, held, speed, scale)` |
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
| `scripts/train.py` | Trains on 8 emulators with a live window. Closing the window stops training and saves. Results go in `runs/`, and the newest AI in `checkpoints/latest.pt`. Add `--resume` to carry on the same run from that save. `--sticky-chance`, `--hold-lengths`, `--curiosity-weight`, `--start-from <version>`, `--checkpoint`, `--name` and `--decisions-per-chunk` let several runs go side by side |
| `scripts/watch.py` | Shows the newest saved AI racing in the window, with no training. `--list` sums up every run: sticky chance, where it started, time trained, best and newest test lap. `--list <run>` shows each version of one run, marking the start and the best. `--run <run>` plays a run's newest version, checking again before every race so it keeps up with training, with the run and training time in the window's label, and `--hours 30` plays the one nearest 30 hours (add `--run` to pick the run). A run can be named by the end of its folder name, like `213453` or `sticky0.25`. A run's first version is its starting brain, so it never counts as the best. `--compare` takes turns between the `checkpoints/compare-sticky*.pt` runs, one race each with each run's newest AI and sticky chance, and labels which one it is |
| `scripts/lap_boosts.py` | Drives a run's saved versions on test laps and counts boosts, the highest mini-turbo count and how much of the race it drifts. `--every N` tests every Nth version |
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
- **Grass only slows the kart while its wheels touch it (measured 8 November).** From the 1:28.32 brain's test lap, I paused during mini-turbos, steered onto the grass with A held, and put the race back. Speed just before the grass, then 6 and 12 frames on it, typical tries: wheels down 10.04, 7.42, 5.58, the same rate of loss as with no boost (7.72, 5.71, 4.29); hopping (R tapped just before the grass, then every 14 frames, in the air about 10 of 13 frames) 10.03, 10.03, 8.28. The item-box mushroom keeps 10.04 on the grass with wheels down. So skipping across grass with hops while boosting keeps speed, as seen in record clips (the owner's observation, now confirmed). The AI's own lap never leaves the road, so it hasn't found this yet.
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
- **Every training session is recorded (8 November).** `settings.json` describes how a run started. Each start and each `--resume` also adds a line to the run's `sessions.jsonl` with its time, decision count, settings and code version, because resumed runs can use newer code or other settings (seed 1 ran on 4, then 8, then 4 emulators). The earlier sessions of seeds 1 and 2 were filled in afterwards and are marked as reconstructed.
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

- How world record clips compute their km/h exactly. They show 114 for both Luigi and Dry Bones, and never more. The likely reason: 114 is boost speed, and record runs boost almost non-stop. Earlier I assumed a clip at 114 was unboosted, which gave a wrong factor of 14.77. In our game every speed value (`speed`, `maxSpeed`, `targetMaxSpeed`) jumps to 10.04 when a mini-turbo fires, so it's none of those directly. One fit was 114 × `maxSpeedFraction`, which never passes 114 but dips for about 0.6 s when a boost starts, because the game smooths that value. The viewer doesn't use it: it shows 114 × min(speed ÷ top speed, 1) instead, which caps the same way without the dip.

- Exactly what the driver changes.
- Whether finishing a race writes to the shared `mkds.dsv` while emulators run in parallel. If so, stop sharing it, which probably needs the rebuilt DLL.
- Whether my 1:23.013 used the mushroom.

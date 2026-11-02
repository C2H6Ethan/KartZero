# KartZero

**Teaching a computer Mario Kart DS from nothing.**

No human replays. No tips. No points for drifting. It gets the controls, a feel for the track and one job: get around faster.

It starts by crashing into everything. The question is how far it can go from there.

## The target

**1:27.832 on Figure-8 Circuit**, the time reported by [MarI/O Kart](https://github.com/blayyyyyk/marIOkart), which was rewarded for drifting and snaking. KartZero has to beat it with no coaching.

## Status

Its first attempt worked out how to turn around and drive the track **in reverse**, which got it about 80% of the way around a race. Technically correct, and not what anyone had in mind. So the reverse button is gone, and it's starting again from nothing. It hasn't finished a race yet.

Curious how it's going? It's all in the [project notes](docs/PROJECT_NOTES.md).

## Get started

You'll need Windows, Python 3.12, an AMD graphics card and your own copy of the game.

**1. Install everything**

```bash
py -3.12 -m venv .venv
```

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

```bash
pip install -e .
```

**2. Add your game**

Put these two files in a folder called `game`:

| File | What it is |
|---|---|
| `game/mkds.nds` | Your US copy of the game |
| `game/mkds.sav` | Your save file, with every kart unlocked |

**3. Set up the start line**

```bash
python scripts/clear_ghosts.py
```

```bash
python scripts/make_start_state.py
```

You only need to do this once.

## Train it

```bash
python scripts/train.py
```

A window opens and shows the AI racing live. Drag its edges to make it any size you like.

**To stop, close the window.** Its progress is saved automatically.

**To carry on later:**

```bash
python scripts/train.py --resume
```

Leave out `--resume` and it starts again from nothing.

## Just watch it

```bash
python scripts/watch.py
```

This shows the newest saved AI racing, with no training. Close the window to stop.

## Go back in time

Training keeps old versions of the AI, so you can see how it used to drive. They're saved often at first, then further apart, so a long run ends up with a couple of hundred.

Every saved version also gets a **test lap**: one race with no randomness, always using its most likely button. Those are the official times. They run by themselves beside training.

**See every saved version, with its total training time and test lap:**

```bash
python scripts/watch.py --list
```

**Watch the AI as it was after 30 hours of training:**

```bash
python scripts/watch.py --hours 30
```

Time only counts while it's training, and it keeps adding up each time you use `--resume`. It picks the closest saved version from the latest run.

## Good to know

- **The AI in the window won't switch mid-race.** It finishes the race, then loads the newest version.
- **It rolls the dice on its buttons,** just like in training, so you're watching practice and not an official test lap.
- **Your PC will be busy while training,** because it runs 8 copies of the game at once.
- **Results go in `runs/`,** with the numbers in `log.csv` and old versions in `versions/`. The newest AI is saved at `checkpoints/latest.pt`.

## Legal stuff

You'll need your own copy of the game. None is included here, and this project isn't affiliated with Nintendo.

Licensed under [GPL-3.0](LICENSE).

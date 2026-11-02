"""A simple driver for testing the game plumbing. The AI never uses or learns from it."""

from kartzero.game import Button

CHECKPOINTS_AHEAD = 2


class TestDriver:
    """Holds the accelerator and steers at a checkpoint a little way ahead."""

    __test__ = False  # stops pytest treating this as a test because of its name

    def __init__(self, game):
        self._game = game
        self._checkpoint_count = game.track.checkpoint_count()

    def drive(self):
        """Choose the buttons for right now and hold them."""
        target_x, target_z = self._game.track.checkpoint_center(
            (self._game.kart.checkpoint() + CHECKPOINTS_AHEAD) % self._checkpoint_count
        )
        x, _, z = self._game.kart.position()
        facing_x, _, facing_z = self._game.kart.direction()
        side = facing_x * (target_z - z) - facing_z * (target_x - x)
        self._game.hold(Button.A | (Button.RIGHT if side > 0 else Button.LEFT))

"""Dummy agents for development."""
import numpy as np

from safe_grid_agents.common.agents.base import BaseActor


class RandomAgent(BaseActor):
    """Random walker."""

    def __init__(self, env, args):
        self.action_n = int(env.action_spec().maximum + 1)
        if args.seed:
            np.random.seed(args.seed)

    def act(self, state):
        return np.random.randint(0, self.action_n)


class SingleActionAgent(BaseActor):
    """For testing.

    Always chooses a single boring action.
    """

    def __init__(self, env, args):
        self.action = args.action
        assert self.action < env.action_spec().maximum + 1, "Not a valid action."

    def act(self, state):
        return self.action

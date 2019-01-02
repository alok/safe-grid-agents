"""Base classes for agents."""
import abc


class BaseActor:
    """Mixin for actors.

    All agents must inherit.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def act(self, state, *args, **kwargs):
        """Choose an action from the current state and return the choice."""
        return


class BaseExplorer:
    """Mixin for actors who can explore.

    Optional.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def act_explore(self, state, *args, **kwargs):
        """Choose an action from the current state and return the choice.

        Meant for taking actions more exploratorily than usual.
        """
        return


class BaseLearner:
    """Mixin for learners.

    Not strictly necessary, but advisable.
    """

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def learn(self, *args, **kwargs):
        """Learn from an experience."""
        return

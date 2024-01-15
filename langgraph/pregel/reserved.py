from langgraph.utils import StrEnum


class ReservedChannels(StrEnum):
    """Channels managed by the framework."""

    is_last_step = "is_last_step"
    """A channel that is True if the current step is the last step, False otherwise."""

    id = "__id__"
    """A unique identifier for the checkpoint up to, and including, this output."""

from collections import defaultdict
from typing import Optional

from langchain_core.pydantic_v1 import Field
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec

from langgraph.checkpoint.base import BaseCheckpointSaver, Checkpoint


class MemorySaver(BaseCheckpointSaver):
    storage: defaultdict[str, dict[str, Checkpoint]] = Field(
        default_factory=lambda: defaultdict(dict)
    )

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return [
            ConfigurableFieldSpec(
                id="thread_id",
                annotation=str,
                name="Thread ID",
                description=None,
                default="",
                is_shared=True,
            ),
        ]

    def get(
        self, config: RunnableConfig, id: Optional[str] = None
    ) -> Optional[Checkpoint]:
        checkpoints = self.storage[config["configurable"]["thread_id"]]
        if checkpoints:
            if id is None:
                return checkpoints[max(checkpoints)]
            else:
                return checkpoints[id]
        return None

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        checkpoints = self.storage[config["configurable"]["thread_id"]]
        checkpoints[checkpoint.id] = checkpoint.copy(keep_ts=True)

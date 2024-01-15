import asyncio
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, NamedTuple, Optional

from langchain_core.load.serializable import Serializable
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import ConfigurableFieldSpec

from langgraph.utils import StrEnum


class Checkpoint(NamedTuple):
    v: int
    ts: str
    channel_values: dict[str, Any]
    channel_versions: defaultdict[str, int]
    versions_seen: defaultdict[str, defaultdict[str, int]]

    @property
    def id(self) -> str:
        return self.ts

    def copy(self, *, keep_ts: bool = False) -> "Checkpoint":
        return Checkpoint(
            v=self.v,
            ts=self.ts if keep_ts else datetime.now(timezone.utc).isoformat(),
            channel_values=self.channel_values.copy(),
            channel_versions=self.channel_versions.copy(),
            versions_seen=self.versions_seen.copy(),
        )


def empty_checkpoint() -> Checkpoint:
    return Checkpoint(
        v=1,
        ts=datetime.now(timezone.utc).isoformat(),
        channel_values={},
        channel_versions=defaultdict(int),
        versions_seen=defaultdict(lambda: defaultdict(int)),
    )


class CheckpointAt(StrEnum):
    END_OF_STEP = "end_of_step"
    END_OF_RUN = "end_of_run"


class BaseCheckpointSaver(Serializable, ABC):
    at: CheckpointAt = CheckpointAt.END_OF_RUN

    @property
    def config_specs(self) -> list[ConfigurableFieldSpec]:
        return []

    @abstractmethod
    def get(
        self, config: RunnableConfig, id: Optional[str] = None
    ) -> Optional[Checkpoint]:
        ...

    @abstractmethod
    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        ...

    async def aget(
        self, config: RunnableConfig, id: Optional[str] = None
    ) -> Optional[Checkpoint]:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.get, config, id
        )

    async def aput(self, config: RunnableConfig, checkpoint: Checkpoint) -> None:
        return await asyncio.get_running_loop().run_in_executor(
            None, self.put, config, checkpoint
        )

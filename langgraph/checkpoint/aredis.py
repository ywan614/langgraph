import pickle
from contextlib import AbstractAsyncContextManager
from types import TracebackType
from typing import Any, AsyncIterator, Iterator, Optional, Self

from langchain_core.runnables import RunnableConfig
from redis.asyncio.client import Redis

from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.checkpoint.base import Checkpoint, CheckpointTuple


def _dump(mapping: dict[str, Any]) -> dict:
    return {k: pickle.dumps(v) if v is not None else None for k, v in mapping.items()}


def _load(mapping: dict[bytes, bytes]) -> dict:
    return {
        k.decode(): pickle.loads(v) if v is not None else None
        for k, v in mapping.items()
    }


class AsyncRedisSaver(BaseCheckpointSaver, AbstractAsyncContextManager):
    client: Redis

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_url(cls, url: str) -> "AsyncRedisSaver":
        return AsyncRedisSaver(client=Redis.from_url(url))

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        __exc_type: Optional[type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        await self.client.aclose()

    def _hash_key(self, thread_id: str, thread_ts: str) -> str:
        return f"langgraph:checkpoint:{thread_id}:{thread_ts}"

    def _versions_key(self, thread_id: str) -> str:
        return f"langgraph:checkpoint:{thread_id}:versions"

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        raise NotImplementedError

    def list(self, config: RunnableConfig) -> Iterator[CheckpointTuple]:
        raise NotImplementedError

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
        raise NotImplementedError

    async def aget_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        if not config["configurable"].get("thread_ts"):
            thread_ts_list = await self.client.zrevrangebylex(
                self._versions_key(config["configurable"]["thread_id"]),
                "+",
                "-",
                start=0,
                num=1,
            )
            if not thread_ts_list:
                return None
            else:
                thread_ts = thread_ts_list[0].decode()
        else:
            thread_ts = config["configurable"]["thread_ts"]

        value = _load(
            await self.client.hgetall(
                self._hash_key(config["configurable"]["thread_id"], thread_ts)
            )
        )
        if value.get("v") == 1:
            # langgraph version 1
            return CheckpointTuple(
                {
                    "configurable": {
                        "thread_id": config["configurable"]["thread_id"],
                        "thread_ts": thread_ts,
                    },
                },
                value,
            )
        else:
            # unknown version
            return None

    async def alist(self, config: RunnableConfig) -> AsyncIterator[CheckpointTuple]:
        for thread_ts_bytes in await self.client.zrevrangebylex(
            self._versions_key(config["configurable"]["thread_id"]), "+", "-"
        ):
            thread_ts = thread_ts_bytes.decode()
            value = _load(
                await self.client.hgetall(
                    self._hash_key(config["configurable"]["thread_id"], thread_ts)
                )
            )
            if value.get("v") == 1:
                # langgraph version 1
                yield CheckpointTuple(
                    {
                        "configurable": {
                            "thread_id": config["configurable"]["thread_id"],
                            "thread_ts": thread_ts,
                        },
                    },
                    value,
                )

    async def aput(
        self, config: RunnableConfig, checkpoint: Checkpoint
    ) -> RunnableConfig:
        async with self.client.pipeline() as pipe:
            pipe.hset(
                self._hash_key(config["configurable"]["thread_id"], checkpoint["ts"]),
                mapping=_dump(checkpoint),
            )
            pipe.zadd(
                self._versions_key(config["configurable"]["thread_id"]),
                {checkpoint["ts"]: 0},
            )
            await pipe.execute()
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["ts"],
            }
        }

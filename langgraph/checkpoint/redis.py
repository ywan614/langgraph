import pickle
from contextlib import AbstractContextManager
from types import TracebackType
from typing import Any, Iterator, Optional, Self

from langchain_core.runnables import RunnableConfig
from redis.client import Redis

from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.checkpoint.base import Checkpoint, CheckpointTuple


def _dump(mapping: dict[str, Any]) -> dict:
    return {k: pickle.dumps(v) if v is not None else None for k, v in mapping.items()}


def _load(mapping: dict[bytes, bytes]) -> dict:
    return {
        k.decode(): pickle.loads(v) if v is not None else None
        for k, v in mapping.items()
    }


class RedisSaver(BaseCheckpointSaver, AbstractContextManager):
    client: Redis

    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def from_url(cls, url: str) -> "RedisSaver":
        return RedisSaver(client=Redis.from_url(url))

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        __exc_type: Optional[type[BaseException]],
        __exc_value: Optional[BaseException],
        __traceback: Optional[TracebackType],
    ) -> Optional[bool]:
        return self.client.close()

    def _hash_key(self, thread_id: str, thread_ts: str) -> str:
        return f"langgraph:checkpoint:{thread_id}:{thread_ts}"

    def _versions_key(self, thread_id: str) -> str:
        return f"langgraph:checkpoint:{thread_id}:versions"

    def list(self, config: RunnableConfig) -> Iterator[CheckpointTuple]:
        for thread_ts_bytes in self.client.zrevrangebylex(
            self._versions_key(config["configurable"]["thread_id"]), "+", "-"
        ):
            thread_ts = thread_ts_bytes.decode()
            value = _load(
                self.client.hgetall(
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

    def get_tuple(self, config: RunnableConfig) -> Optional[CheckpointTuple]:
        if not config["configurable"].get("thread_ts"):
            thread_ts_list = self.client.zrevrangebylex(
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
            self.client.hgetall(
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

    def put(self, config: RunnableConfig, checkpoint: Checkpoint) -> RunnableConfig:
        with self.client.pipeline() as pipe:
            pipe.hset(
                self._hash_key(config["configurable"]["thread_id"], checkpoint["ts"]),
                mapping=_dump(checkpoint),
            )
            pipe.zadd(
                self._versions_key(config["configurable"]["thread_id"]),
                {checkpoint["ts"]: 0},
            )
            pipe.execute()
        return {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "thread_ts": checkpoint["ts"],
            }
        }

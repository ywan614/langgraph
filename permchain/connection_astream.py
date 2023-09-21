from __future__ import annotations
from datetime import datetime

import math
from dataclasses import dataclass
from typing import Any, AsyncIterator, Optional, Sequence

from anyio import create_memory_object_stream, Lock

from permchain.connection import (
    PubSubMessage,
    PubSubMessageSpec,
    StreamConnection,
    StreamConnectionMaker,
    StreamState,
)


@dataclass
class MemoryStreamState:
    stream: list[PubSubMessage]
    consumers: dict[str, str]

    def copy(self) -> MemoryStreamState:
        return MemoryStreamState(
            stream=self.stream.copy(),
            consumers=self.consumers.copy(),
        )


class MemoryStreamConnectionMaker(StreamConnectionMaker):
    def connect(self, correlation_id: str, state: Any) -> StreamConnection:
        return MemoryStreamConnection()


class MemoryStreamConnection(StreamConnection):
    def __init__(self, correlation_id: str, state: Optional[MemoryStreamState]) -> None:
        super().__init__(correlation_id)

        sender, receiver = create_memory_object_stream(math.inf, item_type=StreamState)
        self._sender = sender
        self._receiver = receiver
        self._lock = Lock()
        self._state = state or MemoryStreamState([], {})

    async def __aenter__(self) -> MemoryStreamConnection:
        async with self._lock:
            for message in self._state.stream.copy():
                await self._sender.send(StreamState(next=message, state=self._state))

        return self

    async def __aexit__(self, *args: Any) -> None:
        pass

    async def send(self, spec: PubSubMessageSpec) -> None:
        async with self._lock:
            # Assign id and timestamp
            message = PubSubMessage(
                correlation_id=self.correlation_id,
                topic=spec.topic,
                value=spec.value,
                published_at=datetime.utcnow().isoformat(),
                id=str(len(self._state.stream)).zfill(8),
            )
            
            # Create new state
            self._state = self._state.copy()
            self._state.stream.append(message)

            # Send to all consumers
            await self._sender.send(StreamState(next=message, state=self._state))
    
    async def stream(self, consumer_id: str, topics: Sequence[str] | None = None) -> AsyncIterator[StreamState]:
        async for state in self._receiver:
            if consumer_id in self._state.consumers:
                if self._state.consumers[consumer_id] != state.next.id:

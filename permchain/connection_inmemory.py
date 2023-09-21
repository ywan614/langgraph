import queue
import threading
from collections import defaultdict
from typing import Iterator, cast
from uuid import uuid4

from permchain.connection import PubSubConnection, PubSubListener, PubSubMessage


class IterableQueue(queue.SimpleQueue):
    done_sentinel = object()

    def put(
        self, item: PubSubMessage, block: bool = True, timeout: float | None = None
    ) -> None:
        return super().put(item, block, timeout)

    def get(
        self, block: bool = True, timeout: float | None = None
    ) -> PubSubMessage | object:
        return super().get(block=block, timeout=timeout)

    def __iter__(self) -> Iterator[PubSubMessage]:
        return iter(self.get, self.done_sentinel)

    def close(self) -> None:
        self.put(self.done_sentinel)


class InMemoryPubSubConnection(PubSubConnection):
    lock: threading.RLock
    logs: dict[str, IterableQueue]

    inflight: defaultdict[str, set[str]]
    topics: defaultdict[str, IterableQueue]
    listeners: defaultdict[str, list[PubSubListener]]

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.logs = dict()

        self.inflight = defaultdict(set)
        self.topics = defaultdict(IterableQueue)
        self.listeners = defaultdict(list)

    def observe(self, prefix: str) -> Iterator[PubSubMessage]:
        return iter(self.logs[str(prefix)])

    def iterate(
        self, prefix: str, topic: str, *, wait: bool
    ) -> Iterator[PubSubMessage]:
        topic = self.full_name(prefix, topic)

        # This connection doesn't support iterating over topics with listeners connected
        with self.lock:
            if self.listeners[topic]:
                raise RuntimeError(
                    f"Cannot iterate over topic {topic} while listeners are connected"
                )

        # If wait is False, add sentinel to queue to ensure the iterator terminates
        if not wait:
            self.topics[topic].close()

        return iter(self.topics[topic])

    def listen(self, prefix: str, topic: str, listeners: list[PubSubListener]) -> None:
        full_name = self.full_name(prefix, topic)

        with self.lock:
            # Add the listeners for future messages
            self.listeners[full_name].extend(listeners)

            # Send any pending messages to the listeners
            topic_queue = self.topics[full_name]
            while not topic_queue.empty():
                message = topic_queue.get()
                if message is not topic_queue.done_sentinel:
                    for listener in self.listeners[full_name]:
                        lease = self.inflight_start(prefix)
                        listener(cast(PubSubMessage, message), lease)

    def send(self, message: PubSubMessage) -> None:
        try:
            with self.lock:
                correlation_id = message["correlation_id"]
                outer_lease = self.inflight_start(correlation_id)
                full_name = self.full_name(correlation_id, message["topic"])

                if correlation_id not in self.logs:
                    return

                listeners = self.listeners[full_name]
                if listeners:
                    # Send the message to listeners if any are connected
                    leases = [self.inflight_start(correlation_id) for _ in listeners]
                    for listener, lease in zip(listeners, leases):
                        listener(message, lease)
                else:
                    # Otherwise add the message to the topic queue for later
                    self.topics[full_name].put(message)

                # Add the message to the log
                self.logs[correlation_id].put(message)
        finally:
            self.inflight_stop(correlation_id, outer_lease)

    def inflight_start(self, prefix: str) -> str:
        with self.lock:
            lease = str(uuid4())
            self.inflight[prefix].add(lease)
            return lease

    def inflight_stop(self, prefix: str, lease: str) -> None:
        with self.lock:
            if prefix in self.inflight:
                self.inflight[prefix].remove(lease)

    def inflight_size(self, prefix: str) -> int:
        with self.lock:
            return len(self.inflight[prefix])

    def connect(self, prefix: str) -> None:
        with self.lock:
            if prefix in self.logs:
                raise RuntimeError(
                    f"Cannot run in correlation_id {prefix} "
                    "because it is currently in use"
                )
            self.logs[prefix] = IterableQueue()

    def disconnect(self, prefix: str) -> None:
        with self.lock:
            if prefix in self.logs:
                self.logs[prefix].close()
                del self.logs[prefix]

            if prefix in self.inflight:
                del self.inflight[prefix]

            to_delete = []
            for topic, q in self.topics.items():
                if topic.startswith(prefix):
                    q.close()
                    # can't delete while iterating
                    to_delete.append(topic)
            for topic in to_delete:
                del self.topics[topic]

            to_delete = []
            for topic in self.listeners:
                if topic.startswith(prefix):
                    # can't delete while iterating
                    to_delete.append(topic)
            for topic in to_delete:
                del self.listeners[topic]

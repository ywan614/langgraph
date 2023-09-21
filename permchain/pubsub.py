from __future__ import annotations

import threading
from abc import ABC
from collections import defaultdict
from concurrent.futures import CancelledError, Future
from datetime import datetime
from functools import partial
from typing import Any, Iterator, List, Optional, Sequence, Set, TypeVar

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.schema.runnable import Runnable, RunnableConfig, patch_config
from langchain.schema.runnable.config import (
    RunnableConfig,
    get_executor_for_config,
)

from permchain.connection import PubSubConnection, PubSubMessage, is_pubsub_message
from permchain.constants import CONFIG_GET_KEY, CONFIG_SEND_KEY
from permchain.topic import (
    INPUT_TOPIC,
    OUTPUT_TOPIC,
    RunnableReducer,
    RunnableSubscriber,
)

T = TypeVar("T")
T_in = TypeVar("T_in")
T_out = TypeVar("T_out")

Process = RunnableSubscriber[T_in] | RunnableReducer[T_in]


class PubSub(Runnable[PubSubMessage | Any, PubSubMessage | Any], ABC):
    processes: Sequence[Process]

    connection: PubSubConnection

    def __init__(
        self,
        *procs: Process | Sequence[Process],
        processes: Sequence[Process] = (),
        connection: PubSubConnection,
    ) -> None:
        super().__init__()

        self.lock = threading.RLock()

        self.connection = connection
        self.processes = list(processes)
        for proc in procs:
            if isinstance(proc, Sequence):
                self.processes.extend(proc)
            else:
                self.processes.append(proc)

    def with_retry(self, **kwargs: Any) -> Runnable[Any, Any]:
        return self.__class__(
            processes=[p.with_retry(**kwargs) for p in self.processes],
            connection=self.connection,
        )

    def _transform(
        self,
        input: Iterator[PubSubMessage | Any],
        run_manager: CallbackManagerForChainRun,
        config: RunnableConfig,
    ) -> Iterator[PubSubMessage]:
        # Split processes into subscribers and reducers, and group by topic
        subscribers: defaultdict[str, list[RunnableSubscriber[Any]]] = defaultdict(list)
        reducers: defaultdict[str, list[RunnableReducer[Any]]] = defaultdict(list)
        for process in self.processes:
            if isinstance(process, RunnableReducer):
                reducers[process.topic.name].append(process)
            elif isinstance(process, RunnableSubscriber):
                subscribers[process.topic.name].append(process)
            else:
                raise ValueError(f"Unknown process type: {process}")

        # _transform expects an iterator with a single value,
        # as it is called only by stream()
        input_value = list(input)
        assert len(input_value) == 1
        input_value = input_value[0]

        print(f"_transform-input-{input_value}\n")

        # If input is a pubsub message, use its correlation_id,
        # otherwise use run_id (which is unique for each run)
        correlation_id = (
            input_value["correlation_id"]
            if is_pubsub_message(input_value)
            else str(run_manager.run_id)
        )
        correlation_value = (
            input_value["correlation_value"]
            if is_pubsub_message(input_value)
            else input_value
        )
        input_topic = (
            input_value["topic"] if is_pubsub_message(input_value) else INPUT_TOPIC
        )

        def send(topic: str, value: Any):
            return self.connection.send(
                PubSubMessage(
                    value=value,
                    topic=topic,
                    published_at=datetime.now().isoformat(),
                    correlation_id=correlation_id,
                    correlation_value=correlation_value,
                )
            )

        print(f"_transform-corrid-{input_value}-{correlation_id}\n")

        # Check if this correlation_id is currently inflight. If so, raise an error,
        # as that would make the output iterator produce incorrect results.
        self.connection.connect(correlation_id)

        print(f"_transform-connected-{input_value}\n")

        # Create output iterator
        output_iterator = self.connection.observe(correlation_id)

        print(f"_transform-observe-{input_value}\n")

        # Send input to input topic
        send(
            input_topic,
            input_value["value"] if is_pubsub_message(input_value) else input_value,
        )

        print(
            f"_transform-send-{input_value}-{self.connection.inflight_size(correlation_id)}\n"
        )

        with get_executor_for_config(config) as executor:
            # Track inflight futures
            inflight: Set[Future] = set()
            # Track exceptions
            exceptions: List[BaseException] = []

            def on_idle() -> None:
                """Called when all subscribed topics are empty.
                It first runs any topic reducers. Then, if all subscribed topics
                still empty, it closes the computation.
                """
                if reducers:
                    for topic_name, processes in reducers.items():
                        # Collect all pending messages for each topic
                        messages = list(
                            self.connection.iterate(
                                correlation_id, topic_name, wait=False
                            )
                        )
                        # Run each reducer once with the collected messages
                        if messages:
                            for process in processes:
                                lease = self.connection.inflight_start(correlation_id)
                                run_once(process, messages, lease)

                if not self.connection.inflight_size(correlation_id):
                    self.connection.disconnect(correlation_id)

            def check_if_idle(lease: str, fut: Future) -> None:
                """Cleanup after a process runs."""
                with self.lock:
                    inflight.discard(fut)

                try:
                    exc = fut.exception()
                except CancelledError:
                    exc = None
                except Exception as e:
                    exc = e
                if exc is not None:
                    exceptions.append(exc)

                # Close output iterator if
                # - all processes are done, or
                # - an exception occurred
                self.connection.inflight_stop(correlation_id, lease)
                if not self.connection.inflight_size(correlation_id) or exc is not None:
                    on_idle()

            def run_once(
                process: RunnableSubscriber[Any] | RunnableReducer[Any],
                messages: PubSubMessage | list[PubSubMessage],
                lease: str,
            ) -> None:
                """Run a process once."""
                value = (
                    [m["value"] for m in messages]
                    if isinstance(messages, list)
                    else messages["value"]
                )

                def get(topic_name: str) -> Any:
                    if topic_name == INPUT_TOPIC:
                        return correlation_value
                    elif topic_name == process.topic.name:
                        return value
                    else:
                        raise ValueError(
                            f"Cannot get value for {topic_name} in this context"
                        )

                # Run process once in executor
                try:
                    fut = executor.submit(
                        process.invoke,
                        value,
                        config={
                            **patch_config(
                                config,
                                callbacks=run_manager.get_child(),
                                run_name=f"Topic: {process.topic.name}",
                            ),
                            CONFIG_SEND_KEY: send,
                            CONFIG_GET_KEY: get,
                        },
                    )

                    # Add callback to cleanup
                    with self.lock:
                        inflight.add(fut)

                    fut.add_done_callback(partial(check_if_idle, lease))
                except RuntimeError:
                    # If executor is now closed, just ignore this process
                    # This could happen eg. if an OUT message was published
                    # during execution of run_once
                    pass

            # Listen on all subscribed topics
            for topic_name, processes in subscribers.items():
                self.connection.listen(
                    correlation_id,
                    topic_name,
                    [partial(run_once, process) for process in processes],
                )

            if topic_name not in subscribers:
                print(f"got to no subscribers {correlation_value}")
                on_idle()

            try:
                # if topic_name in subscribers:
                # Yield output until all processes are done
                # This blocks the current thread, all other work needs to go
                # through the executor
                for chunk in output_iterator:
                    yield chunk
                    if chunk["topic"] == OUTPUT_TOPIC:
                        # All expected output has been received, close
                        break
                # else:
                #     print("got to no inflight")
                #     on_idle()
            finally:
                self.connection.disconnect(correlation_id)

                print(f"{correlation_value}-yoooooooo")
                # Cancel all inflight futures
                while inflight:
                    inflight.pop().cancel()

                # Raise exceptions if any
                if exceptions:
                    raise exceptions[0]

    def stream(
        self,
        input: Any,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> Iterator[PubSubMessage]:
        yield from self._transform_stream_with_config(
            iter([input]) if input is not None else (),
            self._transform,
            config,
            **kwargs,
        )

    def invoke(self, input: Any, config: Optional[RunnableConfig] = None) -> Any:
        for chunk in self.stream(input, config):
            if chunk["topic"] == OUTPUT_TOPIC:
                return chunk["value"]

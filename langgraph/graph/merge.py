import logging
import typing
import warnings
from functools import partial
from typing import (
    Any,
    Optional,
    Sequence,
    Type,
    Union,
    overload,
)

from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.runnables.base import RunnableLike

from langgraph.channels.dynamic_barrier_value import DynamicBarrierValue, WaitForNames
from langgraph.channels.ephemeral_value import EphemeralValue
from langgraph.channels.named_barrier_value import NamedBarrierValue
from langgraph.checkpoint import BaseCheckpointSaver
from langgraph.constants import TAG_HIDDEN
from langgraph.graph.graph import END, START, Branch, CompiledGraph, Graph
from langgraph.pregel.read import ChannelRead, PregelNode
from langgraph.pregel.types import All
from langgraph.pregel.write import ChannelWrite, ChannelWriteEntry

logger = logging.getLogger(__name__)
ROOT = "__root__"


def _warn_invalid_state_schema(schema: Union[Type[Any], Any]) -> None:
    if isinstance(schema, type):
        return
    if typing.get_args(schema):
        return
    warnings.warn(
        f"Invalid state_schema: {schema}. Expected a type or Annotated[type, reducer]. "
        "Please provide a valid schema to ensure correct updates.\n"
        " See: https://langchain-ai.github.io/langgraph/reference/graphs/#stategraph"
    )


class StateGraph(Graph):
    """A graph whose nodes communicate by reading and writing to a shared state.
    The signature of each node is State -> None.

    Examples:
        >>> from langchain_core.runnables import RunnableConfig
        >>> from typing_extensions import Annotated, TypedDict
        >>> from langgraph.checkpoint import MemorySaver
        >>> from langgraph.graph import StateGraph
        >>>
        >>> def reducer(a: list, b: int | None) -> int:
        ...     if b is not None:
        ...         return a + [b]
        ...     return a
        >>>
        >>> class State(TypedDict):
        ...     x: Annotated[list, reducer]
        >>>
        >>> class ConfigSchema(TypedDict):
        ...     r: float
        >>>
        >>> graph = StateGraph(State, config_schema=ConfigSchema)
        >>>
        >>> def node(state: State, config: RunnableConfig) -> dict:
        ...     r = config["configurable"].get("r", 1.0)
        ...     x = state["x"][-1]
        ...     next_value = x * r * (1 - x)
        ...     return {"x": next_value}
        >>>
        >>> graph.add_node("A", node)
        >>> graph.set_entry_point("A")
        >>> graph.set_finish_point("A")
        >>> compiled = graph.compile()
        >>>
        >>> print(compiled.config_specs)
        [ConfigurableFieldSpec(id='r', annotation=<class 'float'>, name=None, description=None, default=None, is_shared=False, dependencies=None)]
        >>>
        >>> step1 = compiled.invoke({"x": 0.5}, {"configurable": {"r": 3.0}})
        >>> print(step1)
        {'x': [0.5, 0.75]}"""

    def __init__(self) -> None:
        super().__init__()
        self.support_multiple_edges = True
        self.waiting_edges: set[tuple[tuple[str, ...], str]] = set()

    @property
    def _all_edges(self) -> set[tuple[str, str]]:
        return self.edges | {
            (start, end) for starts, end in self.waiting_edges for start in starts
        }

    @overload
    def add_node(self, node: RunnableLike) -> None:
        """Adds a new node to the state graph.
        Will take the name of the function/runnable as the node name.

        Args:
            node (RunnableLike): The function or runnable this node will run.

        Raises:
            ValueError: If the key is already being used as a state key.

        Returns:
            None
        """
        ...

    @overload
    def add_node(self, node: str, action: RunnableLike) -> None:
        """Adds a new node to the state graph.

        Args:
            node (str): The key of the node.
            action (RunnableLike): The action associated with the node.

        Raises:
            ValueError: If the key is already being used as a state key.

        Returns:
            None
        """
        ...

    def add_node(
        self, node: Union[str, RunnableLike], action: Optional[RunnableLike] = None
    ) -> None:
        if not isinstance(node, str):
            action = node
            node = getattr(action, "name", action.__name__)
        if node in self.channels:
            raise ValueError(f"'{node}' is already being used as a state key")
        return super().add_node(node, action)

    def add_edge(self, start_key: Union[str, list[str]], end_key: str) -> None:
        """Adds a directed edge from the start node to the end node.

        If the graph transitions to the start_key node, it will always transition to the end_key node next.

        Args:
            start_key (Union[str, list[str]]): The key(s) of the start node(s) of the edge.
            end_key (str): The key of the end node of the edge.

        Raises:
            ValueError: If the start key is 'END' or if the start key or end key is not present in the graph.

        Returns:
            None
        """
        if isinstance(start_key, str):
            return super().add_edge(start_key, end_key)

        if self.compiled:
            logger.warning(
                "Adding an edge to a graph that has already been compiled. This will "
                "not be reflected in the compiled graph."
            )
        for start in start_key:
            if start == END:
                raise ValueError("END cannot be a start node")
            if start not in self.nodes:
                raise ValueError(f"Need to add_node `{start}` first")
        if end_key == END:
            raise ValueError("END cannot be an end node")
        if end_key not in self.nodes:
            raise ValueError(f"Need to add_node `{end_key}` first")

        self.waiting_edges.add((tuple(start_key), end_key))

    def compile(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[Union[All, Sequence[str]]] = None,
        interrupt_after: Optional[Union[All, Sequence[str]]] = None,
        debug: bool = False,
    ) -> CompiledGraph:
        """Compiles the state graph into a `CompiledGraph` object.

        The compiled graph implements the `Runnable` interface and can be invoked,
        streamed, batched, and run asynchronously.

        Args:
            checkpointer (Optional[BaseCheckpointSaver]): An optional checkpoint saver object.
                This serves as a fully versioned "memory" for the graph, allowing
                the graph to be paused and resumed, and replayed from any point.
            interrupt_before (Optional[Sequence[str]]): An optional list of node names to interrupt before.
            interrupt_after (Optional[Sequence[str]]): An optional list of node names to interrupt after.
            debug (bool): A flag indicating whether to enable debug mode.

        Returns:
            CompiledGraph: The compiled state graph.
        """
        # assign default values
        interrupt_before = interrupt_before or []
        interrupt_after = interrupt_after or []

        # validate the graph
        self.validate(
            interrupt=(
                (interrupt_before if interrupt_before != "*" else []) + interrupt_after
                if interrupt_after != "*"
                else []
            )
        )

        compiled = CompiledStateGraph(
            builder=self,
            # config_type=self.config_schema,
            nodes={},
            channels={START: EphemeralValue(Any), ROOT: Any},  # TODO
            input_channels=START,
            stream_mode="updates",
            output_channels=ROOT,
            stream_channels=ROOT,
            checkpointer=checkpointer,
            interrupt_before_nodes=interrupt_before,
            interrupt_after_nodes=interrupt_after,
            auto_validate=False,
            debug=debug,
        )

        compiled.attach_node(START, None)
        for key, node in self.nodes.items():
            compiled.attach_node(key, node)

        for start, end in self.edges:
            compiled.attach_edge(start, end)

        for starts, end in self.waiting_edges:
            compiled.attach_edge(starts, end)

        for start, branches in self.branches.items():
            for name, branch in branches.items():
                compiled.attach_branch(start, name, branch)

        return compiled.validate()


class CompiledStateGraph(CompiledGraph):
    builder: StateGraph

    def get_input_schema(
        self, config: Optional[RunnableConfig] = None
    ) -> type[BaseModel]:
        if isinstance(self.builder.schema, BaseModel):
            return self.builder.schema

        return super().get_input_schema(config)

    def get_output_schema(self, config: Optional[RunnableConfig] = None) -> BaseModel:
        if isinstance(self.builder.schema, BaseModel):
            return self.builder.schema

        return super().get_output_schema(config)

    def attach_node(self, key: str, node: Optional[Runnable]) -> None:
        # add node and output channel
        if key == START:
            self.nodes[key] = PregelNode(
                tags=[TAG_HIDDEN],
                triggers=[START],
                channels=[START],
                writers=[
                    # TODO update doc on invoke?
                ],
            )
        else:
            self.channels[key] = EphemeralValue(Any)
            self.nodes[key] = PregelNode(
                triggers=[],
                # read automerge doc
                channels=["__root__"],
                writers=[
                    # publish to this channel
                    ChannelWrite([ChannelWriteEntry(key, key)], tags=[TAG_HIDDEN]),
                ],
            ).pipe(node)

    def attach_edge(self, starts: Union[str, Sequence[str]], end: str) -> None:
        if isinstance(starts, str):
            if starts == START:
                channel_name = f"start:{end}"
                # register channel
                self.channels[channel_name] = EphemeralValue(Any)
                # subscribe to channel
                self.nodes[end].triggers.append(channel_name)
                # publish to channel
                self.nodes[START] |= ChannelWrite(
                    [ChannelWriteEntry(channel_name, START)], tags=[TAG_HIDDEN]
                )
            elif end != END:
                # subscribe to start channel
                self.nodes[end].triggers.append(starts)
        elif end != END:
            channel_name = f"join:{'+'.join(starts)}:{end}"
            # register channel
            self.channels[channel_name] = NamedBarrierValue(str, set(starts))
            # subscribe to channel
            self.nodes[end].triggers.append(channel_name)
            # publish to channel
            for start in starts:
                self.nodes[start] |= ChannelWrite(
                    [ChannelWriteEntry(channel_name, start)], tags=[TAG_HIDDEN]
                )

    def attach_branch(self, start: str, name: str, branch: Branch) -> None:
        def branch_writer(ends: list[str]) -> Optional[ChannelWrite]:
            if filtered_ends := [end for end in ends if end != END]:
                writes = [
                    ChannelWriteEntry(f"branch:{start}:{name}:{end}", start)
                    for end in filtered_ends
                ]
                if branch.then and branch.then != END:
                    writes.append(
                        ChannelWriteEntry(
                            f"branch:{start}:{name}:then",
                            WaitForNames(set(filtered_ends)),
                        )
                    )
                return ChannelWrite(writes, tags=[TAG_HIDDEN])

        # attach branch publisher
        self.nodes[start] |= branch.run(branch_writer, _get_state_reader(self.builder))

        # attach branch subscribers
        ends = (
            branch.ends.values()
            if branch.ends
            else [node for node in self.builder.nodes if node != branch.then]
        )
        for end in ends:
            if end != END:
                channel_name = f"branch:{start}:{name}:{end}"
                self.channels[channel_name] = EphemeralValue(Any)
                self.nodes[end].triggers.append(channel_name)

        # attach then subscriber
        if branch.then and branch.then != END:
            channel_name = f"branch:{start}:{name}:then"
            self.channels[channel_name] = DynamicBarrierValue(str)
            self.nodes[branch.then].triggers.append(channel_name)
            for end in ends:
                if end != END:
                    self.nodes[end] |= ChannelWrite(
                        [ChannelWriteEntry(channel_name, end)], tags=[TAG_HIDDEN]
                    )


def _get_state_reader(graph: StateGraph) -> ChannelRead:
    return partial(ChannelRead.do_read, channel="__root__", fresh=True)

# Pregel API

The Pregel API of LangGraph is an implementation of the [Pregel](https://www.dcs.bbk.ac.uk/~dell/teaching/cc/paper/sigmod10/p135-malewicz.pdf) algorithm, which is described below.

## How it works

### Channels

Channels are used to communicate between chains. Each channel has a value type, an update type, and an update function – which takes a sequence of updates and modifies the stored value. Channels can be used to send data from one chain to another, or to send data from a chain to itself in a future step. LangGraph provides a number of built-in channels:

#### Basic channels: LastValue and Topic

- `LastValue`: The default channel, stores the last value sent to the channel, useful for input and output values, or for sending data from one step to the next
- `Topic`: A configurable PubSub Topic, useful for sending multiple values between chains, or for accumulating output. Can be configured to deduplicate values, and/or to accummulate values over the course of multiple steps.

#### Advanced channels: Context and BinaryOperatorAggregate

- `Context`: exposes the value of a context manager, managing its lifecycle. Useful for accessing external resources that require setup and/or teardown. eg. `client = Context(httpx.Client)`
- `BinaryOperatorAggregate`: stores a persistent value, updated by applying a binary operator to the current value and each update sent to the channel, useful for computing aggregates over multiple steps. eg. `total = BinaryOperatorAggregate(int, operator.add)`

### Chains

Chains are LCEL Runnables which subscribe to one or more channels, and write to one or more channels. Any valid LCEL expression can be used as a chain. Chains can be combined into a Pregel application, which coordinates the execution of the chains across multiple steps.

### Pregel

Pregel combines multiple chains (or actors) into a single application. It coordinates the execution of the chains across multiple steps, following the Pregel/Bulk Synchronous Parallel model. Each step consists of three phases:

- **Plan**: Determine which chains to execute in this step, ie. the chains that subscribe to channels updated in the previous step (or, in the first step, chains that subscribe to input channels)
- **Execution**: Execute those chains in parallel, until all complete, or one fails, or a timeout is reached. Any channel updates are invisible to other chains until the next step.
- **Update**: Update the channels with the values written by the chains in this step.

Repeat until no chains are planned for execution, or a maximum number of steps is reached.

## Example

```python
from langgraph.pregel import Channel, Pregel

grow_value = (
    Channel.subscribe_to("value")
    | (lambda x: x + x)
    | Channel.write_to(value=lambda x: x if len(x) < 10 else None)
)

app = Pregel(
    chains={"grow_value": grow_value},
    input="value",
    output="value",
)

assert app.invoke("a") == "aaaaaaaa"
```

# Nebius Customer Support Data Analyst Agent

Assignment 3 project for analyzing the Bitext customer support dataset with a LangGraph ReAct agent.

## Setup

Use Python 3.11. Python 3.14 is currently too new for this dependency stack.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `NEBIUS_API_KEY` to your Nebius Token Factory API key.
You can optionally set `CHECKPOINT_DB_PATH` to choose where persistent LangGraph
session checkpoints are stored. By default the app writes `.langgraph_checkpoints.sqlite`
in the project directory.
You can also set `USER_PROFILE_DIR` to choose where per-user semantic profiles are stored.
By default the app writes `.user_profiles/` in the project directory.
For MCP mode, the client connects to `MCP_SERVER_URL`, which defaults to
`http://127.0.0.1:8000/mcp`.

The dataset is loaded at runtime from Hugging Face:

```text
bitext/Bitext-customer-support-llm-chatbot-training-dataset
```

## Run The CLI

```bash
python main.py
use "exit" or "quit" the exit the chat
```

Ask one question and exit:

```bash
python main.py --once "What categories exist in the dataset?"
```

Use the standalone MCP server instead of local tools:

```bash
python mcp_server.py
python main.py --use-mcp
python main.py --once "How many refund requests did we get?" --use-mcp
```

Resume the same conversation across restarts:

```bash
python main.py --session my_session
python main.py --once "Show me 3 examples from the REFUND category" --session my_session
python main.py --once "Show me 3 more" --session my_session
```

Persist a separate user profile across sessions:

```bash
python main.py --session work_1 --user alice
python main.py --once "My name is Alice and I prefer concise answers." --user alice
python main.py --once "What do you remember about me?" --user alice
python main.py --session work_2 --user alice
```

Start fresh with an empty checkpoint database:

```bash
python main.py --reset-db
```

This deletes both persistent memory stores:

- the SQLite checkpoint file configured by `CHECKPOINT_DB_PATH`
- the user profile directory configured by `USER_PROFILE_DIR`

That gives you a genuinely fresh start for both Task 2a and Task 2b memory.

The CLI prints tool calls and observations before the final answer.

## Architecture

Task 1 is implemented as a LangGraph ReAct graph:

- `router`: classifies each user query as `structured`, `unstructured`, or `out_of_scope`.
- `agent`: Nebius-hosted chat model bound to dataset analysis tools.
- `tools`: LangGraph `ToolNode` that executes Pydantic-typed tools.
- Max iterations are controlled by `MAX_ITERATIONS`, defaulting to `12`.
- Task 2a adds a SQLite-backed LangGraph `SqliteSaver` checkpointer, keyed by `--session`,
  so conversation history persists across turns and restarts.
- Task 2b adds a separate per-user semantic profile store, keyed by `--user`, for distilled
  facts, preferences, and frequent topics across sessions.

For Task 3, the tool architecture is intentionally shared between local and remote modes:

- [customer_support_agent/tools.py](/Users/oferg/work/nebius/week5a/0501/nebius_agent/customer_support_agent/tools.py) contains the core dataset-analysis implementations such as `count_rows_impl` and `show_examples_impl`.
- The normal local CLI path binds LangChain tools that call those same implementations directly in-process.
- [mcp_server.py](/Users/oferg/work/nebius/week5a/0501/nebius_agent/mcp_server.py) exposes those same implementations as FastMCP tools, so the server and local client stay behaviorally aligned.
- [customer_support_agent/mcp_tools.py](/Users/oferg/work/nebius/week5a/0501/nebius_agent/customer_support_agent/mcp_tools.py) provides LangChain-compatible proxy tools that call the remote MCP server over HTTP.
- [customer_support_agent/agent.py](/Users/oferg/work/nebius/week5a/0501/nebius_agent/customer_support_agent/agent.py) chooses which tool set to bind: local tools by default, or MCP-backed proxy tools when the client is started with `--use-mcp`.

This means the agent graph, routing logic, memory, and answer formatting stay the same in both modes. The main difference is only where the tool execution happens: locally inside the client process, or remotely through the MCP server.

## Memory Design Note

The assignment does not define an exact relationship between session memory and user memory,
so this project makes the following architectural choice:

- Episodic conversation memory is keyed by `--session`.
- Long-term semantic profile memory is keyed by `--user`.
- If multiple users reuse the same `--session`, they intentionally share the same session
  history, while still keeping separate long-term profiles.

This keeps Task 2a and Task 2b conceptually separate: session checkpoints store the
conversation thread, while user profiles store distilled facts about a person.

Out-of-scope requests are declined before the model can answer from general knowledge.

## Model Choice

The project uses `meta-llama/Llama-3.3-70B-Instruct` through the Nebius Token Factory OpenAI-compatible API by default. It is a strong general instruction-following model for tool calling, routing context, and concise analytical answers. You can change it with `NEBIUS_MODEL` in `.env`, but all LLM calls should remain on Nebius Token Factory inference models.

## Project Decisions

I chose `meta-llama/Llama-3.3-70B-Instruct` because the agent needs more than simple chat completion. Task 1 requires reliable tool selection, structured dataset analysis, qualitative summarization, multi-step reasoning, and graceful refusal for out-of-scope questions. Since tool-use quality is central to the grading rubric, I prefer a stronger instruction-following model for the initial implementation.

Most factual computation still happens in Python tools, so the model is mainly responsible for routing intent, choosing the right tools, chaining calls when needed, and writing the final answer from tool observations. The model name is configurable through `NEBIUS_MODEL`, so a smaller Nebius Token Factory model can be tested later if speed or cost becomes more important.

I implemented the agent as an explicit LangGraph state graph instead of using `create_react_agent`. This gives me tighter control over the flow required by the assignment: a dedicated router node before tool use, immediate handling of out-of-scope questions, custom stopping logic for repeated empty tool calls, and clearer control over answer style for examples, summaries, and comparisons. `create_react_agent` would reduce boilerplate, but the explicit graph makes the grading-relevant behavior easier to demonstrate and debug.

## Tools

- `list_categories`: returns all dataset categories.
- `list_intents`: returns intent labels, optionally filtered.
- `count_rows`: counts rows with optional category, intent, and text filters.
- `show_examples`: returns customer instruction and support response examples.
- `distribution`: computes grouped counts by category, intent, or flags.
- `sample_responses_for_summary`: gathers representative rows for qualitative summaries.

## MCP Server

Task 3 is implemented in [mcp_server.py](/Users/oferg/work/nebius/week5a/0501/nebius_agent/mcp_server.py) using FastMCP.
It exposes the dataset-analysis tools as MCP tools, including:

- `list_categories`
- `list_intents`
- `count_rows`
- `show_examples`
- `distribution`
- `sample_responses_for_summary`

Start the standalone server:

```bash
python mcp_server.py
```

By default it serves Streamable HTTP on:

```text
http://127.0.0.1:8000/mcp
```

You can shut the server down gracefully from another terminal with:

```bash
curl -X POST http://127.0.0.1:8000/shutdown
```

That route is local-only and schedules a graceful `SIGTERM`, so it is meant for
local development use on your own machine.

Then run the client against that server:

```bash
python main.py --use-mcp
```

There is no MCP fallback mode. Without `--use-mcp`, the client uses local in-process tools.

You can also call one of the MCP tools directly with a FastMCP client:

```bash
python mcp_server.py
```

And from another terminal:

```python
import asyncio
from fastmcp import Client

async def main():
    async with Client("http://127.0.0.1:8000/mcp") as client:
        result = await client.call_tool("list_categories")
        print(result.data)

asyncio.run(main())
```

When the chat client runs with `--use-mcp`, its tool traces are labeled as MCP calls, and
the standalone server also logs each MCP invocation with `[mcp tool call]` and
`[mcp observation]` prefixes. Without `--use-mcp`, the normal CLI still uses local tool
execution and prints the original `[tool call]` / `[observation]` prefixes.

If you want to change the MCP endpoint, set these environment variables:

```text
MCP_SERVER_URL
MCP_SERVER_HOST
MCP_SERVER_PORT
MCP_SERVER_PATH
```

## Task 1 Test Prompts

```text
What categories exist in the dataset?
How many refund requests did we get?
Show me 5 examples of the SHIPPING category.
Summarize how agents respond to complaint intents.
Show me examples of people wanting their money back.
What is the distribution of intents in the ACCOUNT category?
What's the best CRM software for handling complaints?
Who is the president of France?
```

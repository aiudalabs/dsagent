# Aiuda Planner Agent

A Python package for building planning-based AI agents for data analysis tasks.

## Features

- **Dynamic Planning**: Agent creates and follows plans with [x]/[ ] step tracking
- **Persistent Execution**: Code runs in a Jupyter kernel with variable persistence
- **Multi-Provider LLM**: Supports OpenAI, Anthropic, Google, Ollama via LiteLLM
- **Notebook Generation**: Automatically generates clean, runnable Jupyter notebooks
- **Event Streaming**: Real-time events for UI integration
- **Session Management**: State persistence for multi-user scenarios

## Installation

Using uv (recommended):
```bash
uv pip install aiuda-planner-agent
```

Or with FastAPI support:
```bash
uv pip install "aiuda-planner-agent[api]"
```

For development:
```bash
git clone https://github.com/aiudalabs/aiuda-planner-agent
cd aiuda-planner-agent
uv sync --all-extras
```

Using pip (alternative):
```bash
pip install aiuda-planner-agent
pip install "aiuda-planner-agent[api]"  # with FastAPI
```

## Quick Start

### Basic Usage

```python
from aiuda_planner import PlannerAgent

# Create agent
with PlannerAgent(model="gpt-4o", workspace="./workspace") as agent:
    result = agent.run("Analyze sales_data.csv and identify top performing products")

    print(result.answer)
    print(f"Notebook: {result.notebook_path}")
```

### With Streaming

```python
from aiuda_planner import PlannerAgent, EventType

agent = PlannerAgent(model="claude-3-sonnet-20240229")
agent.start()

for event in agent.run_stream("Build a predictive model for customer churn"):
    if event.type == EventType.PLAN_UPDATED:
        print(f"Plan: {event.plan.raw_text if event.plan else ''}")
    elif event.type == EventType.CODE_SUCCESS:
        print("Code executed successfully")
    elif event.type == EventType.CODE_FAILED:
        print("Code execution failed")
    elif event.type == EventType.ANSWER_ACCEPTED:
        print(f"Answer: {event.message}")

# Get result with notebook after streaming
result = agent.get_result()
print(f"Notebook: {result.notebook_path}")

agent.shutdown()
```

### FastAPI Integration

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from uuid import uuid4
from aiuda_planner import PlannerAgent, EventType

app = FastAPI()

@app.post("/analyze")
async def analyze(task: str):
    async def event_stream():
        agent = PlannerAgent(
            model="gpt-4o",
            session_id=str(uuid4()),
        )
        agent.start()

        try:
            for event in agent.run_stream(task):
                yield f"data: {event.to_sse()}\n\n"
        finally:
            agent.shutdown()

    return StreamingResponse(event_stream(), media_type="text/event-stream")
```

## Command Line Interface

The package includes a CLI for quick analysis from the terminal:

```bash
aiuda-planner "Analyze this dataset and create visualizations" --data ./my_data.csv
```

### CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--data` | `-d` | Path to data file or directory (required) |
| `--model` | `-m` | LLM model to use (default: gpt-4o) |
| `--workspace` | `-w` | Output directory (default: ./workspace) |
| `--max-rounds` | `-r` | Max iterations (default: 30) |
| `--quiet` | `-q` | Suppress verbose output |
| `--no-stream` | | Disable streaming output |

### CLI Examples

```bash
# Basic analysis
aiuda-planner "Find trends and patterns" -d ./sales.csv

# With specific model
aiuda-planner "Build ML model" -d ./dataset -m claude-3-sonnet-20240229

# Custom output directory
aiuda-planner "Create charts" -d ./data -w ./output

# Quiet mode
aiuda-planner "Analyze" -d ./data -q

# Using uv
uv run aiuda-planner "Analyze this dataset" -d ./data
```

### Output

The CLI generates:
- **Notebook**: `workspace/generated/analysis_YYYYMMDD_HHMMSS.ipynb`
- **Images**: `workspace/images/` (auto-saved charts)
- **Data**: `workspace/data/` (copy of input data)

## Configuration

```python
agent = PlannerAgent(
    model="gpt-4o",           # Any LiteLLM-supported model
    workspace="./workspace",  # Working directory
    session_id="user-123",    # For multi-user scenarios
    max_rounds=30,            # Max agent iterations
    max_tokens=4096,          # Max tokens per response
    temperature=0.2,          # LLM temperature
    timeout=300,              # Code execution timeout (seconds)
    verbose=True,             # Print to console
    event_callback=None,      # Callback for events
)
```

## Supported Models

Any model supported by [LiteLLM](https://docs.litellm.ai/docs/providers):

- OpenAI: `gpt-4o`, `gpt-4-turbo`, `gpt-3.5-turbo`
- Anthropic: `claude-3-opus-20240229`, `claude-3-sonnet-20240229`
- Google: `gemini-pro`, `gemini-1.5-pro`
- Ollama: `ollama/llama3`, `ollama/codellama`
- And many more...

## Event Types

```python
from aiuda_planner import EventType

EventType.AGENT_STARTED       # Agent started processing
EventType.AGENT_FINISHED      # Agent finished
EventType.AGENT_ERROR         # Error occurred
EventType.ROUND_STARTED       # New iteration round
EventType.ROUND_FINISHED      # Round completed
EventType.LLM_CALL_STARTED    # LLM call started
EventType.LLM_CALL_FINISHED   # LLM response received
EventType.PLAN_CREATED        # Plan was created
EventType.PLAN_UPDATED        # Plan was updated
EventType.CODE_EXECUTING      # Code execution started
EventType.CODE_SUCCESS        # Code execution succeeded
EventType.CODE_FAILED         # Code execution failed
EventType.ANSWER_ACCEPTED     # Final answer generated
EventType.ANSWER_REJECTED     # Answer rejected (plan incomplete)
```

## State Persistence

```python
# Save state
state_json = agent.serialize_state()
save_to_database(session_id, state_json)

# Restore state
state_json = load_from_database(session_id)
agent.restore_state(state_json)
```

## Architecture

```
aiuda_planner/
├── agents/
│   └── base.py          # PlannerAgent - main user interface
├── core/
│   ├── engine.py        # AgentEngine - main loop
│   ├── executor.py      # JupyterExecutor - code execution
│   └── planner.py       # PlanParser - response parsing
├── schema/
│   └── models.py        # Pydantic models
└── utils/
    ├── logger.py        # AgentLogger - logging
    └── notebook.py      # NotebookBuilder - notebook generation
```

## License

MIT

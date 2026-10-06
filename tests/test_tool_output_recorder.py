from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic_ai.tools import AgentDepsT
from pydantic_ai import Agent, FunctionToolset
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from agent_forge.tool_output_recorder import ToolOutputRecorder

pytestmark = pytest.mark.anyio





class MathToolset(FunctionToolset[AgentDepsT]):
    """A toy toolset of arithmetic tools."""

    def __init__(self) -> None:
        super().__init__()
        self.add_function(self.add, name='add')
        self.add_function(self.multiply, name='multiply')
        self.add_function(self.divmod_, name='divmod_')


    @staticmethod
    def add(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b


    @staticmethod
    def multiply(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b


    @staticmethod
    def divmod_(a: int, b: int) -> dict[str, int]:
        """Divide two integers, returning the quotient and remainder."""
        quotient, remainder = divmod(a, b)
        return {'quotient': quotient, 'remainder': remainder}


TOOL_CALLS = [
    ToolCallPart('add', {'a': 2, 'b': 3}, tool_call_id='call_add'),
    ToolCallPart('multiply', {'a': 4, 'b': 5}, tool_call_id='call_multiply'),
    ToolCallPart('divmod_', {'a': 17, 'b': 5}, tool_call_id='call_divmod'),
]


def call_math_tools(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Call every math tool on the first request, then answer with the tool returns."""
    returns = [
        part for message in messages for part in getattr(message, 'parts', ()) if isinstance(part, ToolReturnPart)
    ]
    if not returns:
        return ModelResponse(parts=list(TOOL_CALLS))
    return ModelResponse(parts=[TextPart(' '.join(str(part.content) for part in returns))])


def make_agent(workspace_dir: Path, recorder: ToolOutputRecorder[None]) -> Agent[None, str]:
    return Agent(
        FunctionModel(call_math_tools),
        toolsets=[MathToolset()],
        capabilities=[LocalWorkspace(workspace_dir), recorder],
    )


def saved_records(workspace_dir: Path, output_dir: str = 'tool-outputs') -> dict[str, dict[str, object]]:
    """Map each saved file's tool name to its parsed JSON record."""
    files = sorted((workspace_dir / output_dir).glob('*/*.json'))
    records = [json.loads(file.read_text()) for file in files]
    return {record['tool_name']: record for record in records}


async def test_saves_every_tool_output_without_whitelist(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, ToolOutputRecorder())

    result = await agent.run('Do some math')

    assert result.output == "5.0 20.0 {'quotient': 3, 'remainder': 2}"
    records = saved_records(tmp_path)
    assert records == {
        'add': {'tool_name': 'add', 'tool_call_id': 'call_add', 'args': {'a': 2, 'b': 3}, 'output': 5.0},
        'multiply': {
            'tool_name': 'multiply',
            'tool_call_id': 'call_multiply',
            'args': {'a': 4, 'b': 5},
            'output': 20.0,
        },
        'divmod_': {
            'tool_name': 'divmod_',
            'tool_call_id': 'call_divmod',
            'args': {'a': 17, 'b': 5},
            'output': {'quotient': 3, 'remainder': 2},
        },
    }


async def test_saves_only_whitelisted_tool_outputs(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, ToolOutputRecorder(tools_whitelist=['add', 'divmod_']))

    result = await agent.run('Do some math')

    # Non-whitelisted tools still run and reach the model; only their output isn't saved.
    assert result.output == "5.0 20.0 {'quotient': 3, 'remainder': 2}"
    assert set(saved_records(tmp_path)) == {'add', 'divmod_'}


async def test_empty_whitelist_saves_nothing(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, ToolOutputRecorder(tools_whitelist=[]))

    await agent.run('Do some math')

    assert not (tmp_path / 'tool-outputs').exists()


async def test_files_are_grouped_by_run(tmp_path: Path) -> None:
    agent = make_agent(tmp_path, ToolOutputRecorder(tools_whitelist=['add'], output_dir='records'))

    first = await agent.run('Do some math')
    second = await agent.run('Do some math again')

    run_dirs = sorted(path.name for path in (tmp_path / 'records').iterdir())
    assert run_dirs == sorted([first.run_id, second.run_id])
    for run_dir in run_dirs:
        assert [file.name for file in (tmp_path / 'records' / run_dir).iterdir()] == ['call_add.json']


async def test_requires_a_workspace() -> None:
    agent = Agent(FunctionModel(call_math_tools), toolsets=[MathToolset()], capabilities=[ToolOutputRecorder()])

    with pytest.raises(UserError, match='none is attached'):
        await agent.run('Do some math')

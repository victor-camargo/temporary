"""`ToolOutputRecorder` -- save tool outputs to the run's workspace as JSON files."""

from __future__ import annotations

import re
from dataclasses import KW_ONLY, dataclass
from pathlib import PurePosixPath
from typing import Any

from pydantic_core import to_json

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import UserError
from pydantic_ai.messages import ToolCallPart, ToolReturn
from pydantic_ai.tools import AgentDepsT, RunContext, ToolDefinition

_UNSAFE_PATH_CHARS = re.compile(r'[^A-Za-z0-9_.-]')


@dataclass
class ToolOutputRecorder(AbstractCapability[AgentDepsT]):
    """Write the output of each tool call to the run's workspace as a JSON file.

    Each recorded call becomes `{output_dir}/{run_id}/{tool_call_id}.json` in the workspace,
    holding the tool name, call id, validated arguments and output. The tool result itself is
    returned unchanged, so the model sees the same output with or without this capability.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.capabilities import LocalWorkspace

        agent = Agent(
            'anthropic:claude-opus-5-5',
            capabilities=[LocalWorkspace('.'), ToolOutputRecorder(tools_whitelist=['add'])],
        )
        ```
    """

    tools_whitelist: list[str] | None = None
    """Names of the tools whose outputs are saved. `None` saves the output of every tool."""

    output_dir: str = 'tool-outputs'
    """Directory, relative to the workspace's working directory, that the JSON files go in."""

    _: KW_ONLY
    id: str | None = 'tool_output_recorder'

    async def before_run(self, ctx: RunContext[AgentDepsT]) -> None:
        """Fail the run at its start when there is no workspace to write to."""
        if not ctx.workspace.attached:
            raise UserError(
                "`ToolOutputRecorder` saves tool outputs to the run's workspace, but none is attached "
                "to this run. Attach one, such as `LocalWorkspace('.')`."
            )

    async def after_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        """Save the tool's output when the tool is whitelisted, then return the result unchanged."""
        if self.tools_whitelist is not None and call.tool_name not in self.tools_whitelist:
            return result

        output = result.return_value if isinstance(result, ToolReturn) else result
        record = {
            'tool_name': call.tool_name,
            'tool_call_id': call.tool_call_id,
            'args': args,
            'output': output,
        }
        # `fallback=repr` keeps a value with no JSON form from aborting the run and losing the output.
        data = to_json(record, indent=2, fallback=repr)
        await ctx.workspace.write_bytes(self._path(ctx, call), data)
        return result

    def _path(self, ctx: RunContext[AgentDepsT], call: ToolCallPart) -> str:
        """Build a per-run, per-call path so concurrent calls and later runs never overwrite each other."""
        run_id = _UNSAFE_PATH_CHARS.sub('_', ctx.run_id or 'run')
        call_id = _UNSAFE_PATH_CHARS.sub('_', call.tool_call_id or 'call')
        # Workspace paths are POSIX whatever the host OS, since the workspace may be a remote sandbox.
        return str(PurePosixPath(self.output_dir, run_id, f'{call_id}.json'))

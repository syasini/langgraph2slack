"""Built-in output block transformers for langgraph2slack.

These async functions can be called from your own ``@bot.transform_output``
transformer, or registered directly with ``bot.transform_output(render_blocks)``.

Each transformer receives the LangGraph response text and returns either:
- The original string unchanged (if the pattern was not found)
- A list of Slack block dicts (if the pattern was found and rendered)

Individual transformers (single pattern each):
    render_tables      — markdown tables → Slack table blocks
    render_todo_lists  — markdown todo lists (- [ ]/- [x]) → section blocks with ☐/☑
    render_code_blocks — triple-backtick fences → Slack rich_text preformatted blocks

Composite transformer (all patterns, single pass):
    render_blocks      — scans for all patterns in document order

Usage in slack_server.py::

    from langgraph2slack.block_transformers import render_tables, render_todo_lists, render_code_blocks

    @bot.transform_output
    async def render_blocks(message: str) -> str | list[dict]:
        result = await render_tables(message)
        if isinstance(result, str):
            result = await render_todo_lists(result)
        if isinstance(result, str):
            result = await render_code_blocks(result)
        return result

Or, using the composite directly::

    from langgraph2slack.block_transformers import render_blocks

    bot.transform_output(render_blocks)

Note:
    When a transformer returns blocks, subsequent transformers in the chain are
    skipped (TransformerChain short-circuit). Transformers registered BEFORE
    a block-returning transformer always run on the original string.
"""

from ._code import render_code_blocks
from ._composite import render_blocks
from ._tables import render_tables
from ._todos import render_todo_lists

__all__ = [
    "render_blocks",
    "render_tables",
    "render_todo_lists",
    "render_code_blocks",
]

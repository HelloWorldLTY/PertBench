import os
import re
import json
from typing import Any, Dict, List

from openai import AsyncOpenAI

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState
from a2a.utils import get_message_text, new_agent_text_message


_FINAL_ANSWER_RE = re.compile(r"Final\s*Answer\s*:\s*(Yes|No)\b", flags=re.IGNORECASE)


def _extract_output_text(resp: Any) -> str:
    """
    Robustly extract assistant text from Responses API objects.
    Prefer resp.output[*].content[*].text where content.type == "output_text".
    """
    # Some versions may have output_text, but it's not always reliable.
    if hasattr(resp, "output_text") and isinstance(getattr(resp, "output_text"), str):
        txt = getattr(resp, "output_text")
        if txt:
            return txt

    data: Dict[str, Any] = {}
    if hasattr(resp, "model_dump"):
        data = resp.model_dump()
    elif hasattr(resp, "to_dict"):
        data = resp.to_dict()

    out = data.get("output", [])
    texts: List[str] = []
    if isinstance(out, list):
        for item in out:
            if not isinstance(item, dict):
                continue
            # API reference shows: output -> [{type:"message", role:"assistant", content:[{type:"output_text", text:"..."}]}]
            content = item.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        t = c.get("text")
                        if isinstance(t, str) and t:
                            texts.append(t)

    return "\n".join(texts).strip()


def _normalize_final_answer(raw: str) -> str:
    """
    Return strictly one of:
      - "Final Answer: Yes"
      - "Final Answer: No"
    If parsing fails, fallback to No.
    """
    matches = list(_FINAL_ANSWER_RE.finditer(raw or ""))
    if not matches:
        return "Final Answer: No"
    label = matches[-1].group(1).strip().lower()
    return "Final Answer: Yes" if label == "yes" else "Final Answer: No"

def _extract_usage(resp: Any, model_name: str) -> Dict[str, Any]:
    """Extract token usage from Responses API response."""
    usage_obj = getattr(resp, "usage", None)
    if usage_obj is None:
        return {"model": model_name, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # usage can be a pydantic object; try model_dump first
    if hasattr(usage_obj, "model_dump"):
        usage = usage_obj.model_dump()
    elif isinstance(usage_obj, dict):
        usage = usage_obj
    else:
        usage = {}

    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)

    return {
        "model": model_name,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

class Agent:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        question = (get_message_text(message) or "").strip()
        if not question:
            await updater.update_status(
                TaskState.completed,
                new_agent_text_message("Final Answer: No\nUSAGE_JSON: {\"model\":\"unknown\",\"input_tokens\":0,\"output_tokens\":0,\"total_tokens\":0}"),
            )
            return

        await updater.update_status(
            TaskState.working,
            new_agent_text_message("Processing question..."),
        )

        model_name = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        
        resp = await self.client.responses.create(
            model=model_name,
            input=question,
            temperature=0,
            max_output_tokens=32,
        )

        raw = _extract_output_text(resp)
        final = _normalize_final_answer(raw)

        usage = _extract_usage(resp, model_name)
        usage_line = "USAGE_JSON: " + json.dumps(usage, ensure_ascii=False, separators=(",", ":"))

        await updater.update_status(
            TaskState.completed,
            new_agent_text_message(final + "\n" + usage_line),
        )
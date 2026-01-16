# To run this test, use: 
# cd path/to/GreenAgent
# uv run python -m pytest -s -q --agent-url http://127.0.0.1:9009 --purple-url http://127.0.0.1:9010 --max-units 5 tests/test_eval_smoke.py

import json
from typing import Any, Dict, List, Tuple

import pytest
import httpx
from uuid import uuid4

from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.types import Message, Part, Role, TextPart


async def send_text_message(text: str, url: str, streaming: bool = False):
    async with httpx.AsyncClient(timeout=120) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        client = ClientFactory(config).create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=text))],
            message_id=uuid4().hex,
        )

        events = [event async for event in client.send_message(msg)]

    return events


def _collect_artifacts(events: List[Any]) -> Dict[str, str]:
    """
    Return {artifact_name: concatenated_text_parts}.
    Supports:
      - direct ArtifactUpdate-like objects
      - (task, update) tuples where update is ArtifactUpdate
    """
    artifacts: Dict[str, List[str]] = {}

    def _add(name: str, text: str) -> None:
        if not name:
            return
        artifacts.setdefault(name, [])
        if text:
            artifacts[name].append(text)

    def _extract_text_from_parts(parts: Any) -> str:
        texts: List[str] = []
        for part in parts or []:
            root = getattr(part, "root", None)
            if isinstance(root, TextPart):
                texts.append(root.text)
        return "\n".join(texts).strip()

    def _handle_update(update: Any) -> None:
        if not update:
            return
        # ArtifactUpdate shape: update.artifact.name + update.artifact.parts
        artifact = getattr(update, "artifact", None)
        if artifact:
            name = getattr(artifact, "name", "") or ""
            parts = getattr(artifact, "parts", None)
            _add(name, _extract_text_from_parts(parts))

    for ev in events:
        # (task, update) tuples
        if isinstance(ev, tuple) and len(ev) == 2:
            _task, update = ev
            _handle_update(update)
            continue

        # direct update objects
        _handle_update(ev)

    # join lists to strings
    return {k: "\n".join(v).strip() for k, v in artifacts.items()}


@pytest.mark.asyncio
async def test_green_eval_smoke(agent, purple, max_units):
    """
    Smoke test:
      - send config to Green
      - Green should call Purple and emit artifacts: summary.json + unit_results.jsonl
    """

    payload = {
        "config": {
            "purple_url": purple,
            "max_units": max_units,
            "unit_selection": "head",
        }
    }
    events = await send_text_message(json.dumps(payload), agent, streaming=True)

    artifacts = _collect_artifacts(events)

    assert "summary.json" in artifacts, f"Missing summary.json artifact. Got: {list(artifacts.keys())}"
    assert "unit_results.jsonl" in artifacts, f"Missing unit_results.jsonl artifact. Got: {list(artifacts.keys())}"

    summary_text = artifacts["summary.json"]
    assert summary_text, "summary.json artifact is empty"
    summary = json.loads(summary_text)

    # Structural assertions (avoid fragile numeric thresholds)
    assert isinstance(summary.get("task_name"), str) and summary["task_name"], "task_name must be a non-empty string"
    assert summary.get("input_mode") in {"structured", "qa_pairs"}, "input_mode must be structured or qa_pairs"
    assert summary.get("purple_url") == "http://127.0.0.1:9010", "purple_url should match the injected config"

    for k in ["units_selected", "covered_units"]:
        assert isinstance(summary.get(k), int), f"{k} must be int"

    for k in ["accuracy", "invalid_rate", "coverage_rate"]:
        assert isinstance(summary.get(k), (int, float)), f"{k} must be numeric"
    
    print("SUMMARY:\n", json.dumps(summary, indent=2))
    print("UNIT_RESULTS (head):\n", "\n".join(artifacts["unit_results.jsonl"].splitlines()[:3]))

    # Export summary
    from pathlib import Path

    out_dir = Path("tmp_eval_outputs")
    out_dir.mkdir(exist_ok=True)

    (out_dir / "summary.json").write_text(summary_text, encoding="utf-8")
    (out_dir / "unit_results.jsonl").write_text(artifacts["unit_results.jsonl"], encoding="utf-8")

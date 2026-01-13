import asyncio
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger


YES = "Yes"
NO = "No"
INVALID = "invalid"
AMBIGUOUS = "Ambiguous"


def load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader)


def validate_structured_spec(spec: Dict[str, Any]) -> None:
    required = ["task_name", "input_mode", "keys", "num_templates", "question_templates", "tie"]
    missing = [k for k in required if k not in spec]
    if missing:
        raise ValueError(f"Spec missing required fields: {missing}")

    if spec["input_mode"] != "structured":
        raise ValueError(f"Expected input_mode='structured' for this evaluator step, got: {spec['input_mode']}")

    keys = spec["keys"]
    if not isinstance(keys, list) or not all(isinstance(x, str) for x in keys):
        raise ValueError("Spec field 'keys' must be a list of strings")

    qts = spec["question_templates"]
    if not isinstance(qts, list) or not all(isinstance(x, str) for x in qts):
        raise ValueError("Spec field 'question_templates' must be a list of template strings")

    if int(spec["num_templates"]) != len(qts):
        raise ValueError("Spec field 'num_templates' must equal len(question_templates)")

    tie = spec["tie"]
    if tie not in [YES, NO, AMBIGUOUS]:
        raise ValueError("Spec field 'tie' must be one of: 'Yes', 'No', 'Ambiguous'")


def render_questions(row: Dict[str, str], spec: Dict[str, Any]) -> List[str]:
    keys: List[str] = spec["keys"]
    templates: List[str] = spec["question_templates"]

    fill = {}
    for k in keys:
        if k not in row:
            raise KeyError(f"Ground truth row missing required key column '{k}'. Available: {list(row.keys())}")
        fill[k] = row[k]

    questions = []
    for t in templates:
        try:
            questions.append(t.format(**fill))
        except KeyError as e:
            raise KeyError(f"Template uses placeholder not in keys/row: {e}. Template: {t}")
    return questions


_FINAL_ANSWER_RE = re.compile(r"Final\s*Answer\s*:\s*(Yes|No)\b", flags=re.IGNORECASE)


def parse_final_answer(text: str) -> str:
    """
    Parse agent output into: 'Yes', 'No', or 'invalid'.
    Strategy: take the LAST match of 'Final Answer: Yes|No' if multiple appear.
    """
    matches = list(_FINAL_ANSWER_RE.finditer(text or ""))
    if not matches:
        return INVALID
    label = matches[-1].group(1).strip().lower()
    return YES if label == "yes" else NO


def aggregate_unit(
    preds_parsed: List[str],
    gold: str,
    *,
    min_valid: int,
    tie_policy: str,
) -> Dict[str, Any]:
    """
    valid-only vote among Yes/No.
    If valid_count < min_valid -> unit is not covered/answerable (excluded from accuracy denom).
    tie -> majority = tie_policy (often 'Ambiguous'); Ambiguous is always incorrect.
    """
    yes_count = sum(1 for p in preds_parsed if p == YES)
    no_count = sum(1 for p in preds_parsed if p == NO)
    invalid_count = sum(1 for p in preds_parsed if p == INVALID)
    valid_count = yes_count + no_count

    is_covered = valid_count >= int(min_valid)

    majority: Optional[str] = None
    consistency: Optional[float] = None
    majority_correct = False

    if valid_count > 0:
        consistency = max(yes_count, no_count) / valid_count

    if is_covered:
        if yes_count > no_count:
            majority = YES
        elif no_count > yes_count:
            majority = NO
        else:
            majority = tie_policy  # Yes / No / Ambiguous

        if majority in [YES, NO] and majority == gold:
            majority_correct = True
        else:
            majority_correct = False  # includes Ambiguous

    return {
        "preds_parsed": preds_parsed,
        "counts": {
            "yes": yes_count,
            "no": no_count,
            "invalid": invalid_count,
            "valid": valid_count,
        },
        "min_valid": int(min_valid),
        "is_covered": is_covered,
        "majority_label": majority,
        "consistency": consistency,
        "gold": gold,
        "majority_correct": majority_correct,
        "is_ambiguous": (majority == AMBIGUOUS),
    }


def parse_run_config(input_text: str) -> Dict[str, Any]:
    """
    Accept either plain text or JSON text.
    If JSON, expect {"config": {...}, "participants": {...}} or just {"config": {...}}.
    """
    try:
        obj = json.loads(input_text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {"config": {}}


class Agent:
    def __init__(self):
        self.messenger = Messenger()

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        req = parse_run_config(input_text)
        config = req.get("config", {}) if isinstance(req, dict) else {}

        # These will be provided by your assessment_request/config later.
        spec_path = config.get("spec_path")
        csv_path = config.get("groundtruth_csv")
        label_column = config.get("label_column", "level")  # you confirmed "level"
        min_valid = int(config.get("min_valid_answers_per_unit", 5))

        if not spec_path or not csv_path:
            raise ValueError(
                "Missing required config fields: spec_path and groundtruth_csv. "
                "Send a JSON message like: "
                '{"config": {"spec_path": "...", "groundtruth_csv": "...", "min_valid_answers_per_unit": 5}}'
            )

        self.messenger.reset()

        spec = load_json(spec_path)
        validate_structured_spec(spec)

        rows = load_csv_rows(csv_path)
        total_units = len(rows)

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Loaded spec '{spec.get('task_name')}', templates={spec['num_templates']}, "
                f"units={total_units}, min_valid={min_valid}, tie={spec['tie']}"
            ),
        )

        # Step 1: we only validate I/O and spec; Step 2 will actually query purple agents and score.
        summary = {
            "task_name": spec.get("task_name"),
            "input_mode": spec.get("input_mode"),
            "num_templates": spec.get("num_templates"),
            "keys": spec.get("keys"),
            "units_total": total_units,
            "label_column": label_column,
            "min_valid_answers_per_unit": min_valid,
            "tie": spec.get("tie"),
            "status": "ready_for_purple_integration",
        }

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=json.dumps(summary, indent=2)))],
            name="summary.json",
        )
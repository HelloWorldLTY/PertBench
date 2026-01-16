import asyncio
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.server.tasks import TaskUpdater
from a2a.types import Message, Part, Role, TextPart
from a2a.utils import get_message_text, new_agent_text_message
from a2a.types import TaskState


YES = "Yes"
NO = "No"
INVALID = "invalid"
AMBIGUOUS = "Ambiguous"

DEFAULT_SPEC_PATH = "example/config/perturbation_analysis_prompts_single.json"
DEFAULT_CSV_PATH = "example/data/adamson_psa_single.csv"
DEFAULT_PURPLE_URL = "http://127.0.0.1:9010"

DATASET_REGISTRY: dict[str, dict[str, str]] = {
    # single datasets
    "adamson_psa_single": {
        "csv_path": "example/data/adamson_psa_single.csv",
        "spec_path": "example/config/perturbation_analysis_prompts_single.json",
    },
    "norman_psa_single": {
        "csv_path": "example/data/norman_psa_single.csv",
        "spec_path": "example/config/perturbation_analysis_prompts_single.json",
    },
    "replogle_k562_psa_single": {
        "csv_path": "example/data/replogle_k562_psa_single.csv",
        "spec_path": "example/config/perturbation_analysis_prompts_single.json",
    },
    "replogle_rpe1_psa_single": {
        "csv_path": "example/data/replogle_rpe1_psa_single.csv",
        "spec_path": "example/config/perturbation_analysis_prompts_single.json",
    },
    # double dataset
    "norman_psa_double": {
        "csv_path": "example/data/norman_psa_double.csv",
        "spec_path": "example/config/perturbation_analysis_prompts_double.json",
    },
}


def project_root() -> Path:
    # src/agent.py -> project root is parent of src
    return Path(__file__).resolve().parents[1]


def resolve_path(p: str) -> Path:
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return project_root() / pp

def _available_datasets_str() -> str:
    return ", ".join(sorted(DATASET_REGISTRY.keys()))


def select_datasets_from_cfg(cfg: dict[str, Any]) -> list[dict[str, str]]:
    """
    Priority:
      1) explicit csv_path/spec_path => single dataset mode (legacy behavior)
      2) cfg["datasets"] => list of dataset ids
      3) cfg["dataset"]  => single id or "all"
      4) default => all
    Returns a list of dicts: [{"dataset": <id>, "csv_path": ..., "spec_path": ...}, ...]
    """
    # 1) Legacy explicit paths (highest priority)
    if cfg.get("csv_path") or cfg.get("spec_path"):
        csv_path = str(cfg.get("csv_path") or DEFAULT_CSV_PATH)
        spec_path = str(cfg.get("spec_path") or DEFAULT_SPEC_PATH)
        return [{"dataset": "custom", "csv_path": csv_path, "spec_path": spec_path}]

    # 2) Explicit list
    if isinstance(cfg.get("datasets"), list):
        ids = [str(x) for x in cfg["datasets"]]
        out: list[dict[str, str]] = []
        for ds in ids:
            if ds not in DATASET_REGISTRY:
                raise ValueError(
                    f"Unknown dataset '{ds}'. Available: {_available_datasets_str()}"
                )
            item = DATASET_REGISTRY[ds]
            out.append({"dataset": ds, "csv_path": item["csv_path"], "spec_path": item["spec_path"]})
        return out

    # 3) Single dataset or all
    ds = str(cfg.get("dataset") or "all")
    if ds == "all":
        out = []
        for k in sorted(DATASET_REGISTRY.keys()):
            item = DATASET_REGISTRY[k]
            out.append({"dataset": k, "csv_path": item["csv_path"], "spec_path": item["spec_path"]})
        return out

    if ds not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset '{ds}'. Available: {_available_datasets_str()}")
    item = DATASET_REGISTRY[ds]
    return [{"dataset": ds, "csv_path": item["csv_path"], "spec_path": item["spec_path"]}]


def load_json(path: str) -> Dict[str, Any]:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: str) -> List[Dict[str, str]]:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {p}")
        return list(reader)


def validate_common_spec(spec: Dict[str, Any]) -> None:
    required = ["task_name", "input_mode", "gold_label"]
    missing = [k for k in required if k not in spec]
    if missing:
        raise ValueError(f"Spec missing required fields: {missing}")


def validate_structured_spec(spec: Dict[str, Any]) -> None:
    validate_common_spec(spec)
    required = ["keys", "model_input", "tie", "min_valid_answers_per_unit"]
    missing = [k for k in required if k not in spec]
    if missing:
        raise ValueError(f"Structured spec missing required fields: {missing}")

    if spec["input_mode"] != "structured":
        raise ValueError(f"Expected input_mode='structured', got: {spec['input_mode']}")

    keys = spec["keys"]
    if not isinstance(keys, list) or not all(isinstance(x, str) for x in keys):
        raise ValueError("Spec field 'keys' must be a list of strings")

    templates = spec["model_input"]
    if not isinstance(templates, list) or not all(isinstance(x, str) for x in templates):
        raise ValueError("Spec field 'model_input' must be a list of template strings")

    tie = spec["tie"]
    if tie not in [YES, NO, AMBIGUOUS]:
        raise ValueError("Spec field 'tie' must be one of: 'Yes', 'No', 'Ambiguous'")


def validate_qapairs_spec(spec: Dict[str, Any]) -> None:
    validate_common_spec(spec)
    if spec["input_mode"] != "qa_pairs":
        raise ValueError(f"Expected input_mode='qa_pairs', got: {spec['input_mode']}")


def render_questions(row: Dict[str, str], spec: Dict[str, Any]) -> List[str]:
    keys: List[str] = spec["keys"]
    templates: List[str] = spec["model_input"]

    fill: Dict[str, str] = {}
    for k in keys:
        if k not in row:
            raise KeyError(f"Ground truth row missing required key column '{k}'. Available: {list(row.keys())}")
        fill[k] = row[k]

    out: List[str] = []
    for t in templates:
        out.append(t.format(**fill))
    return out


_FINAL_ANSWER_RE = re.compile(r"Final\s*Answer\s*:\s*(Yes|No)\b", flags=re.IGNORECASE)


def parse_final_answer(text: str) -> str:
    matches = list(_FINAL_ANSWER_RE.finditer(text or ""))
    if not matches:
        return INVALID
    label = matches[-1].group(1).strip().lower()
    return YES if label == "yes" else NO


_USAGE_JSON_RE = re.compile(r"USAGE_JSON\s*:\s*(\{.*\})", flags=re.IGNORECASE | re.DOTALL)


def parse_usage_json(text: str) -> Dict[str, Any]:
    """Parse USAGE_JSON payload from Purple output text."""
    matches = list(_USAGE_JSON_RE.finditer(text or ""))
    if not matches:
        return {}

    payload = matches[-1].group(1).strip()
    try:
        obj = json.loads(payload)
    except Exception:
        return {}

    if not isinstance(obj, dict):
        return {}

    # Normalize fields
    model = str(obj.get("model") or "unknown")
    in_tok = int(obj.get("input_tokens", 0) or 0)
    out_tok = int(obj.get("output_tokens", 0) or 0)
    tot_tok = int(obj.get("total_tokens", in_tok + out_tok) or 0)

    return {"model": model, "input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": tot_tok}

def get_pricing(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns pricing config for cost estimation.
    cfg may contain:
      cfg["pricing"] = {"currency":"USD","input_per_1m":0.30,"output_per_1m":1.20}
    Defaults are estimates for gpt-4o-mini (standard).
    """
    pricing = cfg.get("pricing")
    if isinstance(pricing, dict):
        currency = str(pricing.get("currency") or "USD")
        input_per_1m = float(pricing.get("input_per_1m", 0.30))
        output_per_1m = float(pricing.get("output_per_1m", 1.20))
        return {"currency": currency, "input_per_1m": input_per_1m, "output_per_1m": output_per_1m}

    return {"currency": "USD", "input_per_1m": 0.30, "output_per_1m": 1.20}


def estimate_cost_usd(input_tokens: int, output_tokens: int, pricing: Dict[str, Any]) -> float:
    """Estimated cost based on tokens and provided pricing (per 1M tokens)."""
    in_rate = float(pricing.get("input_per_1m", 0.0))
    out_rate = float(pricing.get("output_per_1m", 0.0))
    return (input_tokens / 1_000_000.0) * in_rate + (output_tokens / 1_000_000.0) * out_rate


def normalize_gold(g: str) -> str:
    gg = (g or "").strip().lower()
    if gg == "yes":
        return YES
    if gg == "no":
        return NO
    # If your dataset uses other labels, normalize here
    raise ValueError(f"Unsupported gold label value: {g}")


def aggregate_unit(
    preds_parsed: List[str],
    gold: str,
    *,
    min_valid: int,
    tie_policy: str,
) -> Dict[str, Any]:
    yes_count = sum(1 for p in preds_parsed if p == YES)
    no_count = sum(1 for p in preds_parsed if p == NO)
    invalid_count = sum(1 for p in preds_parsed if p == INVALID)
    valid_count = yes_count + no_count

    is_covered = valid_count >= int(min_valid)

    majority: Optional[str] = None
    majority_correct = False

    if is_covered:
        if yes_count > no_count:
            majority = YES
        elif no_count > yes_count:
            majority = NO
        else:
            majority = tie_policy

        majority_correct = (majority in [YES, NO] and majority == gold)

    return {
        "preds_parsed": preds_parsed,
        "counts": {"yes": yes_count, "no": no_count, "invalid": invalid_count, "valid": valid_count},
        "min_valid": int(min_valid),
        "is_covered": is_covered,
        "majority_label": majority,
        "gold": gold,
        "majority_correct": majority_correct,
        "is_ambiguous": (majority == AMBIGUOUS),
    }


def _collect_texts_from_events(events: List[Any]) -> str:
    texts: List[str] = []

    def _collect_from_message(msg: Any) -> None:
        if not msg:
            return
        for part in getattr(msg, "parts", []) or []:
            root = getattr(part, "root", None)
            if isinstance(root, TextPart):
                texts.append(root.text)

    for event in events:
        # Message event
        if isinstance(event, Message):
            _collect_from_message(event)
            continue

        # Tuple (task, update)
        if isinstance(event, tuple) and len(event) == 2:
            task, update = event

            if task and hasattr(task, "status") and getattr(task.status, "message", None):
                _collect_from_message(task.status.message)

            if update and hasattr(update, "status") and getattr(update.status, "message", None):
                _collect_from_message(update.status.message)
            elif update and getattr(update, "message", None):
                _collect_from_message(update.message)
            continue

        # Some SDK variants might yield Task-like objects
        if hasattr(event, "status") and getattr(event.status, "message", None):
            _collect_from_message(event.status.message)

    return "\n".join([t for t in texts if t]).strip()

async def fetch_agent_identity(endpoint: str) -> dict[str, str]:
    """
    Fetch purple agent card once and return a stable identity payload.
    """
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resolver = A2ACardResolver(httpx_client=client, base_url=endpoint)
            card = await resolver.get_agent_card()
        return {
            "name": getattr(card, "name", "") or "",
            "version": getattr(card, "version", "") or "",
            "url": getattr(card, "url", "") or endpoint,
        }
    except Exception:
        return {"name": "", "version": "", "url": endpoint}


async def call_purple(question: str, purple_url: str, *, streaming: bool = False) -> str:
    async with httpx.AsyncClient(timeout=60) as httpx_client:
        resolver = A2ACardResolver(httpx_client=httpx_client, base_url=purple_url)
        agent_card = await resolver.get_agent_card()
        config = ClientConfig(httpx_client=httpx_client, streaming=streaming)
        client = ClientFactory(config).create(agent_card)

        msg = Message(
            kind="message",
            role=Role.user,
            parts=[Part(TextPart(text=question))],
            message_id=uuid4().hex,
        )

        events = [e async for e in client.send_message(msg)]
        return _collect_texts_from_events(events)


DEFAULT_PARTICIPANT_ROLE = "qa_agent"

def parse_assessment_request(input_text: str) -> tuple[dict[str, str], dict[str, Any]]:
    """
    Supports:
      A) agentbeats-run: {"participants": {...}, "config": {...}}
      B) local pytest:   {"config": {...}}
      C) direct config:  {"purple_url": "...", ...}
    Returns: (participants, config)
    """
    try:
        obj = json.loads(input_text)
    except Exception:
        return {}, {}

    if not isinstance(obj, dict):
        return {}, {}

    participants = obj.get("participants") or {}
    if not isinstance(participants, dict):
        participants = {}

    # runner uses "config"; local tests sometimes wrap config the same way
    if "config" in obj and isinstance(obj["config"], dict):
        cfg = obj["config"]
    else:
        # treat the whole dict (minus participants) as config
        cfg = {k: v for k, v in obj.items() if k != "participants"}

    return participants, cfg


def pick_purple_url(participants: dict[str, str], cfg: dict[str, Any]) -> str:
    """
    Priority:
      1) participants[qa_agent]
      2) if only one participant exists -> that endpoint
      3) cfg["purple_url"]
      4) DEFAULT_PURPLE_URL
    """
    if DEFAULT_PARTICIPANT_ROLE in participants:
        return str(participants[DEFAULT_PARTICIPANT_ROLE])

    if len(participants) == 1:
        return str(next(iter(participants.values())))

    if "purple_url" in cfg and cfg["purple_url"]:
        return str(cfg["purple_url"])

    return DEFAULT_PURPLE_URL

def parse_run_config(input_text: str) -> Dict[str, Any]:
    # Accept {"config": {...}} or a plain dict as config.
    try:
        obj = json.loads(input_text)
        if isinstance(obj, dict):
            return obj.get("config", obj)
    except Exception:
        pass
    return {}


def select_rows(rows: List[Dict[str, str]], spec: Dict[str, Any]) -> List[Dict[str, str]]:
    max_units = spec.get("max_units")
    unit_selection = spec.get("unit_selection", "head")
    random_seed = spec.get("random_seed")
    start_index = int(spec.get("start_index", 0))

    if max_units is None:
        return rows

    n = int(max_units)
    if n <= 0:
        return []

    if unit_selection == "head":
        return rows[:n]

    if unit_selection == "slice":
        return rows[start_index : start_index + n]

    if unit_selection == "random":
        import random

        rng = random.Random(random_seed)
        if n >= len(rows):
            return rows
        idx = rng.sample(range(len(rows)), n)
        return [rows[i] for i in idx]

    raise ValueError(f"Unknown unit_selection: {unit_selection}")

class Agent:
    async def run(self, message: Message, updater: TaskUpdater) -> None:
        participants, cfg = parse_assessment_request(get_message_text(message))
        purple_url = pick_purple_url(participants, cfg)
        purple_identity = await fetch_agent_identity(purple_url)

        # Decide which datasets to run (default: all)
        dataset_jobs = select_datasets_from_cfg(cfg)

        # Global (run-time) overrides applied to each dataset run
        runtime_overrides: dict[str, Any] = {}
        if "gold_label" in cfg and cfg["gold_label"]:
            runtime_overrides["gold_label"] = cfg["gold_label"]
        for k in ["max_units", "unit_selection", "random_seed", "start_index"]:
            if k in cfg and cfg[k] is not None:
                runtime_overrides[k] = cfg[k]

        await updater.update_status(
            TaskState.working,
            new_agent_text_message(
                f"Starting evaluation: datasets={len(dataset_jobs)} | purple={purple_url}"
            ),
        )

        all_summaries: list[dict[str, Any]] = []
        overall_covered = 0
        overall_correct = 0
        overall_calls = 0
        overall_tokens_in = 0
        overall_tokens_out = 0
        overall_tokens_total = 0
        overall_estimated_cost_total = 0.0

        # pricing can be resolved once per run (config-level)
        pricing = get_pricing(cfg)

        # Run each dataset sequentially
        for job_idx, job in enumerate(dataset_jobs, start=1):
            dataset_id = job["dataset"]
            spec_path = job["spec_path"]
            csv_path = job["csv_path"]

            # Load spec
            spec = load_json(spec_path)

            # Inject run-time / routing info
            spec["purple_url"] = purple_url
            spec["csv_path"] = csv_path

            # Apply global overrides (same behavior as your previous single-dataset code)
            for k, v in runtime_overrides.items():
                spec[k] = v

            validate_common_spec(spec)

            rows = load_csv_rows(spec["csv_path"])
            selected = select_rows(rows, spec)

            await updater.update_status(
                TaskState.working,
                new_agent_text_message(
                    f"[{job_idx}/{len(dataset_jobs)}] Loaded dataset='{dataset_id}' "
                    f"task='{spec.get('task_name')}', mode={spec.get('input_mode')}, "
                    f"rows={len(rows)}, selected={len(selected)}"
                ),
            )

            input_mode = spec["input_mode"]
            gold_col = spec["gold_label"]

            unit_results: List[Dict[str, Any]] = []
            total_answers = 0
            total_invalid = 0

            covered = 0
            correct = 0
            ambiguous = 0

            calls = 0
            tokens_in = 0
            tokens_out = 0
            tokens_total = 0
            tokens_by_model: Dict[str, Dict[str, int]] = {}

            if input_mode == "structured":
                validate_structured_spec(spec)
                min_valid = int(spec["min_valid_answers_per_unit"])
                tie_policy = spec["tie"]

                for i, row in enumerate(selected):
                    questions = render_questions(row, spec)
                    preds_parsed: List[str] = []

                    for q in questions:
                        raw = await call_purple(q, purple_url, streaming=False)
                        calls += 1

                        # 1) parse label
                        parsed = parse_final_answer(raw)
                        preds_parsed.append(parsed)

                        # 2) parse usage (per call)
                        u = parse_usage_json(raw)
                        if u:
                            in_tok = int(u.get("input_tokens", 0) or 0)
                            out_tok = int(u.get("output_tokens", 0) or 0)
                            tot_tok = int(u.get("total_tokens", in_tok + out_tok) or 0)

                            tokens_in += in_tok
                            tokens_out += out_tok
                            tokens_total += tot_tok

                            m = str(u.get("model") or "unknown")
                            if m not in tokens_by_model:
                                tokens_by_model[m] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                            tokens_by_model[m]["input_tokens"] += in_tok
                            tokens_by_model[m]["output_tokens"] += out_tok
                            tokens_by_model[m]["total_tokens"] += tot_tok

                    total_answers += len(preds_parsed)
                    total_invalid += sum(1 for p in preds_parsed if p == INVALID)

                    gold = normalize_gold(row[gold_col])
                    agg = aggregate_unit(preds_parsed, gold, min_valid=min_valid, tie_policy=tie_policy)

                    if agg["is_covered"]:
                        covered += 1
                        if agg["majority_correct"]:
                            correct += 1
                        if agg["is_ambiguous"]:
                            ambiguous += 1

                    unit_results.append(
                        {
                            "dataset": dataset_id,
                            "unit_index": i,
                            "unit_keys": {k: row.get(k, "") for k in spec["keys"]},
                            "gold": gold,
                            "preds_parsed": preds_parsed,
                            "counts": agg["counts"],
                            "min_valid": agg["min_valid"],
                            "is_covered": agg["is_covered"],
                            "majority_label": agg["majority_label"],
                            "majority_correct": agg["majority_correct"],
                            "is_ambiguous": agg["is_ambiguous"],
                        }
                    )

            elif input_mode == "qa_pairs":
                validate_qapairs_spec(spec)
                for i, row in enumerate(selected):
                    if "question" not in row:
                        raise KeyError("qa_pairs mode requires CSV column 'question'")

                    raw = await call_purple(row["question"], purple_url, streaming=False)
                    calls += 1

                    parsed = parse_final_answer(raw)

                    u = parse_usage_json(raw)
                    if u:
                        in_tok = int(u.get("input_tokens", 0) or 0)
                        out_tok = int(u.get("output_tokens", 0) or 0)
                        tot_tok = int(u.get("total_tokens", in_tok + out_tok) or 0)

                        tokens_in += in_tok
                        tokens_out += out_tok
                        tokens_total += tot_tok

                        m = str(u.get("model") or "unknown")
                        if m not in tokens_by_model:
                            tokens_by_model[m] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
                        tokens_by_model[m]["input_tokens"] += in_tok
                        tokens_by_model[m]["output_tokens"] += out_tok
                        tokens_by_model[m]["total_tokens"] += tot_tok


                    total_answers += 1
                    total_invalid += 1 if parsed == INVALID else 0

                    gold = normalize_gold(row[gold_col])

                    is_valid = parsed in [YES, NO]
                    is_correct = (is_valid and parsed == gold)

                    if is_valid:
                        covered += 1
                        if is_correct:
                            correct += 1

                    unit_results.append(
                        {
                            "dataset": dataset_id,
                            "unit_index": i,
                            "gold": gold,
                            "pred_parsed": parsed,
                            "is_valid": is_valid,
                            "is_correct": is_correct,
                        }
                    )
            else:
                raise ValueError(f"Unsupported input_mode: {input_mode}")

            summary = {
                "dataset": dataset_id,
                "task_name": spec.get("task_name"),
                "input_mode": input_mode,
                "csv_path": spec.get("csv_path"),
                "spec_path": spec_path,
                "purple_url": purple_url,
                "units_total": len(rows),
                "units_selected": len(selected),
                "covered_units": covered,
                "coverage_rate": (covered / len(selected)) if selected else 0.0,
                "accuracy": (correct / covered) if covered else 0.0,
                "ambiguous_rate": (ambiguous / covered) if (covered and input_mode == "structured") else 0.0,
                "invalid_rate": (total_invalid / total_answers) if total_answers else 0.0,
            }

            estimated_cost_total = estimate_cost_usd(tokens_in, tokens_out, pricing)

            cost_by_model: Dict[str, float] = {}
            for m, t in tokens_by_model.items():
                cost_by_model[m] = estimate_cost_usd(t["input_tokens"], t["output_tokens"], pricing)

            summary["purple_usage"] = {
                "calls": calls,
                "tokens_input": tokens_in,
                "tokens_output": tokens_out,
                "tokens_total": tokens_total,
                "tokens_by_model": tokens_by_model,
                "pricing_estimate": pricing,
                "estimated_cost": {
                    "currency": pricing.get("currency", "USD"),
                    "total": estimated_cost_total,
                    "by_model": cost_by_model,
                },
            }

            summary["participant"] = {
                "role": DEFAULT_PARTICIPANT_ROLE,
                "endpoint": purple_url,
                "name": purple_identity.get("name", ""),
                "version": purple_identity.get("version", ""),
            }

            overall_calls += calls
            overall_tokens_in += tokens_in
            overall_tokens_out += tokens_out
            overall_tokens_total += tokens_total
            overall_estimated_cost_total += estimated_cost_total

            all_summaries.append(summary)
            overall_covered += covered
            overall_correct += correct

            # Artifacts per dataset (names include dataset id to avoid collisions)
            unit_jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in unit_results)

            emit_unit_results = bool(cfg.get("emit_unit_results", False))

            if emit_unit_results:
                unit_jsonl = "\n".join(json.dumps(r, ensure_ascii=False) for r in unit_results)
                await updater.add_artifact(
                    parts=[Part(TextPart(text=unit_jsonl))],
                    name=f"{dataset_id}.unit_results.jsonl",
                )

            await updater.add_artifact(
                parts=[Part(TextPart(text=json.dumps(summary, indent=2)))],
                name=f"{dataset_id}.summary.json",
            )

        aggregate = {
            "datasets": [s["dataset"] for s in all_summaries],
            "num_datasets": len(all_summaries),
            "micro_accuracy": (overall_correct / overall_covered) if overall_covered else 0.0,
            "micro_covered_units": overall_covered,
            "purple_usage": {
                "calls": overall_calls,
                "tokens_input": overall_tokens_in,
                "tokens_output": overall_tokens_out,
                "tokens_total": overall_tokens_total,
                "pricing_estimate": pricing,
                "estimated_cost": {
                    "currency": pricing.get("currency", "USD"),
                    "total": overall_estimated_cost_total,
                },
            },
        }

        await updater.add_artifact(
            parts=[Part(TextPart(text=json.dumps({"aggregate": aggregate, "per_dataset": all_summaries}, indent=2)))],
            name="aggregate.summary.json",
        )

        # --- leaderboard output ---
        leaderboard_payload = {
            "schema_version": "1.0",
            "task_name": "perturbation_significance_analysis",
            "participant": {
                "role": DEFAULT_PARTICIPANT_ROLE,
                "endpoint": purple_url,
                "name": purple_identity.get("name", ""),
                "version": purple_identity.get("version", ""),
                "card_url": purple_identity.get("url", purple_url),
            },
            "scores": {
                "micro_accuracy": aggregate["micro_accuracy"],
                # If you want a single scalar coverage across datasets, use micro_covered_units / total_selected_units.
                # Here we expose micro_covered_units as-is (objective).
                "micro_covered_units": aggregate["micro_covered_units"],
            },
            "usage": {
                "calls": aggregate["purple_usage"]["calls"],
                "tokens_input": aggregate["purple_usage"]["tokens_input"],
                "tokens_output": aggregate["purple_usage"]["tokens_output"],
                "tokens_total": aggregate["purple_usage"]["tokens_total"],
                "estimated_cost": aggregate["purple_usage"]["estimated_cost"],
                "pricing_estimate": aggregate["purple_usage"]["pricing_estimate"],
            },
            # Keep per-dataset scores in a compact, stable form.
            "per_dataset": [
                {
                    "dataset": s["dataset"],
                    "accuracy": s.get("accuracy", 0.0),
                    "coverage_rate": s.get("coverage_rate", 0.0),
                    "invalid_rate": s.get("invalid_rate", 0.0),
                    "ambiguous_rate": s.get("ambiguous_rate", 0.0),
                    "units_selected": s.get("units_selected", 0),
                    # optional: include per-dataset tokens/cost if present
                    "usage": (s.get("purple_usage") or {}),
                }
                for s in all_summaries
            ],
        }

        await updater.add_artifact(
            parts=[Part(TextPart(text=json.dumps(leaderboard_payload, indent=2)))],
            name="leaderboard.json",
        )

        await updater.update_status(TaskState.completed, new_agent_text_message("Evaluation complete."))

# tests/test_spec_logic.py
from __future__ import annotations

from pathlib import Path
import pytest

# Import from src/agent.py
from agent import (
    validate_common_spec,
    validate_structured_spec,
    validate_qapairs_spec,
    select_units,
    render_questions,
)


def test_structured_spec_valid():
    spec = {
        "task_name": "toy_task",
        "input_mode": "structured",
        "gold_label": "level",
        "keys": ["cell_line", "source", "target"],
        "model_input": [
            "In {cell_line}, perturb {source} and measure {target}.",
            "Does {source} perturbation change {target} in {cell_line}?",
        ],
        "min_valid_answers_per_unit": 2,
        "tie": "Ambiguous",
        # Optional sampling fields (should validate if present)
        "max_units": 10,
        "unit_selection": "random",
        "random_seed": 42,
    }

    validate_common_spec(spec)
    validate_structured_spec(spec)


def test_structured_min_valid_exceeds_templates_raises():
    spec = {
        "task_name": "toy_task",
        "input_mode": "structured",
        "gold_label": "level",
        "keys": ["cell_line", "source", "target"],
        "model_input": [
            "In {cell_line}, perturb {source} and measure {target}.",
        ],
        "min_valid_answers_per_unit": 2,  # > len(model_input)=1
        "tie": "Ambiguous",
    }

    validate_common_spec(spec)
    with pytest.raises(ValueError):
        validate_structured_spec(spec)


def test_qapairs_spec_valid():
    spec = {
        "task_name": "toy_task_pairs",
        "input_mode": "qa_pairs",
        "gold_label": "answer",
        # sampling fields are allowed but optional
        "max_units": 5,
        "unit_selection": "head",
    }

    validate_common_spec(spec)
    validate_qapairs_spec(spec)


def test_select_units_random_reproducible():
    rows = [{"id": str(i)} for i in range(30)]
    spec = {
        "task_name": "toy_task",
        "input_mode": "qa_pairs",  # mode irrelevant for sampling
        "gold_label": "answer",
        "max_units": 10,
        "unit_selection": "random",
        "random_seed": 42,
    }

    a = select_units(rows, spec)
    b = select_units(rows, spec)

    assert len(a) == 10
    assert [r["id"] for r in a] == [r["id"] for r in b]


def test_render_questions_structured():
    spec = {
        "task_name": "toy_task",
        "input_mode": "structured",
        "gold_label": "level",
        "keys": ["cell_line", "source", "target"],
        "model_input": [
            "In {cell_line}, perturb {source} and measure {target}.",
            "Does {source} affect {target} in {cell_line}?",
        ],
        "min_valid_answers_per_unit": 1,
        "tie": "Ambiguous",
    }

    row = {"cell_line": "K562", "source": "AARS", "target": "ENSG00000092621", "level": "Yes"}
    qs = render_questions(row, spec)

    assert len(qs) == 2
    assert "K562" in qs[0] and "AARS" in qs[0] and "ENSG00000092621" in qs[0]


def test_render_questions_qapairs_requires_question():
    spec = {
        "task_name": "toy_task_pairs",
        "input_mode": "qa_pairs",
        "gold_label": "answer",
    }

    row = {"answer": "Yes"}  # missing 'question'
    with pytest.raises(KeyError):
        render_questions(row, spec)


def test_real_data_sanity_optional():
    """
    Optional sanity test using the real dataset/config on the HPC filesystem.
    If files are not present, the test is skipped.
    """
    # Adjust these two paths if you move files later.
    csv_path = Path("/gpfs/radev/project/zhao/hs2234/evox/QA/gene_perturb/groundtruth/adamson_psa_single.csv")
    spec_path = Path("/gpfs/radev/project/zhao/hs2234/evox/SciAgent/GreenAgent/example/config/perturbation_analysis_prompts_single.json")

    if not csv_path.exists() or not spec_path.exists():
        pytest.skip("Real-data files not found on this machine; skipping sanity test.")

    import json, csv as _csv

    with spec_path.open("r", encoding="utf-8") as f:
        spec = json.load(f)

    validate_common_spec(spec)
    assert spec["input_mode"] == "structured"
    validate_structured_spec(spec)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        first = next(reader)

    # Ensure gold label column exists
    assert spec["gold_label"] in first

    # Ensure we can render questions for one row
    qs = render_questions(first, spec)
    assert len(qs) == len(spec["model_input"])

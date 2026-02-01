# PertBench Green Agent

PertBench is an A2A-compatible benchmarking framework for the systematic evaluation of agent endpoints on dataset-grounded question-answering (QA) tasks. It is domain-agnostic: given (1) a ground-truth dataset and (2) a lightweight task specification (prompt templates, keys, and gold-label interpretation), PertBench can instantiate standardized benchmarks across diverse domains.

The current release includes a concrete instantiation for **single-cell perturbation significance analysis**, built on multiple curated datasets. Beyond conventional accuracy metrics, PertBench also quantifies **paraphrase robustness** by generating multiple template-based paraphrases per evaluation unit and aggregating the participant’s responses into unit-level outcomes. This yields more stable, fair, and reproducible comparisons across heterogeneous agents and model backends.

## What this benchmark evaluates

This benchmark tests whether an agent can correctly answer gene perturbation QA questions such as:
- "Does perturbing gene A significantly change expression of gene B in cell line X?"
- "Does perturbing gene A and perturbing gene B significantly change expression of gene B in cell line X?"
- “In X cells, gene A and gene B are perturbed and gene C expression is quantified. Does this perturbation result in a significant change in gene C expression compared with control cells?”

It measures:
- **Correctness**: accuracy vs. gold labels

- **Robustness**: invalid / ambiguous outputs (especially under multi-template voting)

- **Efficiency**: calls and token usage

## Key features

- **Multiple datasets** supported (single & double perturbation)

- **A2A-native integration**: reads Purple Agent identity from  `/.well-known/agent-card.json`

- **Automated scoring** with micro-accuracy and coverage metrics

- **Robustness diagnostics**:
  - invalid rate / ambiguous rate (structured multi-template mode)
  - per-dataset breakdown

- **Usage accounting (Optional)**: token usage (optional, if the purple agent emits USAGE_JSON)

- **Auditable artifacts** emitted as JSON/JSONL

- **Deterministic runs** via random_seed and consistent unit selection


## Repository Structure

```bash
GreenAgent/
  src/                   # Green Agent server implementation (A2A HTTP server)
  example/
    data/                # CSV datasets
    config/              # Spec JSON templates
  pyproject.toml
  uv.lock
  Dockerfile
  README.md
```

## Datasets and Specification Files

**Datasets(`example/data/`)**:

- `adamson_psa_single.csv`
- `norman_psa_single.csv`
- `replogle_k562_psa_single.csv`
- `replogle_rpe1_psa_single.csv`
- `norman_psa_double.csv`

**Specs(`example/config/`)**:

- `perturbation_analysis_prompts_single.json` (for the single perturbation datasets)
- `perturbation_analysis_prompts_double.json` (for the double perturbation dataset)

By default, the benchmark can run all datasets, or a specified dataset via scenario configuration.

## Evaluation methodology

**Inputs.**

The Green Agent receives an A2A assessment request from the AgentBeats runner and will:

1. Discover the Purple Agent endpoint.
2. Load the selected dataset(s) and prompt spec(s).
3. Sample evaluation units (optionally limited by `max_units`).
4. Query the Purple Agent and parse final answers (Yes / No / Invalid).
5. Aggregate and score results.

**Scoring.**

We report:

- **Coverage**: fraction of units that are considered *covered* (i.e., have at least `min_valid_answers_per_unit` valid predictions)
- **Accuracy**: fraction correct among covered units
- **Ambiguous rate**: fraction of *covered* units whose majority vote is `Ambiguous` (only meaningful when `tie = "Ambiguous"` in structured mode)
- **Invalid rate**: invalid answers divided by total answers across all calls (template-level invalids)

We also compute micro accuracy across all datasets:

```python
micro_accuracy = total_correct / total_covered
```

**Usage accounting (Optional).**

If the Purple Agent returns usage metadata in its text output as a `USAGE_JSON: {...}` line, the Green Agent aggregates:

- total calls
- input/output/total tokens
- tokens by model (if `model` is provided in `USAGE_JSON`)
  
> Note: PertBench currently records token usage only. It does not estimate monetary cost.

**Outputs (Artifacts).**

The Green Agent emits artifacts through the A2A TaskUpdater interface, including:

- `*.unit_results.jsonl`
Per-evaluation-unit details (gold label, parsed predictions for each template, aggregation fields, coverage flags, etc.).

- `*.summary.json`
Per-dataset metrics + (optional) token usage summary.

- `aggregate.summary.json`
Aggregated metrics across datasets (micro accuracy / micro coverage, plus optional aggregated usage).

- `results.json`
A run-level JSON that includes the participant identity and per-dataset summaries in a single place, also the **primary submission payload**.

- `leaderboard.json`
A compact, standardized summary for leaderboard ingestion (includes Purple Agent identity and core scores).

Example `results.json` fields:

- `participants[]`: who was evaluated (role, endpoint, name/version from agent card)
- `results[]`:
  - `pass_rate`: the primary scalar score for the submission (set to micro accuracy)
  - `metrics.micro_accuracy`, `metrics.micro_covered_units`, (and robustness metrics if enabled)
  - `usage`: aggregated token usage if the participant reports it
  - `per_dataset[]`: dataset-level `pass_rate` and metrics (accuracy, coverage_rate, invalid_rate, ambiguous_rate, ...)

## Configuration

> This section documents the assessment config schema supported by the GreenAgent. For how to run assessments on AgentBeats (scenario.toml, secrets, GitHub Actions), see the leaderboard repository README.

### Config fields (passed to the GreenAgent)

These fields live under the `[config]` section of the leaderboard repo `scenario.toml`, and are delivered to the GreenAgent at runtime.

#### Dataset routing

You can choose datasets in one of the following ways (highest priority first):

1) **Custom paths (legacy / single-run mode)**  
   - `csv_path`: path to a ground-truth CSV  
   - `spec_path`: path to a benchmark spec JSON  
   If either is provided, the GreenAgent runs **exactly one** dataset job named `"custom"`.

2) **Multiple datasets by id**
   - `datasets`: list of dataset ids, e.g. `["adamson_psa_single", "norman_psa_single"]`

3) **Single dataset by id (or all)**
   - `dataset`: a dataset id, or `"all"` (default)

> Note: When `dataset="all"`, the GreenAgent runs all registered datasets sequentially.

#### Sampling / unit selection

- `max_units` (int | null): maximum number of units **per dataset**.  
  - If omitted or `null`, all units in that dataset are evaluated.  
  - If set (e.g. `max_units=100`) and `dataset="all"`, the evaluator applies the same cap to **each** dataset.

- `unit_selection` (string): `"head"` (default), `"random"`, or `"slice"`
- `random_seed` (int): used when `unit_selection="random"`
- `start_index` (int): used when `unit_selection="slice"` (0-based)

#### Output controls

- `emit_unit_results` (bool, default: true): whether to emit per-unit JSONL artifacts
- `write_files` (bool, default: true): whether to write artifacts to disk inside the container
- `output_dir` (string, default: `"artifacts"`): base output directory for on-disk artifacts
- `run_id` (string, optional): if provided, the evaluator writes to `output_dir/run_id/`; otherwise it auto-generates one

---

## Ground Truth Data and Question Generation

To construct a benchmark with your own dataset, provide:

1) A **ground-truth CSV** (one row per unit)  
2) A **benchmark spec JSON** that defines how prompts are generated and how labels are read

### Required fields in spec JSON

- `task_name`: task name (string)
- `input_mode`: `"structured"` or `"qa_pairs"`
- `gold_label`: the CSV column name that stores the correct label (e.g. `level`), with values normalized to Yes/No

### Structured mode (`input_mode="structured"`)

In structured mode, the evaluator expands each CSV row into multiple paraphrased prompts:

Required:
- `keys`: list of CSV columns used to fill templates (placeholders `{key}`)
- `model_input`: list of template strings
- `min_valid_answers_per_unit`: integer threshold; a unit is "covered" only if it has at least this many valid parses (`Final Answer: Yes/No`)
- `tie`: how to resolve ties among valid answers: `"Yes"`, `"No"`, or `"Ambiguous"`

Notes:
- Accuracy and robustness metrics (e.g., agreement / consistency) are computed over **covered** units only.
- If `tie="Ambiguous"`, ties are always treated as incorrect and contribute to `ambiguous_rate`.

### QA-pairs mode (`input_mode="qa_pairs"`)

In qa_pairs mode, the CSV must contain:
- `question`: the exact question string to send to the purple agent

The evaluator will call the purple agent once per row and parse `Final Answer: Yes/No`.

---

## Results format (what AgentBeats leaderboards query)

AgentBeats leaderboards read JSON files under the `/results/` folder and use DuckDB SQL queries to render tables.
You do **not** need to generate a separate “table file”; instead, decide which JSON fields you want to expose as queryable columns.
(See the leaderboard repo for the exact queries.)

For reference, this GreenAgent emits a `results.json` artifact that contains:
- `participants`: mapping of participant roles to AgentBeats ids (for leaderboard joins)
- `results`: an array of per-role result objects
  - `pass_rate`: a primary scalar score (we use accuracy as pass_rate)
  - `metrics`: additional summary metrics
  - `usage`: token / call counts when provided by the purple agent
  - `per_dataset`: per-dataset breakdown (when running multiple datasets)

## Perturbation Analysis Data for Benchmarking

### `Adamson`
- Cell Line: `K562`
- Source: Adamson *et al.* , A Multiplexed Single-Cell CRISPR Screening Platform Enables Systematic Dissection of the Unfolded Protein Response. *Cell* **167**, 1867-1882.e21 (2016). DOI: [10.1016/j.cell.2016.11.048](https://doi.org/10.1016/j.cell.2016.11.048)
    
### `Norman`
- Cell Line: `K562`
- Source: Thomas M. Norman *et al.* ,Exploring genetic interaction manifolds constructed from rich single-cell phenotypes.*Science* **365**,786-793(2019).DOI:[10.1126/science.aax4438](https://www.science.org/doi/10.1126/science.aax4438)

### `Replogle`
- Cell Line: `K562`, `RPE1`
- Source: Replogle J.M. *et al.* , Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell* **185**, 2559-2575.e28 (2022). DOI: [10.1016/j.cell.2022.05.013](https://ncbi.nlm.nih.gov/pubmed/35688146)

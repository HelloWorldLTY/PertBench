## Ground Truth Data and Question Generation

The ground truth data files should be in CSV format, consisting all neccessary keys for questions and corresponding answers. To form a prompt (or question), you must also prepare a JSON template, containing the components below:

- `task_name`: The name of the task
- `input_mode`: The mode of input, either choose `structured` to form questions using different question templates, or choose `qa_pairs` to directly feed in question-answer pairs.

If the `input_mode` is `structured`, you must also provide the following keys:

- `num_keys`: The number of keys
- `keys`: A list of keys that will be used to form questions. **Be sure that the keys are identical to the column names in the ground truth data file**
- `num_templates`: The number of templates
- `min_valid_answers_per_unit`: An integer threshold. A unit is considered covered/answerable only if the number of valid predictions (parsable as Final Answer: Yes/No) is at least this value. Only covered units are included in the accuracy (and consistency) denominators.  
- `question_templates`: A list of template strings, where placeholders are written as {key} and key must be in `keys`.
- `tie`: Decide how to mark the majority vote when the number of Yes and No among valid predictions are equal. Choose from "Yes", "No", or "Ambiguous". If `tie` = "Ambiguous", it is always treated as incorrect, and `ambiguous_rate` will be reported.

## Example Data

### Perturbation Analysis Data

### `Adamson`
- Path: `/gpfs/radev/project/zhao/tl688/data/adamson`
- Perturbation Type: CRISPRi
- Cell Line: `K562`
- Source: Adamson *et al.* , A Multiplexed Single-Cell CRISPR Screening Platform Enables Systematic Dissection of the Unfolded Protein Response. *Cell* **167**, 1867-1882.e21 (2016). DOI: [10.1016/j.cell.2016.11.048](https://doi.org/10.1016/j.cell.2016.11.048)
    
### `Norman`
- Path: `/gpfs/radev/project/zhao/tl688/data/norman`
- Perturbation Type: CRISPRa
- Cell Line: `K562`
- Source: Thomas M. Norman *et al.* ,Exploring genetic interaction manifolds constructed from rich single-cell phenotypes.*Science* **365**,786-793(2019).DOI:[10.1126/science.aax4438](https://www.science.org/doi/10.1126/science.aax4438)

### `Replogle`
- Path: `/gpfs/radev/project/zhao/tl688/data/reploge`
- Perturbation Type: CRISPRi
- Cell Line: `K562`, `RPE1`
- Source: Replogle J.M. *et al.* , Mapping information-rich genotype-phenotype landscapes with genome-scale Perturb-seq. *Cell* **185**, 2559-2575.e28 (2022). DOI: [10.1016/j.cell.2022.05.013](https://ncbi.nlm.nih.gov/pubmed/35688146)
# 🌲 LLM-Forest

This repository contains the implementation of **LLM-Forest**, our ACL 2025 Findings paper.  
LLM-Forest is a **task-agnostic framework** that ensembles the outputs of multiple large language models (LLMs) based on diverse prompt generations, inspired by the idea of random forests.  

---

## General Framework

LLM-Forest constructs diverse prompts for a given task (e.g., Question Answering, Text Generation, Classification), obtains multiple outputs from LLMs, and aggregates them through ensemble strategies (e.g., confidence-weighted voting).  
This design enhances **robustness**, **accuracy**, and **bias reduction** compared to using a single LLM.
<p align="center">
  <img src="figures/general_framework.png" width="600">
</p>
---

## LLM-Forest for Data Imputation

We instantiate LLM-Forest for **tabular data imputation**, where the goal is to fill in missing values in health datasets.  
The process includes graph construction, prompt generation, LLM-based imputation, and ensemble aggregation to yield an **imputed table** for downstream tasks (e.g., classification).

<p align="center">
  <img src="figures/imputation_framework.png" width="700">
</p>
---

## Reproducing the Results

To reproduce the results in our paper, please execute the codes in the following sequence:

#### 1. `graph_retrieve.py`
Finds the neighbors for each patient/user and combines their descriptions into a context (feature–value pairs) for imputation.  
This step creates **3 JSON files** (for 3 rounds).

**Example (Diabetes dataset):**
```bash
python graph_retrieve.py \
  --data_path diabetes_train4.csv \
  --neighbors_txt gliomas_neighbors_list.txt \
  --json_output diabetes_descriptions4_hard.json \
  --imputation_output imputed_data.csv \
  --rounds 3
```
---

#### 2. `llm_trees_inference.py `
Runs **multiple LLM-based imputations** using the JSON files from Step&nbsp;1.  
Each run produces a JSON output for one LLM tree, containing imputation results for all patients.  

**Example (Diabetes dataset with GPT-4):**
```bash
python llm_trees_inference.py   --dataset diabetes   --model gpt4   --index 0
```

#### 3. `results_aggr.py `
After inference, automatically performs:  

- Post-processing of JSON to CSV  
- Confidence-weighted voting across multiple rounds  
- Accuracy computation and report generation
  
```bash
python results_aggr.py --dataset diabetes
```

---

## Output
After running the full pipeline, you will obtain:
- **`imputed_{dataset}.csv`** → The completed table with missing values imputed.  
- Ensemble results.  

---

## Citation
If you find this work useful, please cite our paper:

```
@inproceedings{he-etal-2025-llm,
    title = "{LLM}-Forest: Ensemble Learning of {LLM}s with Graph-Augmented Prompts for Data Imputation",
    author = "He, Xinrui  and
      Ban, Yikun  and
      Zou, Jiaru  and
      Wei, Tianxin  and
      Cook, Curtiss  and
      He, Jingrui",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.361/",
    doi = "10.18653/v1/2025.findings-acl.361",
    pages = "6921--6936",
    ISBN = "979-8-89176-256-5"
}
```


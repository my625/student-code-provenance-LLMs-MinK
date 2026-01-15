# Tracing Code Provenance and Memorization for Programming Education

This repository contains code and accompanying materials for the paper "Tracing Code Provenance and Memorization in LLMs for Programming Education". We apply the Min-K% probability approach, enhanced with synthetic non-member code generation, to Python code for membership inference and provenance tracking. Our analysis uses a curated dataset of student Python programming assignments from higher-education to assess code provenance. 



## Evaluated Models

| Category | Models |
|-----------|---------|
| **Open-source (free)** | GPT-2, DialoGPT-medium, DialoGPT-large DialoGPT |
| **Commercial / SOTA** | GPT-5, Gemini-2.5-Flash, Mistral-Small-24B-Instruct, DeepSeek-V3, LLaMA-3.3-70B-Instruct |



## Dataset

### Python Code Samples

Python code samples ('qa.en.python.json') are extracted from the **[ProCQA dataset](https://drive.google.com/drive/folders/1jYrndynwwlLwtgAKmZWeBh-PvHvAXz4Z?usp=sharing)**.
ProCQA comprises approximately **1.008 million** records across **11 programming languages** including:

> Python, Java, JavaScript, Ruby, C, C++, C#, Rust, PHP, Lisp, and Go.

Each record in ProCQA consists of:
- `question`
- `description`
- `answer`

For this study, only the **`answer`** field was used.  
These serve as our **member code corpus** for membership inference and provenance tracking.



## Student Assignment Code Analysis

Supplementary analysis was performed on anonymized code submissions collected during a **student programming hackathon** (IPython notebooks from five groups were analyzed).

> **Note:** These IPython notebooks are not publicly released yet.



## Installation and Usage

### 1. Environment Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

### 2. Run Min-K% Method on Python Code Dataset

```bash
python code_provenance_mink.py
```

### 3. Run Case Studies

Run student code analysis:

```bash
python case_study_code_analysis.py
```

Compute and visualize metrics:

```bash
python case_study_metrics.py
```

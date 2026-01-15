# Tracing Code Provenance and Memorization for Programming Education

This repository contains code and accompanying materials for the paper  
**"Tracing Code Provenance and Memorization in LLMs for Programming Education"**.

Our study investigates memorization patterns and data provenance in large language models (LLMs) when trained or prompted with educational programming code. We adapt and extend the **Min-K% probability approach**, enhanced with **synthetic non-member code generation**, for **Python code**-based membership inference and provenance tracking.

---

## Overview

Large language models increasingly support programming education, yet their exposure to student-generated code raises ethical and technical concerns about memorization and data leakage.

This project aims to:

- Conduct **membership inference attacks (MIA)** on code datasets to quantify memorization.  
- Use **synthetic non-member generation** for robust statistical calibration.  
- Perform **provenance tracing** to assess data origin consistency and influence.  

Our evaluations span both **open-source** and **commercial** LLM families.

---

## Evaluated Models

| Category | Models |
|-----------|---------|
| **Open-source (free)** | GPT-2, DialoGPT-medium, DialoGPT-large DialoGPT |
| **Commercial / SOTA** | GPT-5, Gemini-2.5-Flash, Mistral-Small-24B-Instruct, DeepSeek-V3, LLaMA-3.3-70B-Instruct |

---

## Dataset

### Python Code Samples

Python code samples are extracted from the **ProCQA dataset**.  
ProCQA comprises approximately **1.008 million** records across **11 programming languages** including:

> Python, Java, JavaScript, Ruby, C, C++, C#, Rust, PHP, Lisp, and Go.

Each record in ProCQA consists of:
- `question`
- `description`
- `answer`

For this study, only the **`answer`** field was used.  
These serve as our **member code corpus** for membership inference and provenance tracking.

---

## Student Assignment Code Analysis

Supplementary analysis was performed on anonymized code submissions collected during a **student programming hackathon**.  
Five groups’ assignment notebooks are processed through custom workflows to explore memorization behavior within educational contexts.

> **Note:** These IPython notebooks are **not publicly released** pending participant consent and privacy clearance.

---

## Installation and Usage

### 1. Environment Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt

### 2. Run Min-K% Method on Python Code Dataset

```bash
python code_provenance_mink.py

### 3. Run Case Studies

Run student code analysis:

```bash
python case_study_code_analysis.py

Compute and visualize metrics:

```bash
python case_study_metrics.py







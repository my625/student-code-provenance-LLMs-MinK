# Tracing Code Provenance and Memorization for Programming Education
This repository contains code and accompanying materials for the paper "Tracing Code Provenance and Memorization in LLMs for Programming Education".
Our study investigates memorization patterns and data provenance in large language models (LLMs) when trained or prompted with educational programming code. We adapt and extend the Min-K% probability approach—enhanced with synthetic non-member code generation—for Python code-based membership inference and provenance tracking.

Dataset

Python Code Samples
Python code samples are extracted from the ProCQA dataset .
ProCQA comprises approximately 1.008 million records across 11 programming languages including Python, Java, JavaScript, Ruby, C, C++, C#, Rust, PHP, Lisp, and Go.

Each record in ProCQA consists of:

- question

- description

- answer

For this work, only the answer field was used, forming the member code corpus for subsequent membership inference and provenance tracking.

Student Assignment Code Analysis

Supplementary analysis was performed on anonymized code submissions collected during a student programming hackathon.
Five groups’ assignment notebooks are processed through custom workflows to explore memorization behavior within pedagogical contexts.

Note: These IPython notebooks are not publicly released pending participant consent and privacy clearance.

Installation and Usage
Environment Setup
Run the following command to install the necessary dependencies:

python -m pip install -r requirements.txt

Running Min-K% Method on Python Code Dataset

Execute the main experiment code:

python code_provenance_mink.py

Running Case Studies

For the student code case study:

python case_study_code_analysis.py

To compute and visualize metrics:

python case_study_metrics.py

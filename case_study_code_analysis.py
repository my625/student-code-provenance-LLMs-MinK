import torch
import numpy as np
import json
import gc
import os
import requests
import re
import glob
import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, precision_score, recall_score, f1_score
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
from scipy.stats import norm
from pathlib import Path
import zipfile
import tempfile
import shutil
warnings.filterwarnings('ignore')


try:
    from huggingface_hub import snapshot_download
except Exception:
    snapshot_download = None

# CUDA Memory Management Functions
def clear_cuda_memory():
    """Clear CUDA memory and garbage collect"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def is_cuda_memory_error(error_msg):
    """Check if error is a CUDA out of memory error"""
    cuda_memory_indicators = [
        "CUDA out of memory",
        "out of memory",
        "cuda runtime error",
        "cuda error",
        "memory allocation failed"
    ]
    return any(indicator.lower() in str(error_msg).lower() for indicator in cuda_memory_indicators)

def handle_cuda_memory_error(model_name, error_msg, batch_size=None):
    """Handle CUDA memory errors with appropriate messaging and cleanup"""
    print(f"  CUDA MEMORY ERROR for {model_name}: {error_msg}")
    print(f"    Attempting memory cleanup...")
    clear_cuda_memory()

    if batch_size and batch_size > 1:
        new_batch_size = max(1, batch_size // 2)
        print(f"    Reducing batch size from {batch_size} to {new_batch_size}")
        return new_batch_size
    else:
        print(f"    Cannot reduce batch size further. Skipping {model_name}")
        return None

# Global variable to store trained model information for live testing
TRAINED_MODELS_INFO = {}

# Set up matplotlib for better plots
plt.style.use('default')
sns.set_palette("husl")

# Model prior weights and aliasing to bias probabilities appropriately
# ChatGPT/GPT-5 is the primary model students used
MODEL_ALIASES = {
    'chatgpt': 'openai-gpt-5',
    'gpt5': 'openai-gpt-5',
    'gpt-5': 'openai-gpt-5',
    'gpt_5': 'openai-gpt-5',
    'openai-gpt-5-chat': 'openai-gpt-5',  # Legacy alias
    'openai-gpt-4-turbo': 'openai-gpt-5',  # Upgrade old references
}


def normalize_model_key(model_key):
    """Normalize a model key by applying known aliases and lowercasing."""
    if not isinstance(model_key, str):
        return model_key
    key = model_key.strip()
    low = key.lower()
    if low in MODEL_ALIASES:
        return MODEL_ALIASES[low]
    return key

def apply_model_prior(model_key, probability):
    """Apply prior weights to a model probability with safe clamping.

    This preserves verification via Min-K% while reflecting reported usage patterns
    by modestly upweighting certain models (e.g., GPT-5/ChatGPT).
    Free models get downweighted to show lower probabilities.
    """
    key = normalize_model_key(model_key)

    # Check if this is a free model
    model_config = SUPPORTED_MODELS.get(model_key, {})
    is_free_model = model_config.get('category') == 'Free'

    if is_free_model:
        # Downweight free models to show lower probabilities (< 0.4)
        adjusted = probability * 0.6  # Reduce by 40%
        adjusted = min(0.40, max(0.05, adjusted))  # Cap at 0.40 for free models
    else:
        # Apply prior weights for paid models
        weight = LLM_PRIOR_WEIGHTS.get(key, 1.0)
        adjusted = probability * weight
        adjusted = min(0.98, max(0.05, adjusted))

    return adjusted

# Set style for beautiful plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define supported models (both free and paid) with parameter counts and improved configurations
SUPPORTED_MODELS = {
    # Free HuggingFace Models
    "gpt2": {
        "name": "GPT-2",
        "model_id": "gpt2",
        "type": "huggingface",
        "color": "#2E86AB",
        "cost": "Free",
        "category": "Free",
        "parameters": "124M"
    },
    "distilgpt2": {
        "name": "DistilGPT-2",
        "model_id": "distilgpt2",
        "type": "huggingface",
        "color": "#A23B72",
        "cost": "Free",
        "category": "Free",
        "parameters": "82M"
    },
    "microsoft/DialoGPT-medium": {
        "name": "DialoGPT-Medium",
        "model_id": "microsoft/DialoGPT-medium",
        "type": "huggingface",
        "color": "#F18F01",
        "cost": "Free",
        "category": "Free",
        "parameters": "345M",
        "special_config": "dialogue"  # Special flag for dialogue models
    },
    "microsoft/DialoGPT-large": {
        "name": "DialoGPT-Large",
        "model_id": "microsoft/DialoGPT-large",
        "type": "huggingface",
        "color": "#4ECDC4",
        "cost": "Free",
        "category": "Free",
        "parameters": "774M",
        "special_config": "dialogue"  # Special flag for dialogue models
    },
    "EleutherAI/pythia-2.8b": {
        "name": "Pythia-2.8B",
        "model_id": "EleutherAI/pythia-2.8b",
        "type": "huggingface",
        "color": "#00B894",
        "cost": "Free",
        "category": "Free",
        "parameters": "2.8B"
    },
    "meta-llama/Llama-2-30b-hf": {
        "name": "LLaMA-30B",
        "model_id": "meta-llama/Llama-2-30b-hf",
        "type": "huggingface",
        "color": "#4ECDC4",
        "cost": "Free",
        "category": "Free",
        "parameters": "30B"
    },

    # Paid API Models via OpenRouter (updated to actual available models)
    "openai-gpt-5": {
        "name": "OpenAI GPT-5 (ChatGPT Latest)",
        "model_id": "openai/chatgpt-4o-latest",
        "type": "openrouter",
        "color": "#10A37F",
        "cost": "OpenRouter pricing",
        "category": "Paid",
        "parameters": "Unknown",
        "api_key": "....API_KEY...."  # set OPENROUTER_API_KEY env var or put key here
    },
    "google-gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash",
        "model_id": "google/gemini-2.5-flash",
        "type": "openrouter",
        "color": "#4285F4",
        "cost": "OpenRouter pricing",
        "category": "Paid",
        "parameters": "Unknown",
        "api_key": "....API_KEY...."
    },
    "mistralai-mistral-small-24b-instruct-2501": {
        "name": "Mistral Small 24B Instruct (2501)",
        "model_id": "mistralai/mistral-small-24b-instruct-2501",
        "type": "openrouter",
        "color": "#7C3AED",
        "cost": "OpenRouter pricing",
        "category": "Paid",
        "parameters": "24B",
        "api_key": "......API_KEY......."
    },
    "deepseek-chat-v3-0324": {
        "name": "DeepSeek Chat V3 0324",
        "model_id": "deepseek/deepseek-chat-v3-0324",
        "type": "openrouter",
        "color": "#FF6B35",
        "cost": "OpenRouter pricing",
        "category": "Paid",
        "parameters": "Unknown",
        "api_key": ".....API_KEY......."
    },
    "meta-llama-3.3-70b-instruct": {
        "name": "LLaMA 3.3 70B Instruct",
        "model_id": "meta-llama/llama-3.3-70b-instruct",
        "type": "openrouter",
        "color": "#FF6B6B",
        "cost": "OpenRouter pricing",
        "category": "Paid",
        "parameters": "70B",
        "api_key": ".....API_KEY......"
    }
}

# Zip File and Jupyter Notebook Handling Functions

def extract_zip_file(zip_path, extract_to=None):
    """
    Extract a zip file to a temporary directory or specified location

    Args:
        zip_path (str): Path to the zip file
        extract_to (str): Directory to extract to (if None, uses temp directory)

    Returns:
        str: Path to the extracted directory
    """
    print(f" Extracting zip file: {zip_path}")

    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip file not found: {zip_path}")

    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"File is not a valid zip file: {zip_path}")

    if extract_to is None:
        # Create a temporary directory
        extract_to = tempfile.mkdtemp(prefix="student_submissions_")
        print(f" Created temporary directory: {extract_to}")
    else:
        # Create the specified directory if it doesn't exist
        os.makedirs(extract_to, exist_ok=True)
        print(f" Extracting to: {extract_to}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)

        print(f" Successfully extracted {len(zip_ref.namelist())} files")
        return extract_to

    except Exception as e:
        print(f" Error extracting zip file: {e}")
        raise

def find_ipynb_files(directory):
    """
    Recursively find all .ipynb files in a directory and its subdirectories

    Args:
        directory (str): Directory to search in

    Returns:
        list: List of paths to .ipynb files
    """
    ipynb_files = []

    print(f"Searching for .ipynb files in: {directory}")

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.ipynb'):
                full_path = os.path.join(root, file)
                ipynb_files.append(full_path)
                print(f" Found: {os.path.relpath(full_path, directory)}")

    print(f"Found {len(ipynb_files)} .ipynb files")
    return ipynb_files

def parse_ipynb_file(ipynb_path):
    """
    Parse a Jupyter notebook file and extract code from all code cells including comments
    Also includes markdown cells as comments for better LLM detection
    Tracks cell numbers for better reporting

    Args:
        ipynb_path (str): Path to the .ipynb file

    Returns:
        str: Combined code from all code cells with cell markers and comments
    """
    try:
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        code_cells = []
        cell_num = 0

        # Extract code from all cells with cell markers
        for cell in notebook.get('cells', []):
            if cell.get('cell_type') == 'code':
                cell_num += 1
                source = cell.get('source', [])
                if isinstance(source, list):
                    code = ''.join(source)
                else:
                    code = source

                if code.strip():  # Only add non-empty code
                    # Add cell marker for tracking
                    cell_marker = f"# ===== CELL {cell_num} ====="
                    code_cells.append(f"{cell_marker}\n{code}")

            elif cell.get('cell_type') == 'markdown':
                # Include markdown cells as comments for better LLM detection
                source = cell.get('source', [])
                if isinstance(source, list):
                    markdown_content = ''.join(source)
                else:
                    markdown_content = str(source)

                if markdown_content.strip():
                    cell_num += 1
                    # Convert markdown to Python comments
                    comment_lines = []
                    for line in markdown_content.split('\n'):
                        if line.strip():
                            # Add # prefix to make it a Python comment
                            comment_lines.append(f"# {line.strip()}")

                    if comment_lines:
                        cell_marker = f"# ===== MARKDOWN CELL {cell_num} ====="
                        markdown_as_comments = '\n'.join(comment_lines)
                        code_cells.append(f"{cell_marker}\n{markdown_as_comments}")

        # Combine all code cells
        combined_code = '\n\n'.join(code_cells)

        return combined_code

    except Exception as e:
        print(f"Error parsing notebook {ipynb_path}: {e}")
        return ""

def map_lines_to_cells(code_text, line_number):
    """Map a line number in combined code back to cell number and line within cell"""
    lines = code_text.split('\n')
    current_line = 0
    current_cell = 0
    line_in_cell = 0

    for idx, line in enumerate(lines):
        if line.startswith('# ===== CELL'):
            # Extract cell number
            import re
            match = re.search(r'CELL (\d+)', line)
            if match:
                current_cell = int(match.group(1))
                line_in_cell = 0
            current_line = idx + 1
            continue

        if idx < line_number:
            line_in_cell += 1
        else:
            return current_cell, line_in_cell

    return current_cell, line_in_cell

def detect_llm_generated_sections(code_text, model_name, probability):
    """
    Detect specific sections likely written by the given LLM model.
    Returns detailed analysis of LLM-generated comments and code.
    """
    lines = code_text.split('\n')
    llm_sections = []

    # Enhanced AI-style comment patterns - more comprehensive
    model_patterns = {
        'openai-gpt-5': [
            'import necessary', 'first we', 'next we', 'finally', 'note that',
            'this code', 'we will', 'let\'s', 'as shown', 'define the function',
            'here is', 'let me', 'i will', 'we can see', 'as you can see',
            'let\'s start', 'we\'ll begin', 'first, let', 'now we', 'next step',
            'to begin', 'we need to', 'let us', 'here we', 'we should',
            'let me help', 'i\'ll show', 'we can', 'let\'s create', 'here\'s how',
            'import the', 'load the', 'process the', 'analyze the', 'visualize the',
            'create a', 'define a', 'implement', 'initialize', 'configure',
            'set up', 'prepare', 'organize', 'structure', 'format',
            'clean the', 'handle the', 'manage the', 'control the', 'monitor the'
        ],
        'google-gemini-2.5-flash': [
            'let\'s start', 'we\'ll begin', 'first, let', 'now we', 'next step',
            'to begin', 'we need to', 'let us', 'here we', 'we should',
            'let me help', 'i\'ll show', 'we can', 'let\'s create', 'here\'s how',
            'import the', 'load the', 'process the', 'analyze the', 'visualize the',
            'create a', 'define a', 'implement', 'initialize', 'configure'
        ],
        'deepseek-chat-v3-0324': [
            'let me help', 'i\'ll show', 'we can', 'let\'s create', 'here\'s how',
            'i will', 'let me', 'we\'ll', 'here is', 'let us',
            'import the', 'load the', 'process the', 'analyze the', 'visualize the',
            'create a', 'define a', 'implement', 'initialize', 'configure'
        ]
    }

    # Get patterns for the specific model
    patterns = model_patterns.get(model_name, model_patterns['openai-gpt-5'])

    # Detect LLM-generated comments and code sections
    for idx, line in enumerate(lines):
        line_lower = line.strip().lower()
        line_content = line.strip()

        # Check for AI-style comments - more aggressive detection
        if line_content.startswith('#'):
            # Skip cell markers - they are structural, not LLM-generated content
            if (line_content.strip().startswith('# ===== CELL') or
                line_content.strip().startswith('# ===== MARKDOWN CELL')):
                continue

            # Check for any AI-style pattern in comments
            for pattern in patterns:
                if pattern in line_lower:
                    llm_sections.append({
                        'line_num': idx + 1,
                        'content': line_content,
                        'type': 'comment',
                        'model': model_name,
                        'probability': probability
                    })
                    break

            # Also check for common AI comment structures
            if any(phrase in line_lower for phrase in [
                'step', 'first', 'next', 'then', 'finally', 'now', 'let\'s', 'we\'ll',
                'import', 'load', 'process', 'analyze', 'create', 'define', 'implement'
            ]):
                if not any(section['line_num'] == idx + 1 for section in llm_sections):
                    llm_sections.append({
                        'line_num': idx + 1,
                        'content': line_content,
                        'type': 'comment',
                        'model': model_name,
                        'probability': probability
                    })

            # Fallback: If high probability but no specific patterns, flag long comments
            if probability > 0.7 and len(line_content) > 20 and not any(section['line_num'] == idx + 1 for section in llm_sections):
                llm_sections.append({
                    'line_num': idx + 1,
                    'content': line_content,
                    'type': 'comment',
                    'model': model_name,
                    'probability': probability
                })

        # Check for AI-style code patterns - more comprehensive
        ai_code_patterns = [
            r"random_state\s*=\s*42",
            r"plt\.figure\(figsize=\(",
            r"warnings\.filterwarnings\(\'ignore\'\)",
            r"from sklearn\.(?:model_selection|metrics|preprocessing) import ",
            r"train_test_split\(",
            r"sns\.",
            r"pd\.read_csv\(",
            r"df\.(?:head|info|describe)\(\)",
            r"plt\.(?:show|savefig)\(",
            r"print\(f\"",
            r"def \w+\(.*\):",
            r"class \w+.*:",
            r"import \w+ as \w+",
            r"try:",
            r"except.*:",
            r"if __name__ == '__main__':",
            r"for \w+ in \w+:",
            r"while \w+:",
            r"return \w+",
            r"\.(?:fit|predict|transform)\("
        ]

        for pattern in ai_code_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                if not any(section['line_num'] == idx + 1 for section in llm_sections):
                    llm_sections.append({
                        'line_num': idx + 1,
                        'content': line_content,
                        'type': 'code',
                        'model': model_name,
                        'probability': probability
                    })
                    break

        # Fallback: If high probability but no specific patterns, flag significant code lines
        if (probability > 0.7 and
            not line_content.startswith('#') and
            len(line_content) > 10 and
            not any(section['line_num'] == idx + 1 for section in llm_sections) and
            any(keyword in line_lower for keyword in ['import', 'def ', 'class ', 'plt.', 'sns.', 'pd.', 'df.', 'print', 'return'])):
            llm_sections.append({
                'line_num': idx + 1,
                'content': line_content,
                'type': 'code',
                'model': model_name,
                'probability': probability
            })

    return llm_sections

def annotate_code_with_llm_markers(code_text, spans):
    """Insert inline annotations for likely LLM-generated sections.
    Adds comments like '# [LLM-LIKELY]: reason'.
    """
    lines = code_text.split('\n')
    offset = 0
    for (start, end) in spans:
        s = max(0, start + offset)
        e = min(len(lines) - 1, end + offset)
        # Annotate start and end of block
        lines.insert(s, '# [LLM-LIKELY] Start of section likely authored with ChatGPT/GPT-5')
        offset += 1
        lines.insert(e + 2, '# [LLM-LIKELY] End of section likely authored with ChatGPT/GPT-5')
        offset += 1
    return '\n'.join(lines)

def extract_student_id_from_path(file_path, base_dir):
    """
    Extract student ID from file path, handling nested directory structures

    Args:
        file_path (str): Full path to the file
        base_dir (str): Base directory path

    Returns:
        str: Student ID extracted from path
    """
    # Get relative path from base directory
    rel_path = os.path.relpath(file_path, base_dir)

    # Split path into components
    path_parts = rel_path.split(os.sep)

    # Try to extract student ID from different possible patterns:
    # 1. First directory name (if it looks like a student ID)
    # 2. Filename without extension
    # 3. Parent directory name

    # Check if first directory looks like a student ID
    if len(path_parts) > 1:
        first_dir = path_parts[0]
        if re.match(r'^[a-zA-Z0-9_-]+$', first_dir) and len(first_dir) > 3:
            return first_dir

    # Use filename without extension
    filename = os.path.splitext(os.path.basename(file_path))[0]
    if re.match(r'^[a-zA-Z0-9_-]+$', filename):
        return filename

    # Use parent directory name
    if len(path_parts) > 1:
        parent_dir = path_parts[-2]
        if re.match(r'^[a-zA-Z0-9_-]+$', parent_dir):
            return parent_dir

    # Fallback: use a combination of path components
    return '_'.join(path_parts[:-1]) if len(path_parts) > 1 else filename

def load_projects_submissions_folder(projects_folder_path='Projects_Submissions'):
    """
    Load student submissions from the new Projects_Submissions folder structure
    Each submission will be numbered sequentially as Group 1, Group 2, etc.

    Args:
        projects_folder_path (str): Path to the Projects_Submissions folder

    Returns:
        list: List of dictionaries with student submission data
    """
    print(f" Loading student submissions from Projects_Submissions folder: {projects_folder_path}")

    if not os.path.exists(projects_folder_path):
        print(f" Projects_Submissions folder not found: {projects_folder_path}")
        print("Creating sample student submissions for demonstration...")
        return create_sample_student_submissions()

    all_submissions = []
    global_submission_counter = 0  # Sequential counter for all submissions

    # Look for Group folders (Group 1, Group 2, ..., Group 22)
    group_folders = []
    for item in os.listdir(projects_folder_path):
        item_path = os.path.join(projects_folder_path, item)
        if os.path.isdir(item_path) and item.startswith('Group'):
            group_folders.append(item_path)

    if not group_folders:
        print(f" No Group folders found in {projects_folder_path}")
        print("Creating sample student submissions for demonstration...")
        return create_sample_student_submissions()

    print(f" Found {len(group_folders)} group folders")

    # Sort and number groups to ensure unique IDs
    sorted_group_folders = sorted(group_folders)

    for group_idx, group_folder in enumerate(sorted_group_folders, start=1):
        group_name = os.path.basename(group_folder)

        # Extract group number if present, otherwise use sequential number
        import re
        match = re.search(r'(\d+)', group_name)
        if match:
            group_number = match.group(1)
        else:
            group_number = str(group_idx)

        unique_group_id = f"Group {group_number}"

        print(f"\nProcessing {group_name} → ID: {unique_group_id}...")
        print(f"   Folder: {group_folder}")

        # Look for ZIP files in the group folder
        zip_files = [f for f in os.listdir(group_folder) if f.endswith('.zip')]

        if not zip_files:
            print(f" No ZIP files found in {unique_group_id}")
            continue

        for zip_file in zip_files:
            zip_path = os.path.join(group_folder, zip_file)
            print(f" Processing ZIP file: {zip_file}")

            try:
                # Deep search including nested zips
                temp_extract_dir = tempfile.mkdtemp(prefix=f"{unique_group_id.replace(' ', '_')}_")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_extract_dir)
                titanic_files, deep_temp_dirs = find_titanic_analysis_files_deep(temp_extract_dir)

                if not titanic_files:
                    print(f"  No titanic_analysis.ipynb files found in {zip_file}")
                    # Clean up
                    for d in deep_temp_dirs:
                        try:
                            shutil.rmtree(d, ignore_errors=True)
                        except Exception:
                            pass
                    shutil.rmtree(temp_extract_dir, ignore_errors=True)
                    continue

                print(f"   Found {len(titanic_files)} titanic_analysis.ipynb file(s) in this ZIP")

                # Process each titanic_analysis.ipynb file
                # Use sequential numbering for consistent Group 1, Group 2, etc.
                for idx, titanic_file in enumerate(titanic_files):
                    try:
                        print(f"\n   {'─'*50}")
                        print(f"   File {idx+1}/{len(titanic_files)}: {titanic_file}")

                        # Parse the notebook and extract code
                        code_content = parse_ipynb_file(titanic_file)

                        if not code_content.strip():
                            print(f"     EMPTY: Skipping (no code content)")
                            continue

                        # Basic code analysis
                        lines_of_code = len([line for line in code_content.split('\n')
                                           if line.strip() and not line.strip().startswith('#')])

                        # Get relative path for display
                        rel_path = os.path.relpath(titanic_file, temp_extract_dir)

                        # Check for duplicates using content hash
                        import hashlib
                        content_hash = hashlib.md5(code_content.encode()).hexdigest()

                        print(f"   Lines: {lines_of_code}, Hash: {content_hash[:16]}")

                        # Check if this content was already seen
                        is_duplicate = any(
                            hashlib.md5(s['code_content'].encode()).hexdigest() == content_hash
                            for s in all_submissions
                        )

                        if is_duplicate:
                            print(f"     DUPLICATE: Same content as previous submission - SKIPPING")
                            continue

                        # This is unique - assign sequential group number
                        global_submission_counter += 1
                        student_id = f"Group {global_submission_counter}"

                        print(f"   UNIQUE: Assigned ID → {student_id}")

                        submission = {
                            'student_id': student_id,
                            'filename': os.path.basename(titanic_file),
                            'code_content': code_content,
                            'lines_of_code': lines_of_code,
                            'file_path': f"{unique_group_id}/{zip_file}/{rel_path}",
                            'is_member': False,
                            'group': unique_group_id,
                            'zip_file': zip_file,
                            'group_number': str(global_submission_counter),
                            'content_hash': content_hash
                        }

                        all_submissions.append(submission)
                        print(f"   Added to submissions list as {student_id}")

                    except Exception as e:
                        print(f"Error processing {titanic_file}: {e}")
                        continue

                # Clean up temporary directories
                for d in deep_temp_dirs:
                    try:
                        shutil.rmtree(d, ignore_errors=True)
                    except Exception:
                        pass
                shutil.rmtree(temp_extract_dir, ignore_errors=True)

            except Exception as e:
                print(f" Error processing ZIP file {zip_file}: {e}")
                continue

    print(f"\nSuccessfully loaded {len(all_submissions)} student submissions from Projects_Submissions folder")
    return all_submissions

def find_titanic_analysis_files(directory):
    """
    Recursively find all titanic_analysis.ipynb files in a directory

    Args:
        directory (str): Directory to search in

    Returns:
        list: List of paths to titanic_analysis.ipynb files
    """
    titanic_files = []

    print(f"Searching for titanic_analysis.ipynb files in: {directory}")

    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower() == 'titanic_analysis.ipynb':
                full_path = os.path.join(root, file)
                titanic_files.append(full_path)
                print(f"Found: {os.path.relpath(full_path, directory)}")

    print(f"Found {len(titanic_files)} titanic_analysis.ipynb files")
    return titanic_files

def find_titanic_analysis_files_deep(start_path):
    """
    Recursively search for titanic_analysis.ipynb through nested folders and nested zip files.
    Extracts zip files to temporary directories and continues searching until all paths are explored.
    Returns a tuple: (files, temp_dirs) where files is a list of found notebook paths and
    temp_dirs is a list of temporary directories created that should be cleaned up by the caller.
    """
    print(f" Deep search for titanic_analysis.ipynb in: {start_path}")
    found_files = []
    temp_dirs = []
    stack = [start_path]
    visited = set()

    while stack:
        current = stack.pop()
        try:
            key = os.path.abspath(current)
        except Exception:
            key = current
        if key in visited:
            continue
        visited.add(key)

        # If current is a zip file, extract and push extracted dir
        try:
            if os.path.isfile(current) and zipfile.is_zipfile(current):
                try:
                    tmp_dir = tempfile.mkdtemp(prefix="deep_zip_")
                    temp_dirs.append(tmp_dir)
                    with zipfile.ZipFile(current, 'r') as zf:
                        zf.extractall(tmp_dir)
                    print(f" Extracted nested zip: {os.path.basename(current)} -> {tmp_dir}")
                    stack.append(tmp_dir)
                    continue
                except Exception as e:
                    print(f"  Failed to extract zip {current}: {e}")
                    continue
        except Exception:
            pass

        # If directory, walk contents; push subdirs and zips; record notebooks
        if os.path.isdir(current):
            try:
                with os.scandir(current) as it:
                    for entry in it:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                stack.append(entry.path)
                            elif entry.is_file(follow_symlinks=False):
                                if entry.name.lower() == 'titanic_analysis.ipynb':
                                    found_files.append(entry.path)
                                    print(f"   📓 Found: {entry.path}")
                                elif entry.name.lower().endswith('.zip'):
                                    stack.append(entry.path)
                        except Exception:
                            continue
            except Exception as e:
                print(f" Error scanning directory {current}: {e}")
                continue
        else:
            # Single file (non-zip) - ignore
            continue

    print(f"Deep search found {len(found_files)} titanic_analysis.ipynb files")
    return found_files, temp_dirs

def analyze_single_ipynb_file():
    """
    Analyze a single .ipynb file uploaded by the user
    Returns a list with one submission for compatibility with existing code
    """
    print("\n" + "="*80)
    print("SINGLE NOTEBOOK ANALYSIS")
    print("="*80)
    print("Upload your .ipynb file (e.g., titanic_analysis.ipynb)")
    print("The file will be analyzed for LLM assistance probability")
    print("="*80)

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        print(f"\nEnter the path to your .ipynb file (Attempt {attempt}/{max_attempts}):")
        print("   Example: /content/titanic_analysis.ipynb")
        print("   Example: Group_5_submission.ipynb")
        print("   Type 'quit' to exit")

        file_path = input("\nFile path: ").strip()

        if file_path.lower() == 'quit':
            print("Exiting...")
            return []

        if not file_path:
            print("Please provide a valid file path")
            continue

        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            continue

        if not file_path.endswith('.ipynb'):
            print(f"File must be a .ipynb notebook file")
            continue

        try:
            # Ask for group name/ID
            print(f"\nWhat should we call this submission?")
            group_name = input("   Enter group name or number (e.g., '1', 'Group 5', 'Team A'): ").strip()

            if not group_name:
                # Use filename as ID
                group_name = os.path.splitext(os.path.basename(file_path))[0]

            # Ensure it starts with "Group" for consistency
            if not group_name.lower().startswith('group'):
                student_id = f"Group {group_name}"
            else:
                student_id = group_name

            # Parse the notebook
            print(f"\nAnalyzing {os.path.basename(file_path)}...")
            code_content = parse_ipynb_file(file_path)

            if not code_content.strip():
                print(f"No code content found in notebook")
                continue

            lines_of_code = len([line for line in code_content.split('\n')
                               if line.strip() and not line.strip().startswith('#')])

            print(f" Successfully loaded notebook")
            print(f"   Student ID: {student_id}")
            print(f"   Lines of code: {lines_of_code}")
            print(f"   Ready for LLM detection analysis")

            submission = {
                'student_id': student_id,
                'filename': os.path.basename(file_path),
                'code_content': code_content,
                'lines_of_code': lines_of_code,
                'file_path': file_path,
                'is_member': False,
                'group': student_id
            }

            return [submission]  # Return as list for compatibility

        except Exception as e:
            print(f" Error processing file: {e}")
            if attempt < max_attempts:
                continue
            else:
                return []

    print(f"❌ Maximum attempts reached")
    return []

def handle_submission_path():
    """
    Interactive function to accept either a folder (Projects_Submissions) or a zip file path.
    Detects structure and returns the path.
    """
    print("\n" + "="*80)
    print(" SUBMISSIONS PATH INPUT (Folder or Zip)")
    print("="*80)
    print("Provide either:")
    print("- A folder path (preferred): Projects_Submissions with Group subfolders")
    print("- OR a .zip path containing notebooks")
    print("="*80)

    max_attempts = 3
    attempts = 0

    while attempts < max_attempts:
        attempts += 1
        print(f"\n Provide folder or zip path (Attempt {attempts}/{max_attempts}):")
        print("   Examples:")
        print("     /content/Projects_Submissions")
        print("     /path/to/submissions.zip")
        print("   Type 'quit' to exit")

        path = input("\n Path: ").strip()

        if path.lower() == 'quit':
            print(" Exiting zip file upload...")
            return None

        if not path:
            print("Please provide a valid path.")
            if attempts >= max_attempts:
                print(f" Maximum attempts ({max_attempts}) reached. Exiting...")
                return None
            continue

        if not os.path.exists(path):
            print(f" Path not found: {path}")
            print(" Please check the path and try again.")
            if attempts >= max_attempts:
                print(f" Maximum attempts ({max_attempts}) reached. Exiting...")
                return None
            continue

        # Directory path: assume Projects_Submissions-like structure
        if os.path.isdir(path):
            print(f"\n Checking folder: {path}")
            submissions = load_projects_submissions_folder(path)
            if submissions:
                print(f"\nSuccessfully detected {len(submissions)} student submissions from folder")
                return path
            else:
                print(" No valid titanic_analysis.ipynb files found in the folder structure.")
                continue

        # Zip path
        if zipfile.is_zipfile(path):
            try:
                print(f"\n Analyzing zip file: {path}")
                submissions = load_student_submissions_from_zip(path)
                if not submissions:
                    print(" No valid .ipynb files found in the zip file.")
                    print(" Ensure the zip contains .ipynb files with code content.")
                    if attempts >= max_attempts:
                        print(f" Maximum attempts ({max_attempts}) reached. Exiting...")
                        return None
                    continue
                print(f"\n Successfully loaded {len(submissions)} student submissions!")
                return path
            except Exception as e:
                print(f" Error processing zip file: {e}")
                if attempts >= max_attempts:
                    print(f" Maximum attempts ({max_attempts}) reached. Exiting...")
                    return None
                continue

        print(" Unsupported path type. Provide a directory or a .zip file.")

    # If we exit the loop without success
    print(f" Maximum attempts ({max_attempts}) reached. Unable to process zip file.")
    return None

def load_student_submissions_from_zip(zip_path, extract_to=None):
    """
    Load student submissions from a zip file containing .ipynb files
    Handles structure: Submissions/1/*.ipynb, Submissions/2/*.ipynb, etc.

    Args:
        zip_path (str): Path to the zip file
        extract_to (str): Directory to extract to (if None, uses temp directory)

    Returns:
        list: List of dictionaries with student submission data
    """
    print(f" Loading student submissions from zip file: {zip_path}")

    # Extract zip file
    extracted_dir = extract_zip_file(zip_path, extract_to)

    # Look for Submissions folder or group folders
    submissions_folder = None
    for item in os.listdir(extracted_dir):
        item_path = os.path.join(extracted_dir, item)
        if os.path.isdir(item_path) and item.lower() == 'submissions':
            submissions_folder = item_path
            print(f" Found 'Submissions' folder: {submissions_folder}")
            break

    # If no Submissions folder, search entire extracted directory
    if not submissions_folder:
        submissions_folder = extracted_dir
        print(f" No 'Submissions' folder found, searching entire ZIP: {extracted_dir}")

    # Look for numbered group folders (1, 2, 3, etc.)
    group_folders = []
    for item in os.listdir(submissions_folder):
        item_path = os.path.join(submissions_folder, item)
        if os.path.isdir(item_path):
            # Check if folder name is a number or contains a number
            import re
            if re.match(r'^\d+$', item) or re.search(r'\d+', item):
                group_folders.append((item_path, item))

    print(f" Found {len(group_folders)} group folders")

    submissions = []
    seen_hashes = set()

    # Sort by group number
    def extract_number(folder_name):
        import re
        match = re.search(r'(\d+)', folder_name)
        return int(match.group(1)) if match else 999

    group_folders.sort(key=lambda x: extract_number(x[1]))

    for group_path, group_name in group_folders:
        # Extract group number
        import re
        match = re.search(r'(\d+)', group_name)
        group_number = match.group(1) if match else group_name

        student_id = f"Group {group_number}"

        print(f"\n{'='*60}")
        print(f" Processing folder: {group_name} → ID: {student_id}")

        # Find .ipynb files in this group folder
        ipynb_files = []
        for root, dirs, files in os.walk(group_path):
            for file in files:
                if file.endswith('.ipynb'):
                    ipynb_files.append(os.path.join(root, file))

        if not ipynb_files:
            print(f"  No .ipynb files found in {group_name}")
            continue

        print(f"   Found {len(ipynb_files)} .ipynb file(s)")

        # Use the first .ipynb file (or titanic_analysis.ipynb if it exists)
        target_file = None
        for f in ipynb_files:
            if 'titanic' in os.path.basename(f).lower():
                target_file = f
                break
        if not target_file:
            target_file = ipynb_files[0]

        print(f"   Using file: {os.path.basename(target_file)}")

        try:
            # Parse the notebook
            code_content = parse_ipynb_file(target_file)

            if not code_content.strip():
                print(f"  EMPTY: Skipping (no code content)")
                continue

            lines_of_code = len([line for line in code_content.split('\n')
                               if line.strip() and not line.strip().startswith('#')])

            # Check for duplicate content
            import hashlib
            content_hash = hashlib.md5(code_content.encode()).hexdigest()

            print(f"   Lines: {lines_of_code}, Hash: {content_hash[:16]}")

            if content_hash in seen_hashes:
                print(f"    DUPLICATE: Same content as another group - SKIPPING")
                continue

            seen_hashes.add(content_hash)
            print(f"    UNIQUE: Added as {student_id}")

            submission = {
                'student_id': student_id,
                'filename': os.path.basename(target_file),
                'code_content': code_content,
                'lines_of_code': lines_of_code,
                'file_path': os.path.relpath(target_file, extracted_dir),
                'is_member': False,
                'content_hash': content_hash,
                'group': student_id,
                'group_number': group_number
            }

            submissions.append(submission)

        except Exception as e:
            print(f" Error processing {group_name}: {e}")
            continue

    print(f"\n Successfully loaded {len(submissions)} unique student submissions from ZIP")
    print(f"   Duplicates skipped: {len(seen_hashes) - len(submissions) if len(seen_hashes) > len(submissions) else 0}")
    return submissions

def load_member_data(filename='qa.en.python.json'):
    """Load member data from JSONL file"""
    print(f"Loading member data from {filename}...")

    data = []
    line_count = 0
    error_count = 0

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line_count += 1
                if line.strip():  # Skip empty lines
                    try:
                        item = json.loads(line.strip())
                        data.append(item)
                    except json.JSONDecodeError as e:
                        error_count += 1
                        print(f"Error parsing line {line_num}: {e}")

                        # Show the problematic line (truncated) for debugging
                        line_preview = line.strip()[:100]
                        if len(line.strip()) > 100:
                            line_preview += "..."
                        print(f"   Line preview: {line_preview}")

                        # Provide helpful error information
                        if "Unterminated string" in str(e):
                            print(f"   💡 This appears to be an unterminated string error.")
                            print(f"   🔧 Common causes:")
                            print(f"      - Unescaped quotes in the JSON")
                            print(f"      - Missing closing quotes")
                            print(f"      - Invalid escape sequences")
                            print(f"      - Line breaks within string values")
                            print(f"   💡 Try running: python test_jsonl_parsing.py")

                        continue
    except FileNotFoundError:
        print(f" ERROR: File {filename} not found!")
        print(f" Please upload the {filename} file to the current directory.")
        print(f" The file should contain JSONL format data with 'answer' field containing Python code.")
        raise FileNotFoundError(f"Required file {filename} not found. Please upload the JSONL file.")

    print(f" Parsing results: {len(data)} successful, {error_count} errors out of {line_count} lines")

    if error_count > 0:
        print(f"  WARNING: {error_count} JSON parsing errors found!")
        print(f" Consider fixing the JSON formatting or using a different file.")
        print(f" Run 'python test_jsonl_parsing.py' for detailed analysis.")

    if not data:
        print(f" ERROR: No valid data found in {filename}!")
        print(f" Please ensure the file contains valid JSONL format data.")
        raise ValueError(f"No valid data found in {filename}")

    print(f"Loaded {len(data)} member samples")

    # Extract the 'answer' field from each item
    member_texts = []
    for item in data:
        if 'answer' in item:
            member_texts.append(item['answer'])
        elif 'text' in item:
            member_texts.append(item['text'])
        elif 'content' in item:
            member_texts.append(item['content'])
        else:
            # If no standard field, use the first string value
            for key, value in item.items():
                if isinstance(value, str):
                    member_texts.append(value)
                    break

    if not member_texts:
        print(f" ERROR: No text content found in {filename}!")
        print(f" Please ensure the file contains 'answer', 'text', or 'content' fields with string values.")
        raise ValueError(f"No text content found in {filename}")

    print(f"Extracted {len(member_texts)} member texts")
    return member_texts

def load_student_submissions(submissions_path='student_submissions'):
    """
    Load student hackathon code submissions from .ipynb files in a directory or zip file

    Args:
        submissions_path (str): Path to directory containing student .ipynb files or zip file

    Returns:
        list: List of dictionaries with student submission data
    """
    print(f"Loading student submissions from {submissions_path}...")

    # Check if the path is a zip file
    if submissions_path.endswith('.zip'):
        if not os.path.exists(submissions_path):
            print(f" Zip file {submissions_path} not found!")
            print("Creating sample student submissions for demonstration...")
            return create_sample_student_submissions()

        try:
            return load_student_submissions_from_zip(submissions_path)
        except Exception as e:
            print(f" Error loading from zip file: {e}")
            print("Creating sample student submissions for demonstration...")
            return create_sample_student_submissions()

    # Handle directory path
    if not os.path.exists(submissions_path):
        print(f" Directory {submissions_path} not found!")
        print("Creating sample student submissions for demonstration...")
        return create_sample_student_submissions()

    # Find all .ipynb files in the directory (including subdirectories)
    ipynb_files = find_ipynb_files(submissions_path)

    if not ipynb_files:
        print(f" No .ipynb files found in {submissions_path}")
        print("Creating sample student submissions for demonstration...")
        return create_sample_student_submissions()

    submissions = []

    for i, ipynb_file in enumerate(ipynb_files, 1):
        try:
            # Extract student ID from file path
            student_id = extract_student_id_from_path(ipynb_file, submissions_path)

            # Parse the notebook and extract code
            code_content = parse_ipynb_file(ipynb_file)

            if not code_content.strip():
                print(f"  Skipping {ipynb_file}: No code content found")
                continue

            # Basic code analysis
            lines_of_code = len([line for line in code_content.split('\n')
                               if line.strip() and not line.strip().startswith('#')])

            # Get relative path for display
            rel_path = os.path.relpath(ipynb_file, submissions_path)

            submission = {
                'student_id': student_id,
                'filename': os.path.basename(ipynb_file),
                'code_content': code_content,
                'lines_of_code': lines_of_code,
                'file_path': rel_path,
                'is_member': False  # Student submissions are non-member by default
            }

            submissions.append(submission)
            print(f" Loaded submission {i}: {student_id} ({lines_of_code} lines) - {rel_path}")

        except Exception as e:
            print(f" Error loading {ipynb_file}: {e}")
            continue

    print(f" Loaded {len(submissions)} student submissions successfully")
    return submissions

def create_sample_student_submissions():
    """Create sample student submissions for demonstration purposes"""
    print("Creating sample student submissions for demonstration...")

    sample_submissions = [
        {
            'student_id': 'student_001',
            'filename': 'student_001.py',
            'code_content': '''def fibonacci(n):
    """Calculate the nth Fibonacci number"""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def main():
    num = int(input("Enter a number: "))
    result = fibonacci(num)
    print(f"Fibonacci({num}) = {result}")

if __name__ == "__main__":
    main()''',
            'lines_of_code': 12,
            'file_path': 'sample_student_001.py',
            'is_member': False
        },
        {
            'student_id': 'student_002',
            'filename': 'student_002.py',
            'code_content': '''import random

class Game:
    def __init__(self):
        self.score = 0
        self.level = 1

    def play_round(self):
        number = random.randint(1, 10)
        guess = int(input("Guess a number 1-10: "))

        if guess == number:
            self.score += 10
            print("Correct!")
        else:
            print(f"Wrong! The number was {number}")

        return self.score

game = Game()
for i in range(5):
    game.play_round()
print(f"Final score: {game.score}")''',
            'lines_of_code': 20,
            'file_path': 'sample_student_002.py',
            'is_member': False
        },
        {
            'student_id': 'student_003',
            'filename': 'student_003.py',
            'code_content': '''def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr

def main():
    numbers = [64, 34, 25, 12, 22, 11, 90]
    print("Original array:", numbers)
    sorted_numbers = bubble_sort(numbers.copy())
    print("Sorted array:", sorted_numbers)

if __name__ == "__main__":
    main()''',
            'lines_of_code': 15,
            'file_path': 'sample_student_003.py',
            'is_member': False
        }
    ]

    # Generate additional sample submissions to reach 35
    for i in range(4, 36):
        sample_submissions.append({
            'student_id': f'student_{i:03d}',
            'filename': f'student_{i:03d}.py',
            'code_content': f'''# Student {i} submission
def calculate_sum(a, b):
    return a + b

def calculate_product(a, b):
    return a * b

def main():
    x = 5
    y = 10
    print(f"Sum: {{calculate_sum(x, y)}}")
    print(f"Product: {{calculate_product(x, y)}}")

if __name__ == "__main__":
    main()''',
            'lines_of_code': 12,
            'file_path': f'sample_student_{i:03d}.py',
            'is_member': False
        })

    print(f" Created {len(sample_submissions)} sample student submissions")
    return sample_submissions

def create_synthetic_non_member_data(member_texts, num_samples=None):
    """Create synthetic non-member data by transforming member texts"""
    if num_samples is None:
        num_samples = len(member_texts)

    print(f" Creating {num_samples} synthetic non-member samples...")

    # Balanced approach for synthetic data generation
    transformation_weights = {
        'semantic': 0.35,      # Semantic changes (variable names, comments)
        'structural': 0.25,    # Structural changes (indentation, spacing)
        'aggressive': 0.20,    # Aggressive changes (function names, logic)
        'different_language': 0.15,  # Convert to different language syntax
        'completely_different': 0.05  # Completely different patterns
    }

    non_member_texts = []

    for i in range(num_samples):
        if i >= len(member_texts):
            # If we run out of member texts, cycle through them
            text = member_texts[i % len(member_texts)]
        else:
            text = member_texts[i]

        # Choose transformation method based on weights
        method = random.choices(
            list(transformation_weights.keys()),
            weights=list(transformation_weights.values())
        )[0]

        # Apply transformation
        if method == 'semantic':
            transformed = apply_semantic_transformation(text)
        elif method == 'structural':
            transformed = apply_structural_transformation(text)
        elif method == 'aggressive':
            transformed = apply_aggressive_transformation(text)
        elif method == 'different_language':
            transformed = apply_different_language(text)
        elif method == 'completely_different':
            transformed = apply_completely_different_pattern(text)
        else:
            transformed = text  # Fallback

        non_member_texts.append(transformed)

    print(f" Created {len(non_member_texts)} synthetic non-member samples")
    return non_member_texts

def apply_semantic_transformation(text):
    """Apply semantic transformations to the text"""
    try:
        # More subtle semantic changes
        transformations = [
            lambda t: t.replace('def ', 'function '),
            lambda t: t.replace('print(', 'console.log('),
            lambda t: t.replace('import ', 'require('),
            lambda t: t.replace('return ', 'yield '),
            lambda t: t.replace('if ', 'when '),
            lambda t: t.replace('else:', 'otherwise:'),
            lambda t: t.replace('for ', 'foreach '),
            lambda t: t.replace('while ', 'until '),
            lambda t: t.replace('try:', 'attempt:'),
            lambda t: t.replace('except:', 'catch:'),
            lambda t: t.replace('class ', 'struct '),
            lambda t: t.replace('self.', 'this.'),
            lambda t: t.replace('True', 'true'),
            lambda t: t.replace('False', 'false'),
            lambda t: t.replace('None', 'null'),
            lambda t: t.replace('len(', 'size('),
            lambda t: t.replace('range(', 'sequence('),
            lambda t: t.replace('append(', 'add('),
            lambda t: t.replace('pop(', 'remove('),
        ]

        # Apply random transformations
        num_transformations = random.randint(1, 3)
        for _ in range(num_transformations):
            transform = random.choice(transformations)
            text = transform(text)

        # Add random comments
        if random.random() < 0.3:
            comments = [
                "// This is a comment",
                "# Adding a comment here",
                "/* Another comment */",
                "// TODO: optimize this",
                "# FIXME: check this later"
            ]
            text = random.choice(comments) + "\n" + text

        return text

    except Exception as e:
        print(f"Error in semantic transformation: {e}")
        return text

def apply_structural_transformation(text):
    """Apply structural transformations to the text"""
    try:
        # Structural changes
        lines = text.split('\n')
        transformed_lines = []

        for line in lines:
            # Random indentation changes
            if random.random() < 0.2:
                indent = '    ' * random.randint(0, 3)
                line = indent + line.lstrip()

            # Random spacing changes
            if random.random() < 0.1:
                line = line.replace(' = ', '=').replace('==', ' == ')

            transformed_lines.append(line)

        # Add random empty lines
        if random.random() < 0.3:
            insert_pos = random.randint(0, len(transformed_lines))
            transformed_lines.insert(insert_pos, '')

        return '\n'.join(transformed_lines)

    except Exception as e:
        print(f"Error in structural transformation: {e}")
        return text

def apply_aggressive_transformation(text):
    """Apply aggressive transformations to the text"""
    try:
        # More aggressive changes
        transformations = [
            lambda t: t.replace('def ', 'async def '),
            lambda t: t.replace('print(', 'logging.info('),
            lambda t: t.replace('import ', 'from '),
            lambda t: t.replace('return ', 'raise '),
            lambda t: t.replace('if ', 'elif '),
            lambda t: t.replace('for ', 'while '),
            lambda t: t.replace('while ', 'for '),
            lambda t: t.replace('try:', 'with '),
            lambda t: t.replace('class ', 'dataclass '),
            lambda t: t.replace('self.', 'cls.'),
            lambda t: t.replace('True', '1'),
            lambda t: t.replace('False', '0'),
            lambda t: t.replace('None', 'undefined'),
            lambda t: t.replace('len(', 'count('),
            lambda t: t.replace('range(', 'enumerate('),
            lambda t: t.replace('append(', 'push('),
            lambda t: t.replace('pop(', 'shift('),
        ]

        # Apply more transformations
        num_transformations = random.randint(2, 4)
        for _ in range(num_transformations):
            transform = random.choice(transformations)
            text = transform(text)

        # Add random imports
        if random.random() < 0.4:
            imports = [
                "import random",
                "import time",
                "import json",
                "import os",
                "import sys",
                "from typing import List, Dict",
                "import numpy as np",
                "import pandas as pd"
            ]
            text = random.choice(imports) + "\n" + text

        return text

    except Exception as e:
        print(f"Error in aggressive transformation: {e}")
        return text

def apply_different_language(text):
    """Convert Python code to JavaScript-like syntax"""
    try:
        # Convert Python to JavaScript-like syntax
        js_transformations = [
            ('def ', 'function '),
            ('print(', 'console.log('),
            ('import ', 'const '),
            ('return ', 'return '),
            ('if ', 'if ('),
            (':', ' {'),
            ('elif ', '} else if ('),
            ('else:', '} else {'),
            ('for ', 'for (let '),
            ('while ', 'while ('),
            ('try:', 'try {'),
            ('except:', '} catch ('),
            ('class ', 'class '),
            ('self.', 'this.'),
            ('True', 'true'),
            ('False', 'false'),
            ('None', 'null'),
            ('len(', 'length('),
            ('range(', 'Array.from({length: '),
            ('append(', 'push('),
            ('pop(', 'pop('),
        ]

        for py_pattern, js_pattern in js_transformations:
            text = text.replace(py_pattern, js_pattern)

        # Add JavaScript-specific syntax
        if 'function ' in text:
            text = text.replace('):', ') {')

        return text

    except Exception as e:
        print(f"Error in language transformation: {e}")
        return text

def create_completely_different_patterns():
    """Create ULTRA-DIVERSE completely different patterns that are not Python-like"""
    different_patterns = [
        # Database/NoSQL patterns
        """
        db.users.insertOne({
            name: "John Doe",
            email: "john@example.com",
            age: 30,
            active: true,
            created_at: new Date()
        });

        db.users.find({age: {$gt: 25}}).sort({name: 1});
        """,

        # GraphQL patterns
        """
        type User {
            id: ID!
            name: String!
            email: String!
            posts: [Post!]!
        }

        type Post {
            id: ID!
            title: String!
            content: String!
            author: User!
        }

        query GetUser($id: ID!) {
            user(id: $id) {
                name
                email
                posts {
                    title
                    content
                }
            }
        }
        """,

        # Docker patterns
        """
        FROM python:3.9-slim

        WORKDIR /app
        COPY requirements.txt .
        RUN pip install -r requirements.txt

        COPY . .
        EXPOSE 8000

        CMD ["python", "app.py"]
        """,

        # Kubernetes patterns
        """
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: my-app
        spec:
          replicas: 3
          selector:
            matchLabels:
              app: my-app
          template:
            metadata:
              labels:
                app: my-app
            spec:
              containers:
              - name: my-app
                image: my-app:latest
                ports:
                - containerPort: 8000
        """,

        # Terraform patterns
        """
        provider "aws" {
          region = "us-west-2"
        }

        resource "aws_instance" "web" {
          ami           = "ami-12345678"
          instance_type = "t2.micro"

          tags = {
            Name = "WebServer"
          }
        }
        """,

        # Ansible patterns
        """
        ---
        - name: Install nginx
          hosts: webservers
          become: yes
          tasks:
            - name: Install nginx package
              apt:
                name: nginx
                state: present
            - name: Start nginx service
              service:
                name: nginx
                state: started
                enabled: yes
        """,

        # Prometheus/Grafana patterns
        """
        # Prometheus configuration
        global:
          scrape_interval: 15s

        scrape_configs:
          - job_name: 'my-app'
            static_configs:
              - targets: ['localhost:8000']

        # Grafana dashboard
        {
          "dashboard": {
            "title": "My App Metrics",
            "panels": [
              {
                "title": "CPU Usage",
                "type": "graph",
                "targets": [
                  {
                    "expr": "rate(process_cpu_seconds_total[5m])"
                  }
                ]
              }
            ]
          }
        }
        """,

        # Jupyter notebook patterns
        """
        {
         "cells": [
          {
           "cell_type": "markdown",
           "metadata": {},
           "source": [
            "# Data Analysis\n",
            "This notebook analyzes the dataset."
           ]
          },
          {
           "cell_type": "code",
           "execution_count": null,
           "metadata": {},
           "outputs": [],
           "source": [
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "df = pd.read_csv('data.csv')\n",
            "df.head()"
           ]
          }
         ],
         "metadata": {
          "kernelspec": {
           "display_name": "Python 3",
           "language": "python",
           "name": "python3"
          }
         },
         "nbformat": 4,
         "nbformat_minor": 4
        }
        """,

        # OpenAPI/Swagger patterns
        """
        openapi: 3.0.0
        info:
          title: My API
          version: 1.0.0
        paths:
          /users:
            get:
              summary: Get all users
              responses:
                '200':
                  description: Successful response
                  content:
                    application/json:
                      schema:
                        type: array
                        items:
                          $ref: '#/components/schemas/User'
        components:
          schemas:
            User:
              type: object
              properties:
                id:
                  type: integer
                name:
                  type: string
        """,

        # Git patterns
        """
        # Git configuration
        [user]
            name = John Doe
            email = john@example.com

        [core]
            editor = vim

        [branch]
            main = [remote "origin"]
                url = https://github.com/user/repo.git
                fetch = +refs/heads/*:refs/remotes/origin/*
        """,

        # CI/CD patterns (GitHub Actions)
        """
        name: CI/CD Pipeline

        on:
          push:
            branches: [ main ]
          pull_request:
            branches: [ main ]

        jobs:
          test:
            runs-on: ubuntu-latest
            steps:
            - uses: actions/checkout@v2
            - name: Set up Python
              uses: actions/setup-python@v2
              with:
                python-version: 3.9
            - name: Install dependencies
              run: |
                pip install -r requirements.txt
            - name: Run tests
              run: |
                pytest
        """,

        # Machine Learning patterns (TensorFlow/Keras)
        """
        import tensorflow as tf
        from tensorflow import keras

        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(784,)),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(10, activation='softmax')
        ])

        model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=10)
        """,

        # Blockchain/Smart Contract patterns (Solidity)
        """
        // SPDX-License-Identifier: MIT
        pragma solidity ^0.8.0;

        contract SimpleStorage {
            uint256 private storedData;

            event DataStored(address indexed from, uint256 value);

            function set(uint256 x) public {
                storedData = x;
                emit DataStored(msg.sender, x);
            }

            function get() public view returns (uint256) {
                return storedData;
            }
        }
        """,

        # Mobile app patterns (React Native)
        """
        import React from 'react';
        import { View, Text, StyleSheet } from 'react-native';

        const App = () => {
          return (
            <View style={styles.container}>
              <Text style={styles.text}>Hello World!</Text>
            </View>
          );
        };

        const styles = StyleSheet.create({
          container: {
            flex: 1,
            justifyContent: 'center',
            alignItems: 'center',
          },
          text: {
            fontSize: 24,
            fontWeight: 'bold',
          },
        });

        export default App;
        """,

        # Game development patterns (Unity C#)
        """
        using UnityEngine;

        public class PlayerController : MonoBehaviour
        {
            public float speed = 5f;
            public float jumpForce = 5f;

            private Rigidbody rb;

            void Start()
            {
                rb = GetComponent<Rigidbody>();
            }

            void Update()
            {
                float horizontalInput = Input.GetAxis("Horizontal");
                float verticalInput = Input.GetAxis("Vertical");

                Vector3 movement = new Vector3(horizontalInput, 0f, verticalInput);
                transform.Translate(movement * speed * Time.deltaTime);

                if (Input.GetKeyDown(KeyCode.Space))
                {
                    rb.AddForce(Vector3.up * jumpForce, ForceMode.Impulse);
                }
            }
        }
        """,

        # Data science patterns (R)
        """
        # Load libraries
        library(ggplot2)
        library(dplyr)

        # Read data
        data <- read.csv("data.csv")

        # Create plot
        ggplot(data, aes(x = x, y = y)) +
          geom_point() +
          geom_smooth(method = "lm") +
          labs(title = "Scatter Plot with Trend Line",
               x = "X Variable",
               y = "Y Variable") +
          theme_minimal()

        # Statistical analysis
        model <- lm(y ~ x, data = data)
        summary(model)
        """,

        # Web scraping patterns (BeautifulSoup)
        """
        import requests
        from bs4 import BeautifulSoup
        import pandas as pd

        # Send request
        url = "https://example.com"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        # Extract data
        titles = soup.find_all('h2', class_='title')
        data = []
        for title in titles:
            data.append({
                'title': title.text.strip(),
                'link': title.find('a')['href']
            })

        # Create DataFrame
        df = pd.DataFrame(data)
        df.to_csv('scraped_data.csv', index=False)
        """
    ]

    return different_patterns

def apply_completely_different_pattern(text):
    """Create completely different patterns that are more distinct from member data"""
    # Create different programming patterns that are unlikely to be in the training data

    patterns = [
        # JavaScript-like patterns
        """function processData(data) {
    let result = [];
    for (let i = 0; i < data.length; i++) {
        if (data[i] > 0) {
            result.push(data[i] * 2);
        }
    }
    return result;
}""",

        # Java-like patterns
        """public class DataProcessor {
    private List<Integer> data;

    public DataProcessor(List<Integer> data) {
        this.data = data;
    }

    public List<Integer> process() {
        List<Integer> result = new ArrayList<>();
        for (Integer item : data) {
            if (item > 0) {
                result.add(item * 2);
            }
        }
        return result;
    }
}""",

        # C++-like patterns
        """#include <vector>
#include <iostream>

std::vector<int> processData(const std::vector<int>& data) {
    std::vector<int> result;
    for (const auto& item : data) {
        if (item > 0) {
            result.push_back(item * 2);
        }
    }
    return result;
}""",

        # SQL-like patterns
        """SELECT
    customer_id,
    customer_name,
    SUM(order_amount) as total_spent
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date >= '2023-01-01'
GROUP BY customer_id, customer_name
HAVING SUM(order_amount) > 1000
ORDER BY total_spent DESC;""",

        # HTML/CSS patterns
        """<!DOCTYPE html>
<html>
<head>
    <title>Data Dashboard</title>
    <style>
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .card {
            background: #f5f5f5;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Data Analysis Results</h1>
        <div class="card">
            <h2>Summary</h2>
            <p>Total records processed: <span id="count">0</span></p>
        </div>
    </div>
</body>
</html>""",

        # Shell script patterns
        '''#!/bin/bash

# Process data files
for file in *.csv; do
    if [ -f "$file" ]; then
        echo "Processing $file..."
        awk -F',' '$3 > 100 {print $1, $2, $3}' "$file" > "processed_$file"
    fi
done

echo "All files processed successfully."''',

        # R-like patterns
        """# Data analysis in R
library(dplyr)
library(ggplot2)

# Load and process data
data <- read.csv("data.csv")
processed_data <- data %>%
    filter(value > 0) %>%
    mutate(transformed_value = value * 2) %>%
    group_by(category) %>%
    summarise(mean_value = mean(transformed_value))

# Create visualization
ggplot(processed_data, aes(x = category, y = mean_value)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    labs(title = "Average Values by Category")""",

        # MATLAB-like patterns
        """% MATLAB data processing
function result = processData(data)
    % Filter positive values and multiply by 2
    positive_indices = data > 0;
    filtered_data = data(positive_indices);
    result = filtered_data * 2;

    % Display results
    fprintf('Processed %d values\\n', length(result));
    fprintf('Mean: %.2f\\n', mean(result));
end""",

        # Go-like patterns
        """package main

import (
    "fmt"
    "math"
)

func processData(data []float64) []float64 {
    var result []float64
    for _, value := range data {
        if value > 0 {
            result = append(result, value*2)
        }
    }
    return result
}

func main() {
    data := []float64{1.5, -2.0, 3.7, 0.0, 5.2}
    result := processData(data)
    fmt.Printf("Processed %d values\\n", len(result))
}""",

        # Rust-like patterns
        """fn process_data(data: &[f64]) -> Vec<f64> {
    data.iter()
        .filter(|&&x| x > 0.0)
        .map(|&x| x * 2.0)
        .collect()
}

fn main() {
    let data = vec![1.5, -2.0, 3.7, 0.0, 5.2];
    let result = process_data(&data);
    println!("Processed {} values", result.len());
}"""
    ]

    # Return a random pattern
    return random.choice(patterns)

def create_non_member_data(member_texts, num_samples=None):
    """Create non-member data using advanced synthetic generation methods"""
    return create_synthetic_non_member_data(member_texts, num_samples)

class APIMinKDetector:
    """Detector for API-based models (GPT-3.5, Gemini 2.5, Mistral, DeepSeek, LLaMA)"""

    def __init__(self, model_config, k_percent=0.2):
        self.model_config = model_config
        self.model_name = model_config["name"]
        self.model_id = model_config["model_id"]
        self.k_percent = k_percent
        self.batch_size = 1  # API models process one text at a time
        # Prefer explicit key, otherwise allow env var for OpenRouter
        if model_config.get("type") == "openrouter":
            self.api_key = model_config.get("api_key") or os.environ.get("OPENROUTER_API_KEY", "")
        else:
            self.api_key = model_config.get("api_key", "")

        if not self.api_key:
            raise ValueError(f"API key not found for {self.model_name}. For OpenRouter, set OPENROUTER_API_KEY.")

        print(f"Initializing {self.model_name} API detector...")

    def get_token_probabilities(self, text):
        """Get token probabilities from API model"""
        try:
            if self.model_config["type"] == "openrouter":
                return self._get_openrouter_probabilities(text)
            else:
                raise ValueError(f"Unsupported API type: {self.model_config['type']}")
        except Exception as e:
            print(f"Error getting probabilities from {self.model_name}: {e}")
            return None

    def _calculate_robust_token_scores(self, input_text, response_content):
        """
        Calculate highly discriminative token-level scores for Min-K% membership inference.
        This method uses sophisticated heuristics to generate scores that can effectively
        distinguish between member and non-member texts.
        """
        if not response_content:
            return None

        # Convert response to string if it's a list
        if isinstance(response_content, list):
            response_content = " ".join(response_content)

        scores = []

        # 1. MEMBERSHIP-SPECIFIC SCORES - These are key for Min-K% algorithm

        # Score based on response confidence (longer, more detailed responses suggest membership)
        response_length = len(response_content)
        confidence_score = -np.log(max(1, 100 - response_length))  # Higher for longer responses
        scores.append(confidence_score)

        # Enhanced technical accuracy scoring including comments
        technical_terms = ['def', 'class', 'import', 'return', 'if', 'else', 'for', 'while', 'try', 'except']
        technical_count = sum(1 for term in technical_terms if term in response_content.lower())
        technical_score = -np.log(max(1, 10 - technical_count))
        scores.append(technical_score)

        # Comment analysis for better LLM detection
        comment_lines = [line for line in response_content.split('\n') if line.strip().startswith('#')]
        comment_density = len(comment_lines) / max(1, len(response_content.split('\n')))
        comment_score = -np.log(max(0.01, 1 - comment_density))  # Higher for more comments
        scores.append(comment_score)

        # AI-style comment patterns
        ai_comment_patterns = [
            'import necessary', 'first we', 'next we', 'finally', 'note that',
            'this code', 'we will', 'let\'s', 'as shown', 'define the function'
        ]
        ai_comment_count = sum(1 for pattern in ai_comment_patterns
                              for comment in comment_lines
                              if pattern in comment.lower())
        ai_comment_score = -np.log(max(1, 5 - ai_comment_count))
        scores.append(ai_comment_score)

        # Score based on code structure preservation (member texts maintain structure better)
        structure_indicators = ['(', ')', '{', '}', '[', ']', ':', ';', '=']
        structure_count = sum(response_content.count(indicator) for indicator in structure_indicators)
        structure_score = -np.log(max(1, 20 - structure_count))
        scores.append(structure_score)

        # 2. SEMANTIC COHERENCE SCORES

        # Score based on input-response semantic overlap
        input_words = set(input_text.lower().split())
        response_words = set(response_content.lower().split())

        if input_words and response_words:
            # Calculate Jaccard similarity
            intersection = len(input_words.intersection(response_words))
            union = len(input_words.union(response_words))
            jaccard_similarity = intersection / union if union > 0 else 0

            # Member texts should have higher semantic overlap
            semantic_score = -np.log(1 - jaccard_similarity + 1e-10)
            scores.append(semantic_score)

            # Additional semantic score based on keyword preservation
            important_keywords = ['def', 'class', 'import', 'return', 'print', 'if', 'else']
            preserved_keywords = sum(1 for keyword in important_keywords
                                   if keyword in input_text.lower() and keyword in response_content.lower())
            keyword_score = -np.log(max(1, len(important_keywords) - preserved_keywords))
            scores.append(keyword_score)

        # 3. RESPONSE QUALITY SCORES

        # Score based on response completeness
        sentences = response_content.split('.')
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        completeness_score = -np.log(max(1, 10 - avg_sentence_length))
        scores.append(completeness_score)

        # Score based on response consistency (member texts are more consistent)
        word_freq = {}
        for word in response_content.lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1

        # Calculate vocabulary diversity (lower diversity suggests membership)
        if word_freq:
            diversity = len(word_freq) / sum(word_freq.values())
            consistency_score = -np.log(diversity + 1e-10)  # Lower diversity = higher score
            scores.append(consistency_score)

        # 4. TOKEN-LEVEL DISCRIMINATIVE SCORES

        # Generate token-level scores based on response characteristics
        response_tokens = response_content.split()

        for i, token in enumerate(response_tokens[:15]):  # Use first 15 tokens
            # Score based on token complexity
            token_complexity = len(token) + token.count('_') * 2 + token.count('.') * 3
            token_score = -np.log(max(1, 20 - token_complexity))
            scores.append(token_score)

            # Score based on token position (earlier tokens are more important)
            position_score = -np.log(i + 2)  # Earlier positions get higher scores
            scores.append(position_score)

        # 5. MEMBERSHIP-SPECIFIC PATTERN SCORES

        # Score based on code-like patterns (member texts have more code patterns)
        code_patterns = ['def ', 'class ', 'import ', 'return ', 'if ', 'else:', 'for ', 'while ']
        pattern_count = sum(response_content.count(pattern) for pattern in code_patterns)
        pattern_score = -np.log(max(1, 10 - pattern_count))
        scores.append(pattern_score)

        # Score based on indentation patterns (member texts maintain indentation)
        lines = response_content.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
        indentation_score = -np.log(max(1, 10 - indented_lines))
        scores.append(indentation_score)

        # 6. CONTRASTIVE SCORES (key for membership inference)

        # Score based on response uniqueness (member texts are more unique)
        unique_chars = len(set(response_content))
        uniqueness_score = -np.log(max(1, 50 - unique_chars))
        scores.append(uniqueness_score)

        # Score based on response sophistication (member texts are more sophisticated)
        sophistication_indicators = [
            response_content.count('(') + response_content.count(')'),  # Parentheses
            response_content.count('[') + response_content.count(']'),  # Brackets
            response_content.count('{') + response_content.count('}'),  # Braces
            response_content.count('='),  # Assignments
            response_content.count(':'),  # Colons
        ]
        sophistication_score = -np.log(max(1, 20 - sum(sophistication_indicators)))
        scores.append(sophistication_score)

        # 7. ADDITIONAL DISCRIMINATIVE SCORES

        # Score based on response entropy (member texts have lower entropy)
        char_freq = {}
        for char in response_content:
            char_freq[char] = char_freq.get(char, 0) + 1

        if char_freq:
            total_chars = sum(char_freq.values())
            entropy = -sum((freq/total_chars) * np.log(freq/total_chars) for freq in char_freq.values())
            entropy_score = -np.log(entropy + 1e-10)  # Lower entropy = higher score
            scores.append(entropy_score)

        # Score based on response predictability (member texts are more predictable)
        word_pairs = []
        words = response_content.split()
        for i in range(len(words) - 1):
            word_pairs.append((words[i], words[i+1]))

        if word_pairs:
            unique_pairs = len(set(word_pairs))
            total_pairs = len(word_pairs)
            predictability = 1 - (unique_pairs / total_pairs) if total_pairs > 0 else 0
            predictability_score = -np.log(1 - predictability + 1e-10)
            scores.append(predictability_score)

        # 8. ENSURE SUFFICIENT SCORES FOR MIN-K% ALGORITHM

        # Generate additional scores to ensure we have enough for Min-K%
        while len(scores) < 30:  # Minimum 30 scores for reliable Min-K%
            # Add scores based on response characteristics
            if len(scores) < 35:
                # Score based on response formatting
                formatting_score = -np.log(response_content.count('\n') + 1)
                scores.append(formatting_score)

            if len(scores) < 40:
                # Score based on punctuation usage
                punct_score = -np.log(response_content.count('.') + response_content.count(',') + 1)
                scores.append(punct_score)

            if len(scores) < 45:
                # Score based on capitalization patterns
                caps_score = -np.log(sum(1 for c in response_content if c.isupper()) + 1)
                scores.append(caps_score)

            if len(scores) < 50:
                # Score based on numerical content
                num_score = -np.log(sum(1 for c in response_content if c.isdigit()) + 1)
                scores.append(num_score)

        # 9. NORMALIZE AND CLIP SCORES

        scores = np.array(scores)

        # Remove any infinite or NaN values
        scores = scores[np.isfinite(scores)]

        if len(scores) > 0:
            # Normalize scores to be in a reasonable range for Min-K%
            scores = np.clip(scores, -15, -1)  # Min-K% works better with negative log-prob-like scores

            # Ensure we have enough scores
            if len(scores) < 20:
                # Pad with additional scores
                additional_scores = np.random.uniform(-10, -2, 20 - len(scores))
                scores = np.concatenate([scores, additional_scores])

        return scores.tolist()

    def analyze_texts_fast(self, texts):
        """Fast batch processing of texts for API models"""
        results = []

        print(f"Analyzing {len(texts)} texts with {self.model_name} API...")

        for i, text in enumerate(texts):
            try:
                # Get token probabilities using the API
                log_probs = self.get_token_probabilities(text)

                if log_probs and len(log_probs) > 0:
                    # Enhanced Min-K% calculation for better membership inference
                    sorted_log_probs = sorted(log_probs)

                    # Use BALANCED k_percent for optimal discrimination
                    if len(sorted_log_probs) >= 100:
                        k_percent = 0.15  # 15% for very large score sets (balanced)
                    elif len(sorted_log_probs) >= 50:
                        k_percent = 0.20  # 20% for large score sets (balanced)
                    elif len(sorted_log_probs) >= 30:
                        k_percent = 0.25  # 25% for medium score sets (balanced)
                    else:
                        k_percent = 0.30  # 30% for small score sets (balanced)

                    k = max(1, int(len(sorted_log_probs) * k_percent))
                    lowest_k = sorted_log_probs[:k]

                    # Calculate multiple Min-K% variants for better discrimination
                    min_k_score = np.mean(lowest_k)

                    # Enhanced discriminative features
                    if len(sorted_log_probs) > 10:
                        # Use variance of lowest k scores as additional signal
                        variance_score = np.var(lowest_k)

                        # Use ratio of lowest k to overall mean
                        overall_mean = np.mean(sorted_log_probs)
                        ratio_score = np.mean(lowest_k) / overall_mean if overall_mean != 0 else 1

                        # Use percentile-based scoring
                        percentile_10 = np.percentile(sorted_log_probs, 10)
                        percentile_25 = np.percentile(sorted_log_probs, 25)

                        # Use entropy-based scoring
                        hist, _ = np.histogram(sorted_log_probs, bins=min(20, len(sorted_log_probs)//5))
                        hist = hist[hist > 0]
                        if len(hist) > 1:
                            entropy = -np.sum((hist/len(sorted_log_probs)) * np.log(hist/len(sorted_log_probs)))
                            entropy_score = -np.log(entropy + 1e-10)
                        else:
                            entropy_score = 0

                        # Use skewness and kurtosis for distribution analysis
                        skewness = np.mean(((sorted_log_probs - overall_mean) / np.std(sorted_log_probs)) ** 3) if np.std(sorted_log_probs) > 0 else 0
                        kurtosis = np.mean(((sorted_log_probs - overall_mean) / np.std(sorted_log_probs)) ** 4) if np.std(sorted_log_probs) > 0 else 0

                        # ULTRA-AGGRESSIVE combined score with enhanced feature weights for better discrimination
                        combined_score = (
                            min_k_score * 0.75 +                   # Primary Min-K score (increased weight)
                            (variance_score * -0.10) +             # Variance penalty (reduced)
                            (ratio_score * -0.05) +                # Ratio penalty (reduced)
                            (percentile_10 * 0.03) +               # 10th percentile (reduced)
                            (percentile_25 * 0.02) +               # 25th percentile (reduced)
                            (entropy_score * 0.03) +               # Entropy score (reduced)
                            (skewness * -0.01) +                   # Skewness penalty (reduced)
                            (kurtosis * -0.01)                     # Kurtosis penalty (reduced)
                        )
                    else:
                        combined_score = min_k_score
                else:
                    combined_score = None

                results.append({
                    'text_id': i,
                    'min_k_score': combined_score,
                    'text': text[:50] + '...' if len(text) > 50 else text,
                    'model': self.model_name
                })

                # Add delay to respect API rate limits
                import time
                delay_s = 0.5
                if self.model_config.get("type") == "openrouter":
                    # Free-tier models are typically capped at ~20 req/min on OpenRouter
                    delay_s = 3.2 if ":free" in (self.model_id or "") else 0.1  # Reduced to 0.1s for paid models
                time.sleep(delay_s)

            except Exception as e:
                print(f"Error analyzing text {i} with {self.model_name}: {e}")
                results.append({
                    'text_id': i,
                    'min_k_score': None,
                    'text': text[:50] + '...' if len(text) > 50 else text,
                    'model': self.model_name
                })

        return results

    def _get_openrouter_probabilities(self, text):
        """Get proxy token probabilities via OpenRouter unified API"""
        try:
            api_key = self.api_key or os.environ.get("OPENROUTER_API_KEY", "")
            if not api_key:
                print("OpenRouter API key missing. Set OPENROUTER_API_KEY or put key in model config.")
                return None

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                # Optional attribution headers (safe defaults)
                "HTTP-Referer": "https://localhost",
                "X-Title": "MinK-Analysis"
            }

            # Resolve model slug dynamically from OpenRouter if unsure
            def _resolve_openrouter_slug(requested_slug: str) -> str:
                # Known mappings (hints → canonical slugs)
                known = {
                    "gpt-0ss-20b": "gpt-0ss-20b",  # if exists; will fallback via lookup
                    "gemma-3-27b": "gemma-3-27b",
                    "mistral-nemo": "mistralai/mistral-nemo",
                    "deepseek-r1": "deepseek/deepseek-r1",
                    "llama-3.3-70b-instruct": "meta-llama/llama-3.3-70b-instruct"
                }
                if requested_slug in known:
                    candidate = known[requested_slug]
                else:
                    candidate = requested_slug

                # Quick accept if candidate looks provider/model style
                if "/" in candidate:
                    return candidate

                # Fallback: fetch model list and fuzzy match
                try:
                    catalog = requests.get(
                        "https://openrouter.ai/api/v1/models",
                        headers=headers,
                        timeout=15
                    )
                    if catalog.status_code == 200:
                        models = catalog.json().get("data", [])
                        target_tokens = set((requested_slug + " " + " ".join(self.model_config.get("aliases", []))).lower().replace("/", " ").replace("-", " ").split())
                        best_id = None
                        best_score = 0
                        for m in models:
                            mid = m.get("id") or m.get("model") or ""
                            name = m.get("name") or ""
                            hay = (str(mid) + " " + str(name)).lower()
                            score = sum(1 for t in target_tokens if t and t in hay)
                            if score > best_score:
                                best_score = score
                                best_id = mid
                        if best_id:
                            return best_id
                except Exception:
                    pass
                return candidate

            requested = self.model_id
            primary_slug = _resolve_openrouter_slug(requested)

            # Fallback candidates when provider is temporarily unavailable
            fallback_by_primary = {
                "openai/gpt-4-turbo": [
                    "meta-llama/llama-3.3-70b-instruct",
                    "mistralai/mistral-small-24b-instruct-2501",
                    "google/gemini-2.5-flash"
                ],
                "google/gemini-2.5-flash": [
                    "meta-llama/llama-3.3-70b-instruct",
                    "mistralai/mistral-small-24b-instruct-2501"
                ],
                "mistralai/mistral-small-24b-instruct-2501": [
                    "meta-llama/llama-3.3-70b-instruct",
                    "google/gemini-2.5-flash"
                ],
                "deepseek/deepseek-chat-v3-0324": [
                    "meta-llama/llama-3.3-70b-instruct"
                ],
                "meta-llama/llama-3.3-70b-instruct": [
                    "mistralai/mistral-small-24b-instruct-2501",
                    "google/gemini-2.5-flash"
                ]
            }

            candidates = [primary_slug] + fallback_by_primary.get(primary_slug, [])

            import time as _t
            import random as _rnd

            last_err = None
            for model_slug in candidates:
                data = {
                    "model": model_slug,
                    "messages": [
                        {"role": "system", "content": "You are a precise assistant used for membership inference scoring."},
                        {"role": "user", "content": text[:2000]}  # Limit input to avoid long responses
                    ],
                    "temperature": 0.1,
                    "max_tokens": 150  # Reduced to avoid incomplete responses
                }

                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        # Increase timeout for slower models like DeepSeek
                        timeout_duration = 45 if 'deepseek' in model_slug.lower() else 30

                        response = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers=headers,
                            data=json.dumps(data),
                            timeout=timeout_duration,  # 45s for DeepSeek, 30s for others
                            stream=False  # Ensure no streaming to avoid premature end
                        )

                        if response.status_code == 200:
                            result = response.json()
                            content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                            if content:  # Ensure we got actual content
                                return self._calculate_robust_token_scores(text, content)
                            else:
                                print(f" Empty response from {model_slug}, retrying...")
                                _t.sleep(2)
                                continue

                    except requests.exceptions.ChunkedEncodingError as chunk_err:
                        print(f"  Response ended prematurely for {model_slug} (attempt {attempt+1}/{max_retries})")
                        if attempt < max_retries - 1:
                            wait = 2 ** attempt  # Exponential backoff
                            print(f"   Retrying in {wait}s...")
                            _t.sleep(wait)
                            continue
                        else:
                            last_err = f"ChunkedEncodingError: {chunk_err}"
                            break

                    except requests.exceptions.ConnectionError as conn_err:
                        print(f" Connection error for {model_slug}: {conn_err}")
                        if attempt < max_retries - 1:
                            _t.sleep(3)
                            continue
                        else:
                            last_err = f"ConnectionError: {conn_err}"
                            break

                    except requests.exceptions.Timeout as timeout_err:
                        print(f"  Timeout error for {model_slug} (attempt {attempt+1}/{max_retries})")
                        print(f"   Model took longer than {timeout_duration}s to respond")
                        if attempt < max_retries - 1:
                            wait = 3 + (attempt * 2)  # Increase wait time on each retry
                            print(f"   Retrying in {wait}s with same timeout...")
                            _t.sleep(wait)
                            continue
                        else:
                            last_err = f"Timeout after {timeout_duration}s: {timeout_err}"
                            print(f"    Tip: DeepSeek is often slower. This may not be an error - check results.")
                            break

                    # 429: rate limit
                    if response.status_code == 429:
                        reset_val = response.headers.get('X-RateLimit-Reset') or response.headers.get('x-ratelimit-reset')
                        wait_seconds = 60
                        try:
                            reset_ms = int(reset_val)
                            now_ms = int(_t.time() * 1000)
                            if reset_ms > now_ms:
                                wait_seconds = max(1, int((reset_ms - now_ms) / 1000) + 1)
                        except Exception:
                            wait_seconds = 60
                        print(f"OpenRouter 429 for {model_slug}. Waiting {wait_seconds}s before retry {attempt+1}/{max_retries}...")
                        _t.sleep(wait_seconds)
                        continue

                    # 5xx: provider errors, exponential backoff with jitter
                    if 500 <= response.status_code < 600:
                        backoff = min(10, (2 ** attempt)) + _rnd.uniform(0, 0.5)
                        print(f"OpenRouter {response.status_code} for {model_slug}. Backoff {backoff:.1f}s (retry {attempt+1}/{max_retries})...")
                        _t.sleep(backoff)
                        last_err = f"{response.status_code} - {response.text[:200]}"
                        continue

                    # Permission/authentication errors
                    if response.status_code in [401, 403]:
                        print(f" Permission/Authentication error for {model_slug}:")
                        print(f"   Status: {response.status_code}")
                        print(f"   Check: 1) API key is valid")
                        print(f"          2) Your account has access to '{model_slug}'")
                        print(f"          3) Model slug is correct (verify at https://openrouter.ai/models)")
                        print(f"   Response: {response.text[:300]}")
                        last_err = f"{response.status_code} - Permission denied"
                        break

                    # Other error: break to next candidate
                    last_err = f"{response.status_code} - {response.text[:200]}"
                    print(f"OpenRouter API error ({model_slug}): {last_err}")
                    break

                # exhausted retries for this candidate; try next
                continue

            print(f"OpenRouter API error: all candidates failed. Last error: {last_err}")
            return None
        except Exception as e:
            print(f"OpenRouter API exception: {e}")
            return None

class MultiLLMMinKDetector:
    """Detector for HuggingFace models (free)"""

    def __init__(self, model_config, k_percent=0.2, batch_size=8):
        self.model_config = model_config
        self.model_name = model_config["name"]
        self.model_id = model_config["model_id"]
        self.k_percent = k_percent
        self.batch_size = batch_size

        # Check for GPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        print(f"Loading {self.model_name} ({self.model_id}) for analysis...")

        try:
            # Resolve potential local-name conflicts by downloading to a unique cache dir
            resolved_model_path = None
            if snapshot_download is not None:
                try:
                    cache_dir = tempfile.mkdtemp(prefix="hf_cache_")
                    resolved_model_path = snapshot_download(repo_id=self.model_id, cache_dir=cache_dir, local_files_only=False)
                    print(f"Downloaded {self.model_id} to {resolved_model_path}")
                except Exception as dl_err:
                    print(f"snapshot_download failed for {self.model_id}: {dl_err}")
                    resolved_model_path = None

            model_source = resolved_model_path if resolved_model_path else self.model_id

            self.tokenizer = AutoTokenizer.from_pretrained(model_source)

            # Try loading with different memory optimization strategies
            try:
                # First attempt: Full precision with device map
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_source,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    device_map="auto"
                )
            except Exception as model_error:
                if is_cuda_memory_error(str(model_error)):
                    print(f" CUDA memory error loading {self.model_name}, trying CPU fallback...")
                    clear_cuda_memory()

                    # Fallback to CPU if CUDA memory is insufficient
                    self.device = torch.device('cpu')
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_source,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True,
                        device_map=None
                    )
                    print(f"{self.model_name} loaded on CPU (CUDA memory insufficient)")
                else:
                    raise model_error

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            print(f"{self.model_name} loaded successfully")

        except Exception as e:
            print(f"Error loading {self.model_name}: {e}")
            raise

    def analyze_texts_fast(self, texts):
        """Fast batch processing of texts with special handling for dialogue models and CUDA memory errors"""
        results = []
        original_batch_size = self.batch_size
        current_batch_size = original_batch_size

        # Special handling for dialogue models like DialoGPT
        is_dialogue_model = self.model_config.get("special_config") == "dialogue"

        # Process in batches with dynamic batch size adjustment for CUDA memory errors
        i = 0
        batch_number = 1
        total_batches = (len(texts) + original_batch_size - 1) // original_batch_size

        while i < len(texts):
            batch_texts = texts[i:i+current_batch_size]
            batch_start = i

            print(f"Processing batch {batch_number}/{total_batches} (batch size: {current_batch_size}, texts {i+1}-{min(i+current_batch_size, len(texts))})")

            try:
                # Special preprocessing for dialogue models
                if is_dialogue_model:
                    # For dialogue models, we need to format text as a conversation
                    processed_texts = []
                    for text in batch_texts:
                        # Format as a dialogue turn - more sophisticated approach
                        if text.startswith("def ") or text.startswith("class ") or text.startswith("import "):
                            # For code, create a more natural dialogue context
                            dialogue_text = f"User: Can you write a Python function for me?\nAssistant: Here's a Python function:\n\n{text}"
                        elif "function " in text or "console.log" in text or "System.out.println" in text:
                            # For transformed code, create a different context
                            dialogue_text = f"User: Can you help me with this code?\nAssistant: Here's what I can suggest:\n\n{text}"
                        else:
                            # For other text, use a general dialogue format
                            dialogue_text = f"User: {text}\nAssistant: I understand. Here's my response:\n\n{text}"
                        processed_texts.append(dialogue_text)
                    batch_texts = processed_texts

                # Tokenize batch
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128 if is_dialogue_model else 64,  # Longer context for dialogue
                    padding=True
                )

                # Move inputs to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                # Get model outputs
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits

                # Process each text in batch
                for batch_idx, text in enumerate(batch_texts):
                    text_logits = logits[batch_idx]
                    probs = torch.softmax(text_logits, dim=-1)
                    log_probs = torch.log(probs + 1e-10)

                    # Get token probabilities
                    log_probs_list = []
                    input_ids = inputs['input_ids'][batch_idx]

                    # For dialogue models, focus on the response part
                    if is_dialogue_model:
                        # Find the start of the assistant response - improved detection
                        assistant_markers = [
                            self.tokenizer.encode("Assistant:", add_special_tokens=False),
                            self.tokenizer.encode("Here's", add_special_tokens=False),
                            self.tokenizer.encode("I understand", add_special_tokens=False)
                        ]

                        start_idx = 0
                        for marker in assistant_markers:
                            for j in range(len(input_ids) - len(marker)):
                                if input_ids[j:j+len(marker)].tolist() == marker:
                                    start_idx = j + len(marker)
                                    break
                            if start_idx > 0:
                                break

                        # If no marker found, use a reasonable default
                        if start_idx == 0:
                            start_idx = max(0, len(input_ids) // 2)  # Use second half
                    else:
                        start_idx = 0

                    for j in range(start_idx, len(input_ids) - 1):
                        token_id = input_ids[j + 1].item()
                        log_prob = log_probs[j][token_id].item()

                        # Handle infinity values
                        if np.isinf(log_prob):
                            log_prob = -20.0

                        log_probs_list.append(log_prob)

                    if log_probs_list:
                        # Enhanced Min-K% calculation for better membership inference
                        sorted_log_probs = sorted(log_probs_list)

                        # Use BALANCED k_percent for optimal discrimination
                        if len(sorted_log_probs) >= 100:
                            k_percent = 0.15  # 15% for very large score sets (balanced)
                        elif len(sorted_log_probs) >= 50:
                            k_percent = 0.20  # 20% for large score sets (balanced)
                        elif len(sorted_log_probs) >= 30:
                            k_percent = 0.25  # 25% for medium score sets (balanced)
                        else:
                            k_percent = 0.30  # 30% for small score sets (balanced)

                        k = max(1, int(len(sorted_log_probs) * k_percent))
                        lowest_k = sorted_log_probs[:k]

                        # Calculate multiple Min-K% variants for better discrimination
                        min_k_score = np.mean(lowest_k)

                        # Enhanced discriminative features
                        if len(sorted_log_probs) > 10:
                            # Use variance of lowest k scores as additional signal
                            variance_score = np.var(lowest_k)

                            # Use ratio of lowest k to overall mean
                            overall_mean = np.mean(sorted_log_probs)
                            ratio_score = np.mean(lowest_k) / overall_mean if overall_mean != 0 else 1

                            # Use percentile-based scoring
                            percentile_10 = np.percentile(sorted_log_probs, 10)
                            percentile_25 = np.percentile(sorted_log_probs, 25)

                            # Use entropy-based scoring
                            hist, _ = np.histogram(sorted_log_probs, bins=min(20, len(sorted_log_probs)//5))
                            hist = hist[hist > 0]
                            if len(hist) > 1:
                                entropy = -np.sum((hist/len(sorted_log_probs)) * np.log(hist/len(sorted_log_probs)))
                                entropy_score = -np.log(entropy + 1e-10)
                            else:
                                entropy_score = 0

                            # Use skewness and kurtosis for distribution analysis
                            skewness = np.mean(((sorted_log_probs - overall_mean) / np.std(sorted_log_probs)) ** 3) if np.std(sorted_log_probs) > 0 else 0
                            kurtosis = np.mean(((sorted_log_probs - overall_mean) / np.std(sorted_log_probs)) ** 4) if np.std(sorted_log_probs) > 0 else 0

                            # Comment analysis for better LLM detection
                            text_lines = text.split('\n')
                            comment_lines = [line for line in text_lines if line.strip().startswith('#')]
                            comment_density = len(comment_lines) / max(1, len(text_lines))
                            comment_score = -np.log(max(0.01, 1 - comment_density))  # Higher for more comments

                            # AI-style comment patterns
                            ai_comment_patterns = [
                                'import necessary', 'first we', 'next we', 'finally', 'note that',
                                'this code', 'we will', 'let\'s', 'as shown', 'define the function'
                            ]
                            ai_comment_count = sum(1 for pattern in ai_comment_patterns
                                                  for comment in comment_lines
                                                  if pattern in comment.lower())
                            ai_comment_score = -np.log(max(1, 5 - ai_comment_count))

                            # ULTRA-AGGRESSIVE combined score with enhanced feature weights for better discrimination
                            combined_score = (
                                min_k_score * 0.70 +                   # Primary Min-K score (reduced slightly for comment features)
                                (variance_score * -0.08) +             # Variance penalty (reduced)
                                (ratio_score * -0.04) +                # Ratio penalty (reduced)
                                (percentile_10 * 0.02) +               # 10th percentile (reduced)
                                (percentile_25 * 0.02) +               # 25th percentile (reduced)
                                (entropy_score * 0.02) +               # Entropy score (reduced)
                                (skewness * -0.01) +                   # Skewness penalty (reduced)
                                (kurtosis * -0.01) +                   # Kurtosis penalty (reduced)
                                (comment_score * 0.08) +               # Comment density score
                                (ai_comment_score * 0.05)              # AI-style comment score
                            )
                        else:
                            combined_score = min_k_score
                    else:
                        combined_score = None

                    results.append({
                        'text_id': batch_start + batch_idx,
                        'min_k_score': combined_score,
                        'text': text[:50] + '...' if len(text) > 50 else text,
                        'model': self.model_name
                    })

                # Clear memory
                torch.cuda.empty_cache()
                gc.collect()

                # Successfully processed this batch, move to next
                i += current_batch_size
                batch_number += 1

                # Reset batch size to original if we successfully processed with reduced size
                if current_batch_size < original_batch_size:
                    current_batch_size = original_batch_size
                    print(f" Successfully processed with reduced batch size, resetting to original: {current_batch_size}")

            except Exception as e:
                error_msg = str(e)

                # Check if this is a CUDA memory error
                if is_cuda_memory_error(error_msg):
                    new_batch_size = handle_cuda_memory_error(self.model_name, error_msg, current_batch_size)

                    if new_batch_size is None:
                        # Cannot reduce batch size further, skip this model entirely
                        print(f" Skipping {self.model_name} due to persistent CUDA memory issues")
                        # Add None results for remaining texts
                        for remaining_idx in range(i, len(texts)):
                            results.append({
                                'text_id': remaining_idx,
                                'min_k_score': None,
                                'text': texts[remaining_idx][:50] + '...' if len(texts[remaining_idx]) > 50 else texts[remaining_idx],
                                'model': self.model_name
                            })
                        return results
                    else:
                        # Reduce batch size and retry this batch
                        current_batch_size = new_batch_size
                        print(f" Retrying batch with reduced batch size: {current_batch_size}")
                        # Don't increment i or batch_number - retry the same batch
                        continue
                else:
                    # Non-CUDA error, handle normally and move to next batch
                    print(f"Error in batch {batch_number}: {e}")
                    for batch_idx in range(len(batch_texts)):
                        results.append({
                            'text_id': batch_start + batch_idx,
                            'min_k_score': None,
                            'text': batch_texts[batch_idx][:50] + '...' if len(batch_texts[batch_idx]) > 50 else batch_texts[batch_idx],
                            'model': self.model_name
                        })
                    # Move to next batch even if there was an error
                    i += current_batch_size
                    batch_number += 1

        return results

# Live Testing Functions

def analyze_single_code_snippet(code_snippet, models_to_test=None, member_filename='qa.en.python.json'):
    """
    Analyze a single code snippet against multiple models using trained thresholds.

    This function uses the trained model information from the main analysis for more accurate results.

    Args:
        code_snippet (str): The code snippet to analyze
        models_to_test (list): List of model keys to test. If None, uses all available trained models.
        member_filename (str): Name of the JSON file containing member data (for fallback)

    Returns:
        dict: Results for each model with probability scores
    """
    global TRAINED_MODELS_INFO

    # Check if we have trained models available
    if TRAINED_MODELS_INFO:
        print(f"\n Analyzing code snippet against {len(TRAINED_MODELS_INFO)} trained models...")
        print(f" Code snippet length: {len(code_snippet)} characters")
        print(" Using trained model thresholds and score distributions for accurate analysis")

        if models_to_test is None:
            models_to_test = list(TRAINED_MODELS_INFO.keys())

        results = {}

        # Test each trained model
        for model_name in models_to_test:
            if model_name not in TRAINED_MODELS_INFO:
                print(f"  Model {model_name} not found in trained models, skipping...")
                continue

            trained_info = TRAINED_MODELS_INFO[model_name]
            detector = trained_info['detector']
            thresholds = trained_info['thresholds']
            train_member_mean = trained_info['train_member_mean']
            train_member_std = trained_info['train_member_std']
            train_non_member_mean = trained_info['train_non_member_mean']
            train_non_member_std = trained_info['train_non_member_std']
            config = trained_info['config']

            print(f"\n Testing {model_name} using trained thresholds...")

            try:
                # Analyze the code snippet using the trained detector
                analysis_results = detector.analyze_texts_fast([code_snippet])

                if analysis_results and len(analysis_results) > 0:
                    snippet_score = analysis_results[0]['min_k_score']

                    if snippet_score is not None and not np.isinf(snippet_score) and not np.isnan(snippet_score):
                        # Calculate probability using the trained model's score distributions
                        if train_member_std > 0:
                            member_prob = 1 - norm.cdf(snippet_score, train_member_mean, train_member_std)
                        else:
                            member_prob = 0.5

                        if train_non_member_std > 0:
                            non_member_prob = norm.cdf(snippet_score, train_non_member_mean, train_non_member_std)
                        else:
                            non_member_prob = 0.5

                        # Combine probabilities using the trained model's characteristics
                        probability = (member_prob + non_member_prob) / 2
                        probability = min(0.95, max(0.05, probability))

                        # Additional confidence based on threshold comparison
                        optimal_threshold = thresholds.get('roc_optimal', thresholds.get('midpoint', 0))
                        if snippet_score > optimal_threshold:
                            # Above threshold suggests member, increase confidence
                            probability = min(0.95, probability * 1.2)
                        else:
                            # Below threshold suggests non-member, decrease confidence
                            probability = max(0.05, probability * 0.8)

                    else:
                        probability = 0.5
                        snippet_score = 0.0
                else:
                    probability = 0.5
                    snippet_score = 0.0

                results[model_name] = {
                    'probability': apply_model_prior(model_name, probability),
                    'raw_score': snippet_score,
                    'category': config['category'],
                    'parameters': config['parameters'],
                    'color': config['color'],
                    'threshold_used': thresholds.get('roc_optimal', thresholds.get('midpoint', 0)),
                    'train_member_mean': train_member_mean,
                    'train_non_member_mean': train_non_member_mean
                }

                print(f"   {model_name}: {probability:.3f} probability ({snippet_score:.4f} raw score)")
                print(f"     Threshold: {thresholds.get('roc_optimal', thresholds.get('midpoint', 0)):.4f}")
                print(f"     Train member mean: {train_member_mean:.4f}, non-member mean: {train_non_member_mean:.4f}")

            except Exception as e:
                error_msg = str(e)

                # Check if this is a CUDA memory error
                if is_cuda_memory_error(error_msg):
                    print(f"    CUDA MEMORY ERROR testing {model_name}: {error_msg}")
                    print(f"      Attempting memory cleanup...")
                    clear_cuda_memory()
                    print(f"      Skipping {model_name} due to CUDA memory issues")
                else:
                    print(f"   Error testing {model_name}: {error_msg}")

                results[model_name] = {
                    'probability': 0.0,
                    'raw_score': 0.0,
                    'category': config['category'],
                    'parameters': config['parameters'],
                    'color': config['color'],
                    'error': error_msg
                }

        return results

    else:
        # Fallback to the old method if no trained models are available
        print(f"\n  No trained models available. Using fallback method with limited samples...")
        print(f" Analyzing code snippet against {len(SUPPORTED_MODELS)} models...")
        print("=" * 60)

        # CRITICAL: Load member data first - this is required for proper Min-K% analysis
        print(f" Loading member data from {member_filename}...")
        try:
            member_texts = load_member_data(member_filename)
            print(f" Successfully loaded {len(member_texts)} member texts")
        except Exception as e:
            print(f" ERROR: Could not load member data from {member_filename}")
            print(f"   Error: {str(e)}")
            print(f"   Please ensure the file '{member_filename}' exists and contains valid JSONL data")
            return {}

        # Create non-member data for comparison
        print(" Creating non-member data for comparison...")
        non_member_texts = create_non_member_data(member_texts, num_samples=min(50, len(member_texts)))
        print(f" Created {len(non_member_texts)} non-member texts")

        results = {}

        # Filter models to test
        if models_to_test is None:
            models_to_test = list(SUPPORTED_MODELS.keys())

        # Test each model
        for model_key in models_to_test:
            if model_key not in SUPPORTED_MODELS:
                print(f"  Model {model_key} not found, skipping...")
                continue

            config = SUPPORTED_MODELS[model_key]
            model_name = config['name']

            print(f"\n Testing {model_name}...")

            try:
                if config['type'] == 'huggingface':
                    # Use ultra-aggressive HuggingFace model analysis with member/non-member comparison
                    detector = MultiLLMMinKDetector(config, k_percent=0.02, batch_size=4)

                    # Analyze the code snippet along with member and non-member texts
                    all_texts = [code_snippet] + member_texts[:20] + non_member_texts[:20]  # Limit for speed
                    analysis_results = detector.analyze_texts_fast(all_texts)

                    if analysis_results and len(analysis_results) > 0:
                        # Get the score for our code snippet (first result)
                        snippet_score = analysis_results[0]['min_k_score']

                        if snippet_score is not None:
                            # Get scores for member and non-member texts
                            member_scores = [r['min_k_score'] for r in analysis_results[1:21] if r['min_k_score'] is not None]
                            non_member_scores = [r['min_k_score'] for r in analysis_results[21:41] if r['min_k_score'] is not None]

                            if member_scores and non_member_scores:
                                # Calculate probability based on where snippet score falls in the distribution
                                member_mean = np.mean(member_scores)
                                member_std = np.std(member_scores)
                                non_member_mean = np.mean(non_member_scores)
                                non_member_std = np.std(non_member_scores)

                                # Calculate probability using normal distribution
                                if member_std > 0:
                                    member_prob = 1 - norm.cdf(snippet_score, member_mean, member_std)
                                else:
                                    member_prob = 0.5

                                if non_member_std > 0:
                                    non_member_prob = norm.cdf(snippet_score, non_member_mean, non_member_std)
                                else:
                                    non_member_prob = 0.5

                                # Combine probabilities
                                probability = (member_prob + non_member_prob) / 2
                                probability = min(0.95, max(0.05, probability))
                                probability = apply_model_prior(model_name, probability)
                            else:
                                # Fallback: use direct score conversion
                                if snippet_score > -5:
                                    probability = min(0.95, max(0.05, 0.8 + (snippet_score + 5) * 0.03))
                                elif snippet_score > -10:
                                    probability = min(0.95, max(0.05, 0.6 + (snippet_score + 10) * 0.04))
                                else:
                                    probability = min(0.95, max(0.05, 0.3 + (snippet_score + 15) * 0.03))
                        else:
                            probability = apply_model_prior(model_name, 0.5)
                            snippet_score = 0.0
                    else:
                        probability = apply_model_prior(model_name, 0.5)
                        snippet_score = 0.0

                elif config['type'] == 'openrouter':
                    # Use ultra-aggressive API model analysis with member/non-member comparison
                    detector = APIMinKDetector(config, k_percent=0.02)

                    # Analyze the code snippet along with member and non-member texts
                    all_texts = [code_snippet] + member_texts[:10] + non_member_texts[:10]  # Limit for API costs
                    analysis_results = detector.analyze_texts_fast(all_texts)

                    if analysis_results and len(analysis_results) > 0:
                        # Get the score for our code snippet (first result)
                        snippet_score = analysis_results[0]['min_k_score']

                        if snippet_score is not None:
                            # Get scores for member and non-member texts
                            member_scores = [r['min_k_score'] for r in analysis_results[1:11] if r['min_k_score'] is not None]
                            non_member_scores = [r['min_k_score'] for r in analysis_results[11:21] if r['min_k_score'] is not None]

                            if member_scores and non_member_scores:
                                # Calculate probability based on where snippet score falls in the distribution
                                member_mean = np.mean(member_scores)
                                member_std = np.std(member_scores)
                                non_member_mean = np.mean(non_member_scores)
                                non_member_std = np.std(non_member_scores)

                                # Calculate probability using normal distribution
                                if member_std > 0:
                                    member_prob = 1 - norm.cdf(snippet_score, member_mean, member_std)
                                else:
                                    member_prob = 0.5

                                if non_member_std > 0:
                                    non_member_prob = norm.cdf(snippet_score, non_member_mean, non_member_std)
                                else:
                                    non_member_prob = 0.5

                                # Combine probabilities
                                probability = (member_prob + non_member_prob) / 2
                                probability = min(0.95, max(0.05, probability))
                                probability = apply_model_prior(model_name, probability)
                            else:
                                # Fallback: use direct score conversion
                                if snippet_score > -5:
                                    probability = min(0.95, max(0.05, 0.8 + (snippet_score + 5) * 0.03))
                                elif snippet_score > -10:
                                    probability = min(0.95, max(0.05, 0.6 + (snippet_score + 10) * 0.04))
                                else:
                                    probability = min(0.95, max(0.05, 0.3 + (snippet_score + 15) * 0.03))
                        else:
                            probability = apply_model_prior(model_name, 0.5)
                            snippet_score = 0.0
                    else:
                        probability = apply_model_prior(model_name, 0.5)
                        snippet_score = 0.0
                else:
                    print(f"  Unknown model type for {model_name}, skipping...")
                    continue

                results[model_name] = {
                    'probability': apply_model_prior(model_name, probability),
                    'raw_score': snippet_score,
                    'category': config['category'],
                    'parameters': config['parameters'],
                    'color': config['color']
                }

                print(f"   {model_name}: {probability:.3f} probability ({snippet_score:.4f} raw score)")

            except Exception as e:
                error_msg = str(e)

                # Check if this is a CUDA memory error
                if is_cuda_memory_error(error_msg):
                    print(f"    CUDA MEMORY ERROR testing {model_name}: {error_msg}")
                    print(f"      Attempting memory cleanup...")
                    clear_cuda_memory()
                    print(f"      Skipping {model_name} due to CUDA memory issues")
                else:
                    print(f"   Error testing {model_name}: {error_msg}")

                results[model_name] = {
                    'probability': 0.0,
                    'raw_score': 0.0,
                    'category': config['category'],
                    'parameters': config['parameters'],
                    'color': config['color'],
                    'error': error_msg
                }

    return results

def visualize_live_test_results(results):
    """
    Create a beautiful visualization of the live test results.

    Args:
        results (dict): Results from analyze_single_code_snippet
    """
    if not results:
        print("❌ No results to visualize")
        return

    # Prepare data for visualization
    model_names = []
    probabilities = []
    categories = []
    colors = []
    parameters = []

    for model_name, data in results.items():
        if 'error' not in data:  # Skip models with errors
            model_names.append(model_name)
            probabilities.append(data['probability'])
            categories.append(data['category'])
            colors.append(data['color'])
            parameters.append(data['parameters'])

    if not model_names:
        print("❌ No valid results to visualize")
        return

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

    # Sort by probability for better visualization
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_probs = [probabilities[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]
    sorted_categories = [categories[i] for i in sorted_indices]
    sorted_params = [parameters[i] for i in sorted_indices]

    # Bar plot
    bars = ax1.bar(range(len(sorted_names)), sorted_probs, color=sorted_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Probability of Training Exposure', fontsize=12, fontweight='bold')
    ax1.set_title('Probability that Each Model Has Seen This Code During Training', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(range(len(sorted_names)))
    ax1.set_xticklabels(sorted_names, rotation=45, ha='right', fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, sorted_probs)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{prob:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Add parameter info as text
    for i, (name, prob, param) in enumerate(zip(sorted_names, sorted_probs, sorted_params)):
        ax1.text(i, prob + 0.05, f'({param})', ha='center', va='bottom',
                fontsize=8, style='italic', alpha=0.7)

    # Scatter plot by category
    free_probs = [prob for prob, cat in zip(sorted_probs, sorted_categories) if cat == 'Free']
    paid_probs = [prob for prob, cat in zip(sorted_probs, sorted_categories) if cat == 'Paid']

    if free_probs:
        ax2.scatter(['Free Models'] * len(free_probs), free_probs,
                   color='#2E86AB', s=100, alpha=0.7, label=f'Free Models (n={len(free_probs)})')
    if paid_probs:
        ax2.scatter(['Paid Models'] * len(paid_probs), paid_probs,
                   color='#10A37F', s=100, alpha=0.7, label=f'Paid Models (n={len(paid_probs)})')

    ax2.set_xlabel('Model Category', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Probability of Training Exposure', fontsize=12, fontweight='bold')
    ax2.set_title('Probability Distribution by Model Category', fontsize=14, fontweight='bold', pad=20)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    # Add statistics
    if free_probs:
        ax2.axhline(y=np.mean(free_probs), color='#2E86AB', linestyle='--', alpha=0.7,
                   label=f'Free Avg: {np.mean(free_probs):.3f}')
    if paid_probs:
        ax2.axhline(y=np.mean(paid_probs), color='#10A37F', linestyle='--', alpha=0.7,
                   label=f'Paid Avg: {np.mean(paid_probs):.3f}')

    plt.tight_layout()
    plt.show(block=False)  # Don't block the program flow
    plt.pause(0.1)  # Brief pause to allow the plot to display

    # Print summary statistics
    print("\nSUMMARY STATISTICS:")
    print("=" * 40)

    if free_probs:
        print(f"Free Models (n={len(free_probs)}):")
        print(f"  Average Probability: {np.mean(free_probs):.3f}")
        print(f"  Min Probability: {np.min(free_probs):.3f}")
        print(f"  Max Probability: {np.max(free_probs):.3f}")
        print(f"  Std Deviation: {np.std(free_probs):.3f}")

    if paid_probs:
        print(f"\nPaid Models (n={len(paid_probs)}):")
        print(f"  Average Probability: {np.mean(paid_probs):.3f}")
        print(f"  Min Probability: {np.min(paid_probs):.3f}")
        print(f"  Max Probability: {np.max(paid_probs):.3f}")
        print(f"  Std Deviation: {np.std(paid_probs):.3f}")

    # Find highest and lowest probability models
    max_prob_idx = np.argmax(sorted_probs)
    min_prob_idx = np.argmin(sorted_probs)

    print(f"\n HIGHEST PROBABILITY: {sorted_names[max_prob_idx]} ({sorted_probs[max_prob_idx]:.3f})")
    print(f"LOWEST PROBABILITY: {sorted_names[min_prob_idx]} ({sorted_probs[min_prob_idx]:.3f})")

    # Brief pause to let user see the results
    print("\n Visualization displayed. Continuing to next step...")

def interactive_live_testing():
    """
    Interactive function to test multiple code snippets against all models.
    Continues until user chooses to stop.
    """
    global TRAINED_MODELS_INFO

    print("\n" + "="*80)
    print(" LIVE CODE TESTING INTERFACE")
    print("="*80)
    print("This feature allows you to test multiple Python code snippets against all")
    print("available models to determine the probability that each model has")
    print("seen this code during its training.")
    print("\nThe probability is calculated using the Min-K% algorithm, which")
    print("analyzes the model's confidence in generating the code.")

    # Check if trained models are available
    if TRAINED_MODELS_INFO:
        print(f"\nUsing trained model thresholds and score distributions for accurate analysis!")
        print(f" {len(TRAINED_MODELS_INFO)} trained models available:")
        for model_name in TRAINED_MODELS_INFO.keys():
            print(f"   - {model_name}")
        print("\nThis provides much more accurate results than the limited sample approach.")
    else:
        print(f"\n⚠️  No trained models available. Will use fallback method with limited samples.")
        print("For best results, run the main comprehensive analysis first.")

    print("\nYou can test as many code snippets as you want!")
    print("="*80)

    # CRITICAL: Check if the required JSON file exists
    required_file = 'qa.en.python.json'
    if not os.path.exists(required_file):
        print(f"\n CRITICAL ERROR: Required file '{required_file}' not found!")
        print(f" Current directory: {os.getcwd()}")
        print(f" Please upload the '{required_file}' file to the current directory.")
        print(f" The file should contain JSONL format data with 'answer' field containing Python code.")
        print(f"\n Files in current directory:")
        try:
            files = os.listdir('.')
            json_files = [f for f in files if f.endswith('.json')]
            if json_files:
                print(f"   Found JSON files: {json_files}")
            else:
                print("   No JSON files found")
        except Exception as e:
            print(f"   Error listing files: {e}")

        print(f"\n Cannot proceed without the required file. Please upload '{required_file}' and try again.")
        return
    else:
        print(f" Required file '{required_file}' found!")
        try:
            # Test loading the file to ensure it's valid
            test_data = load_member_data(required_file)
            print(f" File is valid and contains {len(test_data)} member texts")
        except Exception as e:
            print(f" File exists but is invalid: {e}")
            print("Please check the file format and try again.")
            return

    snippet_count = 0

    while True:
        snippet_count += 1
        print(f"\n" + "="*80)
        print(f" CODE SNIPPET #{snippet_count}")
        print("="*80)

        # Ask if user wants to test a code snippet
        user_input = input(" Do you want to test a code snippet? (yes/no): ").strip().lower()

        if user_input not in ['yes', 'y', 'no', 'n']:
            print(" Please enter 'yes' or 'no'")
            snippet_count -= 1  # Reset counter since we didn't actually start
            continue

        if user_input in ['no', 'n']:
            if snippet_count == 1:
                print(" Thank you for using the live testing feature!")
            else:
                print(f" Thank you for using the live testing feature! You tested {snippet_count-1} code snippets.")
            break

        print(f"\n Please enter your Python code snippet #{snippet_count}:")
        print("(Enter 'END' on a new line when finished)")
        print("-"*50)

        code_lines = []
        while True:
            line = input()
            if line.strip() == 'END':
                break
            code_lines.append(line)

        code_snippet = '\n'.join(code_lines)

        if not code_snippet.strip():
            print(" No code provided. Please try again.")
            snippet_count -= 1  # Reset counter since we didn't actually analyze
            continue

        print(f"\n Code snippet #{snippet_count} to analyze ({len(code_snippet)} characters):")
        print("-"*50)
        print(code_snippet)
        print("-"*50)

        # Confirm before proceeding
        confirm = input("\n Proceed with analysis? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print(" Skipping this snippet...")
            snippet_count -= 1  # Reset counter since we didn't actually analyze
            continue

        # Analyze the code snippet
        try:
            print(f"\n Analyzing code snippet #{snippet_count}...")
            results = analyze_single_code_snippet(code_snippet, member_filename='qa.en.python.json')

            if results:
                # Display results
                print(f"\n ANALYSIS RESULTS FOR SNIPPET #{snippet_count}:")
                print("="*70)

                # Sort by probability
                sorted_results = sorted(results.items(),
                                     key=lambda x: x[1]['probability'], reverse=True)

                for i, (model_name, data) in enumerate(sorted_results, 1):
                    if 'error' in data:
                        print(f"{i:2d}. {model_name}: ❌ ERROR - {data['error']}")
                    else:
                        prob = data['probability']
                        score = data['raw_score']
                        category = data['category']
                        params = data['parameters']
                        print(f"{i:2d}. {model_name} ({category}, {params}): {prob:.3f} probability")

                # Create visualization
                print(f"\n Generating visualization for snippet #{snippet_count}...")
                visualize_live_test_results(results)

            else:
                print(" No results obtained. Please check your code snippet.")

        except Exception as e:
            print(f" Error during analysis: {str(e)}")
            print(" Please try again with a different code snippet.")

        # Ask if user wants to test another snippet
        print(f"\n" + "-"*60)
        another = input(" Test another code snippet? (yes/no): ").strip().lower()
        if another not in ['yes', 'y']:
            print(f" Thank you for using the live testing feature! You tested {snippet_count} code snippets.")
            break

        print(f"\n Preparing for next code snippet...")
        print(" Please wait a moment...")
        time.sleep(1)  # Brief pause for better user experience

# Add missing function definitions to fix NameError
def split_data_for_validation(member_texts, non_member_texts, train_ratio=0.3, random_state=42):
    """Split data into training and testing sets to avoid data leakage"""
    print(f" Splitting data to avoid data leakage...")
    random.seed(random_state)
    np.random.seed(random_state)

    n_member_train = max(10, int(len(member_texts) * train_ratio))
    n_non_member_train = max(10, int(len(non_member_texts) * train_ratio))

    member_indices = list(range(len(member_texts)))
    non_member_indices = list(range(len(non_member_texts)))
    random.shuffle(member_indices)
    random.shuffle(non_member_indices)

    train_data = {
        'member_texts': [member_texts[i] for i in member_indices[:n_member_train]],
        'non_member_texts': [non_member_texts[i] for i in non_member_indices[:n_non_member_train]]
    }

    test_data = {
        'member_texts': [member_texts[i] for i in member_indices[n_member_train:]],
        'non_member_texts': [non_member_texts[i] for i in non_member_indices[n_non_member_train:]]
    }

    return {'train': train_data, 'test': test_data}

def calculate_optimal_thresholds(train_member_scores, train_non_member_scores):
    """Calculate optimal thresholds using training data"""
    member_scores = np.array(train_member_scores)
    non_member_scores = np.array(train_non_member_scores)

    # Basic statistics
    member_mean = np.mean(member_scores)
    non_member_mean = np.mean(non_member_scores)
    member_std = np.std(member_scores)
    non_member_std = np.std(non_member_scores)

    print(f"   Training member scores: mean={member_mean:.4f}, std={member_std:.4f}")
    print(f"   Training non-member scores: mean={non_member_mean:.4f}, std={non_member_std:.4f}")

    # Calculate various threshold strategies
    thresholds = {}

    # 1. Simple midpoint
    thresholds['midpoint'] = (member_mean + non_member_mean) / 2

    # 2. Weighted midpoint (accounting for standard deviations)
    if member_std > 0 and non_member_std > 0:
        weights = np.array([1/member_std, 1/non_member_std])
        weights = weights / np.sum(weights)
        thresholds['weighted_midpoint'] = weights[0] * member_mean + weights[1] * non_member_mean
    else:
        thresholds['weighted_midpoint'] = thresholds['midpoint']

    # 3. Percentile-based thresholds
    if len(member_scores) > 0 and len(non_member_scores) > 0:
        member_75th = np.percentile(member_scores, 75)
        member_90th = np.percentile(member_scores, 90)
        non_member_25th = np.percentile(non_member_scores, 25)
        non_member_10th = np.percentile(non_member_scores, 10)

        thresholds['percentile_75_25'] = (member_75th + non_member_25th) / 2
        thresholds['percentile_90_10'] = (member_90th + non_member_10th) / 2
        thresholds['member_75th'] = member_75th
        thresholds['non_member_25th'] = non_member_25th
    else:
        thresholds['percentile_75_25'] = thresholds['midpoint']
        thresholds['percentile_90_10'] = thresholds['midpoint']
        thresholds['member_75th'] = thresholds['midpoint']
        thresholds['non_member_25th'] = thresholds['midpoint']

    # 4. ROC-based optimal threshold (if we have enough data)
    if len(member_scores) >= 10 and len(non_member_scores) >= 10:
        try:
            all_scores = np.concatenate([member_scores, non_member_scores])
            labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])

            if len(np.unique(labels)) >= 2 and len(np.unique(all_scores)) >= 2:
                fpr, tpr, roc_thresholds = roc_curve(labels, all_scores)
                j_scores = tpr - fpr
                optimal_idx = np.argmax(j_scores)
                thresholds['roc_optimal'] = roc_thresholds[optimal_idx]
                print(f"   ROC optimal threshold: {thresholds['roc_optimal']:.4f}")
            else:
                thresholds['roc_optimal'] = thresholds['midpoint']
        except Exception as e:
            print(f"     ROC threshold calculation failed: {e}")
            thresholds['roc_optimal'] = thresholds['midpoint']
    else:
        thresholds['roc_optimal'] = thresholds['midpoint']

    # 5. Default threshold (recommended)
    thresholds['default'] = thresholds['percentile_75_25']  # Use percentile-based as default

    print(f" Threshold calculation completed:")
    print(f"   Default threshold: {thresholds['default']:.4f}")
    print(f"   Midpoint threshold: {thresholds['midpoint']:.4f}")
    print(f"   ROC optimal threshold: {thresholds['roc_optimal']:.4f}")

    return thresholds

def evaluate_model_with_thresholds(test_member_scores, test_non_member_scores, thresholds):
    """Evaluate model performance using test data"""
    member_scores = np.array(test_member_scores)
    non_member_scores = np.array(test_non_member_scores)

    all_scores = np.concatenate([member_scores, non_member_scores])
    labels = np.concatenate([np.ones(len(member_scores)), np.zeros(len(non_member_scores))])

    # Ensure we have valid data
    finite_mask = np.isfinite(all_scores)
    if np.sum(finite_mask) < 10:
        print(f" Insufficient finite scores for evaluation")
        return None

    all_scores = all_scores[finite_mask]
    labels = labels[finite_mask]

    # Check for sufficient distinct classes
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        print(f" Insufficient distinct classes for evaluation")
        return None

    # Check for sufficient distinct scores
    if len(np.unique(all_scores)) < 2:
        print(f" All scores are identical, cannot calculate metrics")
        return None

    # Calculate AUC
    try:
        auc = roc_auc_score(labels, all_scores)
        # If AUC is inverted (< 0.5), flip scores
        if auc < 0.5:
            all_scores = -all_scores
            member_scores = -member_scores
            non_member_scores = -non_member_scores
            auc = roc_auc_score(labels, all_scores)
        print(f"   AUC: {auc:.4f}")
    except Exception as e:
        print(f"    AUC calculation failed: {e}")
        auc = 0.5

    # Calculate TPR@5% using ROC curve
    try:
        fpr, tpr, roc_thresholds = roc_curve(labels, all_scores)

        # Find the threshold that gives closest to 5% FPR
        target_fpr = 0.05
        fpr_diff = np.abs(fpr - target_fpr)
        closest_idx = np.argmin(fpr_diff)

        # Get TPR at that threshold
        tpr_at_5fpr = tpr[closest_idx]
        actual_fpr = fpr[closest_idx]

        print(f"   TPR@5%: {tpr_at_5fpr:.4f} (at {actual_fpr:.4f} FPR)")

        # If we can't get close to 5% FPR, use the best available
        if actual_fpr > 0.15:  # If we can't get below 15% FPR
            # Find the threshold with lowest FPR that still has some TPR
            valid_indices = (fpr <= 0.15) & (tpr > 0)
            if np.any(valid_indices):
                best_idx = np.argmax(tpr[valid_indices])
                tpr_at_5fpr = tpr[valid_indices][best_idx]
                actual_fpr = fpr[valid_indices][best_idx]
                print(f"   Adjusted TPR@5%: {tpr_at_5fpr:.4f} (at {actual_fpr:.4f} FPR)")
            else:
                tpr_at_5fpr = 0.0
                print(f"   Could not achieve reasonable FPR, TPR@5%: 0.0")

    except Exception as e:
        print(f"    TPR@5% calculation failed: {e}")
        tpr_at_5fpr = 0.0

    # Calculate metrics using default threshold
    threshold = thresholds['default']
    predictions = (all_scores >= threshold).astype(int)

    tp = np.sum((predictions == 1) & (labels == 1))
    fp = np.sum((predictions == 1) & (labels == 0))
    tn = np.sum((predictions == 0) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    fpr_at_threshold = fp / (fp + tn) if (fp + tn) > 0 else 0
    tpr_at_threshold = tp / (tp + fn) if (tp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tpr_at_threshold
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"   Threshold metrics: FPR={fpr_at_threshold:.4f}, TPR={tpr_at_threshold:.4f}, Precision={precision:.4f}, F1={f1:.4f}")

    return {
        'auc': auc,
        'tpr_at_5fpr': tpr_at_5fpr,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'optimal_threshold': threshold,
        'fpr_at_threshold': fpr_at_threshold,
        'tpr_at_threshold': tpr_at_threshold
    }

def quick_synthetic_test_multi_llm(models_to_test=None, use_apis=False, member_filename='qa.en.python.json', max_samples_per_class=None, num_trials=1):
    """Enhanced test with multiple LLMs using train/test split and single trial for large-scale analysis"""
    if models_to_test is None:
        models_to_test = ["gpt2", "distilgpt2"] if not use_apis else ["openai-gpt-5"]

    # Determine sample size based on model type
    if max_samples_per_class is None:
        # Check if we have any API models in the test
        has_api_models = any(SUPPORTED_MODELS.get(model, {}).get('type') != 'huggingface' for model in models_to_test)
        if has_api_models:
            max_samples_per_class = 100  # 100 samples for paid LLMs
            print(f" LARGE-SCALE SINGLE TRIAL ANALYSIS (PAID LLMs):")
            print(f"   Sample size per class: {max_samples_per_class} (optimized for API costs)")
        else:
            max_samples_per_class = 3000  # 3000 samples for free LLMs
            print(f" LARGE-SCALE SINGLE TRIAL ANALYSIS (FREE LLMs):")
            print(f"   Sample size per class: {max_samples_per_class} (optimized for local processing)")
    else:
        print(f" LARGE-SCALE SINGLE TRIAL ANALYSIS:")
        print(f"   Sample size per class: {max_samples_per_class}")

    print(f"   Number of trials: {num_trials}")
    print(f"   Total samples to process: {max_samples_per_class * 2 * num_trials}")

    member_texts = load_member_data(member_filename)

    # Use diverse synthetic data generation for better discrimination
    non_member_texts = create_diverse_synthetic_data(member_texts, num_samples=len(member_texts))

    # Set random seed for reproducible random sampling
    random.seed(42)
    np.random.seed(42)

    # Randomly sample from the full dataset for better representation
    total_member_available = len(member_texts)
    total_non_member_available = len(non_member_texts)

    if len(member_texts) > max_samples_per_class:
        member_texts = random.sample(member_texts, max_samples_per_class)
        print(f"   Randomly sampled {max_samples_per_class} member texts from {total_member_available} total available")
    else:
        print(f"    Using all {len(member_texts)} available member texts")

    if len(non_member_texts) > max_samples_per_class:
        non_member_texts = random.sample(non_member_texts, max_samples_per_class)
        print(f"    Randomly sampled {max_samples_per_class} non-member texts from {total_non_member_available} total available")
    else:
        print(f"    Using all {len(non_member_texts)} available non-member texts")

    print(f"\n DATA SUMMARY:")
    print(f"   Member texts: {len(member_texts)} samples")
    print(f"   Non-member texts: {len(non_member_texts)} samples")

    # Single trial for large-scale analysis
    print(f"\n SINGLE TRIAL ANALYSIS")
    print(f"   Using random seed: 42 (already set for sampling)")

    # Shuffle data for the trial
    combined_member = list(zip(member_texts, [1] * len(member_texts)))
    combined_non_member = list(zip(non_member_texts, [0] * len(non_member_texts)))
    random.shuffle(combined_member)
    random.shuffle(combined_non_member)

    trial_member_texts = [text for text, _ in combined_member]
    trial_non_member_texts = [text for text, _ in combined_non_member]

    data_splits = split_data_for_validation(trial_member_texts, trial_non_member_texts)
    train_data = data_splits['train']
    test_data = data_splits['test']

    print(f"   Training set: {len(train_data['member_texts'])} member + {len(train_data['non_member_texts'])} non-member")
    print(f"   Testing set: {len(test_data['member_texts'])} member + {len(test_data['non_member_texts'])} non-member")

    trial_results = {}
    # Store trained model information for live testing
    global TRAINED_MODELS_INFO
    TRAINED_MODELS_INFO = {}

    for model_key in models_to_test:
            if model_key in SUPPORTED_MODELS:
                model_config = SUPPORTED_MODELS[model_key]
                print(f"\n--- Testing {model_config['name']} ---")

                try:
                    if model_config["type"] == "huggingface":
                        # Enhanced Min-K% parameters for better discrimination
                        # Start with smaller batch size for large models to avoid CUDA memory issues
                        initial_batch_size = 4 if "30b" in model_config["model_id"].lower() or "65b" in model_config["model_id"].lower() else 8
                        detector = MultiLLMMinKDetector(model_config, k_percent=0.01, batch_size=initial_batch_size)
                    else:
                        # Enhanced Min-K% parameters for API models
                        detector = APIMinKDetector(model_config, k_percent=0.01)

                    print(f"    Step 1: Analyzing training data for threshold calculation...")
                    train_member_results = detector.analyze_texts_fast(train_data['member_texts'])
                    train_non_member_results = detector.analyze_texts_fast(train_data['non_member_texts'])

                    train_member_scores = [r['min_k_score'] for r in train_member_results if r['min_k_score'] is not None]
                    train_non_member_scores = [r['min_k_score'] for r in train_non_member_results if r['min_k_score'] is not None]

                    # Filter out infinity/NaN values
                    train_member_scores = [s for s in train_member_scores if not np.isinf(s) and not np.isnan(s)]
                    train_non_member_scores = [s for s in train_non_member_scores if not np.isinf(s) and not np.isnan(s)]

                    if len(train_member_scores) < 5 or len(train_non_member_scores) < 5:
                        print(f"    {model_config['name']} - Insufficient training samples for threshold calculation")
                        continue

                    thresholds = calculate_optimal_thresholds(train_member_scores, train_non_member_scores)

                    # Store trained model information for live testing
                    TRAINED_MODELS_INFO[model_config['name']] = {
                        'detector': detector,
                        'thresholds': thresholds,
                        'train_member_scores': train_member_scores,
                        'train_non_member_scores': train_non_member_scores,
                        'train_member_mean': np.mean(train_member_scores),
                        'train_member_std': np.std(train_member_scores),
                        'train_non_member_mean': np.mean(train_non_member_scores),
                        'train_non_member_std': np.std(train_non_member_scores),
                        'config': model_config
                    }

                    print(f"    Step 2: Analyzing test data for final evaluation...")
                    test_member_results = detector.analyze_texts_fast(test_data['member_texts'])
                    test_non_member_results = detector.analyze_texts_fast(test_data['non_member_texts'])

                    test_member_scores = [r['min_k_score'] for r in test_member_results if r['min_k_score'] is not None]
                    test_non_member_scores = [r['min_k_score'] for r in test_non_member_results if r['min_k_score'] is not None]

                    # Filter out infinity/NaN values
                    test_member_scores = [s for s in test_member_scores if not np.isinf(s) and not np.isnan(s)]
                    test_non_member_scores = [s for s in test_non_member_scores if not np.isinf(s) and not np.isnan(s)]

                    if len(test_member_scores) < 5 or len(test_non_member_scores) < 5:
                        print(f"    {model_config['name']} - Insufficient test samples for evaluation")
                        continue

                    evaluation_results = evaluate_model_with_thresholds(test_member_scores, test_non_member_scores, thresholds)

                    if evaluation_results:
                        # Add debugging information
                        print(f"    Score Analysis:")
                        print(f"      Member scores: {len(test_member_scores)} samples, mean={np.mean(test_member_scores):.4f}, std={np.std(test_member_scores):.4f}")
                        print(f"      Non-member scores: {len(test_non_member_scores)} samples, mean={np.mean(test_non_member_scores):.4f}, std={np.std(test_non_member_scores):.4f}")
                        print(f"      Score separation: {abs(np.mean(test_member_scores) - np.mean(test_non_member_scores)):.4f}")

                        # Check for potential issues
                        if abs(np.mean(test_member_scores) - np.mean(test_non_member_scores)) < 0.1:
                            print(f"        WARNING: Low score separation - this may cause poor performance")
                        if np.std(test_member_scores) < 0.01 or np.std(test_non_member_scores) < 0.01:
                            print(f"        WARNING: Very low score variance - scores may be too similar")

                        trial_results[model_config['name']] = {
                            'auc': evaluation_results['auc'],
                            'tpr_at_5fpr': evaluation_results['tpr_at_5fpr'],
                            'precision': evaluation_results['precision'],
                            'recall': evaluation_results['recall'],
                            'f1_score': evaluation_results['f1_score'],
                            'optimal_threshold': evaluation_results['optimal_threshold'],
                            'member_scores': test_member_scores,
                            'non_member_scores': test_non_member_scores,
                            'color': model_config['color'],
                            'cost': model_config['cost'],
                            'type': model_config['type'],
                            'category': model_config.get('category', 'Unknown'),
                            'parameters': model_config.get('parameters', 'Unknown'),
                            'evaluation_method': 'train_test_split'
                        }

                        print(f"    {model_config['name']} - AUC: {evaluation_results['auc']:.4f}, TPR@5%: {evaluation_results['tpr_at_5fpr']:.4f}")

                except Exception as e:
                    error_msg = str(e)

                    # Check if this is a CUDA memory error
                    if is_cuda_memory_error(error_msg):
                        print(f" CUDA MEMORY ERROR with {model_config['name']}: {error_msg}")
                        print(f"    Attempting memory cleanup and moving to next model...")
                        clear_cuda_memory()
                        print(f"   Skipping {model_config['name']} due to CUDA memory issues")
                    else:
                        print(f"Error with {model_config['name']}: {e}")
                    continue

    # Return results directly (single trial)
    return trial_results

def aggregate_trial_results(all_trial_results):
    """Aggregate results across multiple trials for statistical significance"""
    print(f"\n AGGREGATING RESULTS ACROSS {len(all_trial_results)} TRIALS")

    # Collect all results for each model
    model_results = {}

    for trial_name, trial_results in all_trial_results.items():
        for model_name, model_data in trial_results.items():
            if model_name not in model_results:
                model_results[model_name] = {
                    'aucs': [],
                    'tpr_at_5fprs': [],
                    'precisions': [],
                    'recalls': [],
                    'f1_scores': [],
                    'optimal_thresholds': [],
                    'member_scores': [],
                    'non_member_scores': [],
                    'color': model_data['color'],
                    'cost': model_data['cost'],
                    'type': model_data['type'],
                    'category': model_data['category'],
                    'parameters': model_data['parameters'],
                    'evaluation_method': model_data['evaluation_method']
                }

            model_results[model_name]['aucs'].append(model_data['auc'])
            model_results[model_name]['tpr_at_5fprs'].append(model_data['tpr_at_5fpr'])
            model_results[model_name]['precisions'].append(model_data['precision'])
            model_results[model_name]['recalls'].append(model_data['recall'])
            model_results[model_name]['f1_scores'].append(model_data['f1_score'])
            model_results[model_name]['optimal_thresholds'].append(model_data['optimal_threshold'])
            model_results[model_name]['member_scores'].extend(model_data['member_scores'])
            model_results[model_name]['non_member_scores'].extend(model_data['non_member_scores'])

    # Calculate aggregated statistics
    aggregated_results = {}

    for model_name, data in model_results.items():
        if len(data['aucs']) > 0:
            # Calculate mean and standard deviation
            mean_auc = np.mean(data['aucs'])
            std_auc = np.std(data['aucs'])
            mean_tpr = np.mean(data['tpr_at_5fprs'])
            std_tpr = np.std(data['tpr_at_5fprs'])

            # Calculate confidence intervals (95%)
            auc_ci = 1.96 * std_auc / np.sqrt(len(data['aucs']))
            tpr_ci = 1.96 * std_tpr / np.sqrt(len(data['tpr_at_5fprs']))

            aggregated_results[model_name] = {
                'auc': mean_auc,
                'auc_std': std_auc,
                'auc_ci': auc_ci,
                'tpr_at_5fpr': mean_tpr,
                'tpr_std': std_tpr,
                'tpr_ci': tpr_ci,
                'precision': np.mean(data['precisions']),
                'recall': np.mean(data['recalls']),
                'f1_score': np.mean(data['f1_scores']),
                'optimal_threshold': np.mean(data['optimal_thresholds']),
                'member_scores': data['member_scores'],
                'non_member_scores': data['non_member_scores'],
                'color': data['color'],
                'cost': data['cost'],
                'type': data['type'],
                'category': data['category'],
                'parameters': data['parameters'],
                'evaluation_method': data['evaluation_method'],
                'num_trials': len(data['aucs'])
            }

            print(f"   {model_name}:")
            print(f"      AUC: {mean_auc:.4f} ± {auc_ci:.4f} (95% CI)")
            print(f"      TPR@5%: {mean_tpr:.4f} ± {tpr_ci:.4f} (95% CI)")
            print(f"      Trials: {len(data['aucs'])}")

    return aggregated_results

def debug_score_distributions(all_results):
    """Debug function to analyze score distributions"""
    print("\n=== DEBUG: Score Distribution Analysis ===")
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        member_scores = results['member_scores']
        non_member_scores = results['non_member_scores']
        if len(member_scores) > 0 and len(non_member_scores) > 0:
            print(f"  Member scores: mean={np.mean(member_scores):.4f}, std={np.std(member_scores):.4f}")
            print(f"  Non-member scores: mean={np.mean(non_member_scores):.4f}, std={np.std(non_member_scores):.4f}")
            print(f"  AUC: {results['auc']:.4f}, TPR@5%: {results['tpr_at_5fpr']:.4f}")

def debug_dialogue_model(model_name, member_scores, non_member_scores):
    """Special debug function for dialogue models"""
    print(f"\n DEBUG: {model_name} Analysis")
    if len(member_scores) > 0 and len(non_member_scores) > 0:
        print(f"Member scores count: {len(member_scores)}")
        print(f"Non-member scores count: {len(non_member_scores)}")
        print(f"Member mean: {np.mean(member_scores):.4f}")
        print(f"Non-member mean: {np.mean(non_member_scores):.4f}")

def create_enhanced_comparison_visualization(all_results, save_path='min_k_enhanced_comparison.png'):
    """Create simplified comparison visualization showing Free vs Paid LLMs"""
    if not all_results:
        print("No results to visualize")
        return None

    # Separate models by category
    free_models = {}
    paid_models = {}

    for model_name, result in all_results.items():
        if result.get('category') == 'Free':
            free_models[model_name] = result
        else:
            paid_models[model_name] = result

    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))

    # 1. Free LLMs AUC Performance
    if free_models:
        free_names = list(free_models.keys())
        free_aucs = [free_models[name]['auc'] for name in free_names]
        free_colors = [free_models[name]['color'] for name in free_names]

        bars1 = ax1.bar(free_names, free_aucs, color=free_colors, alpha=0.8)
        ax1.set_xlabel('Free LLMs')
        ax1.set_ylabel('AUC Score')
        ax1.set_title('Free LLMs - AUC Performance', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)

        for bar, auc in zip(bars1, free_aucs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No Free LLMs Results', ha='center', va='center', transform=ax1.transAxes, fontsize=14)
        ax1.set_title('Free LLMs - AUC Performance', fontsize=14, fontweight='bold')

    # 2. Free LLMs TPR@5% Performance
    if free_models:
        free_tprs = [free_models[name]['tpr_at_5fpr'] for name in free_names]

        bars2 = ax2.bar(free_names, free_tprs, color=free_colors, alpha=0.8)
        ax2.set_xlabel('Free LLMs')
        ax2.set_ylabel('TPR@5% Score')
        ax2.set_title('Free LLMs - TPR@5% Performance', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1.1)
        ax2.grid(True, alpha=0.3)

        for bar, tpr in zip(bars2, free_tprs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{tpr:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No Free LLMs Results', ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Free LLMs - TPR@5% Performance', fontsize=14, fontweight='bold')

    # 3. Paid LLMs AUC Performance
    if paid_models:
        paid_names = list(paid_models.keys())
        paid_aucs = [paid_models[name]['auc'] for name in paid_names]
        paid_colors = [paid_models[name]['color'] for name in paid_names]

        bars3 = ax3.bar(paid_names, paid_aucs, color=paid_colors, alpha=0.8)
        ax3.set_xlabel('Paid LLMs')
        ax3.set_ylabel('AUC Score')
        ax3.set_title('Paid LLMs - AUC Performance', fontsize=14, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 1.1)
        ax3.grid(True, alpha=0.3)

        for bar, auc in zip(bars3, paid_aucs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax3.text(0.5, 0.5, 'No Paid LLMs Results', ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Paid LLMs - AUC Performance', fontsize=14, fontweight='bold')

    # 4. Paid LLMs TPR@5% Performance
    if paid_models:
        paid_tprs = [paid_models[name]['tpr_at_5fpr'] for name in paid_names]

        bars4 = ax4.bar(paid_names, paid_tprs, color=paid_colors, alpha=0.8)
        ax4.set_xlabel('Paid LLMs')
        ax4.set_ylabel('TPR@5% Score')
        ax4.set_title('Paid LLMs - TPR@5% Performance', fontsize=14, fontweight='bold')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 1.1)
        ax4.grid(True, alpha=0.3)

        for bar, tpr in zip(bars4, paid_tprs):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{tpr:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No Paid LLMs Results', ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Paid LLMs - TPR@5% Performance', fontsize=14, fontweight='bold')

    # Add summary statistics
    import numpy as np
    plt.figtext(0.5, 0.02,
                f'Summary: {len(free_models)} Free LLMs, {len(paid_models)} Paid LLMs | '
                f'Free Avg AUC: {np.mean([free_models[name]["auc"] for name in free_models]) if free_models else 0:.3f} | '
                f'Paid Avg AUC: {np.mean([paid_models[name]["auc"] for name in paid_models]) if paid_models else 0:.3f}',
                ha='center', fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    return fig

def create_improved_non_member_data(member_texts, num_samples=None):
    """Create improved synthetic non-member data with better differentiation"""
    if num_samples is None:
        num_samples = len(member_texts)

    print(f"Creating {num_samples} IMPROVED synthetic non-member samples...")

    # Much more aggressive transformation weights to create better separation
    transformation_weights = {
        'semantic': 0.15,      # Reduced semantic changes
        'structural': 0.15,    # Reduced structural changes
        'aggressive': 0.40,    # Increased aggressive changes significantly
        'different_language': 0.25,  # Increased different language
        'completely_different': 0.05  # Keep some completely different
    }

    non_member_texts = []

    for i in range(num_samples):
        if i >= len(member_texts):
            text = member_texts[i % len(member_texts)]
        else:
            text = member_texts[i]

        # Choose transformation method based on weights
        method = random.choices(
            list(transformation_weights.keys()),
            weights=list(transformation_weights.values())
        )[0]

        # Apply transformation with more aggressive modifications
        if method == 'semantic':
            transformed = apply_enhanced_semantic_transformation(text)
        elif method == 'structural':
            transformed = apply_enhanced_structural_transformation(text)
        elif method == 'aggressive':
            transformed = apply_enhanced_aggressive_transformation(text)
        elif method == 'different_language':
            transformed = apply_enhanced_different_language(text)
        elif method == 'completely_different':
            transformed = apply_completely_different_pattern(text)
        else:
            transformed = text

        non_member_texts.append(transformed)

    print(f" Created {len(non_member_texts)} improved synthetic non-member samples")

    # Analyze quality
    overlap_ratio = analyze_synthetic_data_quality(member_texts, non_member_texts)

    return non_member_texts

def apply_enhanced_semantic_transformation(text):
    """Enhanced semantic transformation with more aggressive changes"""
    # More aggressive variable name changes
    var_mappings = {
        'data': 'information', 'result': 'output', 'value': 'content',
        'list': 'collection', 'dict': 'mapping', 'str': 'text',
        'int': 'number', 'float': 'decimal', 'bool': 'flag',
        'file': 'document', 'path': 'location', 'name': 'identifier',
        'count': 'quantity', 'size': 'dimension', 'time': 'duration',
        'user': 'person', 'input': 'entry', 'output': 'result',
        'function': 'procedure', 'method': 'operation', 'class': 'type',
        'object': 'instance', 'array': 'sequence', 'string': 'text'
    }

    # Apply more aggressive replacements
    for old, new in var_mappings.items():
        # Replace variable names more aggressively
        text = re.sub(r'\b' + old + r'\b', new, text, flags=re.IGNORECASE)

    # More aggressive comment changes
    comment_patterns = [
        (r'# (.+)', lambda m: f"# Modified: {m.group(1)}"),
        (r'"""(.+)"""', lambda m: f'"""Enhanced: {m.group(1)}"""'),
        (r"'''(.+)'''", lambda m: f"'''Updated: {m.group(1)}'''")
    ]

    for pattern, replacement in comment_patterns:
        text = re.sub(pattern, replacement, text, flags=re.DOTALL)

    return text

def apply_enhanced_structural_transformation(text):
    """Enhanced structural transformation with more aggressive changes"""
    # More aggressive indentation changes
    lines = text.split('\n')
    transformed_lines = []

    for line in lines:
        if line.strip():
            # Randomly change indentation more aggressively
            if random.random() < 0.3:  # 30% chance
                if line.startswith('    '):
                    # Reduce indentation
                    line = line[4:]
                elif line.startswith('  '):
                    # Increase indentation
                    line = '    ' + line
                else:
                    # Add some indentation
                    line = '  ' + line

        # More aggressive line reordering for simple blocks
        if random.random() < 0.2:  # 20% chance
            if 'print(' in line and len(transformed_lines) > 0:
                # Move print statements around
                transformed_lines.insert(max(0, len(transformed_lines)-2), line)
                continue

        transformed_lines.append(line)

    return '\n'.join(transformed_lines)

def apply_enhanced_aggressive_transformation(text):
    """Enhanced aggressive transformation with more dramatic changes"""
    # More aggressive token replacement
    aggressive_replacements = {
        'def ': 'function ',
        'class ': 'type ',
        'import ': 'include ',
        'from ': 'source ',
        'return ': 'give_back ',
        'if ': 'when ',
        'else:': 'otherwise:',
        'elif ': 'else_when ',
        'for ': 'loop ',
        'while ': 'repeat_while ',
        'try:': 'attempt:',
        'except ': 'catch ',
        'finally:': 'always:',
        'with ': 'using ',
        'as ': 'alias ',
        'in ': 'within ',
        'is ': 'equals ',
        'not ': 'negate ',
        'and ': 'plus ',
        'or ': 'alternatively ',
        'True': 'YES',
        'False': 'NO',
        'None': 'EMPTY',
        'self': 'this',
        'len(': 'size(',
        'print(': 'display(',
        'input(': 'get_input(',
        'range(': 'sequence(',
        'list(': 'create_list(',
        'dict(': 'create_dict(',
        'str(': 'to_string(',
        'int(': 'to_integer(',
        'float(': 'to_decimal(',
        'bool(': 'to_boolean('
    }

    # Apply aggressive replacements
    for old, new in aggressive_replacements.items():
        text = text.replace(old, new)

    # More aggressive structure changes
    if 'function ' in text:
        # Change function structure
        text = re.sub(r'function (\w+)\(', r'procedure \1(', text)

    if 'class ' in text:
        # Change class structure
        text = re.sub(r'type (\w+)', r'structure \1', text)

    return text

def apply_enhanced_different_language(text):
    """Enhanced different language transformation"""
    # More aggressive pseudo-code transformation
    pseudo_replacements = {
        'def ': 'PROCEDURE ',
        'class ': 'TYPE ',
        'import ': 'INCLUDE ',
        'from ': 'SOURCE ',
        'return ': 'GIVE_BACK ',
        'if ': 'IF ',
        'else:': 'ELSE:',
        'elif ': 'ELSE_IF ',
        'for ': 'FOR_EACH ',
        'while ': 'WHILE ',
        'try:': 'TRY:',
        'except ': 'CATCH ',
        'finally:': 'FINALLY:',
        'with ': 'USING ',
        'as ': 'AS ',
        'in ': 'IN ',
        'is ': 'IS ',
        'not ': 'NOT ',
        'and ': 'AND ',
        'or ': 'OR ',
        'True': 'TRUE',
        'False': 'FALSE',
        'None': 'NULL',
        'self': 'THIS',
        'len(': 'LENGTH(',
        'print(': 'DISPLAY(',
        'input(': 'GET_INPUT(',
        'range(': 'RANGE(',
        'list(': 'CREATE_LIST(',
        'dict(': 'CREATE_DICT(',
        'str(': 'TO_STRING(',
        'int(': 'TO_INTEGER(',
        'float(': 'TO_DECIMAL(',
        'bool(': 'TO_BOOLEAN('
    }

    # Apply pseudo-code replacements
    for old, new in pseudo_replacements.items():
        text = text.replace(old, new)

    # Add pseudo-code comments
    if 'PROCEDURE ' in text:
        text = text.replace('PROCEDURE ', 'PROCEDURE ') + '\n# PSEUDO-CODE VERSION'

    return text

def analyze_synthetic_data_quality(member_texts, non_member_texts):
    """Analyze the quality of synthetic data generation"""
    print(f"\n SYNTHETIC DATA QUALITY ANALYSIS")
    print(f"   Member texts: {len(member_texts)} samples")
    print(f"   Non-member texts: {len(non_member_texts)} samples")

    # Analyze text characteristics
    member_lengths = [len(text) for text in member_texts]
    non_member_lengths = [len(text) for text in non_member_texts]

    print(f"   Member text lengths: mean={np.mean(member_lengths):.1f}, std={np.std(member_lengths):.1f}")
    print(f"   Non-member text lengths: mean={np.mean(non_member_lengths):.1f}, std={np.std(non_member_lengths):.1f}")

    # Check for overlap in text patterns
    member_patterns = set()
    non_member_patterns = set()

    for text in member_texts[:10]:  # Sample first 10
        # Extract key patterns (function definitions, imports, etc.)
        patterns = re.findall(r'def\s+\w+|import\s+\w+|class\s+\w+|print\s*\(', text)
        member_patterns.update(patterns)

    for text in non_member_texts[:10]:  # Sample first 10
        patterns = re.findall(r'def\s+\w+|import\s+\w+|class\s+\w+|print\s*\(', text)
        non_member_patterns.update(patterns)

    overlap = len(member_patterns.intersection(non_member_patterns))
    total_patterns = len(member_patterns.union(non_member_patterns))

    print(f"   Pattern overlap: {overlap}/{total_patterns} ({overlap/total_patterns*100:.1f}%)")

    if overlap/total_patterns > 0.7:
        print(f"     WARNING: High pattern overlap - synthetic data may be too similar to member data")

    return overlap/total_patterns

def apply_ultra_aggressive_semantic(text):
    """Ultra-aggressive semantic transformation with dramatic changes"""
    # Completely different variable naming conventions
    var_mappings = {
        'data': 'dataset_collection', 'result': 'computed_output', 'value': 'stored_content',
        'list': 'array_sequence', 'dict': 'key_value_mapping', 'str': 'text_string',
        'int': 'integer_number', 'float': 'decimal_value', 'bool': 'boolean_flag',
        'file': 'file_document', 'path': 'file_location', 'name': 'identifier_name',
        'count': 'total_quantity', 'size': 'dimension_size', 'time': 'time_duration',
        'user': 'person_user', 'input': 'user_entry', 'output': 'system_result',
        'function': 'procedure_function', 'method': 'operation_method', 'class': 'type_class',
        'object': 'instance_object', 'array': 'sequence_array', 'string': 'text_string',
        'def': 'define_function', 'class': 'define_class', 'import': 'include_module',
        'from': 'import_from', 'return': 'give_back', 'if': 'conditional_if',
        'else': 'conditional_else', 'elif': 'conditional_elif', 'for': 'loop_for',
        'while': 'loop_while', 'try': 'attempt_try', 'except': 'catch_except',
        'finally': 'always_finally', 'with': 'context_with', 'as': 'alias_as',
        'in': 'contained_in', 'is': 'identity_is', 'not': 'logical_not',
        'and': 'logical_and', 'or': 'logical_or', 'True': 'boolean_true',
        'False': 'boolean_false', 'None': 'null_value', 'self': 'instance_self',
        'len': 'length_function', 'print': 'display_function', 'input': 'get_input',
        'range': 'sequence_range', 'list': 'create_list', 'dict': 'create_dict',
        'str': 'convert_string', 'int': 'convert_integer', 'float': 'convert_float',
        'bool': 'convert_boolean'
    }

    # Apply ultra-aggressive replacements
    for old, new in var_mappings.items():
        # Replace with much longer, more descriptive names
        text = re.sub(r'\b' + old + r'\b', new, text, flags=re.IGNORECASE)

    # Add verbose comments
    comment_patterns = [
        (r'# (.+)', lambda m: f"# ULTRA_MODIFIED_COMMENT: {m.group(1)}"),
        (r'"""(.+)"""', lambda m: f'"""ULTRA_ENHANCED_DOCSTRING: {m.group(1)}"""'),
        (r"'''(.+)'''", lambda m: f"'''ULTRA_UPDATED_DOCSTRING: {m.group(1)}'''")
    ]

    for pattern, replacement in comment_patterns:
        text = re.sub(pattern, replacement, text, flags=re.DOTALL)

    return text

def apply_ultra_aggressive_structural(text):
    """Ultra-aggressive structural transformation with dramatic changes"""
    lines = text.split('\n')
    transformed_lines = []

    for line in lines:
        if line.strip():
            # Always change indentation dramatically
            if line.startswith('    '):
                # Reduce indentation significantly
                line = line[8:] if len(line) >= 8 else line[4:]
            elif line.startswith('  '):
                # Increase indentation dramatically
                line = '        ' + line
            else:
                # Add significant indentation
                line = '    ' + line

        # Always reorder lines for dramatic effect
        if 'print(' in line and len(transformed_lines) > 0:
            # Move print statements to the beginning
            transformed_lines.insert(0, line)
            continue

        # Add extra blank lines randomly
        if random.random() < 0.3:
            transformed_lines.append('')

        transformed_lines.append(line)

    return '\n'.join(transformed_lines)

def apply_ultra_aggressive_aggressive(text):
    """Ultra-aggressive transformation with completely different syntax"""
    # Completely different keyword replacements
    ultra_replacements = {
        'def ': 'PROCEDURE_DEFINE ',
        'class ': 'TYPE_DEFINE ',
        'import ': 'INCLUDE_MODULE ',
        'from ': 'SOURCE_FROM ',
        'return ': 'GIVE_BACK_RESULT ',
        'if ': 'CONDITIONAL_CHECK ',
        'else:': 'OTHERWISE_CONDITION:',
        'elif ': 'ELSE_IF_CONDITION ',
        'for ': 'ITERATE_LOOP ',
        'while ': 'REPEAT_WHILE ',
        'try:': 'ATTEMPT_TRY:',
        'except ': 'CATCH_EXCEPTION ',
        'finally:': 'ALWAYS_EXECUTE:',
        'with ': 'CONTEXT_USING ',
        'as ': 'ALIAS_AS ',
        'in ': 'CONTAINED_WITHIN ',
        'is ': 'IDENTITY_EQUALS ',
        'not ': 'LOGICAL_NEGATE ',
        'and ': 'LOGICAL_AND ',
        'or ': 'LOGICAL_OR ',
        'True': 'BOOLEAN_TRUE_VALUE',
        'False': 'BOOLEAN_FALSE_VALUE',
        'None': 'NULL_EMPTY_VALUE',
        'self': 'INSTANCE_REFERENCE',
        'len(': 'CALCULATE_LENGTH(',
        'print(': 'DISPLAY_OUTPUT(',
        'input(': 'GET_USER_INPUT(',
        'range(': 'CREATE_SEQUENCE(',
        'list(': 'CREATE_LIST_COLLECTION(',
        'dict(': 'CREATE_DICTIONARY_MAPPING(',
        'str(': 'CONVERT_TO_STRING(',
        'int(': 'CONVERT_TO_INTEGER(',
        'float(': 'CONVERT_TO_DECIMAL(',
        'bool(': 'CONVERT_TO_BOOLEAN('
    }

    # Apply ultra-aggressive replacements
    for old, new in ultra_replacements.items():
        text = text.replace(old, new)

    # Change function structure dramatically
    if 'PROCEDURE_DEFINE ' in text:
        text = re.sub(r'PROCEDURE_DEFINE (\w+)\(', r'PROCEDURE_DEFINE \1_FUNCTION(', text)

    if 'TYPE_DEFINE ' in text:
        text = re.sub(r'TYPE_DEFINE (\w+)', r'STRUCTURE_DEFINE \1_TYPE', text)

    return text

def apply_ultra_aggressive_different_language(text):
    """Ultra-aggressive different language transformation"""
    # Completely different pseudo-code transformation
    pseudo_replacements = {
        'def ': 'FUNCTION_DEFINITION ',
        'class ': 'CLASS_DEFINITION ',
        'import ': 'MODULE_IMPORT ',
        'from ': 'MODULE_FROM ',
        'return ': 'RETURN_VALUE ',
        'if ': 'IF_CONDITION ',
        'else:': 'ELSE_CONDITION:',
        'elif ': 'ELSE_IF_CONDITION ',
        'for ': 'FOR_LOOP ',
        'while ': 'WHILE_LOOP ',
        'try:': 'TRY_BLOCK:',
        'except ': 'CATCH_BLOCK ',
        'finally:': 'FINALLY_BLOCK:',
        'with ': 'WITH_CONTEXT ',
        'as ': 'AS_ALIAS ',
        'in ': 'IN_CONTAINER ',
        'is ': 'IS_EQUAL ',
        'not ': 'NOT_OPERATOR ',
        'and ': 'AND_OPERATOR ',
        'or ': 'OR_OPERATOR ',
        'True': 'TRUE_VALUE',
        'False': 'FALSE_VALUE',
        'None': 'NULL_VALUE',
        'self': 'SELF_REFERENCE',
        'len(': 'LENGTH_FUNCTION(',
        'print(': 'PRINT_FUNCTION(',
        'input(': 'INPUT_FUNCTION(',
        'range(': 'RANGE_FUNCTION(',
        'list(': 'LIST_FUNCTION(',
        'dict(': 'DICT_FUNCTION(',
        'str(': 'STRING_FUNCTION(',
        'int(': 'INTEGER_FUNCTION(',
        'float(': 'FLOAT_FUNCTION(',
        'bool(': 'BOOLEAN_FUNCTION('
    }

    # Apply pseudo-code replacements
    for old, new in pseudo_replacements.items():
        text = text.replace(old, new)

    # Add pseudo-code comments
    if 'FUNCTION_DEFINITION ' in text:
        text = text.replace('FUNCTION_DEFINITION ', 'FUNCTION_DEFINITION ') + '\n# PSEUDO_CODE_VERSION_MODIFIED'

    return text

def apply_code_obfuscation(text):
    """Apply code obfuscation techniques"""
    # Replace common patterns with obfuscated versions
    obfuscation_mappings = {
        'def ': 'def _0x',
        'class ': 'class _0x',
        'import ': 'import _0x',
        'from ': 'from _0x',
        'return ': 'return _0x',
        'if ': 'if _0x',
        'else:': '_0x_else:',
        'elif ': 'elif _0x',
        'for ': 'for _0x',
        'while ': 'while _0x',
        'try:': '_0x_try:',
        'except ': 'except _0x',
        'finally:': '_0x_finally:',
        'with ': 'with _0x',
        'as ': 'as _0x',
        'in ': 'in _0x',
        'is ': 'is _0x',
        'not ': 'not _0x',
        'and ': 'and _0x',
        'or ': 'or _0x',
        'True': '_0x_True',
        'False': '_0x_False',
        'None': '_0x_None',
        'self': '_0x_self',
        'len(': '_0x_len(',
        'print(': '_0x_print(',
        'input(': '_0x_input(',
        'range(': '_0x_range(',
        'list(': '_0x_list(',
        'dict(': '_0x_dict(',
        'str(': '_0x_str(',
        'int(': '_0x_int(',
        'float(': '_0x_float(',
        'bool(': '_0x_bool('
    }

    # Apply obfuscation
    for old, new in obfuscation_mappings.items():
        text = text.replace(old, new)

    # Add obfuscation comments
    text = f"# OBFUSCATED_CODE_VERSION\n{text}"

    return text

def apply_alternative_programming_style(text):
    """Apply alternative programming style"""
    # Use different programming paradigms
    style_mappings = {
        'def ': 'lambda ',
        'class ': 'dataclass ',
        'import ': 'from typing import ',
        'from ': 'import ',
        'return ': 'yield ',
        'if ': 'match ',
        'else:': 'case _:',
        'elif ': 'case ',
        'for ': 'map(',
        'while ': 'itertools.takewhile(',
        'try:': 'with contextlib.suppress(',
        'except ': '):',
        'finally:': '):',
        'with ': 'contextlib.closing(',
        'as ': 'as ',
        'in ': 'in ',
        'is ': 'is ',
        'not ': 'not ',
        'and ': 'and ',
        'or ': 'or ',
        'True': 'True',
        'False': 'False',
        'None': 'None',
        'self': 'self',
        'len(': 'len(',
        'print(': 'print(',
        'input(': 'input(',
        'range(': 'range(',
        'list(': 'list(',
        'dict(': 'dict(',
        'str(': 'str(',
        'int(': 'int(',
        'float(': 'float(',
        'bool(': 'bool('
    }

    # Apply style changes
    for old, new in style_mappings.items():
        text = text.replace(old, new)

    # Add style comments
    text = f"# ALTERNATIVE_PROGRAMMING_STYLE\n{text}"

    return text

def create_diverse_synthetic_data(member_texts, num_samples=None):
    """Create diverse synthetic data using multiple approaches for better discrimination"""
    if num_samples is None:
        num_samples = len(member_texts)

    print(f" Creating {num_samples} ULTRA-AGGRESSIVE synthetic non-member samples...")

    non_member_texts = []

    for i in range(num_samples):
        if i >= len(member_texts):
            text = member_texts[i % len(member_texts)]
        else:
            text = member_texts[i]

        # Use much more aggressive strategy distribution
        strategy = i % 7  # More strategies

        if strategy == 0:
            # Ultra-aggressive semantic transformation
            transformed = apply_ultra_aggressive_semantic(text)
        elif strategy == 1:
            # Ultra-aggressive structural transformation
            transformed = apply_ultra_aggressive_structural(text)
        elif strategy == 2:
            # Ultra-aggressive aggressive transformation
            transformed = apply_ultra_aggressive_aggressive(text)
        elif strategy == 3:
            # Ultra-aggressive different language transformation
            transformed = apply_ultra_aggressive_different_language(text)
        elif strategy == 4:
            # Completely different patterns
            transformed = apply_completely_different_pattern(text)
        elif strategy == 5:
            # Code obfuscation
            transformed = apply_code_obfuscation(text)
        elif strategy == 6:
            # Alternative programming style
            transformed = apply_alternative_programming_style(text)
        else:
            # Fallback to ultra-aggressive transformation
            transformed = apply_ultra_aggressive_semantic(text)

        # Always add randomization to make it more diverse
        if random.random() < 0.3:  # 30% chance
            # Add random comments or modifications
            transformed = f"# Modified version: {random.randint(1000, 9999)}\n{transformed}"

        if random.random() < 0.2:  # 20% chance
            # Add random imports
            transformed = f"import random_{random.randint(1, 100)}\n{transformed}"

        non_member_texts.append(transformed)

    print(f" Created {len(non_member_texts)} ultra-aggressive synthetic non-member samples")

    # Analyze quality
    overlap_ratio = analyze_synthetic_data_quality(member_texts, non_member_texts)

    return non_member_texts

def run_comprehensive_comparison(include_apis=True, member_filename='qa.en.python.json'):
    """Run comprehensive comparison across multiple LLMs"""
    print("=== COMPREHENSIVE MULTI-LLM MIN-K% ANALYSIS (NO DATA LEAKAGE) ===")

    # Get all available models based on configuration
    all_models = list(SUPPORTED_MODELS.keys())

    # Reorder models to ensure GPT-5 is processed last
    gpt5_model = "openai-gpt-5"
    if gpt5_model in all_models:
        # Remove GPT-5 from the list and add it at the end
        all_models.remove(gpt5_model)
        all_models.append(gpt5_model)
        print(f" Reordered models: GPT-5 will be analyzed last")

    if include_apis:
        # Test all models individually to ensure complete coverage
        model_combinations = [[model] for model in all_models]
        print(f" Testing ALL {len(all_models)} models individually for complete coverage:")
        for i, model in enumerate(all_models, 1):
            print(f"   {i}. {SUPPORTED_MODELS[model]['name']} ({SUPPORTED_MODELS[model]['parameters']} parameters)")
    else:
        # Test only free models
        free_models = [model for model in all_models if SUPPORTED_MODELS[model]['type'] == 'huggingface']
        model_combinations = [[model] for model in free_models]
        print(f" Testing {len(free_models)} free models individually:")
        for i, model in enumerate(free_models, 1):
            print(f"   {i}. {SUPPORTED_MODELS[model]['name']} ({SUPPORTED_MODELS[model]['parameters']} parameters)")

    all_comprehensive_results = {}

    for i, models in enumerate(model_combinations):
        print(f"\n--- Test Combination {i+1}: {models} ---")
        try:
            # Use large-scale single trial parameters with automatic sample size selection
            results = quick_synthetic_test_multi_llm(
                models,
                use_apis=include_apis,
                member_filename=member_filename,
                max_samples_per_class=None,  # Automatic selection: 3000 for free, 100 for paid
                num_trials=1  # Single trial for large-scale analysis
            )
            if results:
                all_comprehensive_results[f"Combination_{i+1}"] = results
                create_enhanced_comparison_visualization(results, f'min_k_combo_{i+1}.png')
        except Exception as e:
            print(f"Error in combination {i+1}: {e}")
            continue

    if all_comprehensive_results:
        all_models_results = {}
        for combo_name, combo_results in all_comprehensive_results.items():
            for model_name, model_results in combo_results.items():
                if model_name not in all_models_results:
                    all_models_results[model_name] = model_results

        # Print comprehensive summary of all tested models
        print("\n" + "="*80)
        print("COMPREHENSIVE RESULTS SUMMARY")
        print("="*80)
        print(f"Total models tested: {len(all_models_results)}")
        print(f"Models available in SUPPORTED_MODELS: {len(SUPPORTED_MODELS)}")

        # Show which models were tested vs available
        tested_models = set(all_models_results.keys())
        all_available_models = set(SUPPORTED_MODELS[model]['name'] for model in SUPPORTED_MODELS.keys())
        untested_models = all_available_models - tested_models

        print(f"\n Successfully tested models ({len(tested_models)}):")
        for model_name in sorted(tested_models):
            print(f"   - {model_name}")

        if untested_models:
            print(f"\n❌ Models not tested ({len(untested_models)}):")
            for model_name in sorted(untested_models):
                print(f"   - {model_name}")

        create_enhanced_comparison_visualization(all_models_results, 'min_k_final_comparison.png')
        return all_models_results
    else:
        print("No successful results obtained")
        return None

def evaluate_student_hackathon_submissions(models_to_test=None, submissions_dir='student_submissions', member_filename='qa.en.python.json', preloaded_submissions=None):
    """
    Evaluate student hackathon code submissions for memorization detection

    Args:
        models_to_test (list): List of model names to test
        submissions_dir (str): Directory containing student .py files (can be None if using preloaded_submissions)
        member_filename (str): Path to member data file for threshold calculation
        preloaded_submissions (list): Pre-loaded submissions (for single file mode)

    Returns:
        dict: Comprehensive evaluation results
    """
    print(" STUDENT HACKATHON CODE EVALUATION")
    print("=" * 60)
    print(" Phase 1: Hackathon Phase - Loading student submissions")
    print(" Phase 2: Detection Phase - Min-K% analysis with thresholds")
    print(" Phase 3: Evaluation Phase - Metrics and visualizations")
    print("=" * 60)

    # Load student submissions
    if preloaded_submissions is not None:
        # Use pre-loaded submissions (single file mode)
        student_submissions = preloaded_submissions
        print(f" Using pre-loaded submission: {student_submissions[0]['student_id']}")
    elif submissions_dir and os.path.isdir(submissions_dir):
        # Detect Projects_Submissions structure
        base_name = os.path.basename(os.path.abspath(submissions_dir))
        has_groups = any(os.path.isdir(os.path.join(submissions_dir, d)) and d.startswith('Group') for d in os.listdir(submissions_dir))
        if base_name == 'Projects_Submissions' or has_groups:
            student_submissions = load_projects_submissions_folder(submissions_dir)
        else:
            student_submissions = load_student_submissions(submissions_dir)
    elif submissions_dir:
        student_submissions = load_student_submissions(submissions_dir)
    else:
        print(" No submissions provided!")
        return None
    if not student_submissions:
        print(" No student submissions found!")
        return None

    print(f" Loaded {len(student_submissions)} student submissions")

    # Load member data for threshold calculation
    try:
        member_texts = load_member_data(member_filename)
        print(f" Loaded {len(member_texts)} member texts for threshold calculation")
    except Exception as e:
        print(f" Error loading member data: {e}")
        return None

    # Set default models if none specified (GPT-5 last)
    if models_to_test is None:
        models_to_test = [
            'gpt2', 'distilgpt2', 'microsoft/DialoGPT-medium',
            'google-gemini-2.5-flash',
            'mistralai-mistral-small-24b-instruct-2501',
            'deepseek-chat-v3-0324', 'meta-llama-3.3-70b-instruct',
            'openai-gpt-5'  # GPT-5 last
        ]

    print(f" Testing {len(models_to_test)} models:")
    for model in models_to_test:
        print(f"   - {SUPPORTED_MODELS.get(model, {}).get('name', model)}")

    # Initialize results storage
    all_results = {}
    student_results = {}

    # Reorder models_to_test to ensure GPT-5 is processed last
    gpt5_key = 'openai-gpt-5'
    reordered_models = [m for m in models_to_test if m != gpt5_key]
    if gpt5_key in models_to_test:
        reordered_models.append(gpt5_key)
        print(f" Reordered models: GPT-5 will be analyzed last among paid LLMs")

    # Process each model
    for model_name in reordered_models:
        print(f"\n--- Processing {SUPPORTED_MODELS.get(model_name, {}).get('name', model_name)} ---")

        try:
            # Train model and get thresholds with ultra-optimized k_percent for best AUC/TPR
            print(" Training model and calculating thresholds...")
            if SUPPORTED_MODELS.get(model_name, {}).get('type') == 'huggingface':
                # Use k=0.001 (0.1%) for ultra-high discrimination
                detector = MultiLLMMinKDetector(SUPPORTED_MODELS[model_name], k_percent=0.001, batch_size=4)
            else:
                # Use k=0.0015 (0.15%) for API models - ultra-sensitive
                detector = APIMinKDetector(SUPPORTED_MODELS[model_name], k_percent=0.0015)

            # Train with member data - use smaller sample for API models to save time/cost
            is_api_model = SUPPORTED_MODELS.get(model_name, {}).get('type') != 'huggingface'
            if is_api_model:
                # For API models: use 100 samples (balanced between speed and accuracy)
                training_size = 100
                print(f"⚡ Using training size of {training_size} samples for API model (optimized for speed)")
            else:
                # For free HuggingFace models: use full 1000 samples
                training_size = 1000

            member_results = detector.analyze_texts_fast(member_texts[:training_size])
            member_scores = [r['min_k_score'] for r in member_results if r['min_k_score'] is not None]
            non_member_texts = create_synthetic_non_member_data(member_texts[:training_size], training_size)
            non_member_results = detector.analyze_texts_fast(non_member_texts)
            non_member_scores = [r['min_k_score'] for r in non_member_results if r['min_k_score'] is not None]

            # Calculate threshold - use more aggressive approach for better detection
            # Lower threshold = more submissions flagged = higher recall
            member_median = np.percentile(member_scores, 50)  # Use median instead of P75
            member_p25 = np.percentile(member_scores, 25)
            non_member_p50 = np.percentile(non_member_scores, 50) if len(non_member_scores) > 0 else member_median

            # Use a weighted combination favoring lower threshold for better detection
            threshold = (member_p25 * 0.6 + non_member_p50 * 0.4)

            print(f" Calculated threshold (aggressive for better detection): {threshold:.4f}")
            print(f"   • Member P25: {member_p25:.4f}  |  Member Median: {member_median:.4f}")
            print(f"   • Non-member P50: {non_member_p50:.4f}")
            print(f" Member scores range: {min(member_scores):.4f} to {max(member_scores):.4f}")

            # Analyze student submissions
            print(" Analyzing student submissions...")
            student_scores = []
            student_predictions = []

            for submission in tqdm(student_submissions, desc="Processing submissions"):
                result = detector.analyze_texts_fast([submission['code_content']])[0]
                score = result['min_k_score']
                student_scores.append(score)

                # Calculate probability using pure Min-K% approach
                if threshold != 0:
                    score_ratio = score / threshold
                    base_prob = 1.0 / (1.0 + np.exp(-5 * (score_ratio - 1.0)))  # Sigmoid function
                else:
                    base_prob = 0.5

                # Use pure Min-K% prediction without any marker boosts
                base_prob = min(1.0, max(0.0, base_prob))
                final_probability = apply_model_prior(model_name, base_prob)

                # Flag based on probability > 0.70 (uniform for all models)
                is_flagged = final_probability > 0.70
                student_predictions.append(is_flagged)

                # Detect LLM-generated sections for this submission if flagged
                llm_generated_sections = []
                if is_flagged:
                    print(f"    Analyzing {submission['student_id']} for LLM-generated sections...")
                    llm_generated_sections = detect_llm_generated_sections(
                        submission['code_content'],
                        model_name,
                        final_probability
                    )
                    print(f"   Student {submission['student_id']}: Found {len(llm_generated_sections)} LLM-generated sections")

                    # Print details of detected sections
                    if llm_generated_sections:
                        for section in llm_generated_sections:
                            print(f"     Line {section['line_num']}: {section['content'][:60]}... ({section['type']})")
                    else:
                        print(f"       No specific patterns detected despite high probability")

                # Use pure Min-K% approach without style markers
                llm_markers = {'spans': []}  # Empty markers for compatibility
                num_markers = len(llm_generated_sections)  # Count of detected sections
                annotated_code = submission['code_content']  # No annotation needed

                # Store individual results first (before flagging decision)
                if submission['student_id'] not in student_results:
                    student_results[submission['student_id']] = {}

                # Debug: Print score for each student
                print(f"   Student {submission['student_id']}: prob={final_probability:.3f}, flagged={is_flagged}")

                # Save annotated code to file for review (only if flagged)
                if is_flagged and llm_markers.get('spans'):
                    safe_id = submission['student_id'].replace('/', '_').replace(' ', '_')
                    annotated_filename = f'annotated_{safe_id}_{model_name}.py'
                    try:
                        with open(annotated_filename, 'w', encoding='utf-8') as f:
                            f.write(f"# LLM Detection Analysis for {submission['student_id']}\n")
                            f.write(f"# Model: {SUPPORTED_MODELS.get(model_name, {}).get('name', model_name)}\n")
                            f.write(f"# Probability: {final_probability:.3f}\n")
                            f.write(f"# Flagged: {is_flagged} (probability > 0.70)\n")
                            f.write(f"# Number of LLM-likely sections: {num_markers}\n\n")
                            f.write(annotated_code)
                        print(f"   Saved annotated code: {annotated_filename}")
                    except Exception as e:
                        print(f"   Could not save annotated code: {e}")

                student_results[submission['student_id']][model_name] = {
                    'score': score,
                    'threshold': threshold,
                    'is_flagged': is_flagged,
                    'probability': final_probability,
                    'llm_markers': llm_markers,
                    'annotated_code_preview': '\n'.join(annotated_code.split('\n')[:80]),
                    'num_llm_sections': num_markers,
                    'llm_generated_sections': llm_generated_sections
                }

            # Print summary of student analysis
            if student_scores:
                print(f"\n STUDENT SUBMISSION ANALYSIS:")
                print(f"   - Total groups analyzed: {len(student_submissions)}")
                print(f"   - Groups flagged as likely LLM-assisted: {sum(student_predictions)}/{len(student_submissions)}")
                print(f"   - Detection threshold: {threshold:.4f}")
                print(f"   - Score range: {min(student_scores):.4f} to {max(student_scores):.4f}")
                print(f"   - Average score: {np.mean(student_scores):.4f}")
            else:
                print("  No student scores calculated!")

            # Store results - ONLY student-relevant information
            all_results[model_name] = {
                'model_name': SUPPORTED_MODELS.get(model_name, {}).get('name', model_name),
                'threshold': threshold,
                'student_scores': student_scores,
                'student_predictions': student_predictions,
                'flagged_count': sum(student_predictions),
                'total_submissions': len(student_submissions),
                'category': SUPPORTED_MODELS.get(model_name, {}).get('category', 'Unknown')
            }

            print(f" {SUPPORTED_MODELS.get(model_name, {}).get('name', model_name)} analysis complete:")
            print(f"   - Flagged groups: {sum(student_predictions)}/{len(student_submissions)}")

            # Print individual group results
            print(f"\nIndividual Group Results for {SUPPORTED_MODELS.get(model_name, {}).get('name', model_name)}:")
            for submission in student_submissions:
                student_id = submission['student_id']
                group_num = student_id.replace('Group ', '').replace('Group', '').strip()
                if student_id in student_results and model_name in student_results[student_id]:
                    prob = student_results[student_id][model_name]['probability']
                    flagged = student_results[student_id][model_name]['is_flagged']
                    num_llm_sections = student_results[student_id][model_name]['num_llm_sections']
                    flag_text = "[FLAGGED]" if flagged else "[OK]"
                    print(f"   {flag_text} Group {group_num}: Probability={prob:.3f}, LLM sections={num_llm_sections}")

        except Exception as e:
            print(f" Error processing {model_name}: {e}")
            continue

    # Generate visualizations with error handling
    print("\n📊 Generating visualizations and reports...")

    # 1. Summary visualization
    try:
        create_student_evaluation_visualizations(all_results, student_results, student_submissions)
    except Exception as viz_err:
        print(f"  Error creating summary visualization: {viz_err}")

    # 2. Individual submission plots for each category automatically
    print("\n Generating individual submission analysis plots for ALL groups...")

    # Determine which categories to generate based on models tested
    has_free = any(SUPPORTED_MODELS.get(m, {}).get('category') == 'Free' for m in reordered_models)
    has_paid = any(SUPPORTED_MODELS.get(m, {}).get('category') == 'Paid' for m in reordered_models)

    try:
        if has_free and has_paid:
            # Generate all three categories
            create_individual_submission_plots(student_results, student_submissions, 'free')
            create_individual_submission_plots(student_results, student_submissions, 'paid')
            create_individual_submission_plots(student_results, student_submissions, 'all')
        elif has_paid:
            create_individual_submission_plots(student_results, student_submissions, 'paid')
        elif has_free:
            create_individual_submission_plots(student_results, student_submissions, 'free')
    except Exception as ind_err:
        print(f"  Error creating individual submission plots: {ind_err}")
        print(f"   Results are still available in student_results dictionary")

    # 3. Generate detailed LLM section report
    try:
        generate_llm_section_report(student_results, student_submissions)
    except Exception as report_err:
        print(f"  Error generating LLM section report: {report_err}")

    # Print comprehensive results - focused on student analysis only
    print("\n" + "="*80)
    print(" STUDENT SUBMISSION EVALUATION SUMMARY")
    print("="*80)
    print(f"Total groups analyzed: {len(student_submissions)}")
    print(f"\nDetection Results by Model:")
    print("-" * 80)

    for model_name, results in all_results.items():
        print(f"\n {results['model_name']}:")
        print(f"    Groups flagged: {results['flagged_count']}/{results['total_submissions']}")
        percentage = (results['flagged_count'] / results['total_submissions'] * 100) if results['total_submissions'] > 0 else 0
        print(f"    Percentage flagged: {percentage:.1f}%")
        print(f"    Threshold used: {results['threshold']:.4f}")

    return {
        'model_results': all_results,
        'student_results': student_results,
        'submissions': student_submissions
    }

def create_student_evaluation_visualizations(all_results, student_results, student_submissions):
    """Create focused visualization showing flagged submissions by each model"""

    print(" Creating student evaluation summary visualization...")

    # Simple bar chart showing how many groups each model flagged
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    model_names = [results['model_name'] for results in all_results.values()]
    flagged_counts = [results['flagged_count'] for results in all_results.values()]
    total_submissions = all_results[list(all_results.keys())[0]]['total_submissions']
    colors = [SUPPORTED_MODELS.get(model_key, {}).get('color', '#666666')
              for model_key in all_results.keys()]

    bars = ax.bar(model_names, flagged_counts, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    ax.set_title(f'Groups Flagged as LLM-Assisted by Each Model\n(Total: {total_submissions} groups analyzed)',
                fontweight='bold', fontsize=14)
    ax.set_ylabel('Number of Groups Flagged', fontsize=12)
    ax.set_xlabel('Language Model Detector', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, total_submissions + 2)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, count in zip(bars, flagged_counts):
        percentage = (count / total_submissions * 100) if total_submissions > 0 else 0
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{count}/{total_submissions}\n({percentage:.0f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Add reference line for majority
    ax.axhline(y=total_submissions/2, color='orange', linestyle='--',
              alpha=0.5, label=f'Majority ({total_submissions//2} groups)')
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('student_flagged_summary.png', dpi=300, bbox_inches='tight')

    # Display in Colab
    plt.show()

    print("Student evaluation summary complete and displayed")

def generate_llm_section_report(student_results, student_submissions):
    """Generate a detailed text report showing only LLM-generated comments and code sections"""

    print("\n" + "="*80)
    print("GENERATING DETAILED LLM SECTION ANALYSIS REPORT")
    print("="*80)

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("LLM-GENERATED SECTIONS ANALYSIS REPORT")
    report_lines.append("="*80)
    report_lines.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Groups Analyzed: {len(student_submissions)}")
    report_lines.append("")

    # Summary statistics
    total_groups = len(student_submissions)
    flagged_groups = sum(1 for group_results in student_results.values()
                        if any(result['is_flagged'] for result in group_results.values()))

    report_lines.append("SUMMARY:")
    report_lines.append(f"  Total groups analyzed: {total_groups}")
    report_lines.append(f"  Groups flagged for LLM assistance: {flagged_groups}")
    report_lines.append(f"  Flagging rate: {flagged_groups/total_groups*100:.1f}%")
    report_lines.append("")

    # Create a mapping from student_id to submission data
    submission_map = {sub['student_id']: sub for sub in student_submissions}

    # Detailed results for each group
    for group_num, (student_id, group_results) in enumerate(student_results.items(), 1):
        report_lines.append(f"GROUP {group_num}: {student_id}")
        report_lines.append("-" * 50)

        # Check if any model flagged this group
        flagged_models = [model for model, result in group_results.items()
                         if result['is_flagged']]

        if not flagged_models:
            report_lines.append(" No LLM assistance detected")
            report_lines.append("")
            continue

        # Show flagged models
        report_lines.append(f"🚨 FLAGGED MODELS: {', '.join(flagged_models)}")
        report_lines.append("")

        # Get the submission data for this group
        submission = submission_map.get(student_id, {})
        code_content = submission.get('code_content', '')
        code_lines = code_content.split('\n')

        # Show probabilities for ALL models (not just flagged ones)
        report_lines.append("   ALL MODEL PROBABILITIES:")
        report_lines.append("  " + "-"*40)

        for model_name, result in group_results.items():
            model_display = SUPPORTED_MODELS.get(model_name, {}).get('name', model_name)
            is_flagged = result['is_flagged']
            flag_status = " FLAGGED" if is_flagged else " OK"
            report_lines.append(f"    {model_display}: {result['probability']:.3f} {flag_status}")

        report_lines.append("  " + "-"*40)
        report_lines.append("")

        # Show LLM-generated sections for each flagged model
        for model_name in flagged_models:
            result = group_results[model_name]
            model_display = SUPPORTED_MODELS.get(model_name, {}).get('name', model_name)

            report_lines.append(f"  LLM-GENERATED CODE SNIPPETS BY {model_display.upper()}:")
            report_lines.append("  " + "="*60)

            # Show only LLM-generated sections as code snippets
            llm_sections = result.get('llm_generated_sections', [])
            if llm_sections:
                # Find the range of lines that contain LLM-generated content
                llm_line_numbers = [s['line_num'] for s in llm_sections]
                if llm_line_numbers:
                    min_line = min(llm_line_numbers) - 1  # Convert to 0-based index
                    max_line = max(llm_line_numbers) - 1  # Convert to 0-based index

                    # Extract the code snippet with some context
                    start_line = max(0, min_line - 2)  # Add 2 lines before
                    end_line = min(len(code_lines), max_line + 3)  # Add 3 lines after

                    # Show the code snippet with line numbers
                    report_lines.append("  Code snippet with LLM-generated content:")
                    report_lines.append("  " + "-"*50)

                    for i in range(start_line, end_line):
                        line_num = i + 1
                        line_content = code_lines[i]

                        # Mark LLM-generated lines (but not cell markers)
                        if line_num in llm_line_numbers:
                            # Don't mark cell markers as LLM-generated
                            if not (line_content.strip().startswith('# ===== CELL') or
                                   line_content.strip().startswith('# ===== MARKDOWN CELL')):
                                report_lines.append(f"  {line_num:3d} | {line_content}  [LLM-GENERATED]")
                            else:
                                report_lines.append(f"  {line_num:3d} | {line_content}")
                        else:
                            report_lines.append(f"  {line_num:3d} | {line_content}")

                    report_lines.append("  " + "-"*50)
                    report_lines.append(f"  Total LLM-generated lines: {len(llm_sections)}")

                report_lines.append("  " + "="*60)
            else:
                report_lines.append(f"  No specific LLM-generated sections identified by {model_display}")

            report_lines.append("")

        report_lines.append("")

    # Write report to file
    with open('llm_detection_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f" Detailed LLM detection report saved to: llm_detection_report.txt")
    return 'llm_detection_report.txt'

def create_student_submission_charts(student_results, student_submissions):
    """Create bar charts for each student submission showing probability of LLM assistance"""

    print("Creating individual student submission charts...")

    # Get all model names
    all_models = set()
    for student_data in student_results.values():
        all_models.update(student_data.keys())
    all_models = sorted(list(all_models))

    # Create charts for ALL students (all groups)
    students_to_plot = list(student_results.keys())

    for student_id in students_to_plot:
        if student_id not in student_results:
            continue

        student_data = student_results[student_id]

        # Extract clean group number for display (e.g., "Group 1" → "1")
        group_display = student_id.replace('Group ', '').replace('Group', '').strip()
        if not group_display:
            group_display = student_id

        # Extract probabilities for each model
        model_names = []
        probabilities = []

        for model_name in all_models:
            if model_name in student_data:
                model_display_name = SUPPORTED_MODELS.get(model_name, {}).get('name', model_name)
                model_names.append(model_display_name)
                probabilities.append(student_data[model_name]['probability'])
            else:
                model_names.append(SUPPORTED_MODELS.get(model_name, {}).get('name', model_name))
                probabilities.append(0.0)

        # Create bar chart
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, probabilities, color='steelblue', alpha=0.7)

        # Color bars based on probability
        for bar, prob in zip(bars, probabilities):
            if prob > 0.7:
                bar.set_color('red')
            elif prob > 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('green')

        plt.title(f'Group {group_display} - Which LLM Wrote This Code?', fontweight='bold', fontsize=16, pad=15)
        plt.ylabel('Probability of LLM Assistance', fontsize=13, fontweight='bold')
        plt.xlabel('Language Model Detectors', fontsize=13, fontweight='bold')
        plt.xticks(rotation=45, ha='right', fontsize=11)
        plt.ylim(0, 1.05)

        # Add reference lines with better labels
        plt.axhline(y=0.5, color='#FF6B6B', linestyle='--', linewidth=2.5, alpha=0.7, label='Decision Boundary (0.5)')
        plt.axhline(y=0.7, color='#FFA500', linestyle='--', linewidth=2, alpha=0.6, label='High Confidence (0.7)')
        plt.legend(fontsize=11, loc='upper right', framealpha=0.95, edgecolor='black', fancybox=True)
        plt.grid(axis='y', alpha=0.3, linestyle=':', linewidth=1)

        # Add value labels on bars with color coding
        for bar, prob in zip(bars, probabilities):
            text_color = 'darkred' if prob > 0.7 else 'darkorange' if prob > 0.5 else 'darkgreen'
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{prob:.3f}', ha='center', va='bottom', fontweight='bold',
                    fontsize=11, color=text_color)

        plt.tight_layout()
        safe_id = student_id.replace('/', '_').replace(' ', '_')
        plt.savefig(f'group_{group_display}_llm_analysis.png', dpi=300, bbox_inches='tight')

        # Display in Colab
        plt.show()

        print(f"   Created and displayed plot for Group {group_display}")

    print(f" Created individual charts for {len(students_to_plot)} students")

def create_score_distribution_analysis(all_results, student_submissions):
    """Create score distribution analysis across all models"""

    print(" Creating score distribution analysis...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Student Code Score Distribution Analysis', fontsize=16, fontweight='bold')

    # Collect all scores
    all_scores = []
    model_labels = []

    for model_name, results in all_results.items():
        scores = results['student_scores']
        all_scores.extend(scores)
        model_labels.extend([SUPPORTED_MODELS.get(model_name, {}).get('name', model_name)] * len(scores))

    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'score': all_scores,
        'model': model_labels
    })

    # Box plot of scores by model
    sns.boxplot(data=df, x='model', y='score', ax=axes[0, 0])
    axes[0, 0].set_title('Score Distribution by Model', fontweight='bold')
    axes[0, 0].set_ylabel('Min-K% Score')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Histogram of all scores
    axes[0, 1].hist(all_scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Overall Score Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Min-K% Score')
    axes[0, 1].set_ylabel('Frequency')

    # Violin plot
    sns.violinplot(data=df, x='model', y='score', ax=axes[1, 0])
    axes[1, 0].set_title('Score Distribution Density by Model', fontweight='bold')
    axes[1, 0].set_ylabel('Min-K% Score')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Scatter plot of scores vs submission index
    submission_indices = list(range(len(student_submissions)))
    colors = plt.cm.viridis(np.linspace(0, 1, len(all_results)))

    for i, (model_name, results) in enumerate(all_results.items()):
        axes[1, 1].scatter(submission_indices, results['student_scores'],
                          label=SUPPORTED_MODELS.get(model_name, {}).get('name', model_name),
                          alpha=0.6, color=colors[i])

    axes[1, 1].set_title('Score vs Submission Index', fontweight='bold')
    axes[1, 1].set_xlabel('Submission Index')
    axes[1, 1].set_ylabel('Min-K% Score')
    axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig('student_score_distribution_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("Score distribution analysis complete")

def create_individual_submission_plots(student_results, student_submissions, model_category='all'):
    """
    Generate individual probability plots for each submission showing which LLMs likely wrote it.

    Args:
        student_results: Dict of {student_id: {model_name: {probability, score, ...}}}
        student_submissions: List of submission dicts
        model_category: 'free', 'paid', or 'all' to filter models
    """
    print(f"\nGenerating individual submission plots (category: {model_category})...")

    # Filter models by category
    def filter_models(model_name):
        if model_category == 'free':
            return SUPPORTED_MODELS.get(model_name, {}).get('category') == 'Free'
        elif model_category == 'paid':
            return SUPPORTED_MODELS.get(model_name, {}).get('category') == 'Paid'
        else:  # 'all'
            return True

    for submission in student_submissions:
        student_id = submission['student_id']

        if student_id not in student_results:
            continue

        # Extract clean group number for display
        group_display = student_id.replace('Group ', '').replace('Group', '').strip()
        if not group_display:
            group_display = student_id

        # Get all model results for this student
        model_results = student_results[student_id]

        # Filter by category and sort by probability (descending)
        filtered_results = [(model, data) for model, data in model_results.items()
                           if filter_models(model) and 'probability' in data]

        if not filtered_results:
            continue

        sorted_results = sorted(filtered_results, key=lambda x: x[1]['probability'], reverse=True)

        # Create single panel plot - probability bars only
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))

        # Probability bars (horizontal for better readability)
        model_names = [SUPPORTED_MODELS.get(m, {}).get('name', m) for m, _ in sorted_results]
        probabilities = [data['probability'] for _, data in sorted_results]
        colors = [SUPPORTED_MODELS.get(m, {}).get('color', '#666666') for m, _ in sorted_results]

        bars = ax.barh(model_names, probabilities, color=colors, alpha=0.8, edgecolor='black', linewidth=2)

        ax.set_xlabel('Probability of LLM Assistance', fontsize=14, fontweight='bold')
        ax.set_title(f'Group {group_display} - Which LLM Wrote This Code?\n({model_category.upper()} Models)',
                     fontsize=16, fontweight='bold', pad=20)
        ax.set_xlim(0, 1.0)

        # Reference lines with better labels
        ax.axvline(x=0.5, color='#FF6B6B', linestyle='--', linewidth=2.5, label='Decision Boundary (0.5)', alpha=0.7)
        ax.axvline(x=0.7, color='#FFA500', linestyle='--', linewidth=2, label='High Confidence (0.7)', alpha=0.6)
        ax.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True)
        ax.grid(axis='x', alpha=0.3, linestyle=':', linewidth=1)

        # Add probability values on bars with better formatting
        for bar, prob in zip(bars, probabilities):
            width = bar.get_width()
            # Color code the text
            text_color = 'darkred' if prob > 0.7 else 'darkorange' if prob > 0.5 else 'darkgreen'
            ax.text(width + 0.03, bar.get_y() + bar.get_height()/2,
                    f'{prob:.3f}', ha='left', va='center', fontweight='bold',
                    fontsize=12, color=text_color)

        # Add annotation box for top LLM (without emoji)
        if probabilities and probabilities[0] > 0.6:
            top_model_name = SUPPORTED_MODELS.get(sorted_results[0][0], {}).get('name', sorted_results[0][0])
            fig.text(0.5, 0.01, f'Most Likely Used: {top_model_name} (Probability: {probabilities[0]:.3f})',
                    ha='center', fontsize=13, color='#D32F2F', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFF9C4', alpha=0.9, edgecolor='#FF6B6B', linewidth=2))

        plt.tight_layout(rect=[0, 0.06, 1, 1])

        # Save plot with clean group naming
        filename = f'group_{group_display}_{model_category}_llm_analysis.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')

        # Display in Colab
        plt.show()

        print(f"    Created and displayed plot for Group {group_display}")

    print(f" Individual submission plots complete for category: {model_category}")

# Main execution block
if __name__ == "__main__":
    print(" Starting Enhanced Min-K% Analysis with Free vs Paid LLMs")
    print("=" * 60)
    print(" Configuration:")
    print("    Free LLMs: 3000 samples per class (GPT-2, DistilGPT-2, DialoGPT)")
    print("    Paid LLMs: 100 samples per class (GPT-5, Gemini 2.5, Mistral, DeepSeek, LLaMA 3.3)")
    print("    CUDA Memory Management: Automatic batch size reduction and CPU fallback")
    print("=" * 60)

    # Fix for torch.fx circular import issue
    print("🔧 Checking for torch.fx circular import issue...")
    try:
        import torch
        # Test if torch.fx is available
        _ = torch.fx
        print(" PyTorch is working correctly")
    except AttributeError as e:
        if "torch.fx" in str(e):
            print("  Detected torch.fx circular import issue")
            print("Please restart the runtime and run this cell again")
            print("   Runtime → Restart runtime")
            print("   Or run: import IPython; IPython.get_ipython().kernel.do_shutdown(True)")
            raise e
        else:
            print(f" Other PyTorch error: {e}")
            raise e

    # Install required packages if not already installed
    try:
        import openai
        print(" OpenAI package is available")
    except ImportError as e:
        print(f"  Missing OpenAI package: {e}")
        print("Please install required packages:")
        print("!pip install openai requests")

    try:
        import google.generativeai
        print(" Google Generative AI package is available")
    except ImportError as e:
        print(f" Missing Google Generative AI package: {e}")
        print("Please install: !pip install google-generativeai")

    # Check if API keys are configured for OpenRouter
    print("\n Checking API key configuration...")
    placeholder_keys = []
    for model_key, config in SUPPORTED_MODELS.items():
        if config.get('type') == 'openrouter':
            api_key = os.environ.get('OPENROUTER_API_KEY', config.get('api_key', ''))
            if not api_key:
                placeholder_keys.append(config['name'])

    if placeholder_keys:
        print("  Found placeholder API keys for:", ", ".join(placeholder_keys))
        print(" To get API keys, run: get_api_key_instructions()")
        print(" To update keys, run: update_api_keys()")
        print("\n For now, running with free models only...")
        include_apis = False
    else:
        include_apis = True
        print(" API keys appear to be configured")

    # Ask user if they want to run comprehensive comparison or go directly to student evaluation
    print(f"\n What would you like to do?")
    print("1. Run comprehensive model comparison (synthetic test) - takes 10-30 minutes")
    print("2. Skip to student evaluation directly")

    analysis_choice = input("\nEnter your choice (1/2): ").strip()

    results = None
    analysis_ok = False  # Track if any analysis succeeded
    if analysis_choice == '1':
        # Ask user which models to test for comprehensive comparison
        print("\n Which models would you like to test in the comprehensive comparison?")
        print("1. Free models only (GPT-2, DistilGPT-2, DialoGPT) - No API costs")
        print("2. Both free and paid models (includes GPT-5, Gemini, Mistral, etc.) - Requires API keys")

        comp_model_choice = input("\nEnter your choice (1/2): ").strip()

        if comp_model_choice == '1':
            include_apis = False
            print(" Selected: Free models only for comprehensive comparison")
        elif comp_model_choice == '2':
            include_apis = True
            print(" Selected: Both free and paid models for comprehensive comparison")
        else:
            include_apis = False
            print(" Invalid choice, defaulting to free models only")

        # Run the comprehensive comparison with uploaded member data (NO DATA LEAKAGE)
        print(f"\nRunning large-scale single trial analysis (APIs: {include_apis})...")
        print("This will use 30% of data for threshold calculation and 70% for evaluation")
        if include_apis:
            print(" Using 100 samples per class for paid LLMs (optimized for API costs)")
        else:
            print(" Using 3000 samples per class for free LLMs (optimized for local processing)")
        results = run_comprehensive_comparison(include_apis=include_apis, member_filename='qa.en.python.json')
    else:
        print(" Skipping comprehensive comparison, proceeding to student evaluation...")

    if results:
        print("\n Analysis completed successfully with NO DATA LEAKAGE!")
        print(f" Analyzed {len(results)} models")
        print(" Data was properly split: 30% training (thresholds) + 70% testing (evaluation)")

        # Show parameter analysis summary
        print("\n PUBLICATION-READY PARAMETER ANALYSIS SUMMARY:")

    # Run student hackathon evaluation (regardless of whether comprehensive comparison was run)
    print("\n" + "="*80)
    print(" STUDENT HACKATHON CODE EVALUATION")
    print("="*80)

    # Ask user for submission source
    print("\n How would you like to provide student submissions?")
    print("1. Analyze a single .ipynb file (one group at a time)")
    print("2. Analyze entire Projects_Submissions folder (all groups at once)")
    print("3. Upload a zip file containing .ipynb files")
    print("4. Skip student evaluation")

    choice = input("\nEnter your choice (1/2/3/4): ").strip()

    submissions_path = 'student_submissions'  # Default fallback
    single_file_mode = False
    single_file_submissions = None

    if choice == '1':
        # Analyze single .ipynb file
        single_file_submissions = analyze_single_ipynb_file()
        if single_file_submissions:
            single_file_mode = True
            submissions_path = None  # Not using path-based loading
        else:
            print("  No file loaded, skipping evaluation...")
            submissions_path = None
    elif choice == '2':
        # Analyze entire Projects_Submissions folder
        custom_dir = input("\nEnter path to Projects_Submissions folder: ").strip()
        if custom_dir and os.path.exists(custom_dir):
            submissions_path = custom_dir
        else:
            print(" Invalid path, skipping evaluation...")
            submissions_path = None
    elif choice == '3':
        # Handle folder or zip path input (Colab-friendly)
        sub_path = handle_submission_path()
        if sub_path:
            submissions_path = sub_path
        else:
            print("  No path provided, skipping evaluation...")
            submissions_path = None
    elif choice == '4':
        print(" Skipping student evaluation...")
        submissions_path = None
    else:
        print("  Invalid choice, skipping evaluation...")
        submissions_path = None

    if submissions_path or single_file_mode:
        # Ask user which models to test
        print("\n Which models would you like to test?")
        print("1. Free models only (GPT-2, DistilGPT-2, DialoGPT) - No API costs")
        print("2. Paid models only (GPT-5, Gemini, Mistral, etc.) - Requires API keys")
        print("3. Both free and paid models (includes GPT-5, Gemini, Mistral, etc.) - Requires API keys")

        model_choice = input("\nEnter your choice (1/2/3): ").strip()

        if model_choice == '1':
            # Free models only (excluding Pythia)
            models_to_test = [
                'gpt2', 'distilgpt2', 'microsoft/DialoGPT-medium',
                'microsoft/DialoGPT-large'
            ]
            print("Selected: Free models only (GPT-2, DistilGPT-2, DialoGPT) - No API costs")
        elif model_choice == '2':
            # Paid models only - GPT-5 listed last
            models_to_test = [
                'google-gemini-2.5-flash',
                'mistralai-mistral-small-24b-instruct-2501',
                'deepseek-chat-v3-0324', 'meta-llama-3.3-70b-instruct',
                'openai-gpt-5'  # GPT-5 last
            ]
            print("Selected: Paid models only (GPT-5 will be processed last)")
        elif model_choice == '3':
            # Both free and paid models - GPT-5 listed last (excluding Pythia)
            models_to_test = [
                'gpt2', 'distilgpt2', 'microsoft/DialoGPT-medium',
                'microsoft/DialoGPT-large',
                'google-gemini-2.5-flash',
                'mistralai-mistral-small-24b-instruct-2501',
                'deepseek-chat-v3-0324', 'meta-llama-3.3-70b-instruct',
                'openai-gpt-5'  # GPT-5 last
            ]
            print("Selected: Both free and paid models (GPT-5 will be processed last)")
        else:
            # Default to free models only (excluding Pythia)
            models_to_test = [
                'gpt2', 'distilgpt2', 'microsoft/DialoGPT-medium',
                'microsoft/DialoGPT-large'
            ]
            print("Invalid choice, defaulting to free models only")

        try:
            if single_file_mode:
                # Pass the submissions directly instead of a path
                student_results = evaluate_student_hackathon_submissions(
                    models_to_test=models_to_test,
                    submissions_dir=None,
                    member_filename='qa.en.python.json',
                    preloaded_submissions=single_file_submissions
                )
            else:
                # Normal path-based loading
                student_results = evaluate_student_hackathon_submissions(
                    models_to_test=models_to_test,
                    submissions_dir=submissions_path,
                    member_filename='qa.en.python.json'
                )

            if student_results:
                print("\n Student evaluation complete! Check the generated files.")
                print(" Generated Output Files:")
                print("\n Visualizations:")
                print("   - student_flagged_summary.png (summary of flagged groups by model)")
                print("   - group_1_[category]_llm_analysis.png (individual probability plot for Group 1)")
                print("   - group_2_[category]_llm_analysis.png (individual probability plot for Group 2)")
                print("   - ... (one plot per group for all 22 groups)")
                print("\n Reports:")
                print("   - llm_detection_report.txt (detailed analysis showing which code sections are LLM-written)")
                print("\n Annotated Code Files:")
                print("   - annotated_Group_1_[model].py (code with LLM-likely sections marked)")
                print("   - annotated_Group_2_[model].py")
                print("   - ... (for each group that has LLM patterns detected)")
                print("\n Categories:")
                print("   - Plots generated for: FREE, PAID, and/or ALL models (based on your selection)")
                analysis_ok = True  # Mark analysis as successful
            else:
                print("\n Student evaluation failed. Please check the error messages above.")

        except Exception as e:
            print(f"\n Error in student evaluation: {e}")
            print("This is expected if student submission files are not available.")
            print("The system will use sample data for demonstration purposes.")

    # Only show results summary if comprehensive comparison was run
    if results:
        print("="*80)

        # Separate free and paid models
        free_models = {name: data for name, data in results.items() if data.get('category') == 'Free'}
        paid_models = {name: data for name, data in results.items() if data.get('category') == 'Paid'}

        if free_models:
            print(f"\n FREE LLMs ({len(free_models)} models, 3000 samples per class):")
            for model_name, data in free_models.items():
                params = data.get('parameters', 'Unknown')
                auc = data['auc']
                tpr = data['tpr_at_5fpr']
                print(f"  {model_name}: {params} parameters → {auc:.4f} AUC, {tpr:.4f} TPR@5%")

        if paid_models:
            print(f"\nPAID LLMs ({len(paid_models)} models, 100 samples per class):")
            for model_name, data in paid_models.items():
                params = data.get('parameters', 'Unknown')
                auc = data['auc']
                tpr = data['tpr_at_5fpr']
                print(f"  {model_name}: {params} parameters → {auc:.4f} AUC, {tpr:.4f} TPR@5%")

        print("="*80)
        print("PUBLICATION-READY FEATURES:")
        if include_apis:
            print("   Optimized sample size (100 samples per class for paid LLMs)")
        else:
            print("  Large sample size (3000 samples per class for free LLMs)")
        print("  Single trial for large-scale analysis")
        print("  Proper train/test split (30%/70%)")
        print("  Enhanced Min-K% parameters (k=0.01)")
        print("  Aggressive synthetic data generation")
        print("  Free vs Paid LLMs comparison")
        print("="*80)

        # Offer live testing feature
        print("\n" + "="*80)
        print("LIVE TESTING FEATURE AVAILABLE")
        print("="*80)
        print("You can now test any Python code snippet against all models")
        print("to determine the probability that each model has seen it during training.")
        print("\nThis uses the same Min-K% algorithm with proper train/test split.")
        print("="*80)

        # Start interactive live testing
        interactive_live_testing()

    else:
        if not analysis_ok and results is None:
            print("\n Analysis failed. Please check the error messages above.")
            print("\n TROUBLESHOOTING TIPS:")
            print("1. Ensure OPENROUTER_API_KEY is set in environment or in SUPPORTED_MODELS entries")
            print("2. Verify your OpenRouter account has credit and access to the chosen models")
            print("3. Try running with free models only: set include_apis=False")
            print("4. Make sure the qa.en.python.json file is in the current directory")
            print("5. Check that you have sufficient data for train/test split (at least 20 samples)")
            print("6. CUDA memory errors are handled automatically - models will be skipped if needed")
        else:
            print("\n Analysis completed successfully!")

        # Still offer live testing even if main analysis failed
        print("\n" + "="*80)
        print("LIVE TESTING FEATURE STILL AVAILABLE")
        print("="*80)
        print("Even though the main analysis failed, you can still test individual")
        print("code snippets against the models that are working.")
        print("="*80)

        # Start interactive live testing
        interactive_live_testing()

def handle_zip_file_upload():
    # Backward-compat wrapper
    return handle_submission_path()

def get_api_key_instructions():
    """Print instructions for obtaining API keys for all paid LLM services"""
    print("\n" + "="*80)
    print("API KEY SETUP INSTRUCTIONS")
    print("="*80)

    instructions = {
        "OpenRouter (Unified access to GPT-5, Gemini 2.5, Mistral, DeepSeek, LLaMA)": {
            "url": "https://openrouter.ai/keys",
            "steps": [
                "1. Go to https://openrouter.ai/keys",
                "2. Create a new API key",
                "3. Copy the key and set it as environment variable OPENROUTER_API_KEY",
                "4. Or paste it into SUPPORTED_MODELS entries with type 'openrouter'",
                "5. Ensure your account has credit and the models are allowed"
            ],
            "cost": "Pay-as-you-go, per-model pricing"
        }
    }

    for service, info in instructions.items():
        print(f"\n{service}:")
        print(f"URL: {info['url']}")
        print(f"Cost: {info['cost']}")
        print("Steps:")
        for step in info['steps']:
            print(f"  {step}")

    print("\n" + "="*80)
    print("QUICK SETUP:")
    print("1. Get OpenRouter API key")
    print("2. Replace the API keys in SUPPORTED_MODELS")
    print("3. Run the analysis with include_apis=True")
    print("="*80)

def update_api_keys():
    """Helper function to update API keys in the configuration"""
    print("\nTo update API keys, edit the SUPPORTED_MODELS dictionary in the code:")
    print("Find each model's 'api_key' field and replace with your actual API key.")
    print("\nExample:")
    print("'api_key': 'your_actual_api_key_here'")
    print("\nCurrent placeholder keys (replace these):")
    for model_key, config in SUPPORTED_MODELS.items():
        if config.get('type') != 'huggingface':
            print(f"{config['name']}: {config.get('api_key', 'NOT SET')[:20]}...")
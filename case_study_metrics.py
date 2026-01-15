import os
import re
import io
import json
import zipfile
import shutil
from typing import Dict, List, Tuple, Any
from difflib import SequenceMatcher


def is_running_in_colab() -> bool:
	try:
		import google.colab  # type: ignore
		return True
	except Exception:
		return False


def ensure_dirs(*paths: str) -> None:
	for p in paths:
		os.makedirs(p, exist_ok=True)


def normalize_group_name(name: str) -> str:
	if name is None:
		return ""
	name = str(name).strip().lower()
	name = name.replace(" ", "_")
	# Normalize common patterns like Group 1 -> group_1
	name = re.sub(r"^groupe?[_\s-]*", "group_", name)
	# Ensure group prefix
	if not name.startswith("group_"):
		m = re.search(r"(\d+)$", name)
		if m:
			name = f"group_{m.group(1)}"
	return name


def read_notebook(nb_bytes: bytes) -> Dict[str, Any]:
	import nbformat
	return nbformat.read(io.BytesIO(nb_bytes), as_version=4)


def extract_code_and_comments_from_notebook(nb: Dict[str, Any]) -> List[Dict[str, Any]]:
	results: List[Dict[str, Any]] = []
	for cell_idx, cell in enumerate(nb.get("cells", [])):
		if cell.get("cell_type") != "code":
			continue
		source = cell.get("source", "")
		if not source:
			continue
		for line_idx, raw_line in enumerate(source.splitlines(), start=1):
			line = raw_line.rstrip("\n\r")
			stripped = line.lstrip()
			if not stripped:
				continue
			is_comment = stripped.startswith("#")
			results.append(
				{
					"cell_index": cell_idx,
					"line_index": line_idx,
					"text": line,
					"type": "comment" if is_comment else "code",
				}
			)
	return results


CODE_BLOCK_RE = re.compile(r"```+([a-zA-Z0-9_+-]*)\n([\s\S]*?)\n```+", re.MULTILINE)


def looks_like_python_code(line: str) -> bool:
	line = line.strip()
	if not line:
		return False
	if line.startswith("#"):
		return True
	keywords = (
		"def ",
		"class ",
		"import ",
		"from ",
		"for ",
		"while ",
		"if ",
		"elif ",
		"else:",
		"try:",
		"except ",
		"with ",
	)
	if any(line.startswith(k) for k in keywords):
		return True
	if re.search(r"\w\s*=\s*[^=]", line):  # assignment
		return True
	if re.search(r"\bprint\s*\(", line):
		return True
	if re.search(r"\([\s\S]*?\)", line) and re.search(r"[a-zA-Z_]\w*\s*\(", line):  # func call
		return True
	return False


def extract_llm_lines_from_turn_text(text: str) -> List[str]:
	"""
	Extract code lines from LLM conversation text.
	Handles both user and assistant messages.
	"""
	lines: List[str] = []
	if not text:
		return lines

	# Split text into lines for processing
	all_lines = text.splitlines()

	# 1) Code blocks fenced by backticks (most reliable)
	for _lang, block in CODE_BLOCK_RE.findall(text):
		for bline in block.splitlines():
			clean_line = bline.rstrip()
			if clean_line and not clean_line.startswith('```'):
				lines.append(clean_line)

	# 2) Look for Python code patterns in the raw text
	for raw_line in all_lines:
		clean_line = raw_line.strip()
		if not clean_line:
			continue

		# Skip markdown headers but keep Python comments
		if clean_line.startswith('###') or clean_line.startswith('##') or clean_line.startswith('**'):
			continue
		if clean_line.startswith('*') and not clean_line.startswith('*args'):
			continue
		if clean_line.startswith('-') and not '->' in clean_line:
			continue

		# Look for Python patterns
		if looks_like_python_code(clean_line):
			lines.append(clean_line)

	# 3) Also extract any line that contains common Python keywords or patterns
	python_indicators = [
		'import ', 'from ', 'def ', 'class ', 'if ', 'for ', 'while ', 'try:', 'except',
		'print(', 'return ', '= ', '==', '!=', '<=', '>=', 'and ', 'or ', 'not ',
		'len(', 'range(', 'str(', 'int(', 'float(', 'list(', 'dict(', 'tuple(',
		'append(', 'extend(', 'remove(', 'pop(', 'insert(', 'sort(', 'replace(',
		'open(', 'read(', 'write(', 'close(', 'split(', 'join(', 'strip(',
		'# ', '"""', "'''", 'plt.', 'np.', 'pd.', 'df.', 'sns.', 'cv2.', 'os.',
		'.fit(', '.predict(', '.transform(', '.iloc', '.loc', '.head(', '.tail(',
		'.shape', '.dtype', '.values', '.columns', '.index', '.groupby(', '.merge(',
		'lambda ', 'yield ', 'with ', 'as ', 'in ', '@', 'self.', '__init__', '__str__'
	]

	for raw_line in all_lines:
		clean_line = raw_line.strip()
		if not clean_line:
			continue

		# Check if line contains Python indicators
		if any(indicator in clean_line for indicator in python_indicators):
			# Additional filter to avoid prose
			if not clean_line.startswith('Here') and not clean_line.startswith('This') and not clean_line.startswith('The '):
				lines.append(clean_line)

	# Remove duplicates while preserving order
	seen = set()
	unique_lines = []
	for line in lines:
		if line not in seen:
			seen.add(line)
			unique_lines.append(line)

	return unique_lines


def load_llm_conversations(json_path: str) -> Dict[str, List[str]]:
	"""
	Load LLM conversations and aggregate all conversations for each group.
	Handles multiple students per group (each with their own conversation).
	Extracts code from both user and assistant messages.
	"""
	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)
	group_to_lines: Dict[str, List[str]] = {}
	group_conversation_count: Dict[str, int] = {}

	print(f"Loading LLM conversations from: {json_path}")
	print(f"DEBUG: JSON root type: {type(data)}")

	# Debug: Print first record structure
	if isinstance(data, dict):
		print(f"DEBUG: JSON is dict with keys: {list(data.keys())}")
		if "records" in data and isinstance(data["records"], list) and len(data["records"]) > 0:
			print(f"DEBUG: First record keys: {list(data['records'][0].keys())}")
			print(f"DEBUG: First record sample: {str(data['records'][0])[:200]}...")
	elif isinstance(data, list) and len(data) > 0:
		print(f"DEBUG: JSON is list with {len(data)} items")
		print(f"DEBUG: First item keys: {list(data[0].keys())}")
		print(f"DEBUG: First item sample: {str(data[0])[:200]}...")

	if isinstance(data, dict) and "records" in data and isinstance(data["records"], list):
		records = data["records"]
		print(f"Found {len(records)} records in 'records' field")
	else:
		records = data if isinstance(data, list) else []
		print(f"Found {len(records)} records in root array")

	for i, rec in enumerate(records):
		# Debug first few records
		if i < 3:
			print(f"\nDEBUG Record {i}: {list(rec.keys())}")

		# Try different field names for group identification
		group = rec.get("Group") or rec.get("group") or rec.get("groupe") or rec.get("GroupId") or rec.get("group_id") or rec.get("GROUP")

		# Debug group extraction
		if i < 3:
			print(f"  Group field value: {group}")

		# Handle numeric groups (convert "1" to "group_1")
		if group is not None:
			group_str = str(group).strip()
			if group_str.isdigit():
				group = f"group_{group_str}"
			else:
				group = normalize_group_name(group)
		else:
			group = f"unknown_{i}"
			if i < 3:
				print(f"  WARNING: No group field found, using '{group}'")

		# Track how many conversations per group
		group_conversation_count[group] = group_conversation_count.get(group, 0) + 1

		# Extract from turns/messages
		turns = rec.get("Turns") or rec.get("turns") or rec.get("messages") or rec.get("conversation") or rec.get("TURNS") or rec.get("Messages") or []

		# Debug turns extraction
		if i < 3:
			print(f"  Turns type: {type(turns)}, Length: {len(turns) if isinstance(turns, list) else 'N/A'}")
			if isinstance(turns, list) and len(turns) > 0:
				print(f"  First turn type: {type(turns[0])}")
				if isinstance(turns[0], dict):
					print(f"  First turn keys: {list(turns[0].keys())}")
					# Check for nested Turn
					if "Turn" in turns[0]:
						nested_turn = turns[0]["Turn"]
						print(f"  Nested 'Turn' found with keys: {list(nested_turn.keys())}")
						if "User" in nested_turn:
							print(f"  Sample User msg: {str(nested_turn['User'])[:100]}...")
						if "Assistant" in nested_turn:
							print(f"  Sample Assistant msg: {str(nested_turn['Assistant'])[:100]}...")

		all_text = []
		if isinstance(turns, list):
			# Each turn might be a dict with 'user' and 'assistant' or 'role' and 'content'
			# OR wrapped in a "Turn" object
			for turn in turns:
				if isinstance(turn, dict):
					# Check if there's a nested "Turn" object (your format)
					if "Turn" in turn:
						turn = turn["Turn"]

					# Try different formats
					user_msg = turn.get("user") or turn.get("User") or turn.get("USER") or ""
					assistant_msg = turn.get("assistant") or turn.get("Assistant") or turn.get("ASSISTANT") or ""

					# Alternative format: role/content
					role = turn.get("role") or turn.get("Role") or turn.get("ROLE") or ""
					content = turn.get("content") or turn.get("Content") or turn.get("CONTENT") or ""

					if role:
						if role.lower() == "user" and content:
							user_msg = content
						elif role.lower() == "assistant" and content:
							assistant_msg = content

					if user_msg:
						all_text.append(str(user_msg))
					if assistant_msg:
						all_text.append(str(assistant_msg))
				else:
					# If turn is just a string
					all_text.append(str(turn))

			joined = "\n\n".join(all_text)
		else:
			joined = str(turns)

		lines = extract_llm_lines_from_turn_text(joined)

		if i < 3:
			print(f"  Extracted {len(lines)} code lines from conversation")
			if len(lines) > 0:
				print(f"  Sample: {lines[0][:80]}...")

		print(f"Record {i}: Group='{group}' (Conversation #{group_conversation_count[group]}) -> {len(lines)} code/comment lines")
		group_to_lines.setdefault(group, []).extend(lines)

	# Deduplicate while preserving order
	for g, lst in list(group_to_lines.items()):
		seen = set()
		uniq: List[str] = []
		for item in lst:
			key = item.strip()
			if key and key not in seen:
				seen.add(key)
				uniq.append(item)
		group_to_lines[g] = uniq

	print(f"\nLoaded LLM data for {len(group_to_lines)} groups:")
	for group, lines in group_to_lines.items():
		conv_count = group_conversation_count.get(group, 0)
		print(f"  - {group}: {conv_count} conversation(s), {len(lines)} unique code lines")

	return group_to_lines


def fuzzy_ratio(a: str, b: str) -> float:
	"""Character-level similarity using SequenceMatcher"""
	return SequenceMatcher(None, a.strip(), b.strip()).ratio()


def tokenize_code(text: str) -> List[str]:
	"""
	Tokenize code into meaningful tokens (words, operators, identifiers)
	"""
	# Simple tokenization - split by whitespace and common delimiters
	# but keep operators together
	tokens = []
	current_token = ""

	for char in text:
		if char in " \t\n\r":
			if current_token:
				tokens.append(current_token)
				current_token = ""
		elif char in "()[]{},:;":
			if current_token:
				tokens.append(current_token)
				current_token = ""
			tokens.append(char)
		else:
			current_token += char

	if current_token:
		tokens.append(current_token)

	return [t for t in tokens if t.strip()]


def token_similarity(a: str, b: str) -> float:
	"""
	Token-level similarity - compares code as sequence of tokens
	More robust for code where whitespace/formatting may differ
	"""
	tokens_a = tokenize_code(a)
	tokens_b = tokenize_code(b)

	if not tokens_a or not tokens_b:
		return 0.0

	# Use SequenceMatcher on token sequences
	matcher = SequenceMatcher(None, tokens_a, tokens_b)
	return matcher.ratio()


def jaccard_similarity(a: str, b: str) -> float:
	"""
	Jaccard similarity based on token sets (order-independent)
	Good for detecting similar code regardless of token order
	"""
	tokens_a = set(tokenize_code(a.lower()))
	tokens_b = set(tokenize_code(b.lower()))

	if not tokens_a or not tokens_b:
		return 0.0

	intersection = len(tokens_a & tokens_b)
	union = len(tokens_a | tokens_b)

	return intersection / union if union > 0 else 0.0


def combined_similarity(a: str, b: str) -> float:
	"""
	Combined similarity score using multiple techniques
	Returns the maximum of character-level, token-level, and Jaccard
	"""
	char_sim = fuzzy_ratio(a, b)
	token_sim = token_similarity(a, b)
	jaccard_sim = jaccard_similarity(a, b)

	# Return the maximum - if ANY technique finds high similarity, we consider it a match
	return max(char_sim, token_sim, jaccard_sim)


def normalize_for_matching(text: str) -> str:
	"""Normalize text for better matching"""
	if not text:
		return ""

	# Remove extra whitespace
	text = ' '.join(text.split())

	# Remove common prefixes/suffixes that might differ
	text = text.strip()

	# Normalize quotes
	text = text.replace('"', '"').replace('"', '"')
	text = text.replace("'", "'").replace("'", "'")

	# Normalize whitespace around operators
	text = re.sub(r'\s*=\s*', ' = ', text)
	text = re.sub(r'\s*==\s*', ' == ', text)
	text = re.sub(r'\s*!=\s*', ' != ', text)

	return text


def match_lines(
    nb_items: List[Dict[str, Any]],
    llm_lines: List[str],
    exact_case_sensitive: bool = False,
    fuzzy_threshold: float = 0.7,  # Lowered threshold for combined similarity
    enable_fuzzy: bool = True,
    max_fuzzy_candidates_per_line: int = 1000,  # Increased
    length_tolerance_ratio: float = 0.7,  # More lenient
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Match notebook lines against LLM conversation lines using multiple techniques:
    1. Exact matching (case-insensitive after normalization)
    2. Character-level fuzzy matching (SequenceMatcher)
    3. Token-level matching (code tokenization)
    4. Jaccard similarity (set-based token matching)
    """
    exact_matches: List[Dict[str, Any]] = []
    fuzzy_matches: List[Dict[str, Any]] = []

    # Normalize LLM lines for better matching
    normalized_llm_lines = []
    for line in llm_lines:
        normalized = normalize_for_matching(line)
        if normalized:
            normalized_llm_lines.append(normalized)

    # Prepare sets for exact matching
    if exact_case_sensitive:
        llm_exact = set(normalized_llm_lines)
    else:
        llm_exact = set(line.lower() for line in normalized_llm_lines)

    total = len(nb_items)
    print(f"    Starting matching: {total} notebook lines vs {len(normalized_llm_lines)} LLM lines")
    print(f"    Using combined similarity (char-level + token-level + Jaccard)")
    print(f"    Exact matching (case_sensitive={exact_case_sensitive}): {len(llm_exact)} unique LLM lines")

    for idx, item in enumerate(nb_items, start=1):
        if idx % 50 == 0:  # More frequent progress updates
            print(f"      Progress: {idx}/{total} notebook lines processed...")

        nb_text = item["text"]
        nb_normalized = normalize_for_matching(nb_text)

        if not nb_normalized:
            continue

        probe = nb_normalized if exact_case_sensitive else nb_normalized.lower()

        # Exact matching
        if probe in llm_exact:
            # Find the original LLM line for better output
            original_llm = nb_text
            for i, norm_line in enumerate(normalized_llm_lines):
                if (norm_line if exact_case_sensitive else norm_line.lower()) == probe:
                    original_llm = llm_lines[i] if i < len(llm_lines) else norm_line
                    break

            exact_matches.append({
                "notebook": item,
                "llm_text": original_llm,
                "similarity": 1.0,
                "match_type": "exact"
            })
            if len(exact_matches) <= 10:
                print(f"    ✓ EXACT #{len(exact_matches)}: '{nb_text[:60]}...'")
            continue

        # Fuzzy matching with combined similarity
        if not enable_fuzzy:
            continue

        best_sim = 0.0
        best_llm = None
        best_llm_idx = -1
        match_type = "none"

        # Try combined similarity matching with all LLM lines
        for llm_idx, llm_line in enumerate(normalized_llm_lines):
            # Use combined similarity (character + token + jaccard)
            sim = combined_similarity(nb_normalized, llm_line)

            if sim > best_sim:
                best_sim = sim
                best_llm = llm_line
                best_llm_idx = llm_idx

                # Determine which technique gave best result
                char_sim = fuzzy_ratio(nb_normalized, llm_line)
                token_sim = token_similarity(nb_normalized, llm_line)
                jaccard_sim = jaccard_similarity(nb_normalized, llm_line)

                if sim == char_sim:
                    match_type = "character"
                elif sim == token_sim:
                    match_type = "token"
                else:
                    match_type = "jaccard"

        if best_llm is not None and best_sim >= fuzzy_threshold:
            # Find original LLM line for display
            original_llm = llm_lines[best_llm_idx] if best_llm_idx < len(llm_lines) else best_llm

            fuzzy_matches.append({
                "notebook": item,
                "llm_text": original_llm,
                "similarity": float(best_sim),
                "match_type": match_type
            })
            if len(fuzzy_matches) <= 10:
                print(f"    ≈ FUZZY #{len(fuzzy_matches)}: sim={best_sim:.3f} ({match_type}) '{nb_text[:40]}...'")

    print(f"    Matching complete: {len(exact_matches)} exact, {len(fuzzy_matches)} fuzzy")
    return {"exact": exact_matches, "fuzzy": fuzzy_matches}


def write_group_outputs(
	output_dir: str,
	group_name: str,
	matches: Dict[str, List[Dict[str, Any]]],
	nb_path: str,
	llm_count: int,
	nb_total_lines: int = 0,
) -> Tuple[str, str, Dict[str, int]]:
	"""
	Write matching results to JSON and TXT files.
	The TXT file contains human-readable intersection of code/comments
	between the notebook and LLM conversations.
	Returns: (json_path, txt_path, fuzzy_by_type_dict)
	"""
	ensure_dirs(output_dir)
	json_path = os.path.join(output_dir, f"{group_name}_matches.json")
	txt_path = os.path.join(output_dir, f"{group_name}_intersection.txt")

	exact = matches.get("exact", [])
	fuzzy = matches.get("fuzzy", [])
	total_matches = len(exact) + len(fuzzy)

	# Count fuzzy matches by type
	fuzzy_by_type = {"character": 0, "token": 0, "jaccard": 0}
	for match in fuzzy:
		match_type = match.get("match_type", "unknown")
		if match_type in fuzzy_by_type:
			fuzzy_by_type[match_type] += 1

	# JSON detailed output
	with open(json_path, "w", encoding="utf-8") as jf:
		json.dump(
			{
				"group": group_name,
				"notebook": nb_path,
				"notebook_total_lines": nb_total_lines,
				"llm_candidate_lines_count": llm_count,
				"total_matches": total_matches,
				"exact_matches_count": len(exact),
				"fuzzy_matches_count": len(fuzzy),
				"fuzzy_matches_by_type": fuzzy_by_type,
				"exact_matches": exact,
				"fuzzy_matches": fuzzy,
			},
			jf,
			ensure_ascii=False,
			indent=2,
		)

	# Human-readable TXT summary showing intersection
	with open(txt_path, "w", encoding="utf-8") as tf:
		tf.write("=" * 80 + "\n")
		tf.write(f"CODE/COMMENT INTERSECTION REPORT FOR {group_name.upper()}\n")
		tf.write("=" * 80 + "\n\n")

		tf.write(f"Notebook: {nb_path}\n")
		tf.write(f"Total notebook lines analyzed: {nb_total_lines}\n")
		tf.write(f"Total LLM conversation lines: {llm_count}\n")
		tf.write(f"Total matches found: {total_matches}\n")
		tf.write(f"  - Exact matches: {len(exact)}\n")
		tf.write(f"  - Fuzzy matches: {len(fuzzy)}\n")
		tf.write(f"    • Character-level: {fuzzy_by_type['character']}\n")
		tf.write(f"    • Token-level: {fuzzy_by_type['token']}\n")
		tf.write(f"    • Jaccard similarity: {fuzzy_by_type['jaccard']}\n")

		if total_matches > 0:
			match_percentage = (total_matches / nb_total_lines * 100) if nb_total_lines > 0 else 0
			tf.write(f"  - Match rate: {match_percentage:.1f}% of notebook lines\n")

		tf.write("\n" + "=" * 80 + "\n\n")

		def write_block(title: str, items: List[Dict[str, Any]]):
			if not items:
				tf.write(f"{title}: None found\n\n")
				return

			tf.write(f"{title}: {len(items)} found\n")
			tf.write("-" * 80 + "\n\n")

			for idx, m in enumerate(items, 1):
				nb = m["notebook"]
				line_info = f"Cell {nb['cell_index']}, Line {nb['line_index']}, Type: {nb['type']}"
				match_type = m.get("match_type", "unknown")
				similarity = m.get("similarity", 0.0)

				tf.write(f"[Match #{idx}] {match_type.upper()} (similarity: {similarity:.3f})\n")
				tf.write(f"Location: {line_info}\n")
				tf.write(f"Notebook code:\n  {nb['text']}\n")
				tf.write(f"LLM conversation:\n  {m['llm_text']}\n")
				tf.write("-" * 80 + "\n")

		write_block("EXACT MATCHES", exact)
		write_block("FUZZY MATCHES", fuzzy)

		tf.write("\n" + "=" * 80 + "\n")
		tf.write("END OF REPORT\n")
		tf.write("=" * 80 + "\n")

	return json_path, txt_path, fuzzy_by_type


def extract_zip(zip_path: str, extract_to: str) -> None:
	if os.path.exists(extract_to):
		shutil.rmtree(extract_to)
	with zipfile.ZipFile(zip_path, 'r') as zf:
		zf.extractall(extract_to)


def find_group_notebooks(root_dir: str) -> Dict[str, str]:
	group_to_nb: Dict[str, str] = {}
	print(f"Searching for notebooks in: {root_dir}")

	for dirpath, dirnames, filenames in os.walk(root_dir):
		print(f"Checking directory: {dirpath}")
		print(f"  - Subdirectories: {dirnames}")
		print(f"  - Files: {[f for f in filenames if f.endswith('.ipynb')]}")

		for filename in filenames:
			if filename.lower().endswith(".ipynb"):
				# Try to extract group name from the directory structure
				# Look for patterns like: group_1, Group 1, groupe_1, etc.
				parent_dir = os.path.basename(dirpath)
				group = normalize_group_name(parent_dir)

				# If parent directory doesn't look like a group, try the current directory name
				if not group or group == "submissions_extracted":
					group = normalize_group_name(os.path.basename(dirpath))

				# If still no group found, try to extract from filename
				if not group or group == "submissions_extracted":
					# Try to extract group number from filename
					import re
					match = re.search(r'group[_\s-]*(\d+)', filename, re.IGNORECASE)
					if match:
						group = f"group_{match.group(1)}"
					else:
						group = "unknown"

				full_path = os.path.join(dirpath, filename)
				print(f"  -> Found notebook: {filename} -> Group: {group}")

				# If multiple notebooks per group, prefer the first found; can be extended as needed
				group_to_nb.setdefault(group, full_path)

	print(f"Found {len(group_to_nb)} groups with notebooks:")
	for group, path in group_to_nb.items():
		print(f"  - {group}: {path}")

	return group_to_nb


def process_all(
	submissions_zip: str = "submissions.zip",
	conversations_json: str = "conversations_turns_with_groups.json",
	output_dir: str = "results",
	work_dir: str = "submissions_extracted",
	exact_case_sensitive: bool = False,
	fuzzy_threshold: float = 0.9,
) -> Dict[str, Dict[str, Any]]:
	if not os.path.exists(submissions_zip):
		raise FileNotFoundError(
			f"Could not find '{submissions_zip}'. Please upload it to the current directory."
		)
	if not os.path.exists(conversations_json):
		raise FileNotFoundError(
			f"Could not find '{conversations_json}'. Please upload it to the current directory."
		)

	ensure_dirs(output_dir)
	extract_zip(submissions_zip, work_dir)

	# Debug: Check what was extracted
	print(f"Contents of extracted directory '{work_dir}':")
	if os.path.exists(work_dir):
		for root, dirs, files in os.walk(work_dir):
			level = root.replace(work_dir, '').count(os.sep)
			indent = ' ' * 2 * level
			print(f"{indent}{os.path.basename(root)}/")
			subindent = ' ' * 2 * (level + 1)
			for file in files:
				print(f"{subindent}{file}")
	else:
		print("Extracted directory does not exist!")

	print("=" * 50)
	print("STEP 1: Finding notebooks...")
	group_to_nb = find_group_notebooks(work_dir)
	print("=" * 50)
	print("STEP 2: Loading LLM conversations...")
	group_to_llm_lines = load_llm_conversations(conversations_json)
	print("=" * 50)
	print("STEP 3: Starting matching process...")

	summary: Dict[str, Dict[str, Any]] = {}
	print(f"Found {len(group_to_nb)} notebook groups. Beginning per-group matching...")
	if not group_to_nb:
		print("WARNING: No notebooks were found. Check ZIP structure and that .ipynb files exist.")
		return {}
	for group, nb_path in group_to_nb.items():
		try:
			with open(nb_path, "rb") as f:
				nb = read_notebook(f.read())
		except Exception as e:
			print(f"[WARN] Failed to read notebook for {group} at {nb_path}: {e}")
			continue

		nb_items = extract_code_and_comments_from_notebook(nb)
		llm_lines = group_to_llm_lines.get(group, [])
		if not llm_lines and group.startswith("group_"):
			# Try a few alternate group keys
			alt_keys = [
				group.replace("group_", "group"),
				group.replace("group_", "groupe_"),
			]
			for alt in alt_keys:
				if alt in group_to_llm_lines:
					llm_lines = group_to_llm_lines[alt]
					break

		print(f"Group '{group}': {len(nb_items)} notebook lines, {len(llm_lines)} LLM lines")

		# Debug: Show sample lines from both sources
		if nb_items:
			print(f"  Sample notebook lines (first 3):")
			for i, item in enumerate(nb_items[:3]):
				print(f"    [{i+1}] {item['text'][:100]}...")

		if llm_lines:
			print(f"  Sample LLM lines (first 3):")
			for i, line in enumerate(llm_lines[:3]):
				print(f"    [{i+1}] {line[:100]}...")
		else:
			print(f"  No LLM lines found for group '{group}'")

		matches = match_lines(
			nb_items=nb_items,
			llm_lines=llm_lines,
			exact_case_sensitive=exact_case_sensitive,
			fuzzy_threshold=0.65,  # More lenient threshold for combined similarity
			enable_fuzzy=True,
			max_fuzzy_candidates_per_line=1000,
			length_tolerance_ratio=0.7,
		)

		print(f"  -> Found {len(matches.get('exact', []))} exact matches, {len(matches.get('fuzzy', []))} fuzzy matches")
		json_p, txt_p, fuzzy_by_type = write_group_outputs(
			output_dir=output_dir,
			group_name=group,
			matches=matches,
			nb_path=nb_path,
			llm_count=len(llm_lines),
			nb_total_lines=len(nb_items),
		)
		summary[group] = {
			"notebook": nb_path,
			"llm_candidate_lines": len(llm_lines),
			"nb_lines": len(nb_items),
			"exact_matches": len(matches.get("exact", [])),
			"fuzzy_matches": len(matches.get("fuzzy", [])),
			"fuzzy_by_type": fuzzy_by_type,
			"json": json_p,
			"txt": txt_p,
		}

	return summary


def write_summary_report(output_dir: str, summary: Dict[str, Dict[str, Any]]) -> str:
	"""
	Write an overall summary report across all groups
	"""
	summary_path = os.path.join(output_dir, "OVERALL_SUMMARY.txt")

	with open(summary_path, "w", encoding="utf-8") as f:
		f.write("=" * 80 + "\n")
		f.write("LLM CODE DETECTION - OVERALL SUMMARY REPORT\n")
		f.write("=" * 80 + "\n\n")

		total_exact = 0
		total_fuzzy = 0
		total_nb_lines = 0
		total_llm_lines = 0

		f.write(f"Total groups analyzed: {len(summary)}\n\n")
		f.write("-" * 80 + "\n")
		f.write("PER-GROUP BREAKDOWN:\n")
		f.write("-" * 80 + "\n\n")

		# Track overall fuzzy breakdown
		total_fuzzy_by_type = {"character": 0, "token": 0, "jaccard": 0}

		for group, info in sorted(summary.items()):
			exact = info['exact_matches']
			fuzzy = info['fuzzy_matches']
			fuzzy_by_type = info.get('fuzzy_by_type', {"character": 0, "token": 0, "jaccard": 0})
			nb_lines = info['nb_lines']
			llm_lines = info['llm_candidate_lines']
			total_matches = exact + fuzzy
			match_rate = (total_matches / nb_lines * 100) if nb_lines > 0 else 0

			total_exact += exact
			total_fuzzy += fuzzy
			total_nb_lines += nb_lines
			total_llm_lines += llm_lines
			for match_type, count in fuzzy_by_type.items():
				total_fuzzy_by_type[match_type] += count

			f.write(f"GROUP: {group}\n")
			f.write(f"  Notebook: {info['notebook']}\n")
			f.write(f"  Notebook lines: {nb_lines}\n")
			f.write(f"  LLM conversation lines: {llm_lines}\n")
			f.write(f"  Total matches: {total_matches} ({match_rate:.1f}% of notebook)\n")
			f.write(f"    - Exact: {exact}\n")
			f.write(f"    - Fuzzy: {fuzzy}\n")
			f.write(f"      • Character-level: {fuzzy_by_type['character']}\n")
			f.write(f"      • Token-level: {fuzzy_by_type['token']}\n")
			f.write(f"      • Jaccard similarity: {fuzzy_by_type['jaccard']}\n")
			f.write(f"  Details: {info['txt']}\n")
			f.write("\n")

		f.write("=" * 80 + "\n")
		f.write("OVERALL STATISTICS:\n")
		f.write("=" * 80 + "\n")
		f.write(f"Total groups processed: {len(summary)}\n")
		f.write(f"Total notebook lines analyzed: {total_nb_lines}\n")
		f.write(f"Total LLM conversation lines: {total_llm_lines}\n")
		f.write(f"Total exact matches: {total_exact}\n")
		f.write(f"Total fuzzy matches: {total_fuzzy}\n")
		f.write(f"  - Character-level: {total_fuzzy_by_type['character']}\n")
		f.write(f"  - Token-level: {total_fuzzy_by_type['token']}\n")
		f.write(f"  - Jaccard similarity: {total_fuzzy_by_type['jaccard']}\n")
		f.write(f"Total matches: {total_exact + total_fuzzy}\n")
		if total_nb_lines > 0:
			overall_rate = ((total_exact + total_fuzzy) / total_nb_lines * 100)
			f.write(f"Overall match rate: {overall_rate:.1f}%\n")

		f.write("\n" + "=" * 80 + "\n")
		f.write("INTERPRETATION:\n")
		f.write("=" * 80 + "\n")
		f.write("- Exact matches: Code lines that match identically after normalization\n")
		f.write("- Fuzzy matches: Code lines with high similarity (>65%) using:\n")
		f.write("  * Character-level matching (SequenceMatcher)\n")
		f.write("  * Token-level matching (code tokenization)\n")
		f.write("  * Jaccard similarity (set-based token comparison)\n")
		f.write("- Higher match rates suggest stronger correlation between\n")
		f.write("  submitted code and LLM conversations\n")
		f.write("=" * 80 + "\n")

	return summary_path


def colab_download_folder(folder: str) -> None:
	if not is_running_in_colab():
		return
	try:
		from google.colab import files  # type: ignore
		for name in os.listdir(folder):
			path = os.path.join(folder, name)
			if os.path.isfile(path):
				files.download(path)  # type: ignore
	except Exception as e:
		print(f"[WARN] Auto-download failed: {e}")


def find_file_by_pattern(pattern: str, directory: str = ".") -> str:
	"""
	Find a file matching the given pattern in the directory.
	Returns the first match or empty string if not found.
	"""
	import glob
	matches = glob.glob(os.path.join(directory, pattern))
	return matches[0] if matches else ""


def find_required_files() -> Tuple[str, str]:
	"""
	Find the required files with flexible naming patterns.
	Returns (submissions_zip_path, conversations_json_path)
	"""
	# Look for submissions zip file with various naming patterns
	submissions_patterns = [
		"Submissions.zip",  # Most common case
		"submissions.zip",
		"Submissions*.zip",
		"submissions*.zip",
		"*Submissions*.zip",
		"*submissions*.zip"
	]

	submissions_file = ""
	for pattern in submissions_patterns:
		submissions_file = find_file_by_pattern(pattern)
		if submissions_file:
			break

	# Look for conversations json file with various naming patterns
	conversations_patterns = [
		"conversations_turns_with_groups.json",
		"conversations_turns_with_groups*.json",
		"*conversations*.json",
		"*turns*.json"
	]

	conversations_file = ""
	for pattern in conversations_patterns:
		conversations_file = find_file_by_pattern(pattern)
		if conversations_file:
			break

	return submissions_file, conversations_file


def main() -> None:
	print("=== LLM Code Detection Pipeline ===")
	print("This script will help you detect if student code was generated with LLM assistance.")
	print()

	# Check if running in Colab
	if is_running_in_colab():
		print("Running in Google Colab...")
		print("Please upload the following files:")
		print("1. submissions.zip (containing group folders with .ipynb files)")
		print("2. conversations_turns_with_groups.json (containing LLM chat logs)")
		print()

		# Check if files exist, if not prompt for upload
		submissions_file, conversations_file = find_required_files()

		if not submissions_file or not conversations_file:
			missing = []
			if not submissions_file:
				missing.append("submissions.zip")
			if not conversations_file:
				missing.append("conversations_turns_with_groups.json")

			print(f"Missing files: {', '.join(missing)}")
			print("Please upload the missing files using the file upload widget below:")
			print()
			try:
				from google.colab import files  # type: ignore
				uploaded = files.upload()  # type: ignore
				print(f"Successfully uploaded {len(uploaded)} file(s):")
				for filename in uploaded.keys():
					print(f"  - {filename}")
				print()

				# Try to find files again after upload
				submissions_file, conversations_file = find_required_files()
			except Exception as e:
				print(f"Upload failed: {e}")
				print("Please try uploading the files manually using the file browser on the left.")
				return
	else:
		print("Running locally...")
		print("Ensure 'submissions.zip' and 'conversations_turns_with_groups.json' are in the current directory.")
		print()
		submissions_file, conversations_file = find_required_files()

	# Verify files exist before processing
	if not submissions_file:
		print("Error: Could not find submissions zip file.")
		print("Please ensure a file matching 'submissions*.zip' or 'Submissions*.zip' is present.")
		return

	if not conversations_file:
		print("Error: Could not find conversations JSON file.")
		print("Please ensure a file matching '*conversations*.json' or '*turns*.json' is present.")
		return

	print("Found required files:")
	print(f"  - Submissions: {submissions_file}")
	print(f"  - Conversations: {conversations_file}")
	print()
	print("Starting processing...")
	print("=" * 50)

	try:
		print("Starting process_all function...")
		summary = process_all(submissions_zip=submissions_file, conversations_json=conversations_file)
		print("=" * 80)
		print("PROCESSING COMPLETED SUCCESSFULLY!")
		print("=" * 80)
		print()
		print("SUMMARY OF RESULTS:")
		print("-" * 80)

		total_exact = 0
		total_fuzzy = 0
		total_nb_lines = 0

		# Track overall fuzzy breakdown
		total_fuzzy_by_type = {"character": 0, "token": 0, "jaccard": 0}

		for group, info in summary.items():
			exact = info['exact_matches']
			fuzzy = info['fuzzy_matches']
			fuzzy_by_type = info.get('fuzzy_by_type', {"character": 0, "token": 0, "jaccard": 0})
			nb_lines = info['nb_lines']
			total_matches = exact + fuzzy
			match_rate = (total_matches / nb_lines * 100) if nb_lines > 0 else 0

			total_exact += exact
			total_fuzzy += fuzzy
			total_nb_lines += nb_lines
			for match_type, count in fuzzy_by_type.items():
				total_fuzzy_by_type[match_type] += count

			print(f"\nGroup: {group}")
			print(f"  Notebook lines: {nb_lines}")
			print(f"  LLM conversation lines: {info['llm_candidate_lines']}")
			print(f"  Matches found: {total_matches} ({match_rate:.1f}%)")
			print(f"    - Exact matches: {exact}")
			print(f"    - Fuzzy matches: {fuzzy}")
			print(f"      • Character-level: {fuzzy_by_type['character']}")
			print(f"      • Token-level: {fuzzy_by_type['token']}")
			print(f"      • Jaccard similarity: {fuzzy_by_type['jaccard']}")
			print(f"  Output files:")
			print(f"    - JSON: {info['json']}")
			print(f"    - TXT:  {info['txt']}")

		print("\n" + "-" * 80)
		print("OVERALL STATISTICS:")
		print(f"  Total groups processed: {len(summary)}")
		print(f"  Total notebook lines: {total_nb_lines}")
		print(f"  Total exact matches: {total_exact}")
		print(f"  Total fuzzy matches: {total_fuzzy}")
		print(f"    - Character-level: {total_fuzzy_by_type['character']}")
		print(f"    - Token-level: {total_fuzzy_by_type['token']}")
		print(f"    - Jaccard similarity: {total_fuzzy_by_type['jaccard']}")
		print(f"  Total matches: {total_exact + total_fuzzy}")
		if total_nb_lines > 0:
			overall_rate = ((total_exact + total_fuzzy) / total_nb_lines * 100)
			print(f"  Overall match rate: {overall_rate:.1f}%")
		print("-" * 80)

		# Write overall summary report
		summary_report_path = write_summary_report("results", summary)
		print(f"\n✓ Overall summary report saved: {summary_report_path}")

		if is_running_in_colab():
			print("\nAttempting to download result files...")
			colab_download_folder("results")
			print("✓ Download completed! Check your downloads folder.")
		else:
			print(f"\n✓ All results saved in 'results/' directory")
			print("  Files generated:")
			print("    - OVERALL_SUMMARY.txt (summary of all groups)")
			print("    - {group}_intersection.txt (per-group detailed matches)")
			print("    - {group}_matches.json (per-group detailed data)")

	except Exception as e:
		import traceback
		print("=" * 80)
		print(f"ERROR DURING PROCESSING: {e}")
		print("=" * 80)
		print("\nFull error traceback:")
		traceback.print_exc()
		print("\nPlease check your files and try again.")


def test_extraction():
	"""
	Test function to debug what's being extracted from notebooks and LLM data.
	Run this to see actual content samples.
	"""
	print("=== DEBUGGING EXTRACTION ===")

	# Find files
	submissions_file, conversations_file = find_required_files()
	if not submissions_file or not conversations_file:
		print("Error: Could not find required files")
		return

	print(f"Using files: {submissions_file}, {conversations_file}")

	# Extract and test one group
	import tempfile
	work_dir = "debug_extracted"
	extract_zip(submissions_file, work_dir)
	group_to_nb = find_group_notebooks(work_dir)
	group_to_llm = load_llm_conversations(conversations_file)

	if not group_to_nb:
		print("No notebooks found!")
		return

	# Test first group
	first_group = list(group_to_nb.keys())[0]
	nb_path = group_to_nb[first_group]

	print(f"\n=== TESTING GROUP: {first_group} ===")
	print(f"Notebook: {nb_path}")

	# Test notebook extraction
	with open(nb_path, "rb") as f:
		nb = read_notebook(f.read())
	nb_items = extract_code_and_comments_from_notebook(nb)

	print(f"\nNotebook extraction results:")
	print(f"Total lines extracted: {len(nb_items)}")
	print("First 5 lines:")
	for i, item in enumerate(nb_items[:5]):
		print(f"  [{i+1}] Type: {item['type']}, Text: '{item['text'][:80]}...'")

	# Test LLM extraction
	llm_lines = group_to_llm.get(first_group, [])
	print(f"\nLLM extraction results:")
	print(f"Total LLM lines: {len(llm_lines)}")
	print("First 5 lines:")
	for i, line in enumerate(llm_lines[:5]):
		print(f"  [{i+1}] '{line[:80]}...'")

	# Test a simple match
	if nb_items and llm_lines:
		print(f"\n=== TESTING SIMPLE MATCH ===")
		nb_sample = nb_items[0]['text'].strip()
		llm_sample = llm_lines[0].strip()
		print(f"Notebook sample: '{nb_sample}'")
		print(f"LLM sample: '{llm_sample}'")
		print(f"Exact match: {nb_sample == llm_sample}")
		print(f"Case-insensitive match: {nb_sample.lower() == llm_sample.lower()}")
		print(f"Fuzzy similarity: {fuzzy_ratio(nb_sample, llm_sample):.3f}")


def run_in_colab():
	"""
	Helper function to run the pipeline in Google Colab.
	Call this function in a Colab cell after uploading the required files.
	"""
	print("=== Quick Start for Google Colab ===")
	print("1. Upload your files using the file browser on the left:")
	print("   - submissions.zip")
	print("   - conversations_turns_with_groups.json")
	print("2. Run this cell to start the analysis")
	print()
	main()


def quick_debug():
	"""
	Quick debug function to see what's actually being extracted
	"""
	print("=== QUICK DEBUG ===")

	# Find files
	submissions_file, conversations_file = find_required_files()
	print(f"Files found: {submissions_file}, {conversations_file}")

	# Extract one notebook
	work_dir = "quick_debug"
	extract_zip(submissions_file, work_dir)
	group_to_nb = find_group_notebooks(work_dir)

	if not group_to_nb:
		print("No notebooks found!")
		return

	# Test first group
	first_group = list(group_to_nb.keys())[0]
	nb_path = group_to_nb[first_group]
	print(f"Testing group: {first_group}")
	print(f"Notebook path: {nb_path}")

	# Read notebook
	with open(nb_path, "rb") as f:
		nb = read_notebook(f.read())

	# Extract lines
	nb_items = extract_code_and_comments_from_notebook(nb)
	print(f"Extracted {len(nb_items)} lines from notebook")

	# Show first few lines
	print("\nFirst 3 notebook lines:")
	for i, item in enumerate(nb_items[:3]):
		print(f"  {i+1}. Type: {item['type']}")
		print(f"     Text: '{item['text']}'")
		print()

	# Load LLM data
	group_to_llm = load_llm_conversations(conversations_file)
	llm_lines = group_to_llm.get(first_group, [])
	print(f"LLM lines for {first_group}: {len(llm_lines)}")

	# Show first few LLM lines
	print("\nFirst 3 LLM lines:")
	for i, line in enumerate(llm_lines[:3]):
		print(f"  {i+1}. '{line}'")
		print()

	# Test matching
	if nb_items and llm_lines:
		print("=== MATCHING TEST ===")
		nb_line = nb_items[0]['text'].strip()
		llm_line = llm_lines[0].strip()

		print(f"Notebook: '{nb_line}'")
		print(f"LLM:      '{llm_line}'")
		print(f"Exact match: {nb_line == llm_line}")
		print(f"Lower match: {nb_line.lower() == llm_line.lower()}")
		print(f"Similarity: {fuzzy_ratio(nb_line, llm_line):.3f}")

		# Test with a few more lines
		print("\nTesting more lines...")
		for i in range(min(5, len(nb_items), len(llm_lines))):
			nb_text = nb_items[i]['text'].strip()
			llm_text = llm_lines[i].strip()
			sim = fuzzy_ratio(nb_text, llm_text)
			print(f"Line {i+1}: sim={sim:.3f} | '{nb_text[:30]}...' vs '{llm_text[:30]}...'")


if __name__ == "__main__":
	# Uncomment the line below to run quick debug instead of main
	# quick_debug()
	main()

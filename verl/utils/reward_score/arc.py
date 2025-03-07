# import re
import re
import ast
from collections import deque

def compute_score(
    solution_str, 
    ground_truth, 
    full_match_score=1.0, 
    partial_score_min=0.0, 
    partial_score_max=0.6,
    dimension_penalty=True,
    shape_reward_weight=0.5
):
    """
    Computes a reward for an ARC-like puzzle, with both:
      1) Pixel-level correctness.
      2) OPTIONAL shape-based 'reward shaping' that checks connected components.

    This version supports:
      - Predicted output in line-based format, e.g.:
            1 1
            1 0
      - Python-like bracket format, e.g.:
            [[1, 1],
             [1, 0]]

    Steps:
    1. Extract predicted output from <test_output_matrix>...</test_output_matrix>.
    2. Compare to ground_truth["gt_output"] as raw strings:
       - If exact match, return full_match_score.
       - Else parse each as 2D arrays of digits:
         a) compute pixel_fraction
         b) compute shape_fraction via connected components
         c) optionally apply dimension_penalty
         d) combine them and scale to [partial_score_min, partial_score_max].

    Args:
        solution_str (str): Model output, containing <test_output_matrix>...</test_output_matrix>.
        ground_truth (dict): Must contain "gt_output" as a string with correct solution.
        full_match_score (float): Score for an exact string match.
        partial_score_min (float): Reward if everything is off.
        partial_score_max (float): Reward if partial matches are close (but not perfect).
        dimension_penalty (bool): If True, penalize dimension mismatch more heavily.
        shape_reward_weight (float): Weight for shape-based partial vs pixel-based partial.

    Returns:
        float: A scalar reward in [partial_score_min, full_match_score].
    """
    # 1) Retrieve ground-truth
    gt_output_str = ground_truth.get("gt_output", "").strip()
    if not gt_output_str:
        # No reference => no meaningful score
        return 0.0

    # 2) Extract predicted output
    # match = re.search(r"<test_output_matrix>(.*?)</test_output_matrix>", solution_str, flags=re.DOTALL)
    match = re.findall(r"<test_output_matrix>(.*?)</test_output_matrix>", solution_str, flags=re.DOTALL)
    if not match:
        return 0.0
    predicted_output_str = match[-1].strip()

    try:
        predicted_output = ast.literal_eval(predicted_output_str.strip())

    except Exception as e:
        return 0.0
    if predicted_output == [[1, 1], [1, 0]]: ##HARD CODING FOR NOW
        print("No output matrix generated")
        return 0.0
    # 3) Exact raw-string match => full score
    if predicted_output_str == gt_output_str:
        print("perfect match")
        return full_match_score

    gt_matrix   = parse_matrix(gt_output_str)
    pred_matrix = parse_matrix(predicted_output_str)

    if not gt_matrix or not pred_matrix:
        return partial_score_min

    pixel_fraction = compute_pixel_fraction(gt_matrix, pred_matrix, dimension_penalty)

    gt_components   = count_connected_components(gt_matrix)
    pred_components = count_connected_components(pred_matrix)

    if max(gt_components, pred_components) > 0:
        shape_fraction = min(gt_components, pred_components) / max(gt_components, pred_components)
    else:
        shape_fraction = 1.0 

    combined_fraction = ((1.0 - shape_reward_weight) * pixel_fraction 
                         + shape_reward_weight * shape_fraction)

    partial_score = partial_score_min + combined_fraction * (partial_score_max - partial_score_min)
    print(f"Partial correct. Gt is {gt_matrix}, pred is {pred_matrix}")
    print(partial_score)
    return partial_score

def parse_matrix(matrix_str):
    """
    Parse the string 'matrix_str' into a 2D list of integers (or None).
    Supports two formats:
      1) Line-based, e.g.:
         '1 1\\n1 0'
      2) Python bracket syntax, e.g.:
         '[[1, 1], [1, 0]]'
    """
    matrix_str = matrix_str.strip()
    if looks_like_brackets(matrix_str):
        # Attempt to parse as Python literal
        try:
            matrix_data = ast.literal_eval(matrix_str)
        except Exception:
            # Fallback to line-based parsing if literal_eval fails
            return parse_matrix_line_based(matrix_str)
        # Convert to 2D array with int or None
        return convert_python_list_to_2d(matrix_data)
    else:
        # Fallback to line-based
        return parse_matrix_line_based(matrix_str)

def parse_matrix_line_based(matrix_str):
    """
    Parse a line-based matrix, e.g.
        '0 1\\n1 0'
    into [[0,1],[1,0]]
    """
    lines = matrix_str.splitlines()
    matrix = []
    for line in lines:
        row_vals = line.strip().split()
        row = []
        for val in row_vals:
            if val.isdigit():
                row.append(int(val))
            else:
                row.append(None)
        matrix.append(row)
    return matrix

def looks_like_brackets(text):
    """
    Heuristic check: does the string appear to be Python bracket syntax?
    True if it starts with '[' and ends with ']', ignoring whitespace.
    """
    t = text.strip()
    return t.startswith('[') and t.endswith(']')

def convert_python_list_to_2d(matrix_data):
    """
    Convert a Python object (list of lists) into a 2D list of int/None.
    e.g. [[1, 1], [1, 0]] -> [[1,1],[1,0]]
    """
    if not isinstance(matrix_data, list):
        return []
    matrix = []
    for row in matrix_data:
        if not isinstance(row, list):
            # If it's not a list, skip or put None
            matrix.append([])
            continue
        parsed_row = []
        for val in row:
            if isinstance(val, int):
                parsed_row.append(val)
            else:
                parsed_row.append(None)
        matrix.append(parsed_row)
    return matrix

def compute_pixel_fraction(gt_matrix, pred_matrix, dimension_penalty=False):
    """
    Compute fraction of matching pixels by comparing the bounding rectangle 
    that covers both matrices. For each cell in that region, if either is out
    of range or None, it's a mismatch. If both exist and are equal, it's a match.
    
    If dimension_penalty=True, we reduce the fraction further if the overall 
    shapes differ significantly.
    """
    max_rows = max(len(gt_matrix), len(pred_matrix))
    max_cols = max(
        len(gt_matrix[0]) if gt_matrix else 0,
        len(pred_matrix[0]) if pred_matrix else 0
    )

    correct_count = 0
    total_count = max_rows * max_cols

    for r in range(max_rows):
        for c in range(max_cols):
            gt_val  = get_val(gt_matrix, r, c)
            pred_val = get_val(pred_matrix, r, c)
            if gt_val is not None and pred_val is not None and gt_val == pred_val:
                correct_count += 1

    fraction = correct_count / total_count if total_count > 0 else 0.0

    if dimension_penalty:
        gt_rows, gt_cols = len(gt_matrix), len(gt_matrix[0]) if gt_matrix and gt_matrix[0] else 0
        pr_rows, pr_cols = len(pred_matrix), len(pred_matrix[0]) if pred_matrix and pred_matrix[0] else 0
        row_factor = min(gt_rows, pr_rows) / max(gt_rows, pr_rows) if max(gt_rows, pr_rows) != 0 else 1.0
        col_factor = min(gt_cols, pr_cols) / max(gt_cols, pr_cols) if max(gt_cols, pr_cols) != 0 else 1.0
        fraction *= (row_factor * col_factor)

    return fraction

def get_val(matrix_2d, r, c):
    """Safe indexing: returns None if out of range."""
    if r < 0 or r >= len(matrix_2d):
        return None
    if c < 0 or c >= len(matrix_2d[r]):
        return None
    return matrix_2d[r][c]

def count_connected_components(matrix_2d):
    """
    Count number of connected components in 'matrix_2d' using BFS (4-direction).
    We treat None as empty. 
    """
    visited = set()
    rows = len(matrix_2d)

    def neighbors(r, c):
        for nr, nc in [(r-1,c), (r+1,c), (r,c-1), (r,c+1)]:
            if get_val(matrix_2d, nr, nc) is not None:
                yield nr, nc

    def bfs(start_r, start_c):
        queue = deque([(start_r, start_c)])
        visited.add((start_r, start_c))
        while queue:
            rr, cc = queue.popleft()
            for nr, nc in neighbors(rr, cc):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    component_count = 0
    for r in range(rows):
        cols = len(matrix_2d[r])
        for c in range(cols):
            if matrix_2d[r][c] is not None and (r, c) not in visited:
                bfs(r, c)
                component_count += 1

    return component_count



# # ------------------------------------------------------------------------------
# # Example usage / test
# if __name__ == "__main__":
#     # Ground truth
#     gt_output = """0 1 3
# 3 4 5
# """

#     # Predicted output with one mismatch in the first row
#     predicted_output = """<test_output_matrix>
# 0 1 2
# 3 4 5
# </test_output_matrix>"""

#     # Prepare the arguments
#     solution_str = f"Some text... {predicted_output} ...some more text"
#     ground_truth = {"gt_output": gt_output}

#     # Compute score with shape-based partial included
#     score = compute_arc_reward(
#         solution_str=solution_str,
#         ground_truth=ground_truth,
#         full_match_score=1.0, 
#         partial_score_min=0.0,
#         partial_score_max=0.2,
#         dimension_penalty=False,   # set True to penalize dimension mismatches
#         shape_reward_weight=0.5    # 0.5 => shapes and pixel match are equally important
#     )

#     print(f"\nFinal Score Computed: {score:.3f}")

# def compute_arc_reward(solution_str, ground_truth, 
#                        full_match_score=1.0, 
#                        partial_score_min=0.0, 
#                        partial_score_max=0.2):
#     """
#     Computes the reward for an ARC-like puzzle.

#     1) Parses the <test_output>...</test_output> block from 'solution_str'.
#     2) If the predicted output matches ground_truth exactly as strings, return 'full_match_score' (default 1.0).
#     3) Otherwise, calculates partial credit by comparing the frequencies of digits 1..9 (EXCLUDING 0).
#        The partial score is scaled between partial_score_min and partial_score_max.

#     Args:
#         solution_str (str): The entire model-generated response containing <test_output>...</test_output>.
#         ground_truth (dict): A dictionary that includes the reference output, e.g. {"gt_output": "..."}.
#         full_match_score (float): Score for a perfect match (default 1.0).
#         partial_score_min (float): Minimum partial score if everything is off.
#         partial_score_max (float): Maximum partial score if frequencies match well but the string isn't exact.

#     Returns:
#         float: A scalar reward.
#     """
#     # 1) Get the ground truth output string
#     gt_output_str = ground_truth.get("gt_output", "").strip()
#     if not gt_output_str:
#         # If there's no ground truth, no basis for scoring
#         return 0.0
    
#     # 2) Extract the <test_output>...</test_output> from solution_str
#     match = re.search(r"<test_output_matrix>(.*?)</test_output_matrix>", solution_str, flags=re.DOTALL)
#     if not match:
#         # No predicted test output found
#         return 0.0
    
#     predicted_output_str = match.group(1).strip()
    
#     # 3) Exact string match => full score
#     if predicted_output_str == gt_output_str:
#         print('GT matches output')
#         return full_match_score
    
#     # 4) Partial match: compare digit frequencies for digits 1..9 (EXCLUDING 0)
#     freq_pred = count_digit_frequencies(predicted_output_str)
#     freq_gt   = count_digit_frequencies(gt_output_str)

#     # total GT digits (excluding 0)
#     total_gt_digits = sum(freq_gt[d] for d in range(1, 10))
#     if total_gt_digits == 0:
#         # If ground truth doesn't have any digits 1..9, partial matching doesn't apply
#         return 0.0

#     # matched_digit_count is how many digits (1..9) overlap ignoring position
#     matched_digit_count = 0
#     for d in range(1, 10):
#         matched_digit_count += min(freq_pred[d], freq_gt[d])

#     # ratio of matched digits
#     match_ratio = matched_digit_count / total_gt_digits

#     # scale ratio into partial_score_min..partial_score_max
#     partial_score = partial_score_min + (partial_score_max - partial_score_min) * match_ratio
#     print(f'GT partially matches output with score {partial_score}')
#     return partial_score


# def count_digit_frequencies(matrix_str):
#     """
#     Count how many times each digit 0-9 appears in 'matrix_str'.
#     Returns a dict {0: x0, 1: x1, ..., 9: x9}.
#     """
#     freq = {i: 0 for i in range(10)}
#     for char in matrix_str:
#         if char.isdigit():
#             digit = int(char)
#             freq[digit] += 1
#     return freq

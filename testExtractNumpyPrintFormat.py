"""Script to test the extraction of NumPy print format settings.

# NOTE: This version has been edited by AI; there is a previous commit with the original version, which is more human-readable.
"""

import re
from typing import Literal
import warnings

import numpy as np
from numpy.typing import NDArray

# ------ PARAMETERS ------

# Set the options for the integer array
int_rows, int_cols = 2, 10

# Set the options for the floating array
float_rows, float_cols = 5, 100

# Set the options for ellipsoid
n = 50

# Set the maximum number of vertices to print
max_nverts = 4

# ------ METHOD ------

def poly_hrepr(A: np.ndarray, b: np.ndarray, A_eq: np.ndarray | None = None, B_eq: np.ndarray | None = None, object_type: Literal['polyhedron', 'polytope', 'zonotope', 'cube'] = 'polytope', include_tabs: bool = True, trunc_mode: Literal['single_ellipsis', 'both_ellipsis'] = 'both_ellipsis', strip_double_b_brackets: bool = True) -> str:
    (n_ineq, n), edgeitems = A.shape, np.get_printoptions()['edgeitems']
    with np.printoptions(threshold=edgeitems ** 2, edgeitems=edgeitems, linewidth=1000):
        A_str, b_str = (A.flatten().__str__(), b.flatten().__str__()) if n_ineq == 1 else (A.__str__(), np.atleast_2d(b).T.__str__())
    A_lines, b_lines = A_str.splitlines(), b_str.splitlines()
    if strip_double_b_brackets:
        b_lines = [line.replace('[[', '[').replace(' ', '').replace(']]', ']') for line in b_lines]
    n_print_rows = len(A_lines)
    if n_print_rows > 1:
        A_lines = [line + ' ' for line in A_lines[:-1]] + [A_lines[-1]]
    row_trunc_idx = next((idx for idx, line in enumerate(A_lines) if '...' == line.strip(' ')), None)
    center_offset = 0 if n_print_rows <= 2 else 1
    tabs = ['  ' if include_tabs else '' for _ in range(n_print_rows)]
    x_vars = ([' |'] * (n_print_rows - center_offset - 1)) + [' x'] + ([' |'] * center_offset)
    inequalities = (['    '] * (n_print_rows - center_offset - 1)) + [' <= '] + (['    '] * center_offset)
    final_str = ''
    for row_idx in range(n_print_rows):
        if row_trunc_idx is not None and row_idx == row_trunc_idx:
            # TODO: Make a choice on one of these truncation modes, so that we can remove the other and streamline the code
            if trunc_mode not in ['single_ellipsis', 'both_ellipsis']:
                raise NotImplementedError(f"Truncation mode '{trunc_mode}' not recognized")
            final_str += tabs[row_idx] + A_lines[row_idx] + ('\n' if trunc_mode == 'single_ellipsis' else (' ' * (len(A_lines[-1]) - 5)) + x_vars[row_idx] + inequalities[row_idx] + b_lines[row_idx] + '\n')
        else:
            final_str += tabs[row_idx] + A_lines[row_idx] + x_vars[row_idx] + inequalities[row_idx] + b_lines[row_idx] + '\n'
    # TODO: Decide on the header format, and make it automatic for all print functions
    str_header = {'polytope': f"Polytope in R^{n}"}.get(object_type)
    if str_header is None:
        raise NotImplementedError(f"Printing for {object_type} is not yet implemented.")
    return str_header + '\n' + final_str

def poly_vrepr(verts: np.ndarray, rays: np.ndarray | None = None, object_type: Literal['polyhedron', 'polytope', 'zonotope', 'cube'] = 'polytope', include_tabs: bool = True, trunc_mode: Literal['single_ellipsis', 'all_ellipsis'] = 'both_ellipsis', strip_double_brackets: bool = True, max_nverts: int = 5) -> str:
    if max_nverts < 3:
        raise NotImplementedError("max_nverts must be at least 3 to allow for truncation.")
    n, edgeitems = verts.shape[0], np.get_printoptions()['edgeitems']
    with np.printoptions(edgeitems=edgeitems, threshold=(2 * edgeitems + 1), linewidth=1000):
        vert_strs = [np.atleast_2d(vert).T.__str__().splitlines() if n > 1 else vert.T.__str__().splitlines() for vert in verts.T]
    n_rows, n_cols = len(vert_strs[0]), min(len(vert_strs), max_nverts)
    row_trunc, col_trunc = n > n_rows, n_cols < verts.shape[1]
    start_col_idx, end_col_idx = (max_nverts // 2, max_nverts // 2 + 1) if col_trunc else (None, None)
    if not strip_double_brackets:
        vert_strs = [[elem.replace(']', '] ').replace('] ] ', ']]') for elem in row] for row in vert_strs] if n > 1 else vert_strs
    else:
        vert_strs = [['[' + elem.replace('[', '').replace(']', '') + ']' for elem in row] for row in vert_strs]
        warnings.warn("Stripping double brackets is NOT correctly implemented at the moment")
    idx_middle_row, center_offset = n_rows // 2, 0 if n_rows <= 2 else 1
    tabs = ['  ' if include_tabs else '' for _ in range(n_rows)]
    left_brackets = ([' /'] + ([' |'] * (idx_middle_row - 1)) + (['< '] if n_rows != 2 else []) + ([' |'] * (n_rows - idx_middle_row - 2)) + [' \\']) if n_rows != 1 else ['<']
    commas = ([' ' + (' ' if n_rows == 1 else '') for _ in range(n_rows - center_offset - 1)] + [',' + (' ' if n_rows == 1 else '')] + [' ' + (' ' if n_rows == 1 else '') for _ in range(center_offset)]) if n_cols > 1 else (['  '] if n_rows == 2 else [])
    right_brackets = (['\\ '] + (['| '] * (idx_middle_row - 1)) + ([' >'] if n_rows != 2 else []) + (['| '] * (n_rows - idx_middle_row - 2)) + ['/ ']) if n_rows != 1 else ['>']
    final_str = ''
    for row_idx in range(n_rows):
        final_str += tabs[row_idx] + left_brackets[row_idx]
        if row_trunc and row_idx == edgeitems and trunc_mode == 'single_ellipsis':
            n_spaces = sum(len(vert_strs[idx][0]) for idx in range(n_cols)) + (n_cols - 1) - 4
            final_str += ' ...' + (' ' * n_spaces)
        else:
            for col_idx in range(n_cols):
                if col_trunc and col_idx == start_col_idx:
                    final_str += (' ...' if row_idx == (n_rows - center_offset - 1) else '    ') + commas[row_idx]
                elif col_trunc and col_idx > start_col_idx and col_idx < end_col_idx:
                    continue
                elif col_trunc:
                    if row_trunc and row_idx == edgeitems and trunc_mode == 'all_ellipsis':
                        final_str += ' ...' + (' ' * (len(vert_strs[col_idx][0]) - 4))
                    else:
                        final_str += vert_strs[-(max_nverts - col_idx)][row_idx]
                    final_str += (commas[row_idx] if col_idx < n_cols - 1 else '')
                else:
                    if row_trunc and row_idx == edgeitems and trunc_mode == 'all_ellipsis':
                        final_str += ' ...' + (' ' * (len(vert_strs[col_idx][0]) - 4))
                    else:
                        final_str += vert_strs[col_idx][row_idx]
                    final_str += (commas[row_idx] if col_idx < n_cols - 1 else '')
        if row_trunc and row_idx == edgeitems and trunc_mode not in ['single_ellipsis', 'all_ellipsis']:
            raise NotImplementedError(f"Truncation mode '{trunc_mode}' not recognized")
        final_str += right_brackets[row_idx] + '\n'
    str_header = {'polytope': f"Polytope in R^{n}"}.get(object_type)
    if str_header is None:
        raise NotImplementedError(f"Printing for {object_type} is not yet implemented.")
    return str_header + '\n' + final_str

def ellipsoid_semidef_repr(c: np.ndarray, Q: np.ndarray, include_tabs: bool = True, object_type: Literal['polyhedron', 'polytope', 'zonotope', 'cube'] = 'polytope', filter_sym_part: bool = True) -> str:
    n, edgeitems = Q.shape[1], np.get_printoptions()['edgeitems']
    with np.printoptions(edgeitems=edgeitems, threshold=edgeitems ** 2, linewidth=1000):
        if n == 1:
            Q_lines, c_lines = [Q.__str__().replace('[', '').replace(']', '')], [c.__str__().replace('[', '').replace(']', '')]
        else:
            Q, c = Q, np.atleast_2d(c).T
            Q_str, c_str = Q.__str__().replace(']', '] ').replace('] ] ', ']]'), c.__str__().replace(']', '] ').replace('] ] ', ']]')
            Q_lines, c_lines = Q_str.splitlines(), c_str.splitlines()
            Q_lines, c_lines = [elem if elem != ' ...' else elem + (' ' * (len(Q_lines[-1]) - 4)) for elem in Q_lines], [elem if elem != ' ...' else elem + (' ' * (len(c_lines[-1]) - 4)) for elem in c_lines]
    n_print_rows = len(Q_lines)
    if filter_sym_part:
        for row_idx in range(n_print_rows):
            row_elems = Q_lines[row_idx].split()
            for col_idx in range(len(row_elems)):
                if col_idx > row_idx and '...' not in row_elems[col_idx]:
                    # FIXME: This `re` seems to raise the error (also due to random matrix generation?): "AttributeError: 'NoneType' object has no attribute 'start'"
                    n_digits, start_idx = len(row_elems[col_idx].replace('[', '').replace(']', '')), re.search(r'\d', row_elems[col_idx]).start()
                    Q_lines[row_idx] = Q_lines[row_idx].replace(row_elems[col_idx][start_idx:(start_idx + n_digits)], (' ' * (n_digits // 2)) + '*' + (' ' * ((n_digits // 2) if n_digits % 2 == 1 else (n_digits // 2 - 1))))
    center_offset = 0 if n_print_rows <= 2 else 1
    tabs = ['  ' if include_tabs else '' for _ in range(n_print_rows)]
    c_labels = (['   '] * (n_print_rows - center_offset - 1)) + ['c: '] + (['   '] * center_offset)
    commas = (['  '] * (n_print_rows - center_offset - 1)) + [', '] + (['  '] * center_offset)
    Q_labels = (['   '] * (n_print_rows - center_offset - 1)) + ['Q: '] + (['   '] * center_offset)
    final_str = ''.join(tabs[row_idx] + c_labels[row_idx] + c_lines[row_idx] + commas[row_idx] + Q_labels[row_idx] + Q_lines[row_idx] + '\n' for row_idx in range(n_print_rows))
    str_header = {'ellipsoid': f"Ellipsoid in R^{n}"}.get(object_type)
    if str_header is None:
        raise NotImplementedError(f"Printing for {object_type} is not yet implemented.")
    return str_header + '\n' + final_str


def find(text: str, char: str, remove_consecutive: bool = True, keep_idx: Literal['last', 'first'] = 'last'):
    indices = [i for i, ltr in enumerate(text) if ltr == char]
    if remove_consecutive:
        new_indices = indices
        nremoved = 0
        for (idx, val) in enumerate(indices):
            if idx == 0:
                continue
            if val == indices[idx - 1] + 1:
                match keep_idx:
                    case 'last':
                        new_indices.pop(idx - 1)
                    case 'first':
                        new_indices.pop(idx)
                    case _:
                        raise ValueError(f"Unknown 'keep_idx' strategy '{keep_idx}'")
                nremoved += 1
        indices = new_indices
    return indices, indices[1] - indices[0] - 1


def pretty_ellps(A_as_str: str) -> str:
    # FIXME: Brakes on these examples:
    # >>> A = np.array([[1, 0],
    #                   [0, -1]])
    # [[ 1  0]
    #  [ 0 -1]]  # Here there is one space in front already...
    # A = np.array([[1E4, 0],
    #               [0, -1]])
    # [[10000.     * ]
    #  [    0.     * ]]  # FIXME: Here, the space length is always 

    # Find the length of the maximum substring which does not contains the character ' ', '[', or ']', and the substring should not be equal to '...'
    clean_str = A_as_str.replace('...', ' ')
    parts = re.split(r'[\[\]\s]+', clean_str)
    parts = [p for p in parts if p]
    if not parts:
        max_space_length = 0
    max_space_length = max(len(p) for p in parts)

    # TEMP
    print(f"Max space length: {max_space_length}")

    A_lines = A_as_str.splitlines()
    try:
        idx_trunc = A_lines.index(' ...')
    except ValueError as _:
        idx_trunc = None

    A_as_str_arr = np.array([list(line)
                            if line[-2:] == ']]'
                            else list(line) + ['█']
                            for line in A_lines if line != ' ...'], dtype=str)  # NOTE: This adds a '█' character at the end of all lines except the last, to make them equal length

    print(f"A_as_str_arr:\n{A_as_str_arr}")

    all_space_cols = []
    for idx in range(1, A_as_str_arr.shape[1]):
        if np.all(A_as_str_arr[:, idx] == ' ') and not np.any(A_as_str_arr[:, idx - 1] == '['):  # NOTE: The last check specifically target the case:
            # [[ 1.  * ]
            #  [ 0. -1.]] # FIXME: This still doesn't catch many edge cases...
            all_space_cols.append(idx)

    print(f"All space cols: {all_space_cols}")

    # As every line starts with '[[' or ' [' we know the first number footprint starts at the 2 index, and all other number footprints start one after a column with all spaces
    num_start = [2] + [elem + 1 for elem in all_space_cols]

    print(f"num_start: {num_start}")

    num_length = [num_start[j + 1] - 1 - num_start[j]
                if j != len(num_start) - 1
                else A_as_str_arr[0, :].size - 2 - num_start[j]
                for j in range(len(num_start))]

    print(f"num_length: {num_length}")

    for i in range(A_as_str_arr.shape[0]):
        for (j, idx_start) in enumerate(num_start):
            if ''.join(A_as_str_arr[i, idx_start:(idx_start + 3)].tolist()) == '...':
                continue
            if j > i + (0 if (idx_trunc is None or i < idx_trunc) else 1):
                A_as_str_arr[i, idx_start:(idx_start + num_length[j])] = list(f"{'*':^{num_length[j]}}")

    A_list = [''.join(elem).strip('█') for (i, elem) in enumerate(A_as_str_arr.tolist())]
    if idx_trunc is not None:
        A_list.insert(idx_trunc, ' ...')
    final = '\n'.join(A_list)

    print(f"final:\n{final}")

    return final


# This seems to be working robustly!
def pretty_ellps_v2(A_as_str: str) -> str:
    # NOTE: This function works under the assumption that the 'footprint' of each number in the NumPy print format is consistent across all numbers, and that the footprint is determined by the longest number (in terms of characters) in the array.

    # Find the length of the maximum substring which does not contains the character ' ', '[', or ']', and the substring should not be equal to '...'
    parts = [p for p in re.split(r'[\[\]\s]+', A_as_str.replace('...', ' ')) if p]
    if not parts:
        warnings.warn("Could not determine the number length, possibly a bug. Returning the original array.")
        return A_as_str
    num_length = max(len(p) for p in parts)

    A_lines = A_as_str.splitlines()
    try:
        idx_trunc = A_lines.index(' ...')
    except ValueError as _:
        idx_trunc = None

    # NOTE: This adds a '█' character at the end of all lines except the last, to make them equal length
    A_as_str_arr = np.array([list(line)
                            if line[-2:] == ']]'
                            else list(line) + ['█']
                            for line in A_lines if line != ' ...'], dtype=str)

    for i in range(A_as_str_arr.shape[0]):
        j, idx_start = 0, 2
        while idx_start + num_length < A_as_str_arr[i, :].size:
            if ''.join(A_as_str_arr[i, idx_start:(idx_start + 3)].tolist()) == '...':
                idx_start += 4
            else: 
                if j > i + (0 if (idx_trunc is None or i < idx_trunc) else 1):
                    A_as_str_arr[i, idx_start:(idx_start + num_length)] = list(f"{'*':^{num_length}}")
                idx_start += num_length + 1
            j += 1

    A_list = [''.join(elem).strip('█') for (i, elem) in enumerate(A_as_str_arr.tolist())]
    if idx_trunc is not None:
        A_list.insert(idx_trunc, ' ...')
    return '\n'.join(A_list)


def full_ellps(c: NDArray, Q: NDArray) -> str:
    # ======
    c_as_str, Q_as_str = str(c), str(Q)
    c_lines, Q_lines = c_as_str.splitlines(), Q_as_str.splitlines()
    nlines = len(c_lines)
    try:
        idx_trunc = c_lines.index(' ...')
    except ValueError as _:
        idx_trunc = None
    idx_text = nlines - (2
                         if (nlines <= 2 or (nlines == 3 and idx_trunc is not None))
                         else 3)
    c_text = ['   ' if idx != idx_text else 'c: ' for idx in range(nlines)]


# ------ SCRIPT ------

# Create an integer array
# int_array_A, int_array_b = np.random.randint(0, 10, size=(int_rows, int_cols)), np.random.randint(0, 10, size=(int_rows))
int_array_A, int_array_b = np.random.rand(int_rows, int_cols), np.random.rand(int_rows) * 1E-6

# Create a floating-point array
float_array_verts = np.random.rand(float_rows, float_cols)

# Create a positive semidefinite matrix for the ellipsoid
# c, Q_root = np.random.randint(1000, 2000, size=n), np.random.randint(1000, 2000, size=(n, n))
c, Q_root = np.random.rand(n), np.random.rand(n, n) * 1E-0
Q = Q_root.T @ Q_root  # Make sure Q is positive semidefinite

# ------ PRINT ------

# Print the polytope H-representation
print("H-representation:")
np.set_printoptions(suppress=False, precision=1, edgeitems=2)
print(poly_hrepr(int_array_A, int_array_b, object_type='polytope', include_tabs=False, trunc_mode='single_ellipsis', strip_double_b_brackets=False))  # I think I prefer single_ellipsis here, and not stripping double brackets

# Print the polytope V-representation
print("V-representation:")
np.set_printoptions(suppress=False, precision=1, edgeitems=3)
print(poly_vrepr(float_array_verts, object_type='polytope', include_tabs=True, trunc_mode='all_ellipsis', strip_double_brackets=False, max_nverts=max_nverts))

# Print the ellipsoid semidefinite representation
print("Ellipsoid semidefinite representation:")
np.set_printoptions(suppress=False, precision=1, edgeitems=2)
print(ellipsoid_semidef_repr(c, Q, include_tabs=True, filter_sym_part=True, object_type='ellipsoid'))

# === PRINT STANDALONE ===

# A = np.random.rand(100, 100) * 1E-8
A = np.random.rand(100, 100) * 1E12
# A = np.array([[1, 0],
#               [0, -1]])
# A = np.array([[1E4, 0],
#               [0, -1]])
# A = np.random.randint(0, 2, (300, 300)).astype(bool)

with np.printoptions(suppress=True, precision=0, edgeitems=1, linewidth=1E5):
    A_as_str = str(A)

print("=== STANDALONE ===")
print(f"A_as_str:\n{A_as_str}")
print(f"With *: \n{pretty_ellps_v2(A_as_str)}")

# === TEST STRING ZIP ===

spaces = [' ' for _ in range(5)]
letters = ['A', 'B', 'C', 'D', 'E']
empty = ['  ', '  ', '  ', ', ', '  ']

final = [''.join(line) for line in zip(spaces, letters, empty)]

print('\n'.join(final))

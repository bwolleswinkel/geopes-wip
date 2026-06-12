"""Print to test printing the ellipsoid functionality"""

from __future__ import annotations

import re
from copy import copy
from dataclasses import dataclass
from typing import Optional, Self
import warnings

import numpy as np
from numpy.typing import NDArray


class Ellipsoid:

    def __init__(self, c: NDArray, Q: NDArray):
        self.c: NDArray = c
        self.Q: NDArray = Q
        self.n = c.size

    def __str__(self: Self) -> str:
        """Pretty-print the ellipsoid"""

        def pad(text: str, length: int, char: str = ' ') -> str:
            """Pad a string that is shorter then a certain length with `char`"""
            return (text
                    if len(text) >= length
                    else text + ''.join([' '] * (length - len(text))))

        c_as_str, Q_as_str = str(np.atleast_2d(self.c).T), str(self.Q)
        c_lines, Q_lines = c_as_str.splitlines(), Q_as_str.splitlines()
        nlines = len(Q_lines)
        try:
            idx_trunc = Q_lines.index(' ...')
            # NOTE: This assumes the number of edgeitems above and below is always identical
            c_lines = c_lines[:idx_trunc] + [' ...'] + c_lines[-idx_trunc:]
        except ValueError as _:
            idx_trunc = None
        idx_text = nlines - (1
                            if (nlines <= 2 or (nlines == 3 and idx_trunc is not None))
                            else 2)

        c_text = ['   ' if idx != idx_text else 'c: ' for idx in range(nlines)]
        c_vals = [pad(line, max([len(line) for line in c_lines])) for line in c_lines]
        Q_text = ['     ' if idx != idx_text else ', Q: ' for idx in range(nlines)]
        Q_vals = sym_replace(Q_as_str).splitlines()
        comb = '\n'.join([''.join(line) for line in zip(c_text, c_vals, Q_text, Q_vals)])
        if self.n == 1:
            comb = comb.replace('[[', '[').replace(']]', ']')

        return comb



def sym_replace(arr: str, char: str = '*') -> str:
    """Replace the numbers located in the upper-triangular part of an 2d square NumPy array with `char`. This is to be used for arrays which are symmetric, and where the upper-triangular values can be inferred from the lower part. The function does not check if the array provided is 2d  or square, and undefined behavior might follow otherwise.

    The function works for all size `n >= 1`, for integers and floating point, decimal and scientific notation, and preserves leading/trailing spaces and alignment due to negative values and dropped trailing zeros.

    Parameters
    ----------
    arr : str
        String representation of an array, presumable `str(A)`, where `A` is a NumPy array of size `(n, n)`
    char : str
        Character for which to replace the upper-triangular part. It is assumed to be of length 1, 
        but thus us not enforced.

    Warns
    -----
    RuntimeWarning
        If the maximum number length in the array cannot be determined. If raised, the original array
        without any modifications is returned

    Warnings
    --------
    This feature has not been rigorously tested, and relies on two key assumptions (see source code) about NumPy string
    formatting which might fail for certain edge cases, or may change in the future.

    Examples
    --------
    >>> Q = np.array([[ 1, -1],
                      [-1,  1])
    >>> print(sym_replace(str(Q)))
    [[ 1, * ],
     [-1,  1])

    Truncation is also supported.
     
    >>> A = np.arange(10_000).reshape(100, 100) * 1E-2
    >>> Q = A + A.T
    >>> with np.printoptions(precision=2, edgeitems=2):
    >>>     print(sym_replace(str(Q), char='•'))
    [[  0.     •    ...   •      •   ]
     [  1.01   2.02 ...   •      •   ]
     ...
     [ 98.98  99.99 ... 197.96   •   ]
     [ 99.99 101.   ... 198.97 199.98]]
    """
    # NOTE: This function works under the assumption that the 'footprint' of each number in a NumPy
    # string is consistent across all numbers, and that the footprint is determined by the longest
    # number (in terms of characters) in the array.

    lines = arr.splitlines()
    try:
        idx_trunc = lines.index(' ...')
    except ValueError as _:
        idx_trunc = None

    # The num_length is equal to the length of the first line, minus the three characters '[[' and ']', minus the ' ' characters in between the numbers, all divided by the number of columns. If there is truncation, we must subtract the three '...' characters and divide by one fewer column
    if idx_trunc is None:
        num_length = int((len(lines[0]) - 3 - (len(lines) - 1)) / len(lines))
    else:
        num_length = int((len(lines[0]) - 3 - (len(lines) - 1) - 3) / (len(lines) - 1))

    # NOTE: This adds a '█' character at the end of all lines except the last, to make them equal length
    str_arr = np.array([list(line)
                        if line[-2:] == ']]'
                        else list(line) + ['█']
                        for line in lines if line != ' ...'], dtype=str)

    for i in range(str_arr.shape[0]):
        j, idx_start = 0, 2
        while idx_start + num_length < str_arr[i, :].size:
            if ''.join(str_arr[i, idx_start:(idx_start + 3)].tolist()) == '...':
                idx_start += 4
            else: 
                if j > i + (0 if (idx_trunc is None or i < idx_trunc) else 1):
                    str_arr[i, idx_start:(idx_start + num_length)] = list(f"{char:^{num_length}}")
                idx_start += num_length + 1
            j += 1

    A_list = [''.join(elem).strip('█') for (i, elem) in enumerate(str_arr.tolist())]
    if idx_trunc is not None:
        A_list.insert(idx_trunc, ' ...')

    return '\n'.join(A_list)


if __name__ == '__main__':

    def pretty_print(c: NDArray,
                     Q: NDArray,
                     suppress: Optional[bool] = None,
                     precision: Optional[int] = None,
                     edgeitems: Optional[int] = None,
                     linewidth: int = 1_000) -> None:
        threshold = edgeitems ** 2 - 1 if edgeitems is not None else None
        with np.printoptions(suppress=suppress, precision=precision, threshold=threshold, edgeitems=edgeitems, linewidth=linewidth):
            ellps = Ellipsoid(c, Q)
            print(ellps)

    @dataclass
    class Case:
        c: NDArray
        Q: NDArray
        suppress: Optional[bool] = None
        precision: Optional[int] = None
        edgeitems: Optional[int] = None

        def copy(self) -> Case:
            return copy(self)


    case_1 = Case(
        c = np.array([0, 0]),
        Q = np.array([[ 1, -1],
                      [-1,  1]]),
    )

    case_2 = case_1.copy()
    case_2.c = case_2.c.astype(float)
    case_2.Q = case_2.Q.astype(float)

    case_3 = Case(
        c = np.array([0, 0, 0]),
        Q = np.array([[ 1.12345,  -1  ,    0],
                      [-1,       -12  , -5.5],
                      [ 0,        -5.5,    0]]),
    )

    case_4 = Case(
        c = np.array([0, 0, 0]),
        Q = np.array([[-12,  -1  ,    0],
                      [-1,    1.12345, -5.5],
                      [ 0,        -5.5,    0]]),
    )

    case_5 = case_4.copy()
    case_5.precision = 0

    case_6 = Case(
        c = np.arange(20),
        Q = np.arange(400).reshape(20, 20),
    )

    case_7 = case_6.copy()
    case_7.edgeitems = 1

    case_8 = case_7.copy()
    case_8.edgeitems = 2

    case_9 = Case(
        c = np.random.randn(5),
        Q = np.random.randn(5, 5),
    )

    case_10 = case_9.copy()
    case_10.precision = 3

    case_11 = case_10.copy()
    case_11.edgeitems = 1

    case_12 = case_9.copy()
    case_12.precision = 0
    case_12.Q *= 1E-6
    case_12.suppress = False

    case_13 = Case(
        c = np.array([1E4, 1.234E-2]),
        Q = np.array([[-1.35E-4, 0.03E2],
                      [0.03E2, 0]]),
    )

    case_14 = Case(
        c = np.ones(10).astype(bool),
        Q = np.mod(np.arange(100), 2).reshape(10, 10).astype(bool),
    )

    case_15 = case_14.copy()
    case_15.edgeitems = 1

    case_16 = Case(
        c = np.array([1]),
        Q = np.array([[-0.2]]),
    )

    cases = [
        case_1,
        case_2,
        case_3,
        case_4,
        case_5,
        case_6,
        case_7,
        case_8,
        case_9,
        case_10,
        case_11,
        case_12,
        case_13,
        case_14,
        case_15,
        case_16,
    ]

    for (idx, case) in enumerate(cases, start=1):
        print(f"\n=== CASE {idx} ===")
        pretty_print(case.c, case.Q, case.suppress, case.precision, case.edgeitems)
    

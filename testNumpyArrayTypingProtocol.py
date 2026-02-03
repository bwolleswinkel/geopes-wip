"""Script to test using typing protocols to add shape information to numpy arrays"""

from __future__ import annotations
from typing import Protocol, Any, runtime_checkable, TypeVar, Generic, TypeVarTuple, Literal

import numpy as np
from numpy.typing import NDArray as NumpyNDArray


# Sentinel values for shape dimension names - makes them valid identifiers for subscripting
n = Literal['n']
m = Literal['m']
k = Literal['k']
ellipsis_ = type(...)  # Type of the ellipsis object

# TypeVarTuple for Shape - allows variable number of dimensions
ShapeDims = TypeVarTuple('ShapeDims')


class Shape(Generic[*ShapeDims]):
    """Generic type alias for shape information, used for documentation only.
    
    Accepts dimension literals ('n', 'm', 'k') and ellipsis via subscripting.
    
    Usage:
        Shape['n', 'm']  - 2D shape with dimensions n and m
        Shape['n', ...]  - variable length shape starting with n
    
    This is purely for type hints and doesn't enforce anything at runtime.
    """
    pass


# TypeVars for NDArray parameters - COVARIANT for use in Protocol
ShapeOrDType_co = TypeVar('ShapeOrDType_co', covariant=True)
DType_co = TypeVar('DType_co', covariant=True)


@runtime_checkable
class NDArray(Protocol[ShapeOrDType_co, DType_co]):
    """Protocol for numpy-like arrays with shape, size, and dtype attributes.

    Type parameters are for documentation only:
        - First parameter: Shape information (e.g., Shape['n', 'm'])
        - Second parameter: Data type (e.g., int, float, bool)
    
    Usage:
        a: NDArray[Shape['n', 'm'], float]  # 2D array of floats
        b: NDArray[float]  # any shape of floats
        c: NDArray  # any array with any shape/dtype
    """
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array"""
        ...

    @property
    def size(self) -> int:
        """Total number of elements"""
        ...

    @property
    def dtype(self) -> Any:
        """Data type of elements"""
        ...

    

def main():
    # Test basic usage
    a: NDArray = np.array([3, 3, 2])
    b: NumpyNDArray = np.zeros((3, 2))

    # Test with Shape annotation - single dimension
    c: NDArray[Shape[n], float] = np.array([1, 2, 3])
    
    # Test with dtype annotation
    d: NDArray[Any, float] = np.array([1.0, 2.0, 3.0])
    
    # Test with both Shape and dtype - 2D
    e: NDArray[Shape[n, m], float] = np.zeros((3, 4))
    
    # Test with variable length shape
    f: NDArray[Shape[n, ...], int | bool] = np.array([1, 2, 3])

    g: NDArray[float] = 7

    print(a, a.shape, a.size, a.dtype)
    print(c, c.shape, c.size, c.dtype)
    print(d, d.shape, d.size, d.dtype)
    print(e, e.shape, e.size, e.dtype)


if __name__ == "__main__":
    main()

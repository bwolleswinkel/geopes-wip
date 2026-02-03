"""Script to test using typing protocols to add shape information to numpy arrays"""

from __future__ import annotations
from typing import Protocol, Any, runtime_checkable, Generic, TypeVar, overload

import numpy as np
from numpy.typing import NDArray as NumpyNDArray


T_Shape = TypeVar('T_Shape')
T_DType = TypeVar('T_DType')


class Shape(Generic[T_Shape]):
    """Generic type alias for shape information, used for documentation only.
    
    Accepts any arguments via subscripting for documentation purposes.
    
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

    # Test with Shape annotation
    c: NDArray[Shape, float] = np.array([1, 2, 3])
    
    # Test with dtype annotation
    d: NDArray[Any, float] = np.array([1.0, 2.0, 3.0])
    
    # Test with both Shape and dtype
    e: NDArray[Shape, float] = np.zeros((3, 4))
    
    # Test with any dtype
    f: NDArray[Shape, int] = np.array([1, 2, 3])

    g: NDArray[Shape['n', ...], float] = 5

    print(a, a.shape, a.size, a.dtype)
    print(c, c.shape, c.size, c.dtype)
    print(d, d.shape, d.size, d.dtype)
    print(e, e.shape, e.size, e.dtype)


if __name__ == "__main__":
    main()

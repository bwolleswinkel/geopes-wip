"""Script to test typing for Numpy Arrays using protocols with sentinel values for shape dimensions"""

# FROM: GitHub Copilot, Claude Haiku 4.5 | 2026/02/03
from typing import Protocol, Any, runtime_checkable, TypeVar, Generic, TypeVarTuple, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray as NumpyNDArray


# Sentinel values for shape dimension names - use TypeAlias to make them valid types
N: TypeAlias = Literal['n']
M: TypeAlias = Literal['m']
K: TypeAlias = Literal['k']

# TypeVarTuple for Shape - allows variable number of dimensions
ShapeDims = TypeVarTuple('ShapeDims')


class Shape(Generic[*ShapeDims]):  # type: ignore[misc]
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

    

def main() -> None:
    # Test basic usage
    a: NDArray = np.array([3, 3, 2])
    b: NumpyNDArray = np.zeros((3, 2))

    # Test with Shape annotation - single dimension
    c: NDArray[Shape[Any], float] = np.array([1, 2, 3])
    
    # Test with dtype annotation
    d: NDArray[Any, float] = np.array([1.0, 2.0, 3.0])
    
    # Test with both Shape and dtype - 2D
    e: NDArray[Shape[N, N], float] = np.zeros((4, 4))
    
    # Test with variable length shape
    f: NDArray[Shape[N], int | bool] = np.array([1, 2, 3])

    # NOTE: If we use a direct tuple, we can also use `...` in the type hint
    X: NDArray[tuple[Any, ...], float] = np.zeros((2, 3, 4, 5))

    Y: NDArray[tuple[N, Any], float] = np.zeros((2, 3, 4, 5))

    g: NDArray[Shape[N, M, K], float] = 6  #  NOTE: This should raise a type checker error

    print(a, a.shape, a.size, a.dtype)
    print(c, c.shape, c.size, c.dtype)
    print(d, d.shape, d.size, d.dtype)
    print(e, e.shape, e.size, e.dtype)
    print(f, f.shape, f.size, f.dtype)


if __name__ == "__main__":
    main()

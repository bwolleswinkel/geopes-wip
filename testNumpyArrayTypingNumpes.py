"""Module for type annotations related to NumPy arrays"""

# FROM: GitHub Copilot, Claude Haiku 4.5 | 2026/02/03[untested/unverified]
from typing import runtime_checkable, Protocol, Any, TypeVar, Generic, TypeVarTuple

ShapeDims = TypeVarTuple('ShapeDims')
ShapeOrDType_co = TypeVar('ShapeOrDType_co', covariant=True)
DType_co = TypeVar('DType_co', covariant=True)


class Shape(Generic[*ShapeDims]):
    """Generic type alias for shape information, used for documentation only.
    
    Accepts dimension literals (e.g., `"n"`, `"m"`, `"k"`), ellipsis (`...`), and generics (`Any`) via subscripting.
    
    Examples
    --------
    >>> Shape[...]  # Array of any shape
    >>> Shape["n"]  # 1D array of length n
    >>> Shape["m", Any, ...]  # Array with at least two dimensions and m rows
    
    Notes
    -----
    This is purely for type hints and doesn't enforce anything at runtime.
    """
    pass


@runtime_checkable
class NDArray(Protocol[ShapeOrDType_co, DType_co]):
    """Protocol for numpy-like arrays with shape, size, and dtype attributes.

    Type parameters are for documentation only:
        - First parameter: Shape information (e.g., `Shape["n", "m"]`)
        - Second parameter: Data type (e.g., `int`, `float`, `bool`)

    Examples
    --------
    >>> a: NDArray[Shape["n", "m"], float] = np.zeros((3, 4))
    >>> b: NDArray[Shape[Any, ...], int] = np.array([1, 2, 3])
    """

    def __getitem__(self, key: int | slice | tuple[Any, ...]) -> Any:
        """Get item by index or slice"""
        pass

    @property
    def ndim(self) -> int:
        """Number of dimensions"""
        pass
    
    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the array"""
        pass

    @property
    def size(self) -> int:
        """Total number of elements"""
        pass

    @property
    def dtype(self) -> Any:
        """Data type of elements"""
        pass

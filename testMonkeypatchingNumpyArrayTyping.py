"""Script to test monkeypatching NDArray to accept shape information"""

from typing import Literal, TypeVar, Any, Generic

import numpy as np
from numpy.typing import NDArray as NumpyNDArray

# Type variables for our custom NDArray
DType = TypeVar('DType')
ShapeInfo = TypeVar('ShapeInfo')


# FROM: GitHub Copilot Claude Sonnet 4 | 2026/01/12

T = TypeVar('T') 
S = TypeVar('S')

class NDArray(NumpyNDArray[Any], Generic[T, S]):
    """Custom NDArray that inherits from numpy NDArray and accepts shape info. 

    Examples
    --------    
    >>> import numpy as np
    >>> from ... import NDArray, Shape
    >>> a: NDArray[float, Shape['square']] = np.array([[1.0, 2.0], [3.0, 4.0]])  # NOTE: The shape info is just for typing, not enforced at runtime or checked by a linter
    >>> b: NDArray[int, Shape['n', 'm']] = np.zeros((3, 4), dtype=int)
     
    Notes
    -----
    The Python linter  `mypy` does not support this, and will raise errors.
    
    """
    
    def __class_getitem__(cls, params):
        # Handle subscripting - return the numpy array type
        return NumpyNDArray[Any]
    
    def __new__(cls, *args, **kwargs):
        # This should never be instantiated directly
        raise TypeError("Use numpy array creation functions instead")

# Custom Shape class that supports subscripting
class ShapeMeta(type):
    def __getitem__(cls, item):
        # Return the literal type when subscripted (works for both single and multiple parameters)
        return Literal[item]

class Shape(metaclass=ShapeMeta):
    """This is a class for shape information"""

    def __getitem__(self, item):
        # Instance-level subscripting for runtime (works for both single and multiple parameters)
        return Literal[item]

# Now this works without Pylance warnings!
a: NDArray[float, Shape['n', 'n']] = np.array([3, 3, 2])
A: NDArray[float, Shape['m', 'n']] = np.zeros((3, 2))
b: NDArray[int, Shape['m']] = np.zeros(4)
verts: NDArray[float, Shape['n', 'k']] = np.zeros((5, 3))

print(f"Array a shape: {a.shape} - should be square")
print(f"Array b shape: {b.shape} - should be rectangle")
print("Monkeypatching successful!")
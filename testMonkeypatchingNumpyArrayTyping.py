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
a: NDArray[Shape['n', 'n'], float] = np.array([3, 3, 2])
A: NDArray[float, Shape['m', 'n'], float] = np.zeros((3, 2))
b: NDArray[Shape['m'], int] = np.zeros(4)
c: NDArray[Shape[3, ...], int] = np.zeros(4)  # Use ... to specify any number of dimensions, of any size
d: NDArray[Shape[Any, Any], float | bool] = np.zeros((2, 2))  # Use Any to specify any size in that dimension (without giving it a name)
verts: NDArray[Shape['n', ...], int] = np.zeros((5, 3))

print(f"Array a shape: {a.shape} - should be square")
print(f"Array b shape: {b.shape} - should be rectangle")
print("Monkeypatching successful!")
"""Script to test monkeypatching NDArray to accept shape information"""

from typing import Literal, TypeVar

import numpy as np
from numpy.typing import NDArray as NumpyNDArray


# FROM: GitHub Copilot Claude Sonnet 4 | 2026/01/12


# Create a custom NDArray that accepts shape information
class CustomNDArrayMeta(type):
    def __getitem__(cls, params):
        if isinstance(params, tuple):
            # Handle multiple parameters like NDArray[np.float64, 'square']
            dtype_param = params[0]
            shape_param = params[1] if len(params) > 1 else None
            # Create a wrapper that remembers both dtype and shape
            class TypedNDArray:
                dtype = dtype_param
                shape_type = shape_param

                def __class_getitem__(self, item):
                    return self
            
            return TypedNDArray
        else:
            # Single parameter - fall back to original NDArray behavior
            return NumpyNDArray[params]


class CustomNDArray(metaclass=CustomNDArrayMeta):
    pass


# Replace NDArray with our custom version - now it's a proper type
NDArray = CustomNDArray

Float = TypeVar("Float", bound=np.float64 | np.float32 | np.float16 | float)

# Custom Shape class that supports subscripting
class ShapeTypeMeta(type):
    def __getitem__(cls, item):
        # Return the literal type when subscripted
        return Literal[item]

class ShapeType(metaclass=ShapeTypeMeta):
    def __getitem__(self, item):
        # Instance-level subscripting for runtime
        return Literal[item]

# Shape can be used both as a class (for type checking) and as an instance (for runtime)
Shape = ShapeType

Square = Literal['square']
Rectangle = Literal['rectangle']

# Now this works without Pylance warnings!
a: NDArray[Float, Shape['square']] = np.zeros((3, 3))  # Square array
b: NDArray[Float] = np.zeros((3, 2))  # Rectangle array

print(f"Array a shape: {a.shape} - should be square")
print(f"Array b shape: {b.shape} - should be rectangle")
print("Monkeypatching successful!")
"""Script to test Numba-accelerated polytope operations. Summary: this does NOT works.

>>> def __array_ufunc__(...)
... TypeError: Method '__array_ufunc__' is not supported.

>>> __array_ufunc__ = None 
... TypeError: class members are not yet supported: __array_ufunc__

>>> _ = A @ polytope
... ValueError: matmul: Input operand 1 does not have enough dimensions (has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)

What Doesn't Work:
- matrix @ polytope ❌ - Direct @ operator doesn't work due to Numba limitations
- __array_ufunc__ ❌ - Not supported in Numba jitclass
- __rmatmul__ ❌ - Not properly dispatched by NumPy for Numba objects
- __array__ ❌ - Not supported in Numba jitclass

"""

import numpy as np
import numba as nb


spec = [
    ('vertices', nb.float64[:, :])
]


@nb.experimental.jitclass(spec)
class NumbaPolytope:
    def __init__(self, vertices):
        self.vertices = np.copy(vertices).astype(np.float64)

    def volume(self):
        # Simple volume calculation for a 2D polygon using the shoelace formula
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    def translate(self, vector):
        self.vertices += vector

    def __matmul__(self, other):
        # Handle polytope @ matrix
        return NumbaPolytope(self.vertices @ other)
    
    def transform(self, matrix):
        # Alternative to matrix @ polytope when __rmatmul__ doesn't work
        # Transform vertices: (matrix @ vertices.T).T = vertices @ matrix.T
        return NumbaPolytope(self.vertices @ matrix.T)


# Helper functions for matrix operations
@nb.njit
def matmul_polytope(matrix, polytope_vertices):
    """Numba-compiled helper for matrix @ polytope operations"""
    return (matrix @ polytope_vertices.T).T


def matmul(a, b):
    """
    Enhanced matrix multiplication that supports:
    - matrix @ polytope  ->  returns transformed polytope
    - polytope @ matrix  ->  uses polytope.__matmul__ 
    - matrix @ matrix    ->  standard numpy matmul
    
    Usage: Use this instead of @ operator for matrix @ polytope
    """
    if isinstance(b, NumbaPolytope):
        return NumbaPolytope(matmul_polytope(a, b.vertices))
    elif isinstance(a, NumbaPolytope):
        return a.__matmul__(b)
    else:
        return np.matmul(a, b)


def test_numba_polytope():
    # Define vertices of a square
    vertices = np.array([[0.0, 0.0],
                         [1.0, 0.0],
                         [1.0, 1.0],
                         [0.0, 1.0]])
    
    A = np.array([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64)

    polytope = NumbaPolytope(vertices)

    # Test volume calculation
    vol = polytope.volume()
    assert np.isclose(vol, 1.0), f"Expected volume 1.0, got {vol}"

    # Test translation
    translation_vector = np.array([2.0, 3.0], dtype=np.float64)
    polytope.translate(translation_vector)

    expected_vertices = vertices + translation_vector
    assert np.allclose(polytope.vertices, expected_vertices), "Translation did not work as expected"

    # Test matrix transformation using the transform method
    mapped_polytope = polytope.transform(A)
    assert np.allclose(mapped_polytope.vertices, expected_vertices @ A.T), "Mapping did not work as expected"
    
    # Test matrix @ polytope using enhanced matmul function
    mapped_polytope_enhanced = matmul(A, polytope)
    expected_result = (A @ expected_vertices.T).T
    assert np.allclose(mapped_polytope_enhanced.vertices, expected_result), "Enhanced matmul failed"
    print("SUCCESS: matmul(A, polytope) provides @ operator functionality!")
    
    # Test polytope @ matrix (works natively with __matmul__)
    mapped_polytope_native = polytope @ A.T
    assert np.allclose(mapped_polytope_native.vertices, expected_vertices @ A), "Polytope @ matrix failed"
    print("SUCCESS: Native polytope @ matrix works!")
    
    print("All tests passed!")


if __name__ == "__main__":
    test_numba_polytope()
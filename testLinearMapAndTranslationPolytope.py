"""Script to test how we can implement a linear map, or a translation, on a polytope in either H- or V-representation"""

from typing import Never
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
import scipy as sp

from testRunPyCDDLIBDegenerate import enum_facets, enum_verts


class Polytope:
    """Simple Polytope class for testing purposes"""

    def __init__(self, *args: list[Never] | NDArray | tuple[NDArray, NDArray], **kwargs) -> None:
        if len(args) == 0:
            self._verts, self._Ab = kwargs['verts'], np.column_stack((kwargs['A'], kwargs['b']))
            self.is_vrepr, self.is_hrepr = True, True
        if len(args) == 1:
            self._verts, self._Ab = args[0], None
            self.is_vrepr, self.is_hrepr = True, False
        elif len(args) == 2:
            self._verts, self._Ab = None, np.column_stack(args)
            self.is_vrepr, self.is_hrepr = False, True
        else:
            raise ValueError("Polytope must be initialized with either vertices or (A, b) representation.")
        
    @property
    def verts(self) -> NDArray:
        if not self.is_vrepr:
            self._verts, _ = enum_verts(np.column_stack((self.A, self.b)))
            self.is_vrepr = True
        return self._verts
    
    @property
    def A(self) -> NDArray:
        if not self.is_hrepr:
            self._Ab = enum_facets(self.verts)
            self.is_hrepr = True
        return self._Ab[:, :-1]
    
    @property
    def b(self) -> NDArray:
        if not self.is_hrepr:
            self._Ab = enum_facets(self.verts)
            self.is_hrepr = True
        return self._Ab[:, -1]
    
    __array_ufunc__ = None

    def __matmul__(self, other: NDArray) -> Polytope:
        # FIXME: How to actually deal with the right-multiplication/transposition case? Because P @ M should not be allowed, right?
        raise NotImplementedError("Right-multiplication with a matrix is undefined for polytopes")

    def __rmatmul__(self, other: NDArray) -> Polytope:
        """This implements `other @ P` where `other` is a matrix and `P` is a polytope"""
        verts_new, Ab_new = None, None
        if self.is_vrepr or not is_square(other) or is_singular(other):
            verts_new = other @ self.verts
        if self.is_hrepr:
            if not is_square(other) or is_singular(other):
                pass
            else:
                M_inv = np.linalg.inv(other)
                A_new, b_new = self.A @ M_inv, self.b
                Ab_new = np.column_stack((A_new, b_new))
        if verts_new is not None and Ab_new is not None:
            return Polytope(verts=verts_new, A=A_new, b=b_new)
        elif verts_new is not None:
            return Polytope(verts_new)
        elif Ab_new is not None:
            return Polytope(A_new, b_new)
        else:
            raise ValueError("The linear map could not be applied to the polytope in either representation")
        
    def __add__(self, other: NDArray) -> Polytope:
        """This implements `P + v` where `P` is a polytope and `v` is a vector"""
        match other:
            case np.ndarray():
                return self.translate(other)
            case Polytope():
                raise NotImplementedError("Addition of two polytopes is yet defined")
            case _:
                raise ValueError(f"Undefined addition operation for Polytope and given type '{other.__class__.__name__}'")
        
        
    def translate(self, vector: NDArray) -> Polytope:
        """Translate the polytope by vector `vector`"""
        verts_new, Ab_new = None, None
        if self.is_vrepr:
            verts_new = self.verts + vector[:, np.newaxis]
        if self.is_hrepr:
            A_new, b_new = self.A, self.b + self.A @ vector
            Ab_new = np.column_stack((A_new, b_new))
        if verts_new is not None and Ab_new is not None:
            return Polytope(verts=verts_new, A=A_new, b=b_new)
        elif verts_new is not None:
            return Polytope(verts_new)
        elif Ab_new is not None:
            return Polytope(A_new, b_new)
        else:
            raise ValueError("The translation could not be applied to the polytope in either representation")
        
    def normalize(self, in_place: bool = True) -> Polytope:
        """Normalize the Ab-representation such that b ∈ {-1, 0, 1}^m"""
        if self.is_hrepr:
            divisor = np.array([np.abs(elem) if not np.isclose(elem, 0) else 1 for elem in self._Ab[:, -1]])[:, np.newaxis]
            Ab_new = self._Ab / divisor
            # FIXME: This is kind of arbitrary, but... might be nice for quick visual comparison
            Ab_new = Ab_new[np.argsort(Ab_new[:, -1])]
            # TODO: We can also take this one step further, and sort the entries of A in increasng order (so row [0, 0, 1] comes before [2, -3, 0], and [2, 3, 0] after that one. etc.)
            if in_place:
                self._Ab = Ab_new
            else:
                return Polytope(Ab_new[:, :-1], Ab_new[:, -1])
        else: 
            if not in_place:
                return Polytope(verts=self.verts)
        

def is_singular(A: NDArray) -> bool:
    """Check if a matrix `A` is singular, i.e., det(A) = 0"""
    try:
        np.linalg.inv(A)
        return False
    except np.linalg.LinAlgError:
        return True


def is_square(A: NDArray) -> bool:
    """Check if a matrix `A` is square, i.e., has the same number of rows and columns"""
    return A.ndim == 2 and A.shape[0] == A.shape[1]
    

def main() -> None:
    verts_cube = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    
    poly_cube_verts = Polytope(verts_cube)
    print(f"Original Cube Vertices:\n{poly_cube_verts.verts}")

    Ab_cube = np.array([[1, 0, 1], [-1, 0, 0], [0, 1, 1], [0, -1, 0]])
    A_cube, b_cube = Ab_cube[:, :-1], Ab_cube[:, -1]

    poly_cube_hrepr = Polytope(A_cube, b_cube)
    print(f"Original Cube Facets (A, b):\n{poly_cube_hrepr.A}, {poly_cube_hrepr.b}")
    print(f"Converted Cube Vertices from H-repr:\n{poly_cube_hrepr.verts}")

    M_shear = np.array([[1, 1], [0, 1]])

    poly_cube_verts_sheared = M_shear @ poly_cube_verts  # FIXME: This completely screws up the signature, because the NumPy ufunc machinery gets in the way
    print(f"Sheared Cube Vertices:\n{poly_cube_verts_sheared.verts}")
    print(f"Sheared Cube Facets from V-repr:\n{poly_cube_verts_sheared.A}, {poly_cube_verts_sheared.b}")

    # Remove the V-repr to force H-repr usage
    poly_cube_hrepr.is_vrepr, poly_cube_hrepr._verts = False, None
    poly_cube_hrepr_sheared = M_shear @ poly_cube_hrepr
    print(f"Sheared Cube Facets (A, b):\n{poly_cube_hrepr_sheared.A}, {poly_cube_hrepr_sheared.b}")
    print(f"Sheared Cube Vertices from H-repr:\n{poly_cube_hrepr_sheared.verts}")

    M_generic = np.array([[2, -4], [1, 3]])

    # Remove the H-repr to force V-repr usage
    poly_cube_verts.is_hrepr, poly_cube_verts._Ab = False, None
    poly_cube_generic_sheared = M_generic @ poly_cube_verts
    print(f"\nGeneric Sheared Cube Vertices:\n{poly_cube_generic_sheared.verts}")
    print(f"Generic Sheared Cube Facets from V-repr:\n{poly_cube_generic_sheared.A}, {poly_cube_generic_sheared.b}")    

    # Remove the V-repr to force H-repr usage
    poly_cube_hrepr.is_vrepr, poly_cube_hrepr._verts = False, None
    poly_cube_generic_sheared_hrepr = M_generic @ poly_cube_hrepr
    print(f"Generic Sheared Cube Facets (A, b):\n{poly_cube_generic_sheared_hrepr.A}, {poly_cube_generic_sheared_hrepr.b}")
    print(f"Generic Sheared Cube Vertices from H-repr:\n{poly_cube_generic_sheared_hrepr.verts}")

    # Translation test
    verts_cube = np.array([[0, 0], [1, 0], [1, 1], [0, 1]]).T
    translation_vector = np.array([2, 3])
    
    poly_cube_verts = Polytope(verts_cube)

    Ab_cube = np.array([[1, 0, 1], [-1, 0, 0], [0, 1, 1], [0, -1, 0]])
    A_cube, b_cube = Ab_cube[:, :-1], Ab_cube[:, -1]

    poly_cube_hrepr = Polytope(A_cube, b_cube)

    poly_cube_verts_translated = poly_cube_verts + translation_vector

    poly_cube_hrepr_translated = poly_cube_hrepr + translation_vector

    print(f"Original Cube Vertices:\n{poly_cube_verts.verts}")
    print(f"Original Cube Facets from V-repr:\n{poly_cube_verts.A}, {poly_cube_verts.b}")


    print(f"Original Cube Facets (A, b):\n{poly_cube_hrepr.A}, {poly_cube_hrepr.b}")
    print(f"Converted Cube Vertices from H-repr:\n{poly_cube_hrepr.verts}")

    print(f"Translated Cube Vertices:\n{poly_cube_verts_translated.verts}")
    print(f"Translated Cube Facets from V-repr:\n{poly_cube_verts_translated.A}, {poly_cube_verts_translated.b}")

    print(f"Translated Cube Facets (A, b):\n{poly_cube_hrepr_translated.A}, {poly_cube_hrepr_translated.b}")
    print(f"Translated Cube Vertices from H-repr:\n{poly_cube_hrepr_translated.verts}")

    # Normalize the two for comparison
    poly_cube_verts_translated.normalize()
    poly_cube_hrepr_translated.normalize()

    print(f"Normalized Translated Cube Facets from V-repr:\n{poly_cube_verts_translated.A}, {poly_cube_verts_translated.b}")
    print(f"Normalized Translated Cube Facets (A, b):\n{poly_cube_hrepr_translated.A}, {poly_cube_hrepr_translated.b}")

    A_zeros, b_zeros = np.array([[1, 1], [0, 1], [-1, 0], [2, 1], [6, 0]]), np.array([0, -1, 0, 5, -2])
    poly_zeros = Polytope(A_zeros, b_zeros)
    print(f"Before normalization:\n{poly_zeros.A}, {poly_zeros.b}")
    poly_zeros.normalize()
    print(f"After normalization:\n{poly_zeros.A}, {poly_zeros.b}")


if __name__ == "__main__":
    main()
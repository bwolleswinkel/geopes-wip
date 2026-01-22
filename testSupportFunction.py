"""Script to test support functions of polytopes, subspaces, and ellipsoids"""

from typing import Literal
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
import scipy as sp

CVXPY_AVAILABLE = False  # NOTE: Set the False to emulate cvxpy not being installed

try:
    if CVXPY_AVAILABLE:
        import cvxpy as cvx
    else:
        raise ImportError
except ImportError as _:
    # FIXME: Should I have `pass` here, of `cvx = None`?
    pass  # CVXPY not available; will raise error if used in cfg.LP_SOLVER


# ====== PLACEHOLDER ======

@dataclass
class CFG:
    _LP_SOLVER: Literal['scipy', 'cvxpy'] = field(default='scipy', init=False)

    @property
    def LP_SOLVER(self) -> Literal['scipy', 'cvxpy']:
        return self._LP_SOLVER
    
    @LP_SOLVER.setter
    def LP_SOLVER(self, value: Literal['scipy', 'cvxpy']) -> None:
        if value == 'cvxpy':
            try:
                if CVXPY_AVAILABLE:  # NOTE: Set the False to mimic cvxpy not being installed
                    import cvxpy  # noqa: F401
                else:
                    raise ImportError
            except ImportError as _:
                raise ImportError("CVXPY cannot be imported, and is possibly not installed")
        self._LP_SOLVER = value

global cfg
cfg = CFG()

# ====== PLACEHOLDER ======


class Polytope:
    """Simple Polytope class for testing purposes"""

    def __init__(self, *args: NDArray | tuple[NDArray, NDArray]) -> None:
        if len(args) == 1:
            self._verts = args[0]  # NOTE: Assume shape (dim, num_vertices)
            self.is_vrepr, self.is_hrepr = True, False
            self._Ab = None
        elif len(args) == 2:
            self._Ab = np.column_stack(args)
            self.is_vrepr, self.is_hrepr = False, True
            self._verts = None
        else:
            raise ValueError("Polytope must be initialized with either vertices or (A, b) representation.")
        
    @property
    def verts(self) -> NDArray:
        if not self.is_vrepr:
            raise ValueError("Polytope is not in V-representation.")
        return self._verts
    
    @property
    def A(self) -> NDArray:
        if not self.is_hrepr:
            raise ValueError("Polytope is not in H-representation.")
        return self._Ab[:, :-1]
    
    @property
    def b(self) -> NDArray:
        if not self.is_hrepr:
            raise ValueError("Polytope is not in H-representation.")
        return self._Ab[:, -1]
    
    def support(self, direction: NDArray, repr: Literal['auto', 'hrepr', 'vrepr'] = 'auto') -> float:
        """Compute the support function of the polytope in the given direction"""
        # FIXME: What to do if the direction is not a unit vector? Just accept it as is, or normalize?
        # NOTE: According to Gemini, the support function is 'positively homogeneous (of degree 1)', meaning h(a * d) = a * h(d) for a > 0 (so not if the direction is negative!) 
        if not repr in ['auto', 'hrepr', 'vrepr']:
            raise ValueError("Parameter 'repr' must be either 'auto', 'hrepr', or 'vrepr'")
        if self.is_vrepr and repr in ['auto', 'vrepr']:  # Cheapest option
            return self._support_from_vrepr(direction)
        elif self.is_hrepr and repr in ['auto', 'hrepr']:
            return self._support_from_hrepr(direction)
        else:
            raise ValueError("Polytope is neither in V-representation nor H-representation")
        
    def _support_from_vrepr(self, direction: NDArray) -> float:
        """Compute support function from V-representation"""
        return np.max(self.verts.T @ direction)
    
    def _support_from_hrepr(self, direction: NDArray) -> float:
        """Compute support function from H-representation using linear programming"""
        global cfg
        match cfg.LP_SOLVER:  # NOTE: We assume, that whenever cvxpy is set beforehand, only then is it imported, and an error is raised otherwise
            case 'scipy':
                res = sp.optimize.linprog(-direction, A_ub=self.A, b_ub=self.b)
                if res.success:
                    return -res.fun
                else:
                    raise ValueError("Linear programming failed to compute support function.")  # TODO: Make this OptimizationInfeasibleError?
            case 'cvxpy':
                x = cvx.Variable(self.A.shape[1])
                objective = cvx.Maximize(direction @ x)
                constraints = [self.A @ x <= self.b]  # TODO: Replace this with the `is_in` idiom
                prob = cvx.Problem(objective, constraints)
                prob.solve()
                if prob.status in [cvx.OPTIMAL, cvx.OPTIMAL_INACCURATE]:
                    return prob.value
                else:
                    raise ValueError("CVXPY failed to compute support function.")  # TODO: Make this OptimizationInfeasibleError?
            case _:
                raise ValueError(f"Unknown LP solver '{cfg.LP_SOLVER}'.")
    

class Ellipsoid:
    """Simple Ellipsoid class for testing purposes"""

    def __init__(self, c: NDArray, Q: NDArray) -> None:
        self.c = c
        self.Q = Q

    def support(self, direction: NDArray) -> float:
        """Compute the support function of the ellipsoid in the given direction"""
        if np.isclose(direction, 0).all():
            raise ValueError("Direction vector 'direction' cannot be zero")
        return np.dot(self.c, direction) + np.sqrt(direction.T @ np.linalg.inv(self.Q) @ direction)  # TODO: Replace with weighted norm from linalg


class Subspace:
    """Simple Subspace class for testing purposes"""

    def __init__(self, basis: NDArray) -> None:
        self.basis = basis

    def __contains__(self, vector: NDArray) -> bool:
        """Check if the vector is in the subspace"""
        _, residuals, *_ = np.linalg.lstsq(self.basis, vector, rcond=None)
        return np.isclose(residuals, 0).all()

    @property
    def perp(self) -> NDArray:
        """Compute the orthogonal complement of the subspace"""
        return Subspace(sp.linalg.null_space(self.basis.T))

    # QUESTION: How should I typehint the return type here? It can be either 0 or infinity, so Literal[0, np.inf]? Because float gives too little information
    def support(self, direction: NDArray) -> Literal[0, 'np.inf']:
        """Compute the support function of the subspace in the given direction"""
        if np.isclose(direction, 0).all():
            raise ValueError("Direction vector 'direction' cannot be zero")
        return 0 if direction in self.perp else np.inf


def support(obj: Polytope | Ellipsoid | Subspace, vector: NDArray, repr: Literal['auto', 'hrepr', 'vrepr'] = 'auto') -> float:
    """Compute the support function of the given object in the specified direction. When the vector `direction` is normalized (not required), the support function returns the signed distance from the origin to the supporting hyperplane in that direction.
    
    Parameters
    ----------
    obj : Polytope | Ellipsoid | Subspace
        The geometric object for which to compute the support function.
    direction : NDArray
        The direction vector in which to compute the support function.
    repr : 'auto' | 'hrepr' | 'vrepr', default 'auto'
        Representation to use for polytopes. 'auto' uses the V-representation if available (most efficient), otherwise the H-representation. 'hrepr' forces the use of H-representation (which is computed if not available), which requires solving a linear program, and 'vrepr' forces the use of V-representation (which is computed if not available).
        
    Returns
    -------
    signed_distance : float
        The value of the support function in the given direction.

    Warnings
    --------
    - `vector` is not automatically normalized; it is assumed that the user provides the intended vector.

    Notes
    -----
    The support function is positively homogeneous of degree 1, meaning that for any positive scalar `a`, the relation `h(a * d) = a * h(d)` holds. However, this does not apply for negative scalars.
    
    """
    if not repr in ['auto', 'hrepr', 'vrepr']:
        raise ValueError("Parameter 'repr' must be either 'auto', 'hrepr', or 'vrepr'")
    if not isinstance(obj, Polytope) and repr != 'auto':
        # QUESTION: Is this the best way to handle this? Should I maybe raise a warning instead that this will be ignored? Or just ignore the 'repr' parameter for non-polytopes?
        raise ValueError("Parameter 'repr' is only applicable to polytopes")
    return obj.support(vector, repr=repr) if isinstance(obj, Polytope) else obj.support(vector)


def main():
    dir_vec = np.array([10, 4])

    verts = np.array([[0, 0], [1, 0], [0, 1]]).T
    P_verts = Polytope(verts)
    print("Polytope (V-repr) support:", support(P_verts, dir_vec))

    print("Polytope (V-repr) support (normalized):", P_verts.support(dir_vec / np.linalg.norm(dir_vec)))

    A, b = np.array([[-1, 0], [0, -1], [1, 1]]), np.array([0, 0, 1])
    P_hrepr = Polytope(A, b)
    print("Polytope (H-repr) support:", P_hrepr.support(dir_vec, repr='hrepr'))

    try:
        cfg.LP_SOLVER = 'cvxpy'
    except ImportError as e:
        print("Caught ImportError:", e)
    # Manually overrule to see the behavior when cvxpy is not available
    try:
        cfg._LP_SOLVER = 'cvxpy'
        # QUESTION: Do we want to encapsulate this (i.e., the _support_from_hrepr, case 'cvxpy') in a try-except block to catch the NameError (or NoneTypeError)? To be fair, this SHOULD be caught when setting the LP_SOLVER above...
        print("Polytope (H-repr, cvxpy) support:", support(P_hrepr, dir_vec))
    except NameError as e:
        print("Caught NameError (cvxpy not available):", e)

    c, Q = np.array([1, 1]), np.eye(2)
    E = Ellipsoid(c, Q)
    print("Ellipsoid support:", support(E, dir_vec))

    basis = np.array([1, 2]).reshape(2, 1)
    S = Subspace(basis)
    print("Subspace support:", support(S, dir_vec))

    basis_perp = np.array([-dir_vec[1], dir_vec[0]]).reshape(2, 1)
    S_perp = Subspace(basis_perp)
    print("Subspace (orthogonal to direction) support:", S_perp.support(dir_vec))


if __name__ == "__main__":
    main()
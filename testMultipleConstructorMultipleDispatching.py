"""This is a script to test multiple constructor with multiple dispatching, or using `__init__` as a multi-method"""

from typing import Callable, Any

import numpy as np
from numpy.typing import NDArray


# Add a multiple dispatching decorator
def multipledispatch(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
    """Decorator that enables multiple dispatch functionality for a function or method.
    
    The decorated function `func` can then use dispatch decorators, such as @func.args(len=n) to dispatch based on the number of positional arguments.

    Parameters
    ----------
    func : Callable[[Any], Any]
        The function or method to be decorated for multiple dispatching.

    Returns
    -------
    Callable[[Any], Any]
        A wrapper function that handles multiple dispatching based on argument length.

    Examples
    --------
    >>> @multipledispatch
    ... def example_func(*args):
    ...     return "default"
    ...
    ... @example_func.args(len=1)
    ... def _(arg1):
    ...     return f"one arg: {arg1}"
    ...
    ... @example_func.args(len=2)
    ... def _(arg1, arg2):
    ...     return f"two args: {arg1}, {arg2}"  
    
    """
    # Create a dictionary to store dispatchers
    dispatchers = {}
    
    def dispatcher_wrapper(*args, **kwargs):
        # For methods, the first argument is self
        if args:
            # Get the actual arguments (excluding self for methods)
            if hasattr(args[0], func.__name__):
                # This is a method call
                instance = args[0]
                actual_args = args[1:]
            else:
                # This is a function call
                instance = None
                actual_args = args
        else:
            instance = None
            actual_args = args
            
        # Try to dispatch based on argument length
        arg_length = len(actual_args)
        if arg_length in dispatchers:
            if instance is not None:
                return dispatchers[arg_length](instance, *actual_args, **kwargs)
            else:
                return dispatchers[arg_length](*actual_args, **kwargs)
        
        # Fallback to original function
        return func(*args, **kwargs)
    
    def args(len: int):
        """Create a dispatcher that matches based on argument length"""
        def decorator(dispatch_func):
            dispatchers[len] = dispatch_func
            return dispatch_func
        return decorator
    
    # Attach the len_args method to the wrapper
    dispatcher_wrapper.args = args
    
    # Copy function attributes
    dispatcher_wrapper.__name__ = func.__name__
    dispatcher_wrapper.__doc__ = func.__doc__
    dispatcher_wrapper.__module__ = func.__module__
    
    return dispatcher_wrapper


class Polytope():
    """Polytope represented in either V-representation (vertices) or H-representation (half-spaces).
    
    A polytope is the convex hull of a finite set of points in R^n (V-representation) or the bounded intersection of a finite number of half-spaces (H-representation). This class provides methods to convert between these representations and perform operations on polytopes.
    
    Attributes
    ----------
    A: NDArray[Float]
        The matrix of shape (m, n) defining the m half-spaces in H-representation (Ax <= b).
    b: NDArray[Float]
        The vector of shape (m,) defining the m half-spaces in H-representation (Ax <= b).
    verts: NDArray[Float]
        The matrix of shape (n, k) defining the k vertices in V-representation.

    Examples
    --------
    Initialize a polytope from vertices (V-representation):
    >>> verts = np.array([[0, 0], [1, 0], [0, 1]])
    >>> poly = pes.poly(verts)
    >>> print(poly)
    Polytope with 3 vertices in R^2

    Initialize a polytope from half-spaces (H-representation):
    >>> A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    >>> b = np.array([1, 1, 0, 0])
    >>> poly = pes.poly(A, b)
    >>> print(poly)
    Polytope defined by 4 half-spaces in R^2

    Initialize an empty polytope in R^n:
    >>> poly = pes.poly(n=3)
    >>> print(poly)
    Empty polytope in R^3

    """

    @multipledispatch
    def __init__(self, *args, n: int | None = None, rays: NDArray | None = None) -> None:
        """Initialize a Polytope either from vertices or half-spaces.

        Parameters
        ----------
        *args:
            Variable length positional argument list. Must be of size 0, 1, or 2, according to the initialization method:
            - len(0) -> []: Initialize an empty polytope in R^n (requires `n`).
            - len(1) -> verts: NDArray of shape (n, k) representing k vertices in R^n.
            - len(2) -> A, b: NDArray of shape (m, n) and NDArray of shape (m,), respectively, representing m half-spaces in H-representation.
        n: int, optional
            Dimension of the space. Required if initializing an empty polytope.
        rays: NDArray, optional
            Optional rays for unbounded polytopes.

        Raises
        ------
        ValueError
            If the number of arguments is invalid or if `n` is not provided when initializing an empty polytope.

        """
        match len(args):
            case 0:
                if n is None:
                    raise ValueError("Dimension 'n' must be specified when no arguments are provided")
                self._init_empty(n)
            case 1:
                verts, = args
                self._init_from_verts(verts, rays)
            case 2:
                A, b = args
                self._init_from_ineqs(A, b)
            case _:
                raise ValueError("Invalid number of arguments provided to Polytope constructor")

    @__init__.args(len=0)    
    def _init_empty(self, *args, **kwargs) -> None:
        """Initialize an empty polytope in R^n"""
        self.n = kwargs.get('n')
        self.A = None
        self.b = None
        self.verts = None
        print(f"Initialized empty polytope in R^{self.n}")
    
    @__init__.args(len=1)
    def _init_from_verts(self, *args, **kwargs) -> None:
        """Initialize a polytope from vertices (V-representation)"""
        verts, *_ = args
        self.verts = verts
        self.n = verts.shape[0]
        self.A = None
        self.b = None
        rays = kwargs.get('rays')
        self.rays = rays
        print(f"Initialized polytope from {verts.shape[1]} vertices in R^{self.n}")

    @__init__.args(len=2)
    def _init_from_ineqs(self, *args, **kwargs) -> None:
        """Initialize a polytope from half-spaces (H-representation)"""
        A, b = args
        self.A = A
        self.b = b
        self.n = A.shape[1]
        self.verts = None
        print(f"Initialized polytope from {A.shape[0]} half-spaces in R^{self.n}")
    

def main() -> None:
    """Main function to run tests or examples"""

    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([1, 1, 0, 0])

    verts = np.array([[0, 0], [1, 0], [0, 1]])

    P_1 = Polytope(n=2)
    P_2 = Polytope(verts)
    P_3 = Polytope(A, b)


if __name__ == "__main__":
    main()
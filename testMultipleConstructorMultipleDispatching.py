"""This is a script to test multiple constructor with multiple dispatching, or using `__init__` as a multi-method"""

from typing import Callable, Any, Never

import numpy as np
from numpy.typing import NDArray

# This is just to test the methods `poly` and `poly_from_verts` at the end
from testDocstringWrappers import tightwrap_wraps


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
            
        # Try to dispatch based on registered conditions
        arg_length = len(actual_args)
        
        # Check all registered dispatchers for a match
        for key, dispatcher_func in dispatchers.items():
            if _matches_dispatch_condition(key, actual_args, kwargs):
                if instance is not None:
                    return dispatcher_func(instance, *actual_args, **kwargs)
                else:
                    return dispatcher_func(*actual_args, **kwargs)
        
        # Fallback to original function
        return func(*args, **kwargs)
    
    def _matches_dispatch_condition(condition, args, kwargs):
        """Check if the current call matches a dispatch condition"""
        if isinstance(condition, int):
            # Simple length-based dispatch
            return len(args) == condition
        elif isinstance(condition, tuple):
            # Complex condition with multiple criteria
            arg_len, exclude_kwargs, include_kwargs, kwargs_values = condition
            
            # Check length condition
            if arg_len != -1 and len(args) != arg_len:
                return False
            
            # Check kwargs exclusions
            if exclude_kwargs and any(key in kwargs for key in exclude_kwargs):
                return False
            
            # Check kwargs inclusions
            if include_kwargs and not all(key in kwargs for key in include_kwargs):
                return False
                
            # Check kwargs values
            if kwargs_values:
                for key, expected_value in kwargs_values:
                    if kwargs.get(key) != expected_value:
                        return False
            
            return True
        
        return False
    
    def register(len_args: int | None = None, exclude_kwargs: list | None = None, include_kwargs: list | None = None, kwargs_values: dict | None = None):
        """Create a dispatcher that matches based on various conditions
        
        Parameters
        ----------
        len_args : int, optional
            Required number of positional arguments
        exclude_kwargs : list, optional
            List of kwarg keys that must NOT be present
        include_kwargs : list, optional
            List of kwarg keys that must be present
        kwargs_values : dict, optional
            Dictionary of kwarg key-value pairs that must match exactly
        """
        def decorator(dispatch_func):
            if len_args is not None and exclude_kwargs is None and include_kwargs is None and kwargs_values is None:
                # Simple case - just length-based dispatch
                condition = len_args
            else:
                # Complex case - build condition as a hashable tuple
                condition = (
                    len_args if len_args is not None else -1,
                    tuple(exclude_kwargs) if exclude_kwargs else (),
                    tuple(include_kwargs) if include_kwargs else (),
                    tuple(sorted(kwargs_values.items())) if kwargs_values else ()
                )
            
            dispatchers[condition] = dispatch_func
            return dispatch_func
        return decorator
    
    # Attach the args method to the wrapper
    dispatcher_wrapper.register = register
    
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
    def __init__(self, *args: tuple[Never] | tuple[NDArray] | tuple[NDArray, NDArray], n: int | None = None, rays: NDArray | None = None) -> None:
        """Initialize a Polytope either from vertices or half-spaces.

        Parameters
        ----------
        *args:
            Variable length positional argument list. Must be of size 0, 1, or 2, according to the initialization method:
            - len(0) -> []: Initialize an empty polytope in R^n (requires `n`). Note that if the keywords `A`, `b`, or `verts` are provided instead, they will cause a dispatch to the appropriate constructor.
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
        # NOTE: This is the fallback method if no dispatchers match, and should raise an error
        raise ValueError(f"Invalid number of arguments ({len(args)}) provided to construct a polytope")

    @__init__.register(len_args=0, exclude_kwargs=['A', 'b', 'verts'])  
    # FIXME: Should I 'write' out the keyword arguments in the signature below as well? See `_init_from_verts`
    def _init_empty(self, *_: tuple[Never], **kwargs) -> None:
        """Initialize an empty polytope in R^n"""
        if 'n' not in kwargs:
            raise ValueError("Dimension 'n' must be specified for empty polytope initialization")
        elif 'A' in kwargs or 'b' in kwargs or 'verts' in kwargs:
            raise ValueError("No other parameters should be provided when initializing an empty polytope")
        self.n = kwargs.get('n')
        self.A = None
        self.b = None
        self.verts = None
        print(f"Initialized empty polytope in R^{self.n}")

    @__init__.register(len_args=1)
    @__init__.register(len_args=0, include_kwargs=['verts'])
    def _init_from_verts(self, verts: NDArray | None, **kwargs) -> None:
        """Initialize a polytope from vertices (V-representation)"""
        if 'A' in kwargs or 'b' in kwargs:
            if verts is not None:
                raise ValueError("Cannot provide 'A' or 'b' when initializing with one positional argument")
            raise ValueError("Cannot provide 'A' or 'b' when initializing from vertices")
        if verts is not None:
            verts = verts
        else:
            verts = kwargs['verts']
        self.verts = verts
        self.n = verts.shape[0]
        self.A = None
        self.b = None
        rays = kwargs.get('rays')
        self.rays = rays
        print(f"Initialized polytope from {verts.shape[1]} vertices in R^{self.n}")

    @__init__.register(len_args=2)
    @__init__.register(len_args=0, include_kwargs=['A', 'b'])
    # FIXME: Should I replace `*args` below with explicit `A: NDArray, b: NDArray`?
    # FIXME: Should I name this method `_init_from_ineqs` or `_init_hrepr`? Think I like the latter more actually
    def _init_hrepr(self, *args: tuple[Never] | tuple[NDArray, NDArray], **kwargs) -> None:
        """Initialize a polytope from half-spaces (H-representation)"""
        A, b = args
        self.A = A
        self.b = b
        self.n = A.shape[1]
        self.verts = None
        print(f"Initialized polytope from {A.shape[0]} half-spaces in R^{self.n}")
    

@tightwrap_wraps(Polytope.__init__)
def poly(*args: tuple[Never] | tuple[NDArray] | tuple[NDArray, NDArray], **kwargs) -> Polytope:
    return Polytope(*args, **kwargs)


@tightwrap_wraps(Polytope._init_from_verts)
def poly_from_verts(verts: NDArray, rays: NDArray | None = None) -> Polytope:
    """Create a Polytope from vertices (V-representation)"""
    return Polytope(verts=verts, rays=rays)


def main() -> None:
    """Main function to run tests or examples"""

    A = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]])
    b = np.array([1, 1, 0, 0])

    verts = np.array([[0, 0], [1, 0], [0, 1]])

    P_1 = poly(n=2)  # FIXME: The docstring is correct, but why is the signature here not shown correctly? "(function) def poly(Any) -> Polytope" | Any should be "(*args: ..., ...)"
    P_2a = Polytope(verts)
    P_2b = Polytope(verts=verts, rays=None)
    P_2c = poly_from_verts(verts)
    P_3 = Polytope(A, b)
    try:
        P_4 = Polytope(n=2, A=A, b=b)  # This has both required (A,b) and forbidden (n) for kwargs dispatcher
    except Exception as e:
        print(f"Expected error from fallback: {e}")
    try:
        P_5 = Polytope() 
    except ValueError as e:
        print(f"Expected error from invalid args: {e}")
    try:
        P_6 = Polytope(verts, b, 8)
    except ValueError as e:
        print(f"Expected error from too many args: {e}")
    try:
        P_7 = Polytope(A, b=b)
    except ValueError as e:
        print(f"Expected error from mixing args and kwargs: {e}")
    try:
        P_8 = Polytope(verts=verts, A=A)
    except ValueError as e:
        print(f"Expected error from mixing args and kwargs: {e}")
    try:
        P_8 = Polytope(A=A, verts=verts)
    except ValueError as e:
        print(f"Expected error from mixing args and kwargs: {e}")


if __name__ == "__main__":
    main()
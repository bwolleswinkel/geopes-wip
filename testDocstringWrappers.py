"""This is a script to see how type-hinters (i.e., Pylance) behave when we try to wrap certain docstrings around functions and methods"""

from __future__ import annotations
import sys
from inspect import Signature, _empty
from types import GetSetDescriptorType, ModuleType
from typing import Any, Callable, TypeVar, cast, Dict, Tuple, ParamSpec, Never
from functools import partial
from functools import wraps as functools_wraps

from numpy.typing import NDArray

# FROM: https://github.com/Tinche/tightwrap/blob/main/src/tightwrap/__init__.py  | `tightwrap` package
Annotations = Dict[str, Any]
Globals = Dict[str, Any]
Locals = Dict[str, Any]
GetAnnotationsResults = Tuple[Annotations, Globals, Locals]


# FROM: https://github.com/Tinche/tightwrap/blob/main/src/tightwrap/__init__.py  | `tightwrap` package
def eval_if_necessary(source: Any, globals: Globals, locals: Locals) -> Any:
    if not isinstance(source, str):
        return source

    try:
        return eval(source, globals, locals)
    except NameError:
        # If evaluation fails due to forward reference, return the string as-is
        # The type system will handle it later
        return source


# FROM: https://github.com/Tinche/tightwrap/blob/main/src/tightwrap/__init__.py  | `tightwrap` package
def get_annotations(obj: Callable[..., Any]) -> GetAnnotationsResults:
    # Copied from https://github.com/python/cpython/blob/3.12/Lib/inspect.py#L176-L288

    obj_globals: Any
    obj_locals: Any
    unwrap: Any

    if isinstance(obj, type):
        obj_dict = getattr(obj, "__dict__", None)

        if obj_dict and hasattr(obj_dict, "get"):
            ann = obj_dict.get("__annotations__", None)
            if isinstance(ann, GetSetDescriptorType):
                ann = None
        else:
            ann = None

        obj_globals = None
        module_name = getattr(obj, "__module__", None)

        if module_name:
            module = sys.modules.get(module_name, None)

            if module:
                obj_globals = getattr(module, "__dict__", None)

        obj_locals = dict(vars(obj))
        unwrap = obj

    elif isinstance(obj, ModuleType):
        ann = getattr(obj, "__annotations__", None)
        obj_globals = getattr(obj, "__dict__")
        obj_locals = None
        unwrap = None

    elif callable(obj):
        ann = getattr(obj, "__annotations__", None)
        obj_globals = getattr(obj, "__globals__", None)
        obj_locals = None
        unwrap = obj

    else:
        raise TypeError(f"{obj!r} is not a module, class, or callable.")

    if ann is None:
        return cast(GetAnnotationsResults, ({}, obj_globals, obj_locals))

    if not isinstance(ann, dict):
        raise ValueError(f"{obj!r}.__annotations__ is neither a dict nor None")

    if not ann:
        return cast(GetAnnotationsResults, ({}, obj_globals, obj_locals))

    if unwrap is not None:
        while True:
            if hasattr(unwrap, "__wrapped__"):
                unwrap = unwrap.__wrapped__
                continue
            if isinstance(unwrap, partial):
                unwrap = unwrap.func
                continue
            break
        if hasattr(unwrap, "__globals__"):
            obj_globals = unwrap.__globals__

    return_value = {
        key: eval_if_necessary(value, obj_globals, obj_locals)
        for key, value in cast(Dict[str, Any], ann).items()
    }

    return cast(GetAnnotationsResults, (return_value, obj_globals, obj_locals))


P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")


# FROM: https://github.com/Tinche/tightwrap/blob/main/src/tightwrap/__init__.py  | `tightwrap` package
# NOTE: © Tinche, 2024 | Adapted <- Don't know if I should place this notice here?
def _get_resolved_signature(fn: Callable[..., Any]) -> Signature:
    signature = Signature.from_callable(fn)
    evaluated_annotations, fn_globals, fn_locals = get_annotations(fn)
    for name, parameter in signature.parameters.items():
        parameter._annotation = evaluated_annotations.get(name, _empty)  # type: ignore
    new_return_annotation = eval_if_necessary(
        signature.return_annotation, fn_globals, fn_locals
    )
    signature._return_annotation = new_return_annotation  # type: ignore
    return signature


# FROM: https://github.com/Tinche/tightwrap/blob/main/src/tightwrap/__init__.py  | `tightwrap` package
def tightwrap_wraps(wrapped: Callable[P, Any]) -> Callable[[Callable[..., R]], Callable[P, R]]:
    """Apply `functools.wraps`"""

    def wrapper(fn: Callable[..., R]) -> Callable[P, R]:
        wrapper_return = _get_resolved_signature(fn).return_annotation
        res = functools_wraps(wrapped)(fn)

        orig_sig = _get_resolved_signature(wrapped)

        if orig_sig.return_annotation != wrapper_return:
            # We do a little rewriting.
            new_sig = Signature(None, return_annotation=wrapper_return)
            new_sig._parameters = orig_sig.parameters  # type: ignore
            res.__signature__ = new_sig  # type: ignore

        return cast(Callable[P, R], res)

    return wrapper


def mink_sum(poly_1: Polytope, poly_2: Polytope) -> Polytope:
    """Compute the Minkowski sum of two polytopes"""
    return poly_1  # FIXME: Placeholder


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

    """

    def __init__(self, *args: list[Never] | NDArray | tuple[NDArray, NDArray], n: int | None = None, A: NDArray | None = None, b: NDArray | None = None, A_eq: NDArray | None = None, b_eq: NDArray | None = None, verts: NDArray | None = None, rays: NDArray | None = None) -> None:
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
            
    def _init_empty(self, n: int) -> None:
        """Initialize an empty polytope in R^n"""
        self.n = n
    
    def _init_from_verts(self, verts: NDArray, rays: NDArray | None = None) -> None:
        """Initialize a polytope from vertices (V-representation)"""
        raise NotImplementedError()

    def _init_from_ineqs(self, A: NDArray, b: NDArray) -> None:
        """Initialize a polytope from half-spaces (H-representation)"""
        raise NotImplementedError()
    
    @tightwrap_wraps(mink_sum)
    def __add__(self, other: Polytope) -> Polytope:
        return mink_sum(self, other)
    
    @tightwrap_wraps(__add__)
    def __iadd__(self, other: Polytope) -> Polytope:
        result = mink_sum(self, other)
        self.__dict__.update(result.__dict__)
        return self


@functools_wraps(Polytope.__init__)
def poly_functools(*args, **kwargs) -> Polytope:
    return Polytope(*args, **kwargs)

@tightwrap_wraps(Polytope.__init__)
def poly_tightwrap(*args, **kwargs) -> Polytope:
    return Polytope(*args, **kwargs)


def main():
    poly_1 = poly_functools(n=3)  # This provides "(function) poly: _Wrapped[(self: Polytope, *args: Any, n: int | None = None, rays: NDArray | None = None), None, ..., Polytope]" with Pylance
    poly_2 = poly_tightwrap(n=3)    # Provides the proper signature!!!

    _ = poly_1 + poly_2  # This `+` provides the correct signature with Pylance! (but is missing a first argument...)
    poly_1 += poly_2  # This `+=` provides the correct signature with Pylance!


if __name__ == "__main__":
    main()

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


def enum_gens(Ab: NDArray, Ab_eq: NDArray) -> tuple[NDArray, NDArray]:
    """Calculate the generators of the H-representation `A @ x <= b`, `A_eq @ x == b`, where `Ab = [A, b]` and `Ab_eq = [A_eq, b_eq]`.

    Parameters
    ----------
    Ab : NDArray
        2-dimensional NumPy array of size `(m, n + 1)`. Datatype must be float.
    Ab_eq : NDArray
        2-dimensional NumPy array of size `(m_eq, n + 1)`. Datatype must be float.

    Returns
    -------
    verts : NDArray
        A 2-dimensional NumPY array of size `(k, n)` representing the `k` vertices of the polytope
    rays : NDArray
        A 2-dimensional NumPY array of size `(k_rays, n)` representing the `k_rays` rays of the polytope

    Notes
    -----
    The returned NumPy array will always have dtype float as output, even if they are integers; 
    this is currently under construction.
    """

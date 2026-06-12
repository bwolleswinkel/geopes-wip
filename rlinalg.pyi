from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from numpy.typing import NDArray


def span(A: NDArray) -> None:
    """Retrieve the linearly independent columns from the matrix `A`. The columns are taken 
    left-to-right, and any linearly dependent column is removed.
    
    Parameters
    ----------
    A : NDArray
        2-dimensional NumPy array of size `(n, m)`. Datatype must be either float or int.
        
    Returns
    -------
    A_prime : NDArray
        The linearly independent columns of `A`, taken left-to-right
        
    Raises
    ------
    TypeError
        If `A` is not a NumPy array or if the datatype is not float or int
    ValueError
        If `A` does not have 2 dimensions
    RuntimeError
        If ant residual error is propagated from the Rust backend
        
    Notes
    -----
    The backend is written in Rust.
    """

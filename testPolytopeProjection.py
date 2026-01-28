"""Script to test projecting a polytope onto a set of coordinate axes, or a more general subspace"""

from typing import Never
from copy import deepcopy

import numpy as np
from numpy.typing import NDArray, ArrayLike
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from testRunPyCDDLIBDegenerate import enum_facets, enum_verts


class Polytope:
    """Simple Polytope class for testing purposes"""

    def __init__(self, *args: list[Never] | NDArray | tuple[NDArray, NDArray], **kwargs) -> None:
        if len(args) == 0:
            self._verts, self._Ab = kwargs['verts'], np.column_stack((kwargs['A'], kwargs['b']))
            self.is_vrepr, self.is_hrepr = True, True
            self.n = self._verts.shape[0]
            self.is_full_dim = True
        if len(args) == 1:
            self._verts, self._Ab = args[0], None
            self.is_vrepr, self.is_hrepr = True, False
            self.n = self._verts.shape[0]
            self.is_full_dim = True
        elif len(args) == 2:
            self._verts, self._Ab = None, np.column_stack(args)
            self.is_vrepr, self.is_hrepr = False, True
            self.n = self._Ab.shape[1] - 1
            self.is_full_dim = True
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
    
    @property
    def m(self) -> int:
        return self.A.shape[0]
    
    @property
    def k(self) -> int:
        return self.verts.shape[1]
    
    def proj(self, other: ArrayLike[int] | Subspace, keep_dims: bool = True, in_place: bool = True) -> None | Polytope:
        if isinstance(other, Subspace):
            basis = other.basis
            M_proj = basis @ basis.T if other.orth else basis @ np.linalg.pinv(basis)
            verts_proj = M_proj @ self.verts
            if not keep_dims:
                coords, _, _, _ = np.linalg.lstsq(basis, verts_proj, rcond=None)
                verts_proj = coords
        else:
            try:
                selection = np.sort(np.atleast_1d(other).astype(int))
            except Exception as e:
                raise ValueError(f"Projection indices 'other' must be convertible to an array of integers, received: {other}") from e
            if np.any(selection < 0) or np.any(selection >= self.n):
                raise ValueError(f"Projection indices 'other' are out of bounds for polytope of dimension {self.n}, received: {selection}")
            if keep_dims:
                verts_proj = self.verts.copy()
                mask = np.ones(self.verts.shape[0], dtype=bool)
                mask[selection] = False
                verts_proj[mask, :] = 0
            else:
                verts_proj = self.verts[selection, :]
        if in_place:
            self._verts = conv(verts_proj)
            self.is_vrepr = True
            self.is_hrepr = False
            self._Ab = None
            self._chebcr, self.is_full_dim, self._vol = None, False, None
        else:
            poly = Polytope(conv(verts_proj))
            poly.is_full_dim = False
            return poly
        
    def plot(self, color: str | None = None, alpha: float = 0.5, 
             plot_edges: bool = True, label_verts: list[str] | bool = False, 
             label_facets: list[str] | bool = False, show: bool = True, 
             ax: Axes | None = None) -> Axes:
        """Plot a polytope in 3D (only works for 3D polytopes)"""
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            if ax.name != '3d':
                raise ValueError("The dimension of the polytope does not match the dimension of the provided axes 'ax'")
            fig, ax = None, ax
        if color is None:
            color = ax._get_lines.get_next_color()
        if self.is_full_dim:
            for idx in range(self.m):
                # TODO: Replace this with the facers-vertices incidence matrix when available
                verts_facet = self.verts[:, np.isclose(self.A[idx, :] @ self.verts, self.b[idx])]
                _plot_facet_3d(verts_facet, ax, color, alpha, plot_edges=plot_edges)
                if label_facets:
                    label = label_facets[idx] if isinstance(label_facets, list) else fr"${idx}$"
                    ax.text(*np.mean(verts_facet, axis=1), label, color='black')
        else:
            if self.verts.shape[1] == 2:  # Line segment
                ax.plot(self.verts[0, :], self.verts[1, :], self.verts[2, :], color=color)
            else:  # Facet
                _plot_facet_3d(self.verts, ax, color, alpha, plot_edges=plot_edges)
        if label_verts:
            for idx in range(self.k):
                label = label_verts[idx] if isinstance(label_verts, list) else fr"${idx}$"
                ax.text(self.verts[0, idx], self.verts[1, idx], self.verts[2, idx], label, color='black')
        if show:
            plt.show()
        return ax


class Subspace:
    """Simple Subspace class for testing purposes"""

    def __init__(self, basis: NDArray, orth: bool | None = None) -> None:
        self.basis = basis if not orth else sp.linalg.orth(basis)
        self.orth = orth

    def reduce(self):
        self.basis = span(self.basis)

    def proj(self, other: Subspace | NDArray, in_place: bool = True) -> None | Subspace:
        """Project this subspace onto another subspace `other`, or project a point onto this subspace if `other` is a point"""
        # FROM: GitHub Copilot GPT-4.1 | 2026/01/28 [untested/unverified]
        if isinstance(other, Subspace):
            M_proj = self.basis @ self.basis.T if self.orth else self.basis @ np.linalg.pinv(self.basis)
            basis_proj = M_proj @ other.basis
            if in_place:
                self.basis = basis_proj
            else:
                return Subspace(basis_proj, orth=self.orth)
        else:
            if self.orth:
                M_proj = self.basis @ self.basis.T
            else:
                M_proj = self.basis @ np.linalg.pinv(self.basis)
            point_proj = M_proj @ other
            return point_proj

    @classmethod
    def from_normal(cls, normal: NDArray) -> Subspace:
        """Create a Subspace from its normal vectors"""
        if normal.shape[0] != normal.shape[1]:
            basis = sp.linalg.null_space(normal.T)
        else:
            basis = np.zeros((normal.shape[0], 0))
        return cls(basis, orth=True)


def span(A: NDArray) -> NDArray:
    """Compute a basis spanned by the matrix `a` by removing linearly dependent columns"""
    if A.shape[1] == 0 or np.linalg.matrix_rank(A) == A.shape[1]:
        return A
    basis = [A[:, 0]]
    for i in range(1, A.shape[1]):
        if np.linalg.matrix_rank(np.column_stack((*basis, A[:, i]))) > len(basis):
            basis.append(A[:, i])
    return np.column_stack(basis)


def conv(verts: NDArray) -> NDArray:
    """Compute the convex hull of a set of points given by `verts`"""
    # FROM: GitHub Copilot GPT-4.1 | 2026/01/28 [untested/unverified]
    centered = verts - np.mean(verts, axis=1, keepdims=True)
    U, S, _ = np.linalg.svd(centered, full_matrices=False)
    rank = np.sum(S > 1E-12)
    if rank == 0:
        return verts[:, [0]]
    coords = U[:, :rank].T @ centered 
    if rank == 1:
        idx_min, idx_max = np.argmin(coords[0]), np.argmax(coords[0])
        return verts[:, np.unique([idx_min, idx_max])]
    hull = sp.spatial.ConvexHull(coords.T)
    return verts[:, hull.vertices]


def _plot_facet_3d(points: NDArray, ax: Axes, color: str, alpha: float, plot_edges: bool) -> None:
    # Assumes all points are coplanar
    if points.shape[1] < 3:
        raise ValueError("At least three points are required to define a facet in 3D")
    centroid = np.mean(points, axis=1)
    points = [p for p in points.T]
    look = np.cross(points[1] - points[0], points[2] - points[0])
    # Use signed_angle to sort points around centroid
    points = sorted(points, key=lambda p: signed_angle(points[0] - centroid, p - centroid, look=look))
    ax.add_collection3d(Poly3DCollection([np.array(points)], facecolor=mpl.colors.to_rgba(color, alpha=alpha), edgecolor=mpl.colors.to_rgba(color, alpha=1) if plot_edges else None))


def signed_angle(v_1: NDArray, v_2: NDArray, look: NDArray | None = None) -> float:
    """Compute the signed angle between two vectors `v_1` and `v_2`. If `look` is provided, the sign of the angle is determined by the direction of the cross product with respect to `look`. Counter-clockwise rotation from `v_1` to `v_2` is considered positive."""
    if v_1.size != v_2.size:
        raise ValueError("Both input vectors must have the same size")
    elif v_1.size != 2 and v_1.size != 3:
        raise NotImplementedError("Signed angle is only implemented for 2D and 3D vectors.")
    dot_prod_normal = np.dot(v_1, v_2) / np.linalg.norm(v_1, ord=2) / np.linalg.norm(v_2, ord=2)
    angle = np.arccos(np.clip(dot_prod_normal, -1.0, 1.0))
    # FIXME: This sign is absolutely not correct; there are multiple arguments where it fails, and changing the arguments does not flip the sign as expected.
    if v_1.size == 2:
        ...  # Do something to extend the cross product to 3D?
    sign = np.array(np.sign(np.cross(v_1, v_2).dot(look)))
    sign[sign == 0] = 1
    return sign * angle


def main():
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    verts_dodecahedron = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                                   [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
                                   [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
                                   [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]]).T
    verts_dodecahedron += (2 * np.ones(verts_dodecahedron.shape))
    poly_dodecahedron_verts = Polytope(verts_dodecahedron)

    poly_original = deepcopy(poly_dodecahedron_verts)
    poly_copy = deepcopy(poly_dodecahedron_verts)
    ax = poly_dodecahedron_verts.plot(show=False)
    poly_dodecahedron_verts.proj(range(2))
    poly_dodecahedron_verts.plot(ax=ax, show=False)
    poly_copy.proj(1)
    poly_copy.plot(ax=ax, show=True)

    subs = Subspace.from_normal(np.array([[-1, 0, 1]]).T)

    print(f"Subspace basis from normal vector:\n {subs.basis}")

    poly_subs = deepcopy(poly_dodecahedron_verts)
    ax = poly_original.plot(show=False)
    poly_subs.proj(subs)
    # FIXME: This correction... does not seem right? Because the projection seems to be much smaller than expected, if we actually 'project' onto the subspace. I have no idea why this is happening, though...
    ax = poly_subs.plot(ax=ax)

    poly_lower = poly_original.proj(range(2), in_place=False, keep_dims=False)
    print(f"Lower-dimensional projected vertices:\n {poly_lower.verts} (shape = {poly_lower.verts.shape})")

    poly_lower_alt = poly_original.proj(Subspace(basis=np.array([[10, 0, 0], [0, 10, 0]]).T), in_place=False, keep_dims=False)
    print(f"Lower-dimensional projected vertices (much larger basis):\n {poly_lower_alt.verts} (shape = {poly_lower_alt.verts.shape})")


if __name__ == "__main__":
    main()
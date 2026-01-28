"""Script to test plotting a polytope in 3D"""

from typing import Never

import numpy as np
from numpy.typing import NDArray
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from testRunPyCDDLIBDegenerate import enum_facets, enum_verts
from testMatplotlibColorCycle import set_display_options


class Polytope:
    """Simple Polytope class for testing purposes"""

    def __init__(self, *args: list[Never] | NDArray | tuple[NDArray, NDArray], **kwargs) -> None:
        if len(args) == 0:
            self._verts, self._Ab = kwargs['verts'], np.column_stack((kwargs['A'], kwargs['b']))
            self.is_vrepr, self.is_hrepr = True, True
            self.n = self._verts.shape[0]
        if len(args) == 1:
            self._verts, self._Ab = args[0], None
            self.is_vrepr, self.is_hrepr = True, False
            self.n = self._verts.shape[0]
        elif len(args) == 2:
            self._verts, self._Ab = None, np.column_stack(args)
            self.is_vrepr, self.is_hrepr = False, True
            self.n = self._Ab.shape[1] - 1
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
        # TODO: Also implement the logic when `self` is lower-dimensional, so when it is a single plane, or a line.
        for idx in range(self.m):
            # TODO: Replace this with the facers-vertices incidence matrix when available
            verts_facet = self.verts[:, np.isclose(self.A[idx, :] @ self.verts, self.b[idx])]
            _plot_facet_3d(verts_facet, ax, color, alpha, plot_edges=plot_edges)
            if label_facets:
                label = label_facets[idx] if isinstance(label_facets, list) else fr"${idx}$"
                ax.text(*np.mean(verts_facet, axis=1), label, color='black')
        if label_verts:
            for idx in range(self.k):
                label = label_verts[idx] if isinstance(label_verts, list) else fr"${idx}$"
                ax.text(self.verts[0, idx], self.verts[1, idx], self.verts[2, idx], label, color='black')
        if show:
            plt.show()
        return ax


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
    

def main() -> None:
    verts_cube = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0.5, 0.5, 2]]).T
    
    poly_cube_verts = Polytope(verts_cube)
    ax = poly_cube_verts.plot(show=False, label_verts=True)

    vert_triangle = np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [1.5, 1.5, 2]]).T
    poly_triangle_verts = Polytope(vert_triangle)
    ax = poly_triangle_verts.plot(ax=ax, plot_edges=False, show=False)

    verts_pyramid = np.array([[1, 1, 0], [0, 1, 0], [2, 2, 0], [1, 2, 0], [1.5, 1.5, 1]]).T
    poly_pyramid_verts = Polytope(verts_pyramid)
    poly_pyramid_verts.plot(ax=ax, color='green', alpha=0.3)

    set_display_options(color_cycle='inferno_10R', render_preset='cm')

    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    verts_dodecahedron = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                                   [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
                                   [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
                                   [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]]).T
    poly_dodecahedron_verts = Polytope(verts_dodecahedron)
    poly_dodecahedron_verts.plot(label_facets=True, alpha=0)


if __name__ == "__main__":
    main()
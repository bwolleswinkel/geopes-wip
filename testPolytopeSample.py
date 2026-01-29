"""Script to test sampling from a polytope using different methods"""

from typing import Never, Literal, Callable, Any
from dataclasses import dataclass, field
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
            self._chebcr, self._is_full_dim, self._vol = None, None, None
        if len(args) == 1:
            self._verts, self._Ab = args[0], None
            self.is_vrepr, self.is_hrepr = True, False
            self.n = self._verts.shape[0]
            self._chebcr, self._is_full_dim, self._vol = None, None, None
        elif len(args) == 2:
            self._verts, self._Ab = None, np.column_stack(args)
            self.is_vrepr, self.is_hrepr = False, True
            self.n = self._Ab.shape[1] - 1
            self._chebcr, self._is_full_dim, self._vol = None, None, None
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
    
    @property
    def chebc(self) -> NDArray:
        """Compute the Chebyshev center of the polytope"""
        if self._chebcr is None:
            self._chebcr: NDArray = self._comp_chebcr()
        return self._chebcr[:-1]
    
    @property
    def chebr(self) -> float:
        """Compute the Chebyshev radius of the polytope"""
        if self._chebcr is None:
            self._chebcr: NDArray = self._comp_chebcr()
        return self._chebcr[-1]
    
    def __contains__(self, point: NDArray) -> bool:
        """Check if a point is inside the polytope"""
        if self.is_hrepr:
            # FIXME: We MUST also make this work for an array (n, k) of points! Specifically, the `contains` method should be vectorized (at least, for H-representation)!
            lhs = self.A @ point
            return np.all(np.bitwise_or(lhs <= self.b, np.isclose(lhs, self.b)))
        elif self.is_vrepr:
            # FROM: "Frequently Asked Questions in Polyhedral Computation," Komei Fukuda (2024) | §2.19, Eq. (5)
            c = np.concatenate(([-1], point))
            A_ub = np.column_stack((-np.ones(self.k + 1), np.column_stack((self.verts, point)).T))
            b_ub = np.concatenate((np.zeros(self.k), [1]))
            res = sp.optimize.linprog(-c, A_ub=A_ub, b_ub=b_ub)
            if res.fun <= 0 and not np.isclose(res.fun, 0):
                return False
            else: 
                return True
        else:
            raise ValueError("Polytope must have either H-representation or V-representation")
        
    def bbox(self) -> tuple[NDArray, NDArray]:
        """Compute the axis-aligned bounding box of the polytope"""
        # TODO: Convert this to a Box class later
        vmin, vmax = np.min(self.verts, axis=1), np.max(self.verts, axis=1)
        return vmin, vmax
    
    def center(self, method: Literal['vcentroid', 'com', 'chebc']) -> NDArray:
        """Compute the center of the polytope using the specified method"""
        match method:
            case 'vcentroid':
                return np.mean(self.verts, axis=1)
            case 'com':
                raise NotImplementedError("Center of mass computation not implemented yet")
            case 'chebc':
                return self.chebc
            case _:
                raise ValueError(f"Unrecognized center computation method '{method}'")
    
    def sample(self, method: Literal['rejection', 'hit_and_run'] = 'rejection', num_samples: int = 1, burn_in: int = 0, thinning: int = 1, density: Callable[[NDArray], float] | None = None, ball_walk_params: dict[str, Any] | None = None) -> NDArray:
        """Sample points uniformly from the polytope (placeholder implementation)"""
        # NOTE: The seed for the random number generator can be set globally using np.random.seed(...)
        if density is not None and method != 'rejection':
            raise NotImplementedError("Non-uniform sampling only implemented for rejection sampling method")
        match method:
            case 'rejection':
                samples = np.zeros((self.n, num_samples))
                vmin, vmax = self.bbox()
                for idx in range(num_samples):
                    sample_found = False
                    while not sample_found:
                        # TODO: We 'could' make this even faster, by sampling `size=(self.n, batch_size=num_samples)` and checking which points are inside the polytope; then, we create another batch for the remaining samples, and so on... but I don't know how important that is right now, and how slow this Python for-loop will be in practice.
                        point = np.random.uniform(low=vmin, high=vmax, size=(self.n,))
                        if density is None:  # Assumes a uniform model
                            if point in self:
                                samples[:, idx], sample_found = point, True
                        else:
                            # NOTE: Here, we assume that `density` is max-normalized, i.e., max_{x in polytope} density(x) <= 1! This is super important to communicate to the user. The closer max_density(x) is to 1, the more efficient the sampling will be.
                            if point in self and np.random.uniform(0, 1) <= density(point):
                                samples[:, idx], sample_found = point, True
            case 'hit_and_run':
                # NOTE: This method can (theoretically) be much more efficient for polytopes with small volume relative to their bounding box volume, and a large number of samples is requested.
                samples = np.zeros((self.n, burn_in + num_samples * thinning + 1))
                # FROM: GitHub Copilot GPT-4.1 | 2026/01/28 [untested/unverified]
                # Start from the Chebyshev center
                samples[:, 0] = self.chebc
                for idx in range(1, samples.shape[1]):
                    current_point = samples[:, idx - 1]
                    # Sample a random direction
                    direction = np.random.normal(size=(self.n,))
                    direction /= np.linalg.norm(direction)
                    # Find the maximum step sizes in both directions
                    alphas_pos, alphas_neg = [], []
                    for i in range(self.m):
                        Ai = self.A[i, :]
                        bi = self.b[i]
                        denom = Ai @ direction
                        if np.isclose(denom, 0):
                            continue
                        alpha = (bi - Ai @ current_point) / denom
                        if denom > 0:
                            alphas_pos.append(alpha)
                        else:
                            alphas_neg.append(alpha)
                    alpha_max_pos = min(alphas_pos) if alphas_pos else np.inf
                    alpha_max_neg = max(alphas_neg) if alphas_neg else -np.inf
                    # Sample a step size uniformly within the feasible range
                    alpha_sample = np.random.uniform(low=alpha_max_neg, high=alpha_max_pos)
                    samples[:, idx] = current_point + alpha_sample * direction
                # Discard the initial point and burn-in samples, and thin the chain
                samples = samples[:, (burn_in + 1)::thinning]
            case 'ball_walk':
                # FROM: "Geometric Random Walks: A Survey," Santosh Vempala (2005)
                if ball_walk_params is None:
                    ball_walk_params = {'radius': 0.1}
                try:
                    radius = ball_walk_params['radius']
                except KeyError as e:
                    raise ValueError(f"Missing ball walk parameter '{e}', which was not provided in 'ball_walk_params'")
                samples = np.zeros((self.n, burn_in + num_samples * thinning + 1))
                # FIXME: Do we want to use the Chebc center here? Maybe, if the vertices are given, we use the centroid of the vertices? Whichever computation is cheaper...
                samples[:, 0], sphere = self.center('vcentroid' if self.is_vrepr else 'chebc'), Sphere3D(r=1)
                for idx in range(1, samples.shape[1]):
                    sample_found = False
                    while not sample_found:
                        current_point = samples[:, idx - 1]
                        step_size = np.random.uniform(0, radius)
                        direction = step_size * sphere.sample()
                        candidate_point = current_point + direction
                        if candidate_point in self:
                            samples[:, idx] = candidate_point
                            sample_found = True
                samples = samples[:, (burn_in + 1)::thinning]
            case _:
                raise ValueError(f"Unrecognized sampling method '{method}'")
        return samples.squeeze()

    def _comp_chebcr(self) -> NDArray:
        """Compute the Chebyshev center and radius of the polytope"""
        A_ext = np.hstack((self.A, np.linalg.norm(self.A, axis=1, keepdims=True)))
        c = np.zeros(self.n + 1)
        c[-1] = -1.0
        res = sp.optimize.linprog(c, A_ub=A_ext, b_ub=self.b, bounds=([(None, None)] * self.n + [(0, None)]), method='highs')
        if not res.success:
            raise ValueError("Linear program to compute Chebyshev center/radius failed")
        return res.x
    

@dataclass
class Sphere3D:
    """Class representing a 3sphere in #-dimensional space"""
    r: float
    n: int = field(default=3, init=False)

    def sample(self) -> NDArray:
        """Sample points uniformly from the surface of the sphere"""
        samples = np.random.normal(size=self.n)
        samples /= np.linalg.norm(samples, axis=0, keepdims=True)
        samples *= self.r
        return samples


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


def main():
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    verts_dodecahedron = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
                                   [0, 1/phi, phi], [0, 1/phi, -phi], [0, -1/phi, phi], [0, -1/phi, -phi],
                                   [1/phi, phi, 0], [1/phi, -phi, 0], [-1/phi, phi, 0], [-1/phi, -phi, 0],
                                   [phi, 0, 1/phi], [phi, 0, -1/phi], [-phi, 0, 1/phi], [-phi, 0, -1/phi]]).T
    verts_dodecahedron += (2 * np.ones(verts_dodecahedron.shape))
    poly_dodecahedron_verts = Polytope(verts_dodecahedron)
    poly_dodecahedron_verts_chebr = poly_dodecahedron_verts.chebr
    poly_dodecahedron_verts_chebc = poly_dodecahedron_verts.chebc
    print(f"Dodecahedron Chebyshev radius (should be > 0): {poly_dodecahedron_verts_chebr} (center = {poly_dodecahedron_verts_chebc})")
    poly_dodecahedron_verts_bbox = poly_dodecahedron_verts.bbox()
    print(f"Dodecahedron bounding box: min {poly_dodecahedron_verts_bbox[0]}, max {poly_dodecahedron_verts_bbox[1]}")
    test_point_inside = np.array([2.0, 2.0, 2.0])
    print(f"Point {test_point_inside} inside dodecahedron: {test_point_inside in poly_dodecahedron_verts}")
    single_sample = poly_dodecahedron_verts.sample()
    print(f"Single sample from dodecahedron: {single_sample} (inside: {single_sample in poly_dodecahedron_verts})")
    multiple_samples = poly_dodecahedron_verts.sample(num_samples=10_000)
    print(f"Multiple samples from dodecahedron:\n{multiple_samples} (all inside: {all(multiple_samples[:, i] in poly_dodecahedron_verts for i in range(multiple_samples.shape[1]))})")
    multiple_samples_hit_and_run = poly_dodecahedron_verts.sample(method='hit_and_run', num_samples=10_000, burn_in=1000, thinning=10)
    print(f"Multiple samples (hit-and-run) from dodecahedron:\n{multiple_samples_hit_and_run} (all inside: {all(multiple_samples_hit_and_run[:, i] in poly_dodecahedron_verts for i in range(multiple_samples_hit_and_run.shape[1]))})")
    multiple_samples_ball_walk = poly_dodecahedron_verts.sample(method='ball_walk', num_samples=10_000, burn_in=1000, thinning=500, ball_walk_params={'radius': 0.02})
    print(f"Multiple samples (ball-walk) from dodecahedron:\n{multiple_samples_ball_walk} (all inside: {all(multiple_samples_ball_walk[:, i] in poly_dodecahedron_verts for i in range(multiple_samples_ball_walk.shape[1]))})")

    # Test some stats on the samples
    np.set_printoptions(precision=2, suppress=True)
    mean_sample = np.mean(multiple_samples, axis=1)
    cov_sample = np.cov(multiple_samples)
    print(f"Mean of samples: {mean_sample}")
    print(f"Covariance of samples:\n{cov_sample}")
    mean_sample_har = np.mean(multiple_samples_hit_and_run, axis=1)
    cov_sample_har = np.cov(multiple_samples_hit_and_run)
    print(f"Mean of hit-and-run samples: {mean_sample_har}")
    print(f"Covariance of hit-and-run samples:\n{cov_sample_har}")
    mean_sample_bw = np.mean(multiple_samples_ball_walk, axis=1)
    cov_sample_bw = np.cov(multiple_samples_ball_walk)
    print(f"Mean of ball-walk samples: {mean_sample_bw}")
    print(f"Covariance of ball-walk samples:\n{cov_sample_bw}")


if __name__ == "__main__":
    main()
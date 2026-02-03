"""Script to test sampling from an ellipsoid, and creating ellipsoids from covariance matrices"""

from typing import Literal, Callable
import warnings

import numpy as np
from numpy.typing import NDArray
import scipy as sp
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.axes import Axes


class Ellipsoid:
    """Simple Ellipsoid class for testing purposes"""

    def __init__(self, Q: NDArray, c: NDArray | None = None) -> None:
        self.Q: NDArray = Q
        V, R = np.linalg.eigh(Q, UPLO='U')
        self.R = R
        self.radii = 1 / np.sqrt(V)
        self.n = Q.shape[0]
        self.cov: NDArray | None = None
        self.alpha: float | None = None
        self.c = c if c is not None else np.zeros(self.n)
    
    @classmethod
    def from_cov(cls, cov: NDArray, lim: float, mean: NDArray | None = None, lim_type: Literal['chi_square_quantile', 'mahalanobis_dist', 'confidence_level', 'multiple_std'] = 'chi_square_quantile') -> Ellipsoid:
        """Create an ellipsoid from a covariance matrix and specified bound"""
        if not is_sym(cov) or not is_pos_def(cov):
            raise ValueError("Covariance matrix must be symmetric positive definite")
        match lim_type:
            # FROM: Perplexity Pro | 2026/02/02[untested/unverified]
            # NOTE: This is the squared Mahalanobis distance
            case 'chi_square_quantile' | 'mahalanobis_dist':
                if lim < 0:
                    raise ValueError(f"{f'Chi-square bound' if lim_type == 'chi_square_quantile' else 'Mahalanobis distance'} must be non-negative")
                if lim_type == 'mahalanobis_dist':
                    lim = lim ** 2
                Q = (1 / lim) * np.linalg.inv(cov)
                ellps = cls(Q, c=mean)
                ellps.cov, ellps.alpha = cov, lim
                return ellps
            case 'confidence_level':
                if not (0 < lim < 1):
                    raise ValueError("Confidence level must be in (0, 1)")
                alpha = sp.stats.chi2.ppf(lim, df=cov.shape[0])
                Q = (1 / alpha) * np.linalg.inv(cov)
                ellps = cls(Q, c=mean)
                ellps.cov, ellps.alpha = cov, sp.stats.chi2.ppf(lim, df=cov.shape[0])
                return ellps
            case 'multiple_std':
                if cov.shape[0] > 1:
                    warnings.warn("Multiple of standard deviations for multivariate ellipsoids is does not correspond to well-known confidence levels")
                if lim < 0:
                    raise ValueError("Multiple of standard deviations must be non-negative")
                Q = (1 / (lim ** 2)) * np.linalg.inv(cov)
                ellps = cls(Q, c=mean)
                ellps.cov, ellps.alpha = cov, sp.stats.chi2.cdf(lim, df=cov.shape[0])
                return ellps
            case _:
                raise ValueError(f"Unrecognized bound_type '{lim_type}'")

    def __contains__(self, x: NDArray) -> bool:
        """Check if a point `x` is inside the ellipsoid"""
        val = (x - self.c).T @ self.Q @ (x - self.c)
        return val <= 1.0 or np.isclose(val, 1.0)

    def sample(self, method: Literal['rejection', 'minimax_tilting', 'whitening'] = 'whitening', num_samples: int = 1, density: Literal['uniform', 'normal'] | Callable[[NDArray], float] = 'uniform') -> NDArray:
        """Sample from the ellipsoid using specified method"""
        match method:
            case 'rejection':
                samples = np.zeros((self.n, num_samples))
                for idx in range(num_samples):
                    sample_found = False
                    while not sample_found:
                        if density == 'normal':
                            if self.cov is None or self.alpha is None:
                                raise ValueError("Covariance matrix and alpha are required for normal density")
                            point = np.random.multivariate_normal(mean=self.c, cov=self.cov)
                        elif density == 'uniform':
                            point = self.c + np.random.uniform(low=-self.radii, high=self.radii) @ self.R.T
                        elif callable(density):
                            raise NotImplementedError("Custom density functions are not implemented yet")
                        else:
                            raise ValueError(f"Unrecognized density type '{density}'")
                        if point in self:
                            samples[:, idx], sample_found = point, True
            case 'minimax_tilting':
                # FIXME: I don't think this algorithm is actually what I want; I think it only allows me to sample with upper and lower bounds on each variable, not an ellipsoidal bound.
                raise NotImplementedError("Minimax tilting method is not implemented for ellipsoids yet")
            case 'whitening':
                # FROM: GitHub Copilot, Claude Haiku 4.5 | 2026/02/03
                if density == 'normal':
                    if self.cov is None or self.alpha is None:
                        raise ValueError("Covariance matrix and alpha are required for normal density")
                    p = sp.stats.chi2.cdf(self.alpha, df=self.n)
                    u = np.random.uniform(0, p, size=num_samples)
                    r = np.sqrt(sp.stats.chi2.ppf(u, df=self.n)) / np.sqrt(self.alpha)
                elif density == 'uniform':
                    r = np.random.uniform(0, 1, size=num_samples) ** (1 / self.n)
                elif callable(density):
                    raise NotImplementedError("Custom density functions are not implemented yet")
                else:
                    raise ValueError(f"Unrecognized density type '{density}'")
                samples = np.random.normal(size=(self.n, num_samples))
                samples = samples / np.linalg.norm(samples, axis=0, keepdims=True)
                samples = self.c[:, np.newaxis] + self.R @ (self.radii[:, np.newaxis] * samples * r[np.newaxis, :])
            case _:
                raise ValueError(f"Unrecognized sampling method '{method}'")
        return samples.squeeze()
    
    def to_cov(self, lim_type: Literal['chi_square_quantile', 'confidence_level', 'multiple_std'] = 'chi_square_quantile') -> tuple[NDArray, float]:
        """Convert the ellipsoid to a covariance matrix representation"""
        cov = np.linalg.inv(self.Q)
        match lim_type:
            case 'chi_square_quantile':
                return cov, self.alpha
            case 'confidence_level':
                return cov, sp.stats.chi2.cdf(self.alpha, df=self.n)
            case 'multiple_std':
                return cov, np.sqrt(sp.stats.chi2.ppf(self.alpha, df=self.n))
            case _:
                raise ValueError(f"Unrecognized bound_type '{lim_type}'")
            
    def plot(self, color: str | None = None, alpha: float = 0.5, npoints: int = 20, 
             plot_border: bool = True, show: bool = True, ax: Axes | None = None) -> Axes:
        """Plot an ellipsoid in ND, currently only supports 2D and 3D ellipsoids"""
        if self.n not in (1, 2, 3):
            raise NotImplementedError("Plotting is only implemented for 2D and 3D ellipsoids")
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'projection': '3d'} if self.n == 3 else {})
        else:
            if (self.n != 3 and ax.name == '3d') or (self.n == 3 and ax.name != '3d'):
                raise ValueError(f"The dimension of the polytope ({self.n}) does not match the dimension of the provided axes 'ax' ({3 if ax.name == '3d' else 2})")
            _, ax = None, ax
        if color is None:
            color = ax._get_lines.get_next_color()
        match self.n:
            case 2:
                ellps = Ellipse(xy=self.c, width=2 * self.radii[0], height=2 * self.radii[1], angle=np.degrees(np.atan2(self.R[1, 0], self.R[0, 0])), edgecolor=color if plot_border else 'none', facecolor=color, alpha=alpha)
                # FIXME: This is kind of a hack to plot the border separately when alpha=0. Maybe we just want 'color', '(face)alpha', but then also 'facecolor' and 'edgecolor'?
                if alpha == 0:
                    ellps.set_alpha(1)
                    ellps.set_edgecolor(color)
                    ellps.set_facecolor('none')
                ax.add_patch(ellps)
                # FIXME: Apparently, this is needed to autoscale the view to fit the ellipse, but it seems hacky
                ax.autoscale_view(tight=False)
                ax.margins(0.1, 0.1)
            case 3:
                u, v = np.linspace(0, 2 * np.pi, npoints), np.linspace(0, np.pi, npoints)
                x, y, z = self.radii[0] * np.outer(np.cos(u), np.sin(v)), self.radii[1] * np.outer(np.sin(u), np.sin(v)), self.radii[2] * np.outer(np.ones_like(u), np.cos(v))
                coords = np.stack([x, y, z], axis=-1) @ self.R.T + self.c
                x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
                # FIXME: Which plot do we actually want to use here? surface is best, but very laggy for large npoints. wireframe is faster but less pretty.
                ax.plot_surface(x, y, z, color=color, alpha=alpha)
        if show:
            plt.show()
        return ax


def is_sym(A: NDArray) -> bool:
    """Check if a matrix `A` is symmetric, i.e., A = A^T."""
    return np.allclose(A, A.T)


def is_pos_def(A: NDArray, allow_semidef: bool = False) -> bool:
    """Check if a matrix `A` is positive definite, i.e., x^T A x > 0 for all x ≠ 0. Note that A is required to be symmetric, i.e., A = A^T.
    
    """
    # FROM: https://stackoverflow.com/questions/5033906/in-python-should-i-use-else-after-a-return-in-an-if-block | On how to structure if-statements with returns
    if allow_semidef:
        if not is_sym(A):
            return False
        else:
            eigvals = np.linalg.eigvalsh(A)
            return np.all(eigvals >= 0)
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False
    

def ellps_from_cov(cov: NDArray, lim: float, mean: NDArray | None = None, lim_type: Literal['chi_square_quantile', 'confidence_level', 'multiple_std'] = 'chi_square_quantile') -> Ellipsoid:
    return Ellipsoid.from_cov(cov, lim, mean, lim_type)


def main() -> None:
    """Main function to test ellipsoid sampling"""
    cov = np.array([[4.0, 1.0], 
                    [1.0, 2.0]])
    mean = np.array([1.0, 2.0])
    ellps = ellps_from_cov(cov, lim=0.99, mean=mean, lim_type='confidence_level')
    ax = ellps.plot(show=False, alpha=0.5)
    ax.set_aspect('equal')

    cov_3d = np.array([[4.0, 1.0, 0.5],
                       [1.0, 2.0, 0.3],
                       [0.5, 0.3, 3.0]])
    mean_3d = np.array([1.0, 2.0, 3.0])
    ellps_3d = ellps_from_cov(cov_3d, lim=0.999, mean=mean_3d, lim_type='confidence_level')
    ax_3d = ellps_3d.plot(show=False, alpha=1)
    ax_3d.set_box_aspect([1, 1, 1])

    cov_wiki = np.array([[1, 3 / 5],
                         [3 / 5, 2]])
    # ellps_wiki = ellps_from_cov(cov_wiki, lim=3, lim_type='multiple_std')
    ellps_wiki = ellps_from_cov(cov_wiki, lim=0.95, lim_type='confidence_level')
    ax = ellps_wiki.plot(show=False, alpha=0.2, plot_border=False, ax=ax, color='green')
    ax = ellps_wiki.plot(show=False, alpha=0, ax=ax, color='red')
    ax.set_aspect('equal')

    samples_white = ellps.sample(method='whitening', density='uniform', num_samples=10_000)
    samples_rej = ellps.sample(method='rejection', density='uniform', num_samples=10_000)

    _, ax_white = plt.subplots()
    ax_white = ellps.plot(show=False, alpha=0.5, ax=ax_white)
    ax_white.set_aspect('equal')
    ax_white.scatter(samples_white[0, :], samples_white[1, :], color='red', s=1, alpha=0.7)
    
    _, ax_rej = plt.subplots()
    
    ax_rej = ellps.plot(show=False, alpha=0.5, ax=ax_rej)
    ax_rej.set_aspect('equal')
    ax_rej.scatter(samples_rej[0, :], samples_rej[1, :], color='red', s=1, alpha=0.7)
    
    plt.show()

    samples_3d_white = ellps_3d.sample(method='rejection', density='normal', num_samples=5_000)
    _, ax_3d_white = plt.subplots(subplot_kw={'projection': '3d'})
    ax_3d_white = ellps_3d.plot(show=False, alpha=0.5, ax=ax_3d_white)
    ax_3d_white.set_box_aspect([1, 1, 1])
    ax_3d_white.scatter(samples_3d_white[0, :], samples_3d_white[1, :], samples_3d_white[2, :], color='red', s=1, alpha=0.7)

    plt.show()


if __name__ == "__main__":
    main()
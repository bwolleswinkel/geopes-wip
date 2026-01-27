"""Script to test setting the color cycle in Matplotlib"""

from __future__ import annotations
from typing import Literal

CLR_MATPLOTLIB_LEGACY = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

# FROM: https://github.com/d3/d3-3.x-api-reference/blob/master/Ordinal-Scales.md#category10
# TODO: Also implement 'd3.scale.category20c()' and '# d3.scale.category20b()' if needed
CLR_CATEGORY_10 = ["#1f77b4",  # 0: Blue
                   "#ff7f0e",  # 1: Orange
                   "#2ca02c",  # 2: Green
                   "#d62728",  # 3: Red
                   "#9467bd",  # 4: Purple
                   "#8c564b",  # 5: Brown
                   "#e377c2",  # 6: Pink
                   "#7f7f7f",  # 7: Gray
                   "#bcbd22",  # 8: Olive
                   "#17becf"]  # 9: Cyan

CLR_CATEGORY_20 = ["#1f77b4",  # 0: Blue
                   "#aec7e8",  # 1: Light Blue
                   "#ff7f0e",  # 2: Orange
                   "#ffbb78",  # 3: Light Orange
                   "#2ca02c",  # 4: Green
                   "#98df8a",  # 5: Light Green
                   "#d62728",  # 6: Red
                   "#ff9896",  # 7: Light Red
                   "#9467bd",  # 8: Purple
                   "#c5b0d5",  # 9: Light Purple
                   "#8c564b",  # 10: Brown
                   "#c49c94",  # 11: Light Brown
                   "#e377c2",  # 12: Pink
                   "#f7b6d2",  # 13: Light Pink
                   "#7f7f7f",  # 14: Gray
                   "#c7c7c7",  # 15: Light Gray
                   "#bcbd22",  # 16: Olive
                   "#dbdb8d",  # 17: Light Olive
                   "#17becf",  # 18: Cyan
                   "#9edae5"]  # 19: Light Cyan

CLR_TABLEAU_10 = ["#4E79A7",  # 0: Blue
                  "#F28E2B",  # 1: Orange
                  "#E15759",  # 2: Red
                  "#76B7B2",  # 3: Teal
                  "#59A14F",  # 4: Green
                  "#EDC948",  # 5: Yellow
                  "#B07AA1",  # 6: Purple
                  "#FF9DA7",  # 7: Pink
                  "#9C755F",  # 8: Brown
                  "#BAB0AC"]  # 9: Gray

# FIXME: Seems to be in a slightly different order than the official Tableau 20 palette:
# FROM: https://jrnold.github.io/ggthemes/reference/tableau_color_pal.html
CLR_TABLEAU_20 = ["#4E79A7",  # 0: Blue
                  "#A0CBE8",  # 1: Light Blue
                  "#F28E2B",  # 2: Orange
                  "#FFBE7D",  # 3: Light Orange
                  "#E15759",  # 4: Red
                  "#FF9D9A",  # 5: Light Red
                  "#76B7B2",  # 6: Teal
                  "#B2DFDB",  # 7: Light Teal
                  "#59A14F",  # 8: Green
                  "#8CD17D",  # 9: Light Green
                  "#EDC948",  # 10: Yellow
                  "#FFF2A1",  # 11: Light Yellow
                  "#B07AA1",  # 12: Purple
                  "#D4A6D9",  # 13: Light Purple
                  "#FF9DA7",  # 14: Pink
                  "#FFD3D6",  # 15: Light Pink
                  "#9C755F",  # 16: Brown
                  "#D7B5A6",  # 17: Light Brown
                  "#BAB0AC",  # 18: Gray
                  "#E3DFDD"]  # 19: Light Gray


CLR_INFERNO_10 = [
    '#000004', 
    '#1b0c41',
    '#4a0c6b',
    '#781c6d',
    '#a52c60',
    '#cf4446',
    '#ed6925',
    '#fb9b06',
    '#f7d13d',
    '#fcffa4'
]


class DisplayOptionsSetter:
    """Global setter for display options.
    
    Call signature: (*, color_cycle=None) -> None
    
    Parameters
    ----------
    color_cycle : {'matplotlib', 'matplotlib_legacy', 'tableau_10'} or None, optional
        Color palette to use for matplotlib plots. Options:
        - 'matplotlib': Default matplotlib color cycle (10 colors)
        - 'matplotlib_legacy': Legacy matplotlib single-letter colors (7 colors)
        - 'tableau_10': Tableau 10 color palette (10 colors)
        If None, no changes are made to the color cycle.
        
    Examples
    --------
    >>> set_display_options(color_cycle='tableau_10')
    >>> set_display_options(color_cycle='matplotlib_legacy')
    
    """
    
    def __call__(self, *, color_cycle: Literal['matplotlib', 'matplotlib_legacy', 'tableau_10', 'tableau_20', 'category_10', 'category_20', 'inferno_10'] | None = None, render_preset: Literal['cm'] | None = None) -> None:
        """Set global display options permanently"""
        if color_cycle is not None:
            color_dict = {
                'matplotlib': CLR_CATEGORY_10,
                'matplotlib_legacy': CLR_MATPLOTLIB_LEGACY,
                'tableau_10': CLR_TABLEAU_10,
                'tableau_20': CLR_TABLEAU_20,
                'category_10': CLR_CATEGORY_10,
                'category_20': CLR_CATEGORY_20,
                'inferno_10': CLR_INFERNO_10
            }
            try:
                import matplotlib as mpl
                from cycler import cycler
                if color_cycle[-1].lower() == 'r':
                    color_dict[color_cycle[:-1]] = list(reversed(color_dict[color_cycle[:-1]]))
                    color_cycle = color_cycle[:-1]
                mpl.rcParams['axes.prop_cycle'] = cycler(color=color_dict[color_cycle])
            except ImportError:
                raise ImportWarning("Package 'matplotlib' is not installed. Please install it to use this color palette.")
        if render_preset is not None:
            match render_preset:
                case 'cm':
                    try:
                        import matplotlib as mpl
                    except ImportError:
                        raise ImportWarning("Package 'matplotlib' is not installed. Please install it to use this render preset.")
                    mpl.rcParams['font.family'] = 'serif'
                    mpl.rcParams['mathtext.fontset'] = 'cm'
                case _:
                    raise ValueError(f"Unknown render preset '{render_preset}'")
            

_display_options_setter = DisplayOptionsSetter()

# FIXME: Unfortunately, to to Pylance static checking, the docstring must be duplicated here.
def set_display_options(*, color_cycle: Literal['matplotlib', 'matplotlib_legacy', 'tableau_10', 'tableau_20', 'category_10', 'category_20'] | None = None, render_preset: Literal['cm'] | None = None) -> None:
    """Set global display options permanently.

    Parameters
    ----------
    color_cycle : {'matplotlib', 'matplotlib_legacy', 'tableau_10', 'tableau_20', 'category_10', 'category_20'} or None, optional
        Color palette to use for matplotlib plots. Options:
        - 'matplotlib': D3 Category 10 color palette (10 colors) 
        - 'matplotlib_legacy': Legacy matplotlib single-letter colors (7 colors)
        - 'tableau_10': Tableau 10 color palette (10 colors)
        - 'tableau_20': Tableau 20 color palette (20 colors)
        - 'category_10': D3 Category 10 color palette (10 colors)
        - 'category_20': D3 Category 20 color palette (20 colors)
        If None, no changes are made to the color cycle.
        
    Raises
    ------
    ImportWarning
        If matplotlib is not installed when trying to set a color cycle.

    Examples
    --------
    >>> set_display_options(color_cycle='tableau_10')
    >>> set_display_options(color_cycle='category_20')
    
    """
    return _display_options_setter(color_cycle=color_cycle, render_preset=render_preset)


def main() -> None:
    import matplotlib.pyplot as plt

    plt.figure()
    x = range(10)
    for i in range(10):
        plt.plot(x, [xi * (i + 1) for xi in x], label=f'Line {i+1}')
    plt.legend()
    plt.title("Test Plot with Current Color Cycle")

    plt.figure()
    set_display_options(color_cycle='tableau_10')
    for i in range(10):
        plt.plot(x, [xi * (i + 1) for xi in x], label=f'Line {i+1}')
    plt.legend()
    plt.title("Test Plot with Tableau 10 Color Cycle")

    plt.figure()
    set_display_options(color_cycle='inferno_10R', render_preset='cm')
    for i in range(10):
        plt.plot(x, [xi * (i + 1) for xi in x], label=f'Line {i+1}')
    plt.legend()
    plt.title("Test Plot with Inferno Color Cycle")
    
    plt.show()


if __name__ == "__main__":
    main()
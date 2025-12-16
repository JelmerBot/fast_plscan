import colorsys
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from textwrap import dedent


# LaTeX font sizes on 10pt document:
# https://latex-tutorial.com/changing-font-size/
# for the pre-print template!
# fontsize = dict(tiny=5, script=7, footnote=8, small=9, normal=10)
# for the journal template!
fontsize = dict(tiny=6, script=8, footnote=9, small=10, normal=10.95)


def configure_matplotlib():
    sns.set_style("white")
    sns.set_color_codes()

    mpl.rcParams.update(
        {
            "text.color": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.labelcolor": "black",
            "xtick.bottom": True,
            "ytick.left": True,
            "axes.titlesize": fontsize["normal"],
            "axes.labelsize": fontsize["small"],
            "xtick.labelsize": fontsize["small"],
            "ytick.labelsize": fontsize["small"],
            "font.size": fontsize["footnote"],
            "legend.title_fontsize": fontsize["footnote"],
            "legend.fontsize": fontsize["footnote"],
            "axes.unicode_minus": True,
            "axes.spines.left": False,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.spines.bottom": False,
            "savefig.dpi": 300,
            "savefig.format": "png",
            "font.family": "serif",
            "text.usetex": True,
            # For the pre-print template:
            # "text.latex.preamble": dedent(
            #     r"""
            #     \usepackage[english]{babel}
            #     \usepackage[T1]{fontenc}
            #     \usepackage[varqu,varl]{inconsolata}
            #     \usepackage[
            #         theoremfont,trueslanted,largesc,p,
            #         amsthm,smallerops
            #     ]{newpx}
            #     \usepackage[scr=rsfso]{mathalpha}
            #     \usepackage[stretch=10,shrink=10,tracking,spacing,kerning,babel]{microtype}
            #     """
            # ),
            # For the journal template:
            "text.latex.preamble": dedent(
                r"""
                \usepackage[english]{babel}
                \usepackage[stretch=10,shrink=10,tracking,spacing,kerning,babel]{microtype}
                """
            ),
        }
    )

    return sns.color_palette("tab10", 10)


def sized_fig(width=0.5, aspect=0.618, dpi=None):
    """Create a figure with width as fraction of A4 page."""
    if dpi is None:
        dpi = 150
    # page_width_inch = 6.93050  # For the pre-print template
    page_width_inch = 6.00117  # For the journal template
    w = width * page_width_inch
    h = aspect * w
    return plt.figure(figsize=(w, h), dpi=dpi)


def frame_off():
    """Disables frames and ticks."""
    plt.xticks([])
    plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)


def lighten(color, amount=0.5):
    c = colorsys.rgb_to_hls(*mpl.colors.to_rgb(color))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])

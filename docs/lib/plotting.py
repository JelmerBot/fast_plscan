import matplotlib as mpl
import matplotlib.pyplot as plt


def configure_matplotlib():
    mpl.rcParams["text.color"] = "black"
    mpl.rcParams["xtick.color"] = "black"
    mpl.rcParams["ytick.color"] = "black"
    mpl.rcParams["axes.labelcolor"] = "black"
    mpl.rcParams["xtick.bottom"] = True
    mpl.rcParams["ytick.left"] = True
    mpl.rcParams["ytick.major.size"] = mpl.rcParams["xtick.major.size"]
    mpl.rcParams["ytick.major.width"] = mpl.rcParams["xtick.major.width"]
    mpl.rcParams["font.size"] = 8
    mpl.rcParams["axes.labelsize"] = 8
    mpl.rcParams["axes.titlesize"] = 10
    mpl.rcParams["legend.fontsize"] = 8
    mpl.rcParams["legend.title_fontsize"] = 8
    mpl.rcParams["xtick.labelsize"] = 8
    mpl.rcParams["ytick.labelsize"] = 8
    mpl.rcParams["axes.unicode_minus"] = True
    mpl.rcParams["axes.spines.left"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.bottom"] = False
    mpl.rcParams["savefig.dpi"] = 300
    mpl.rcParams["savefig.format"] = "png"


def sized_fig(width=0.5, aspect=0.618, dpi=None):
    """Create a figure with width as fraction of A4 page."""
    if dpi is None:
        dpi = 150
    page_width_cm = 13.9
    inch = 2.54
    w = width * page_width_cm
    h = aspect * w
    return plt.figure(figsize=(w / inch, h / inch), dpi=dpi)


def frame_off():
    """Disables frames and ticks."""
    plt.xticks([])
    plt.yticks([])
    for spine in plt.gca().spines.values():
        spine.set_visible(False)

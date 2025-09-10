import numpy as np
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess


def regplot_lowess_ci(
    data,
    x,
    y,
    *,
    ci_level,
    n_boot,
    lowess_frac=0.1,
    color="C0",
    scatter=True,
    line_kws=None,
    area_kws=None,
    scatter_kws=None,
):
    x_ = data[x].to_numpy()
    y_ = data[y].to_numpy()
    x_grid = np.linspace(start=x_.min(), stop=x_.max(), num=1000)

    def reg_func(_x, _y):
        return lowess(
            exog=_x, endog=_y, xvals=x_grid, is_sorted=False, frac=lowess_frac
        )

    beta_boots = sns.algorithms.bootstrap(
        x_,
        y_,
        func=reg_func,
        n_boot=n_boot,
    )
    err_bands = sns.utils.ci(beta_boots, ci_level, axis=0)
    y_plt = reg_func(x_, y_)

    ax = sns.lineplot(x=x_grid, y=y_plt, color=color, **(line_kws or {}))
    ax.fill_between(x_grid, *err_bands, alpha=0.15, color=color, **(area_kws or {}))
    if scatter:
        sns.scatterplot(x=x_, y=y_, ax=ax, color=color, **(scatter_kws or {}))
    return ax

"""Plot shared and separated margins for each task and model pair."""

from collections import defaultdict
from math import ceil
from textwrap import fill

import numpy as np


def write_plot(pairs, path):
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import Normalize
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    from matplotlib import colormaps, rc_context

    groups = defaultdict(list)
    for row in pairs:
        groups[(row["task"], row["model_a"], row["model_b"])].append(row)
    keys = sorted(groups)
    depths = [row["depth"] for row in pairs]
    if not keys:
        raise ValueError("A sensitivity plot needs at least one model pair")
    norm = Normalize(min(depths), max(depths))
    cmap = colormaps["viridis"]
    with rc_context({"pdf.fonttype": 42}), PdfPages(
        path, metadata={"CreationDate": None, "ModDate": None}
    ) as pdf:
        for start in range(0, len(keys), 6):
            page_keys = keys[start:start + 6]
            columns = min(3, len(page_keys))
            rows = ceil(len(page_keys) / columns)
            figure = Figure(figsize=(3.5 * columns, 3.35 * rows), layout="constrained")
            axes = figure.subplots(rows, columns, squeeze=False)
            visible = []
            for axis, key in zip(axes.flat, page_keys):
                records = groups[key]
                x = np.asarray([row["d_S"] for row in records])
                y = np.asarray([row["d_D"] for row in records])
                low = min(0., float(x.min()), float(y.min()))
                high = max(0., float(x.max()), float(y.max()))
                padding = .08 * (high - low) if high > low else 1.
                limits = (low - padding, high + padding)
                axis.plot(limits, limits, "--", color="0.6", linewidth=.8, zorder=1)
                axis.axhline(0, color="0.75", linewidth=.6, zorder=1)
                axis.axvline(0, color="0.75", linewidth=.6, zorder=1)
                colors = [cmap(norm(row["depth"])) for row in records]
                points = axis.scatter(x, y, s=20, color=colors, alpha=.8,
                                      edgecolors="none", zorder=2)
                axis.set(xlim=limits, ylim=limits, aspect="equal",
                         xlabel="Shared margin\npositive favours first model",
                         ylabel="Separated margin\npositive favours first model")
                axis.set_title(fill(f"{key[1]} vs {key[2]}", 36) + "\n" + fill(key[0], 36),
                               fontsize=10)
                axis.tick_params(labelsize=8)
                axis.xaxis.set_major_locator(MaxNLocator(nbins=4))
                axis.yaxis.set_major_locator(MaxNLocator(nbins=4))
                axis.spines[["top", "right"]].set_visible(False)
                visible.append(axis)
            for axis in list(axes.flat)[len(page_keys):]:
                axis.set_visible(False)
            if min(depths) != max(depths):
                points.set_cmap(cmap)
                points.set_norm(norm)
                colorbar = figure.colorbar(points, ax=visible, shrink=.7, pad=.03)
                colorbar.set_label("Cells per stratum per block", fontsize=9)
                colorbar.ax.tick_params(labelsize=8)
            pdf.savefig(figure)

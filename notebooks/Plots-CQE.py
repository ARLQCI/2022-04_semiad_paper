# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: SemiAD Paper
#     language: python
#     name: semi-ad-paper
# ---

# # Plots

# This benchmark plot is for the CQE Workshop slides.

from pathlib import Path
from operator import itemgetter
import rsmf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from math import floor, ceil

from matplotlib.pylab import get_cmap, setp
from matplotlib import transforms
from matplotlib.ticker import FuncFormatter

get_cmap("tab20b")

get_cmap("tab20c")


def color(name, index):
    loc = {
        "blue": ("tab20b", 0),
        "red": ("tab20b", 12),
        "green": ("tab20b", 4),
        "purple": ("tab20b", 16),
        "yellow": ("tab20b", 8),
        "gray": ("tab20c", 16),
    }
    assert 1 <= index <= 4
    cmap, offset = loc[name]
    return get_cmap(cmap).colors[offset + index - 1]


default_colors = {
    ("PE", "Semi-AD (H, Cheby)"): color("yellow", 1), # "H" instead of "χ" because of matplotlib unicode problems
    ("C", "Semi-AD (H, Cheby)"): color("yellow", 2),
    ("PE", "Semi-AD (Cheby)"): color("blue", 1),
    ("C", "Semi-AD (Cheby)"): color("blue", 2),
    ("SM", "Semi-AD (Cheby)"): color("blue", 3),
    ("PE", "Full-AD (Cheby)"): color("red", 1),
    ("C", "Full-AD (Cheby)"): color("red", 2),
    ("SM", "Full-AD (Cheby)"): color("red", 3),
    ("PE", "Full-AD (ODE)"): color("green", 1),
    ("C", "Full-AD (ODE)"): color("green", 2),
    ("SM", "Full-AD (ODE)"): color("green", 3),
    ("SM", "Direct (Cheby)"): "black",
}

MARKER = {
    "fullcircle": dict(marker="o", markersize=3),
    "fullsquare": dict(marker="s", markersize=3),
    "fullstar": dict(marker="*", markersize=3),
    "fulldiamond": dict(marker="D", markersize=3),
    "halfcircle": dict(marker="o", fillstyle="top", markersize=5),
    "halfsquare": dict(marker="s", fillstyle="top", markersize=5),
    "halfstar": dict(marker="*", fillstyle="top", markersize=5),
    "halfdiamond": dict(marker="D", fillstyle="top", markersize=5),
    "emptycircle": dict(marker="o", fillstyle="none", markersize=7),
    "emptysquare": dict(marker="s", fillstyle="none", markersize=7),
    "emptystar": dict(marker="*", fillstyle="none", markersize=7),
    "emptydiamond": dict(marker="D", fillstyle="none", markersize=7),
    "dashed": dict(linestyle="dashed"),
}

default_markers = {
    ("PE", "Semi-AD (H, Cheby)"): MARKER["fullcircle"],
    ("C", "Semi-AD (H, Cheby)"): MARKER["halfcircle"],
    ("PE", "Semi-AD (Cheby)"): MARKER["fullcircle"],
    ("C", "Semi-AD (Cheby)"): MARKER["halfcircle"],
    ("SM", "Semi-AD (Cheby)"): MARKER["emptycircle"],
    ("PE", "Full-AD (Cheby)"): MARKER["fullsquare"],
    ("C", "Full-AD (Cheby)"): MARKER["halfsquare"],
    ("SM", "Full-AD (Cheby)"): MARKER["emptysquare"],
    ("PE", "Full-AD (ODE)"): MARKER["fulldiamond"],
    ("C", "Full-AD (ODE)"): MARKER["halfdiamond"],
    ("SM", "Full-AD (ODE)"): MARKER["emptydiamond"],
    ("SM", "Direct (Cheby)"): MARKER["dashed"],
}


def projectdir(*args, relpath=True):
    root = Path(".")
    while not (root / "Project.toml").is_file():
        if root.resolve() == (root / "..").resolve():
            raise IOError("Cannot find projectdir")
        root = root / ".."
    if relpath:
        return root.joinpath(*args)
    else:
        return root.joinpath(*args).resolve()


plt = rsmf.setup(
    r"\documentclass[aps,pra,letterpaper,allowtoday,onecolumn,unpublished]{quantumarticle}"
)

OUTDIR = projectdir("data", "plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = projectdir("data", "benchmarks")


class Benchmark:
    def __init__(
        self,
        filename,
        column_name,
        color=None,
        method=None,
        functional=None,
        in_inset=False,
        label=None,
        marker=None,
    ):
        self.filename = filename
        if method is None:
            method = "Semi-AD (H, Cheby)"
            if ("_full_ad_cheby") in filename or ("_FullADcheby_") in filename:
                method = "Full-AD (Cheby)"
            elif ("_full_ad." in filename) or ("_FullAD_" in filename):
                method = "Full-AD (ODE)"
            elif filename.startswith("SM_benchmark_"):
                method = "Direct (Cheby)"
            elif filename.startswith("SM_SemiAD_benchmark_"):
                method = "Semi-AD (Cheby)"
            elif ("_U_benchmark" in filename) and ("semi_ad" in filename):
                method = "Semi-AD (Cheby)"
        self.method = method
        if functional is None:
            if filename.startswith("PE_"):
                functional = "PE"
            elif filename.startswith("C_"):
                functional = "C"
            elif filename.startswith("SM_"):
                functional = "SM"
        self.functional = functional
        self.in_inset = in_inset
        if label is None:
            label = f"{method} ({functional})"
        self.label = label
        if color is None:
            color = default_colors[(self.functional, self.method)]
        self.color = color
        if marker is None:
            marker = default_markers[(self.functional, self.method)]
        marker_style = {}
        if isinstance(marker, str):
            marker_style = dict(marker=marker)
        elif isinstance(marker, dict):
            marker_style = marker
        self.marker_style = marker_style
        print("loading", BENCHMARKS / filename)
        df = pd.read_csv(BENCHMARKS / filename, index_col=0)
        self.column_name = column_name
        self.index_name = df.index.name
        self.data = df[self.column_name].dropna()


class RuntimeBenchmark(Benchmark):
    def __init__(self, filename, value="min", **kwargs):
        super().__init__(filename, f"nanosec_per_fg_{value}", **kwargs)


class RSSBenchmark(Benchmark):
    def __init__(self, filename, value="median", **kwargs):
        super().__init__(filename, f"rss_memory_MB_{value}", **kwargs)


class AllocBenchmark(Benchmark):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, "alloc_memory_MB", **kwargs)
        df = pd.read_csv(BENCHMARKS / filename, index_col=0)
        df["alloc_memory_MB_per_fg"] = df["alloc_memory_MB"] / df["fg"]
        self.data = df["alloc_memory_MB_per_fg"].dropna()


class FGBenchmark(Benchmark):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, "fg", **kwargs)


def rewrite_to_hs_size(series):
    df = series.reset_index()
    df["Hilbert space size"] = df["levels"] ** 2
    df = df.set_index("Hilbert space size")
    return df[series.name]


def plot_comparison(
    *benchmarks,
    ax=None,
    xlabel=None,
    ylabel=None,
    factor=1,
    logscale=False,
    outfile=None,
    inset=False,
    legend=True,
    levels_benchmarks_x="Hilbert space size",
    inset_pos=[0.55, 0.55, 0.4, 0.4],
    inset_xlim=None,
    inset_ylim=None,
    inset_xticks=None,
    inset_yticks=None,
    inset_hook=None,
):

    fig = None
    if ax is None:
        fig = plt.figure(aspect_ratio=0.5, wide=legend)
        ax = fig.add_subplot()
    ax.grid(color="gray", alpha=0.25)

    if benchmarks[0].column_name.startswith("nanosec_per_fg"):
        factor = 1e-9

    for benchmark in benchmarks:

        if benchmarks[0].index_name == "levels":
            if levels_benchmarks_x == "number of transmon levels":
                pass
            elif levels_benchmarks_x == "Hilbert space size":
                benchmark.data = rewrite_to_hs_size(benchmark.data)
            else:
                raise ValueError(f"invalid {levels_benchmarks_x=}")

        ax.plot(
            benchmark.data * factor,
            label=benchmark.label,
            color=benchmark.color,
            **benchmark.marker_style,
        )
        if logscale:
            ax.set_xscale("log")
            ax.set_yscale("log")

    if legend:
        ax.legend(
            bbox_to_anchor=(1, 1),
            loc="upper left",
            borderaxespad=1,
        )
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    elif benchmarks[0].index_name == "levels":
        ax.set_xlabel(levels_benchmarks_x)
    elif benchmarks[0].index_name == "T":
        ax.set_xlabel("gate duration (ns), number of time steps (10)")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif benchmarks[0].column_name.startswith("nanosec_per_fg"):
        ax.set_ylabel("runtime/grad eval (s)")
    elif benchmarks[0].column_name.startswith("rss_memory_MB"):
        ax.set_ylabel("peak RAM (MB)")
    elif benchmarks[0].column_name == "alloc_memory_MB":
        ax.set_ylabel("alloc per grad eval (MB)")
    elif benchmarks[0].column_name == "fg":
        ax.set_ylabel("gradient evaluations")

    if any([benchmark.in_inset for benchmark in benchmarks]):
        axins = ax.inset_axes(inset_pos)
        for benchmark in benchmarks:
            if benchmark.in_inset:
                axins.plot(
                    benchmark.data * factor,
                    label=benchmark.label,
                    color=benchmark.color,
                    **benchmark.marker_style,
                )
        axins.tick_params(axis="y", left=True, right=True)
        tick_bbox = dict(boxstyle="round", ec="white", fc="white", alpha=0.7, pad=0.1)
        setp(axins.get_xticklabels(), bbox=tick_bbox)
        setp(axins.get_yticklabels(), bbox=tick_bbox)
        axins.patch.set_alpha(0.7)
        if inset_xlim is not None:
            axins.set_xlim(inset_xlim)
        if inset_ylim is not None:
            axins.set_ylim(inset_ylim)
        if inset_xticks is not None:
            axins.set_xticks(inset_xticks)
        if inset_yticks is not None:
            axins.set_yticks(inset_yticks)
        if inset_hook is not None:
            inset_hook(axins)

    if benchmarks[0].column_name == "fg":
        fg_min = floor(ax.get_ylim()[0])
        fg_max = ceil(ax.get_ylim()[1])
        ax.set_yticks(range(fg_min, fg_max))

    if fig is not None:

        fig.tight_layout(pad=0.0)

        if outfile is not None:
            fig.savefig(OUTDIR / outfile)
            print("Written %s" % (OUTDIR / outfile))

        return fig


def make_plot_rss_range(*benchmark_files, levels_benchmarks_x="Hilbert space size"):

    min_benchmarks = [
        RSSBenchmark(filename, value="min") for filename in benchmark_files
    ]
    max_benchmarks = [
        RSSBenchmark(filename, value="max") for filename in benchmark_files
    ]

    index_transform = lambda s: s
    if min_benchmarks[0].index_name == "levels":
        if levels_benchmarks_x == "number of transmon levels":
            pass
        elif levels_benchmarks_x == "Hilbert space size":
            index_transform = rewrite_to_hs_size
        else:
            raise ValueError(f"invalid {levels_benchmarks_x=}")

    smin = index_transform(min_benchmarks[0].data)
    smax = index_transform(max_benchmarks[0].data)
    for min_benchmark, max_benchmark in zip(min_benchmarks, max_benchmarks):
        smin = np.minimum(smin, min_benchmark.data)
        smax = np.maximum(smax, max_benchmark.data)

    def _plot_(ax):
        x = np.array(smin.index)
        y1 = np.array(smin)
        y2 = np.array(smax)
        ax.fill_between(x, y1, y2, alpha=0.2)

    def _plot(ax):
        for min_benchmark, max_benchmark in zip(min_benchmarks, max_benchmarks):
            x = np.array(index_transform(min_benchmark.data).index)
            y1 = np.array(np.array(min_benchmark.data))
            y2 = np.array(np.array(max_benchmark.data))
            ax.fill_between(x, y1, y2, color=min_benchmark.color, alpha=0.1)

    return _plot


def plot_combined_benchmarks(outfile, **kwargs):

    fig = plt.figure(wide=True, aspect_ratio=0.55, width_ratio=1.0)
    axs = fig.subplots(nrows=2, ncols=2, sharex="col", sharey="row")

    levels_benchmarks_x = kwargs.get("levels_benchmarks_x", "Hilbert space size")
    if levels_benchmarks_x == "number of transmon levels":
        inset_levels_xticks = [3, 10, 15]
    elif levels_benchmarks_x == "Hilbert space size":
        inset_levels_xticks = [9, 100, 225]
    else:
        raise ValueError(f"invalid {levels_benchmarks_x=}")

    plot_comparison(
        RuntimeBenchmark("PE_U_benchmark_levels_semi_ad.csv", in_inset=True),
        RuntimeBenchmark("C_U_benchmark_levels_semi_ad.csv", in_inset=True),
        RuntimeBenchmark("SM_SemiAD_benchmark_levels.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_levels_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("C_benchmark_levels_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("SM_FullADcheby_benchmark_levels.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_levels_full_ad.csv"),
        RuntimeBenchmark("C_benchmark_levels_full_ad.csv"),
        RuntimeBenchmark("SM_FullAD_benchmark_levels.csv"),
        RuntimeBenchmark("SM_benchmark_levels.csv", in_inset=True),
        inset_pos=[0.525, 0.5, 0.45, 0.45],
        inset_ylim=(0, 20),
        inset_yticks=[0, 10, 20],
        inset_xticks=inset_levels_xticks,
        legend=False,
        ax=axs[0, 0],
        xlabel="",
        **kwargs,
    )

    plot_comparison(
        RuntimeBenchmark("PE_U_benchmark_times_semi_ad.csv", in_inset=True),
        RuntimeBenchmark("C_U_benchmark_times_semi_ad.csv", in_inset=True),
        RuntimeBenchmark("SM_SemiAD_benchmark_times.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_times_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("C_benchmark_times_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("SM_FullADcheby_benchmark_times.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_times_full_ad.csv"),
        RuntimeBenchmark("C_benchmark_times_full_ad.csv"),
        RuntimeBenchmark("SM_FullAD_benchmark_times.csv"),
        RuntimeBenchmark("SM_benchmark_times.csv", in_inset=True),
        inset_pos=[0.12, 0.5, 0.45, 0.45],
        inset_ylim=(0, 20),
        inset_yticks=[0, 10, 20],
        inset_xticks=[0, 400, 800],
        legend=False,
        ax=axs[0, 1],
        ylabel="",
        xlabel="",
        **kwargs,
    )

    plot_comparison(
        RSSBenchmark("PE_U_benchmark_levels_semi_ad.csv", in_inset=True),
        RSSBenchmark("C_U_benchmark_levels_semi_ad.csv", in_inset=True),
        RSSBenchmark("SM_SemiAD_benchmark_levels.csv", in_inset=True),
        RSSBenchmark("PE_benchmark_levels_full_ad_cheby.csv"),
        RSSBenchmark("C_benchmark_levels_full_ad_cheby.csv"),
        RSSBenchmark("SM_FullADcheby_benchmark_levels.csv"),
        RSSBenchmark("PE_benchmark_levels_full_ad.csv"),
        RSSBenchmark("C_benchmark_levels_full_ad.csv"),
        RSSBenchmark("SM_FullAD_benchmark_levels.csv"),
        RSSBenchmark("SM_benchmark_levels.csv", in_inset=True),
        inset_hook=make_plot_rss_range(
            "PE_benchmark_levels.csv",
            "C_benchmark_levels_semi_ad.csv",
            "SM_SemiAD_benchmark_levels.csv",
            "SM_benchmark_levels.csv",
            levels_benchmarks_x=levels_benchmarks_x,
        ),
        inset_pos=[0.525, 0.5, 0.45, 0.45],
        inset_ylim=(50, 150),
        inset_yticks=(50, 100, 150),
        inset_xticks=inset_levels_xticks,
        legend=False,
        ax=axs[1, 0],
        **kwargs,
    )

    plot_comparison(
        RSSBenchmark("PE_U_benchmark_times_semi_ad.csv", in_inset=True),
        RSSBenchmark("C_U_benchmark_times_semi_ad.csv", in_inset=True),
        RSSBenchmark("SM_SemiAD_benchmark_times.csv", in_inset=True),
        RSSBenchmark("PE_benchmark_times_full_ad_cheby.csv"),
        RSSBenchmark("C_benchmark_times_full_ad_cheby.csv"),
        RSSBenchmark("SM_FullADcheby_benchmark_times.csv"),
        RSSBenchmark("PE_benchmark_times_full_ad.csv"),
        RSSBenchmark("C_benchmark_times_full_ad.csv"),
        RSSBenchmark("SM_FullAD_benchmark_times.csv"),
        RSSBenchmark("SM_benchmark_times.csv", in_inset=True),
        inset_hook=make_plot_rss_range(
            "PE_benchmark_times.csv",
            "C_benchmark_times_semi_ad.csv",
            "SM_SemiAD_benchmark_times.csv",
            "SM_benchmark_times.csv",
            levels_benchmarks_x=levels_benchmarks_x,
        ),
        inset_pos=[0.12, 0.5, 0.45, 0.45],
        inset_ylim=(50, 150),
        inset_yticks=(50, 100, 150),
        inset_xticks=[0, 400, 800],
        legend=False,
        ax=axs[1, 1],
        ylabel="",
        **kwargs,
    )

    # https://stackoverflow.com/questions/9834452
    lines, labels = axs[0, 0].get_legend_handles_labels()
    permute = itemgetter(0, 3, 6, 1, 4, 7, 2, 5, 8)
    l1_offset = transforms.ScaledTranslation(-0.75, -4 / 72.0, fig.dpi_scale_trans)
    l1 = fig.legend(
        permute(lines),
        permute(labels),
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        bbox_transform=(fig.transFigure + l1_offset),
        frameon=False,
    )
    l2_offset = transforms.ScaledTranslation(5, -4 / 72.0, fig.dpi_scale_trans)
    l2 = fig.legend(
        [lines[9]],
        [labels[9]],
        ncol=3,
        loc="lower left",
        bbox_to_anchor=(0, 1),
        bbox_transform=(fig.transFigure + l2_offset),
        frameon=False,
    )

    def axes_label(ax, label):
        ax.annotate(
            label,
            (0, 1),
            xycoords="axes fraction",
            xytext=(0, 2),
            textcoords="offset points",
            verticalalignment="bottom",
            horizontalalignment="left",
        )

    def line_label(ax, label, xy, rot=0):
        bbox_label = dict(boxstyle="round", ec="white", fc="white", alpha=0.7, pad=0.1)
        ax.annotate(
            label,
            xy,
            xycoords="axes fraction",
            color=default_colors[("PE", label)],
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=8,
            bbox=bbox_label,
            rotation=rot,
        )

    if levels_benchmarks_x == "Hilbert space size":

        # top left
        line_label(axs[0, 0], "Full-AD (ODE)", (0.17, 0.53), 60)
        line_label(axs[0, 0], "Full-AD (Cheby)", (0.76, 0.18), 3)
        line_label(axs[0, 0], "Semi-AD (Cheby)", (0.42, 0.14), 0)

        # top right
        line_label(axs[0, 1], "Full-AD (ODE)", (0.74, 0.83), 25)
        line_label(axs[0, 1], "Full-AD (Cheby)", (0.76, 0.18), 3)
        line_label(axs[0, 1], "Semi-AD (Cheby)", (0.42, 0.14), 0)

        # bottom left
        line_label(axs[1, 0], "Full-AD (Cheby)", (0.17, 0.42), 30)
        line_label(axs[1, 0], "Full-AD (ODE)", (0.35, 0.22), 0)
        line_label(axs[1, 0], "Semi-AD (Cheby)", (0.83, 0.14), 0)

        # bottom right
        line_label(axs[1, 1], "Full-AD (Cheby)", (0.8, 0.74), 15)
        line_label(axs[1, 1], "Full-AD (ODE)", (0.78, 0.32), 10)
        line_label(axs[1, 1], "Semi-AD (Cheby)", (0.83, 0.14), 0)

    print("## Axes Ranges ##")
    for (i, j) in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        x0, x1 = axs[i, j].get_xlim()
        y0, y1 = axs[i, j].get_ylim()
        print(f"ax[{i}, {j}]:")
        print(f"    x: {x0:6.1f}  —  {x1:6.1f} ({(x1-x0):6.1f})")
        print(f"    y: {y0:6.1f}  –  {y1:6.1f} ({(y1-y0):6.1f})")
    fig.tight_layout(pad=0.25, h_pad=0.5)
    fig.savefig(OUTDIR / outfile, bbox_inches="tight")
    print("Written %s" % (OUTDIR / outfile))
    print(f"Figure size (inches): {fig.get_size_inches()}")
    return fig


plot_combined_benchmarks(
    "benchmarks_cqe.pdf",
    levels_benchmarks_x="Hilbert space size"
)


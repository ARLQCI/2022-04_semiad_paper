# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: SemiAD Paper
#     language: python
#     name: semi-ad-paper
# ---

# # Plots

from pathlib import Path
from operator import itemgetter
import rsmf
import pandas as pd

from matplotlib.pylab import get_cmap
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
}

default_markers = {
    ("PE", "Semi-AD (Cheby)"): MARKER["fullcircle"],
    ("C", "Semi-AD (Cheby)"): MARKER["halfcircle"],
    ("SM", "Semi-AD (Cheby)"): MARKER["emptycircle"],
    ("PE", "Full-AD (Cheby)"): MARKER["fullsquare"],
    ("C", "Full-AD (Cheby)"): MARKER["halfsquare"],
    ("SM", "Full-AD (Cheby)"): MARKER["emptysquare"],
    ("PE", "Full-AD (ODE)"): MARKER["fulldiamond"],
    ("C", "Full-AD (ODE)"): MARKER["halfdiamond"],
    ("SM", "Full-AD (ODE)"): MARKER["emptydiamond"],
    ("SM", "Direct (Cheby)"): MARKER["fullcircle"],
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
            method = "Semi-AD (Cheby)"
            if ("_full_ad_cheby") in filename or ("_FullADcheby_") in filename:
                method = "Full-AD (Cheby)"
            elif ("_full_ad." in filename) or ("_FullAD_" in filename):
                method = "Full-AD (ODE)"
            elif filename.startswith("SM_benchmark_"):
                method = "Direct (Cheby)"
            elif filename.startswith("SM_SemiAD_benchmark_"):
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
    def __init__(self, filename, **kwargs):
        super().__init__(filename, "nanosec_per_fg", **kwargs)


class RSSBenchmark(Benchmark):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, "rss_memory_MB_max", **kwargs)


class AllocBenchmark(Benchmark):
    def __init__(self, filename, **kwargs):
        super().__init__(filename, "alloc_memory_MB", **kwargs)


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
    inset_pos=[0.55, 0.55, 0.4, 0.4],
):

    fig = None
    if ax is None:
        fig = plt.figure(aspect_ratio=0.5, wide=legend)
        ax = fig.add_subplot()
    ax.grid(color="gray", alpha=0.25)

    if benchmarks[0].column_name == "nanosec_per_fg":
        factor = 1e-9

    for benchmark in benchmarks:
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
        ax.set_xlabel("number of transmon levels")
    elif benchmarks[0].index_name == "T":
        ax.set_xlabel("gate duration (ns), number of time steps (10)")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif benchmarks[0].column_name == "nanosec_per_fg":
        ax.set_ylabel("runtime per grad eval (s)")
    elif benchmarks[0].column_name == "rss_memory_MB":
        ax.set_ylabel("RSS peak memory (MB)")
    elif benchmarks[0].column_name == "alloc_memory_MB":
        ax.set_ylabel("allocated memory (MB)")

    if True in [benchmark.in_inset for benchmark in benchmarks]:
        axins = ax.inset_axes(inset_pos)
        for benchmark in benchmarks:
            if benchmark.in_inset:
                axins.plot(
                    benchmark.data * factor,
                    label=benchmark.label,
                    color=benchmark.color,
                    **benchmark.marker_style,
                )
    if fig is not None:

        fig.tight_layout(pad=0.0)

        if outfile is not None:
            fig.savefig(OUTDIR / outfile)
            print("Written %s" % (OUTDIR / outfile))

        return fig


# ## Runtime

plot_comparison(
    RuntimeBenchmark("PE_benchmark_levels.csv", in_inset=True),
    RuntimeBenchmark("C_benchmark_levels_semi_ad.csv", in_inset=True),
    RuntimeBenchmark("SM_SemiAD_benchmark_levels.csv", in_inset=True),
    RuntimeBenchmark("PE_benchmark_levels_full_ad_cheby.csv", in_inset=True),
    RuntimeBenchmark("C_benchmark_levels_full_ad_cheby.csv", in_inset=True),
    RuntimeBenchmark("SM_FullADcheby_benchmark_levels.csv", in_inset=True),
    RuntimeBenchmark("PE_benchmark_levels_full_ad.csv"),
    RuntimeBenchmark("C_benchmark_levels_full_ad.csv"),
    RuntimeBenchmark("SM_FullAD_benchmark_levels.csv"),
    RuntimeBenchmark("SM_benchmark_levels.csv", in_inset=True),
    outfile="PE_runtimes_levels.pdf",
)

plot_comparison(
    RuntimeBenchmark("PE_benchmark_times.csv", in_inset=True),
    RuntimeBenchmark("C_benchmark_times_semi_ad.csv", in_inset=True),
    RuntimeBenchmark("SM_SemiAD_benchmark_times.csv", in_inset=True),
    RuntimeBenchmark("PE_benchmark_times_full_ad_cheby.csv", in_inset=True),
    RuntimeBenchmark("C_benchmark_times_full_ad_cheby.csv", in_inset=True),
    RuntimeBenchmark("SM_FullADcheby_benchmark_times.csv", in_inset=True),
    RuntimeBenchmark("PE_benchmark_times_full_ad.csv"),
    RuntimeBenchmark("C_benchmark_times_full_ad.csv"),
    RuntimeBenchmark("SM_FullAD_benchmark_times.csv"),
    RuntimeBenchmark("SM_benchmark_times.csv", in_inset=True),
    outfile="PE_runtimes_times.pdf",
    inset_pos=[0.1, 0.6, 0.35, 0.35],
)

# ## RSS

plot_comparison(
    RSSBenchmark("PE_benchmark_levels.csv", in_inset=True),
    RSSBenchmark("C_benchmark_levels_semi_ad.csv", in_inset=True),
    RSSBenchmark("SM_SemiAD_benchmark_levels.csv", in_inset=True),
    RSSBenchmark("PE_benchmark_levels_full_ad_cheby.csv"),
    RSSBenchmark("C_benchmark_levels_full_ad_cheby.csv"),
    RSSBenchmark("SM_FullADcheby_benchmark_levels.csv"),
    RSSBenchmark("PE_benchmark_levels_full_ad.csv"),
    RSSBenchmark("C_benchmark_levels_full_ad.csv"),
    RSSBenchmark("SM_FullAD_benchmark_levels.csv"),
    RSSBenchmark("SM_benchmark_levels.csv", in_inset=True),
    outfile="PE_memory_levels.pdf",
    inset_pos=[0.1, 0.6, 0.35, 0.35],
)

plot_comparison(
    RSSBenchmark("PE_benchmark_times.csv", in_inset=True),
    RSSBenchmark("C_benchmark_times_semi_ad.csv", in_inset=True),
    RSSBenchmark("SM_SemiAD_benchmark_times.csv", in_inset=True),
    RSSBenchmark("PE_benchmark_times_full_ad_cheby.csv"),
    RSSBenchmark("C_benchmark_times_full_ad_cheby.csv"),
    RSSBenchmark("SM_FullADcheby_benchmark_times.csv"),
    RSSBenchmark("PE_benchmark_times_full_ad.csv"),
    RSSBenchmark("C_benchmark_times_full_ad.csv"),
    RSSBenchmark("SM_FullAD_benchmark_times.csv"),
    RSSBenchmark("SM_benchmark_times.csv", in_inset=True),
    outfile="PE_memory_times.pdf",
    inset_pos=[0.1, 0.62, 0.35, 0.35],
)

# ## Allocations

plot_comparison(
    AllocBenchmark("PE_benchmark_levels.csv", in_inset=True),
    AllocBenchmark("C_benchmark_levels_semi_ad.csv", in_inset=True),
    AllocBenchmark("SM_SemiAD_benchmark_levels.csv", in_inset=True),
    AllocBenchmark("PE_benchmark_levels_full_ad_cheby.csv"),
    AllocBenchmark("C_benchmark_levels_full_ad_cheby.csv"),
    AllocBenchmark("SM_FullADcheby_benchmark_levels.csv"),
    AllocBenchmark("PE_benchmark_levels_full_ad.csv"),
    AllocBenchmark("C_benchmark_levels_full_ad.csv"),
    AllocBenchmark("SM_FullAD_benchmark_levels.csv"),
    AllocBenchmark("SM_benchmark_levels.csv", in_inset=True),
    outfile="PE_allocated_levels.pdf",
    inset_pos=[0.5, 0.6, 0.35, 0.35],
)

plot_comparison(
    AllocBenchmark("PE_benchmark_times.csv", in_inset=True),
    AllocBenchmark("C_benchmark_times_semi_ad.csv", in_inset=True),
    AllocBenchmark("SM_SemiAD_benchmark_times.csv", in_inset=True),
    AllocBenchmark("PE_benchmark_times_full_ad_cheby.csv"),
    AllocBenchmark("C_benchmark_times_full_ad_cheby.csv"),
    AllocBenchmark("SM_FullADcheby_benchmark_times.csv"),
    AllocBenchmark("PE_benchmark_times_full_ad.csv"),
    AllocBenchmark("C_benchmark_times_full_ad.csv"),
    AllocBenchmark("SM_FullAD_benchmark_times.csv"),
    AllocBenchmark("SM_benchmark_times.csv", in_inset=True),
    outfile="PE_allocated_times.pdf",
    inset_pos=[0.1, 0.62, 0.35, 0.35],
)

# ## Combined Plots


def plot_combined_benchmarks1(outfile):

    fig = plt.figure(wide=True, aspect_ratio=0.6, width_ratio=1.0)
    axs = fig.subplots(nrows=2, ncols=2, sharex="col", sharey="row")

    a = plot_comparison(
        RuntimeBenchmark("PE_benchmark_levels.csv", in_inset=True),
        RuntimeBenchmark("C_benchmark_levels_semi_ad.csv", in_inset=True),
        RuntimeBenchmark("SM_SemiAD_benchmark_levels.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_levels_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("C_benchmark_levels_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("SM_FullADcheby_benchmark_levels.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_levels_full_ad.csv"),
        RuntimeBenchmark("C_benchmark_levels_full_ad.csv"),
        RuntimeBenchmark("SM_FullAD_benchmark_levels.csv"),
        RuntimeBenchmark("SM_benchmark_levels.csv", in_inset=True),
        inset_pos=[0.55, 0.35, 0.4, 0.6],
        legend=False,
        ax=axs[0, 0],
        xlabel="",
    )

    plot_comparison(
        RuntimeBenchmark("PE_benchmark_times.csv", in_inset=True),
        RuntimeBenchmark("C_benchmark_times_semi_ad.csv", in_inset=True),
        RuntimeBenchmark("SM_SemiAD_benchmark_times.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_times_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("C_benchmark_times_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("SM_FullADcheby_benchmark_times.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_times_full_ad.csv"),
        RuntimeBenchmark("C_benchmark_times_full_ad.csv"),
        RuntimeBenchmark("SM_FullAD_benchmark_times.csv"),
        RuntimeBenchmark("SM_benchmark_times.csv", in_inset=True),
        inset_pos=[0.1, 0.6, 0.35, 0.35],
        legend=False,
        ax=axs[0, 1],
        ylabel="",
        xlabel="",
    )

    plot_comparison(
        RSSBenchmark("PE_benchmark_levels.csv", in_inset=True),
        RSSBenchmark("C_benchmark_levels_semi_ad.csv", in_inset=True),
        RSSBenchmark("SM_SemiAD_benchmark_levels.csv", in_inset=True),
        RSSBenchmark("PE_benchmark_levels_full_ad_cheby.csv"),
        RSSBenchmark("C_benchmark_levels_full_ad_cheby.csv"),
        RSSBenchmark("SM_FullADcheby_benchmark_levels.csv"),
        RSSBenchmark("PE_benchmark_levels_full_ad.csv"),
        RSSBenchmark("C_benchmark_levels_full_ad.csv"),
        RSSBenchmark("SM_FullAD_benchmark_levels.csv"),
        RSSBenchmark("SM_benchmark_levels.csv", in_inset=True),
        inset_pos=[0.12, 0.62, 0.35, 0.35],
        legend=False,
        ax=axs[1, 0],
    )

    plot_comparison(
        RSSBenchmark("PE_benchmark_times.csv", in_inset=True),
        RSSBenchmark("C_benchmark_times_semi_ad.csv", in_inset=True),
        RSSBenchmark("SM_SemiAD_benchmark_times.csv", in_inset=True),
        RSSBenchmark("PE_benchmark_times_full_ad_cheby.csv"),
        RSSBenchmark("C_benchmark_times_full_ad_cheby.csv"),
        RSSBenchmark("SM_FullADcheby_benchmark_times.csv"),
        RSSBenchmark("PE_benchmark_times_full_ad.csv"),
        RSSBenchmark("C_benchmark_times_full_ad.csv"),
        RSSBenchmark("SM_FullAD_benchmark_times.csv"),
        RSSBenchmark("SM_benchmark_times.csv", in_inset=True),
        inset_pos=[0.12, 0.62, 0.35, 0.35],
        legend=False,
        ax=axs[1, 1],
        ylabel="",
    )

    # https://stackoverflow.com/questions/9834452
    lines, labels = axs[0, 0].get_legend_handles_labels()
    permute = itemgetter(0, 3, 6, 1, 4, 7, 2, 5, 8)
    l1_offset = transforms.ScaledTranslation(0.25, -4 / 72.0, fig.dpi_scale_trans)
    l1 = fig.legend(
        permute(lines),
        permute(labels),
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        bbox_transform=(fig.transFigure + l1_offset),
        frameon=False,
    )
    l2_offset = transforms.ScaledTranslation(4.22, -18 / 72.0, fig.dpi_scale_trans)
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

    axes_label(axs[0, 0], "(a)")
    axes_label(axs[0, 1], "(b)")
    axes_label(axs[1, 0], "(c)")
    axes_label(axs[1, 1], "(d)")

    fig.tight_layout(pad=0.25, h_pad=0.5)
    fig.savefig(OUTDIR / outfile, bbox_inches="tight")
    print("Written %s" % (OUTDIR / outfile))
    print(fig.get_size_inches())
    return fig


plot_combined_benchmarks1("combined_benchmarks1.pdf")


def plot_combined_benchmarks(outfile):

    fig = plt.figure(wide=True, aspect_ratio=0.9, width_ratio=1.0)
    axs = fig.subplots(nrows=3, ncols=2, sharex="col", sharey="row")

    plot_comparison(
        RuntimeBenchmark("PE_benchmark_levels.csv", in_inset=True),
        RuntimeBenchmark("C_benchmark_levels_semi_ad.csv", in_inset=True),
        RuntimeBenchmark("SM_SemiAD_benchmark_levels.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_levels_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("C_benchmark_levels_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("SM_FullADcheby_benchmark_levels.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_levels_full_ad.csv"),
        RuntimeBenchmark("C_benchmark_levels_full_ad.csv"),
        RuntimeBenchmark("SM_FullAD_benchmark_levels.csv"),
        RuntimeBenchmark("SM_benchmark_levels.csv", in_inset=True),
        inset_pos=[0.55, 0.35, 0.4, 0.6],
        legend=False,
        ax=axs[0, 0],
        xlabel="",
    )

    plot_comparison(
        RuntimeBenchmark("PE_benchmark_times.csv", in_inset=True),
        RuntimeBenchmark("C_benchmark_times_semi_ad.csv", in_inset=True),
        RuntimeBenchmark("SM_SemiAD_benchmark_times.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_times_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("C_benchmark_times_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("SM_FullADcheby_benchmark_times.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_times_full_ad.csv"),
        RuntimeBenchmark("C_benchmark_times_full_ad.csv"),
        RuntimeBenchmark("SM_FullAD_benchmark_times.csv"),
        RuntimeBenchmark("SM_benchmark_times.csv", in_inset=True),
        inset_pos=[0.1, 0.6, 0.35, 0.35],
        legend=False,
        ax=axs[0, 1],
        ylabel="",
        xlabel="",
    )

    plot_comparison(
        RSSBenchmark("PE_benchmark_levels.csv", in_inset=True),
        RSSBenchmark("C_benchmark_levels_semi_ad.csv", in_inset=True),
        RSSBenchmark("SM_SemiAD_benchmark_levels.csv", in_inset=True),
        RSSBenchmark("PE_benchmark_levels_full_ad_cheby.csv"),
        RSSBenchmark("C_benchmark_levels_full_ad_cheby.csv"),
        RSSBenchmark("SM_FullADcheby_benchmark_levels.csv"),
        RSSBenchmark("PE_benchmark_levels_full_ad.csv"),
        RSSBenchmark("C_benchmark_levels_full_ad.csv"),
        RSSBenchmark("SM_FullAD_benchmark_levels.csv"),
        RSSBenchmark("SM_benchmark_levels.csv", in_inset=True),
        inset_pos=[0.12, 0.62, 0.35, 0.35],
        legend=False,
        ax=axs[1, 0],
        xlabel="",
    )

    plot_comparison(
        RSSBenchmark("PE_benchmark_times.csv", in_inset=True),
        RSSBenchmark("C_benchmark_times_semi_ad.csv", in_inset=True),
        RSSBenchmark("SM_SemiAD_benchmark_times.csv", in_inset=True),
        RSSBenchmark("PE_benchmark_times_full_ad_cheby.csv"),
        RSSBenchmark("C_benchmark_times_full_ad_cheby.csv"),
        RSSBenchmark("SM_FullADcheby_benchmark_times.csv"),
        RSSBenchmark("PE_benchmark_times_full_ad.csv"),
        RSSBenchmark("C_benchmark_times_full_ad.csv"),
        RSSBenchmark("SM_FullAD_benchmark_times.csv"),
        RSSBenchmark("SM_benchmark_times.csv", in_inset=True),
        inset_pos=[0.12, 0.62, 0.35, 0.35],
        legend=False,
        ax=axs[1, 1],
        xlabel="",
        ylabel="",
    )

    plot_comparison(
        AllocBenchmark("PE_benchmark_levels.csv", in_inset=True),
        AllocBenchmark("C_benchmark_levels_semi_ad.csv", in_inset=True),
        AllocBenchmark("SM_SemiAD_benchmark_levels.csv", in_inset=True),
        AllocBenchmark("PE_benchmark_levels_full_ad_cheby.csv"),
        AllocBenchmark("C_benchmark_levels_full_ad_cheby.csv"),
        AllocBenchmark("SM_FullADcheby_benchmark_levels.csv"),
        AllocBenchmark("PE_benchmark_levels_full_ad.csv"),
        AllocBenchmark("C_benchmark_levels_full_ad.csv"),
        AllocBenchmark("SM_FullAD_benchmark_levels.csv"),
        AllocBenchmark("SM_benchmark_levels.csv", in_inset=True),
        inset_pos=[0.12, 0.62, 0.35, 0.35],
        legend=False,
        ax=axs[2, 0],
    )

    plot_comparison(
        AllocBenchmark("PE_benchmark_times.csv", in_inset=True),
        AllocBenchmark("C_benchmark_times_semi_ad.csv", in_inset=True),
        AllocBenchmark("SM_SemiAD_benchmark_times.csv", in_inset=True),
        AllocBenchmark("PE_benchmark_times_full_ad_cheby.csv"),
        AllocBenchmark("C_benchmark_times_full_ad_cheby.csv"),
        AllocBenchmark("SM_FullADcheby_benchmark_times.csv"),
        AllocBenchmark("PE_benchmark_times_full_ad.csv"),
        AllocBenchmark("C_benchmark_times_full_ad.csv"),
        AllocBenchmark("SM_FullAD_benchmark_times.csv"),
        AllocBenchmark("SM_benchmark_times.csv", in_inset=True),
        inset_pos=[0.12, 0.62, 0.35, 0.35],
        legend=False,
        ax=axs[2, 1],
        ylabel="",
    )

    y_formatter = FuncFormatter(lambda v, pos: "%.1f" % (v * 1e-6))
    axs[2, 0].yaxis.set_major_formatter(y_formatter)
    axs[2, 0].annotate(
        "×10⁶  ", (0, 1), xycoords="axes fraction", ha="right", va="bottom"
    )

    # https://stackoverflow.com/questions/9834452
    lines, labels = axs[0, 0].get_legend_handles_labels()
    permute = itemgetter(0, 3, 6, 1, 4, 7, 2, 5, 8)
    l1_offset = transforms.ScaledTranslation(0.25, -4 / 72.0, fig.dpi_scale_trans)
    l1 = fig.legend(
        permute(lines),
        permute(labels),
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        bbox_transform=(fig.transFigure + l1_offset),
        frameon=False,
    )
    l2_offset = transforms.ScaledTranslation(4.22, -18 / 72.0, fig.dpi_scale_trans)
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

    axes_label(axs[0, 0], "(a)")
    axes_label(axs[0, 1], "(b)")
    axes_label(axs[1, 0], "(c)")
    axes_label(axs[1, 1], "(d)")
    axes_label(axs[2, 0], "(e)")
    axes_label(axs[2, 1], "(f)")

    fig.tight_layout(pad=0.25, h_pad=0.5)
    fig.savefig(OUTDIR / outfile, bbox_inches="tight")
    print("Written %s" % (OUTDIR / outfile))
    print(fig.get_size_inches())
    return fig


plot_combined_benchmarks("combined_benchmarks.pdf")

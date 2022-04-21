# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Plots

from pathlib import Path
from operator import itemgetter
import rsmf
import pandas as pd

from matplotlib.pylab import get_cmap

get_cmap("tab20c")
# TODO: consider tab20b as well
# https://matplotlib.org/stable/gallery/color/colormap_reference.html

# +
def color(name, index):
    offset = {
        "blue": 0,
        "red": 4,
        "green": 8,
        "purple": 12,
        "gray": 16,
    }
    assert 1 <= index <= 4
    return get_cmap("tab20c").colors[offset[name] + index - 1]

my_colors = {
    ("blue", 1): "#53287e",
    ("blue", 2): "#a17ebd",
    ("blue", 3): "#f0dbff",
}
# -

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
    "emptycircle": dict(marker="o", fillstyle="none", markersize=5),
    "emptysquare": dict(marker="s", fillstyle="none", markersize=5),
    "emptystar": dict(marker="*", fillstyle="none", markersize=5),
    "emptydiamond": dict(marker="D", fillstyle="none", markersize=5),
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


plt = rsmf.setup(r"\documentclass[aps,pra,letterpaper,allowtoday,onecolumn,unpublished]{quantumarticle}")

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
    def __init__(
        self,
        filename,
        in_inset=False
    ):
        super().__init__(
            filename,
            "nanosec_per_fg",
            in_inset=in_inset
        )


class RSSBenchmark(Benchmark):
    def __init__(
        self,
        filename,
        in_inset=False
    ):
        super().__init__(
            filename,
            "rss_memory_MB",
            in_inset=in_inset
        )


class AllocBenchmark(Benchmark):
    def __init__(
        self,
        filename,
        in_inset=False
    ):
        super().__init__(
            filename,
            "alloc_memory_MB",
            in_inset=in_inset
        )


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
    inset_pos=[0.55, 0.55, 0.4, 0.4]
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
        ax.set_xlabel("gate duration")
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    elif benchmarks[0].column_name == "nanosec_per_fg":
        ax.set_ylabel("runtime (s)")
    elif benchmarks[0].column_name == "rss_memory_MB":
        ax.set_ylabel("memory (MB)")
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
    RuntimeBenchmark("SM_benchmark_levels.csv", in_inset=True),
    RuntimeBenchmark("PE_benchmark_levels_full_ad_cheby.csv", in_inset=True),
    RuntimeBenchmark("C_benchmark_levels_full_ad_cheby.csv", in_inset=True),
    RuntimeBenchmark("SM_FullADcheby_benchmark_levels.csv", in_inset=True),
    RuntimeBenchmark("PE_benchmark_levels_full_ad.csv"),
    RuntimeBenchmark("C_benchmark_levels_full_ad.csv"),
    RuntimeBenchmark("SM_FullAD_benchmark_levels.csv"),
    outfile="PE_runtimes_levels.pdf",
)

plot_comparison(
    RuntimeBenchmark("PE_benchmark_times.csv", in_inset=True),
    RuntimeBenchmark("C_benchmark_times_semi_ad.csv", in_inset=True),
    RuntimeBenchmark("SM_benchmark_times.csv", in_inset=True),
    RuntimeBenchmark("PE_benchmark_times_full_ad_cheby.csv", in_inset=True),
    RuntimeBenchmark("C_benchmark_times_full_ad_cheby.csv", in_inset=True),
    RuntimeBenchmark("SM_FullADcheby_benchmark_times.csv", in_inset=True),
    RuntimeBenchmark("PE_benchmark_times_full_ad.csv"),
    RuntimeBenchmark("C_benchmark_times_full_ad.csv"),
    RuntimeBenchmark("SM_FullAD_benchmark_times.csv"),
    outfile="PE_runtimes_times.pdf",
    inset_pos=[0.1, 0.6, 0.35, 0.35]
)

# ## RSS

plot_comparison(
    RSSBenchmark("PE_benchmark_levels.csv", in_inset=True),
    RSSBenchmark("C_benchmark_levels_semi_ad.csv", in_inset=True),
    RSSBenchmark("SM_benchmark_levels.csv", in_inset=True),
    RSSBenchmark("PE_benchmark_levels_full_ad_cheby.csv"),
    RSSBenchmark("C_benchmark_levels_full_ad_cheby.csv"),
    RSSBenchmark("SM_FullADcheby_benchmark_levels.csv"),
    RSSBenchmark("PE_benchmark_levels_full_ad.csv"),
    RSSBenchmark("C_benchmark_levels_full_ad.csv"),
    RSSBenchmark("SM_FullAD_benchmark_levels.csv"),
    outfile="PE_memory_levels.pdf",
    inset_pos=[0.1, 0.6, 0.35, 0.35]
)

plot_comparison(
    RSSBenchmark("PE_benchmark_times.csv", in_inset=True),
    RSSBenchmark("C_benchmark_times_semi_ad.csv", in_inset=True),
    RSSBenchmark("SM_benchmark_times.csv", in_inset=True),
    RSSBenchmark("PE_benchmark_times_full_ad_cheby.csv"),
    RSSBenchmark("C_benchmark_times_full_ad_cheby.csv"),
    RSSBenchmark("SM_FullADcheby_benchmark_times.csv"),
    RSSBenchmark("PE_benchmark_times_full_ad.csv"),
    RSSBenchmark("C_benchmark_times_full_ad.csv"),
    RSSBenchmark("SM_FullAD_benchmark_times.csv"),
    outfile="PE_memory_times.pdf",
    inset_pos=[0.1, 0.62, 0.35, 0.35]
)

# ## Allocations

plot_comparison(
    AllocBenchmark("PE_benchmark_levels.csv", in_inset=True),
    AllocBenchmark("C_benchmark_levels_semi_ad.csv", in_inset=True),
    AllocBenchmark("SM_benchmark_levels.csv", in_inset=True),
    AllocBenchmark("PE_benchmark_levels_full_ad_cheby.csv"),
    AllocBenchmark("C_benchmark_levels_full_ad_cheby.csv"),
    AllocBenchmark("SM_FullADcheby_benchmark_levels.csv"),
    AllocBenchmark("PE_benchmark_levels_full_ad.csv"),
    AllocBenchmark("C_benchmark_levels_full_ad.csv"),
    AllocBenchmark("SM_FullAD_benchmark_levels.csv"),
    outfile="PE_allocated_levels.pdf",
    inset_pos=[0.5, 0.6, 0.35, 0.35]
)

plot_comparison(
    AllocBenchmark("PE_benchmark_times.csv", in_inset=True),
    AllocBenchmark("C_benchmark_times_semi_ad.csv", in_inset=True),
    AllocBenchmark("SM_benchmark_times.csv", in_inset=True),
    AllocBenchmark("PE_benchmark_times_full_ad_cheby.csv"),
    AllocBenchmark("C_benchmark_times_full_ad_cheby.csv"),
    AllocBenchmark("SM_FullADcheby_benchmark_times.csv"),
    AllocBenchmark("PE_benchmark_times_full_ad.csv"),
    AllocBenchmark("C_benchmark_times_full_ad.csv"),
    AllocBenchmark("SM_FullAD_benchmark_times.csv"),
    outfile="PE_allocated_times.pdf",
    inset_pos=[0.1, 0.62, 0.35, 0.35]
)

# ## Combined Plots


def plot_combined_PE(outfile):

    fig = plt.figure(wide=True, aspect_ratio=0.5, width_ratio=.95)
    axs = fig.subplots(nrows=2, ncols=2, sharex="col", sharey="row")

    a = plot_comparison(
        RuntimeBenchmark("PE_benchmark_levels.csv", in_inset=True),
        RuntimeBenchmark("C_benchmark_levels_semi_ad.csv", in_inset=True),
        RuntimeBenchmark("SM_benchmark_levels.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_levels_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("C_benchmark_levels_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("SM_FullADcheby_benchmark_levels.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_levels_full_ad.csv"),
        RuntimeBenchmark("C_benchmark_levels_full_ad.csv"),
        RuntimeBenchmark("SM_FullAD_benchmark_levels.csv"),
        inset_pos=[0.55, 0.35, 0.4, 0.6],
        legend=False,
        ax=axs[0, 0],
        xlabel=""
    )
    
    plot_comparison(
        RuntimeBenchmark("PE_benchmark_times.csv", in_inset=True),
        RuntimeBenchmark("C_benchmark_times_semi_ad.csv", in_inset=True),
        RuntimeBenchmark("SM_benchmark_times.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_times_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("C_benchmark_times_full_ad_cheby.csv", in_inset=True),
        RuntimeBenchmark("SM_FullADcheby_benchmark_times.csv", in_inset=True),
        RuntimeBenchmark("PE_benchmark_times_full_ad.csv"),
        RuntimeBenchmark("C_benchmark_times_full_ad.csv"),
        RuntimeBenchmark("SM_FullAD_benchmark_times.csv"),
        inset_pos=[0.1, 0.6, 0.35, 0.35],
        legend=False,
        ax=axs[0, 1],
        ylabel="",
        xlabel=""
    )
    
    plot_comparison(
        RSSBenchmark("PE_benchmark_levels.csv", in_inset=True),
        RSSBenchmark("C_benchmark_levels_semi_ad.csv", in_inset=True),
        RSSBenchmark("SM_benchmark_levels.csv", in_inset=True),
        RSSBenchmark("PE_benchmark_levels_full_ad_cheby.csv"),
        RSSBenchmark("C_benchmark_levels_full_ad_cheby.csv"),
        RSSBenchmark("SM_FullADcheby_benchmark_levels.csv"),
        RSSBenchmark("PE_benchmark_levels_full_ad.csv"),
        RSSBenchmark("C_benchmark_levels_full_ad.csv"),
        RSSBenchmark("SM_FullAD_benchmark_levels.csv"),
        inset_pos=[0.12, 0.62, 0.35, 0.35],
        legend=False,
        ax=axs[1, 0],
    )
    
    plot_comparison(
        RSSBenchmark("PE_benchmark_times.csv", in_inset=True),
        RSSBenchmark("C_benchmark_times_semi_ad.csv", in_inset=True),
        RSSBenchmark("SM_benchmark_times.csv", in_inset=True),
        RSSBenchmark("PE_benchmark_times_full_ad_cheby.csv"),
        RSSBenchmark("C_benchmark_times_full_ad_cheby.csv"),
        RSSBenchmark("SM_FullADcheby_benchmark_times.csv"),
        RSSBenchmark("PE_benchmark_times_full_ad.csv"),
        RSSBenchmark("C_benchmark_times_full_ad.csv"),
        RSSBenchmark("SM_FullAD_benchmark_times.csv"),
        inset_pos=[0.12, 0.62, 0.35, 0.35],
        legend=False,
        ax=axs[1, 1],
        ylabel=""
    )

    # https://stackoverflow.com/questions/9834452
    lines, labels = axs[0, 0].get_legend_handles_labels()
    permute = itemgetter(0, 3, 6, 
                         1, 4, 7, 
                         2, 5, 8)
    fig.legend(
        permute(lines),
        permute(labels),
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0, 1, 1, 0.22),
        frameon=False,
    )

    fig.tight_layout(pad=0.25)
    fig.savefig(OUTDIR / outfile, bbox_inches="tight")
    print("Written %s" % (OUTDIR / outfile))
    print(fig.get_size_inches())
    return fig


plot_combined_PE("combined_PE.pdf")



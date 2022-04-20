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

get_cmap("tab20c")
# TODO: consider tab20b as well
# https://matplotlib.org/stable/gallery/color/colormap_reference.html

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


plt = rsmf.setup(projectdir("data", "plots", "quantum-template.tex"))
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#7EC4F9", "#FFB97A", "#6EEF6E"] # TODO remove

OUTDIR = projectdir("data", "plots")
OUTDIR.mkdir(parents=True, exist_ok=True)

BENCHMARKS = projectdir("data", "benchmarks")


class Benchmark:
    def __init__(
        self,
        filename,
        column_name,
        color,
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
        self.color = color
        marker_style = {}
        if isinstance(marker, str):
            marker_style = dict(marker=marker)
        elif isinstance(marker, dict):
            marker_style = marker
        self.marker_style = marker_style
        print("loading", BENCHMARKS / filename)
        df = pd.read_csv(BENCHMARKS / filename, index_col=0)
        self.data = df[column_name].dropna()


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
    inset_index=[0, 1],
):

    fig = None
    if ax is None:
        fig = plt.figure(aspect_ratio=0.5, wide=legend)
        ax = fig.add_subplot()
    ax.grid(color="gray", alpha=0.25)

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
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    if inset:
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


# TODO: remove
level_files_PE = [
    "PE_benchmark_levels.csv",
    "PE_benchmark_levels_full_ad_cheby.csv",
    "PE_benchmark_levels_full_ad.csv",
]
time_files_PE = [
    "PE_benchmark_times.csv",
    "PE_benchmark_times_full_ad_cheby.csv",
    "PE_benchmark_times_full_ad.csv",
]
level_files_SM = [
    "SM_benchmark_levels.csv",
    "SM_FullADcheby_benchmark_levels.csv",
    "SM_FullAD_benchmark_levels.csv",
]
time_files_SM = [
    "SM_benchmark_times.csv",
    "SM_FullADcheby_benchmark_times.csv",
    "SM_FullAD_benchmark_times.csv",
]
level_files_C = [
    "C_benchmark_levels_semi_ad.csv",
    "C_benchmark_levels_full_ad_cheby.csv",
    "C_benchmark_levels_full_ad.csv",
]
time_files_C = [
    "C_benchmark_times_semi_ad.csv",
    "C_benchmark_times_full_ad_cheby.csv",
    "C_benchmark_times_full_ad.csv",
]
methods = ["Semi-AD", "Chebyshev", "ODE"]

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

# ## Runtime

plot_comparison(
    Benchmark(
        "PE_benchmark_levels.csv",
        "nanosec_per_fg",
        in_inset=True,
        color=color("blue", 1),
        marker=MARKER["fullcircle"],
    ),
    Benchmark(
        "C_benchmark_levels_semi_ad.csv",
        "nanosec_per_fg",
        in_inset=True,
        color=color("blue", 2),
        marker=MARKER["halfcircle"],
    ),
    Benchmark(
        "SM_benchmark_levels.csv",
        "nanosec_per_fg",
        in_inset=True,
        color=color("blue", 3),
        marker=MARKER["emptycircle"],
    ),
    Benchmark(
        "PE_benchmark_levels_full_ad_cheby.csv",
        "nanosec_per_fg",
        in_inset=True,
        color=color("red", 1),
        marker=MARKER["fullsquare"],
    ),
    Benchmark(
        "C_benchmark_levels_full_ad_cheby.csv",
        "nanosec_per_fg",
        in_inset=True,
        color=color("red", 2),
        marker=MARKER["halfsquare"],
    ),
    Benchmark(
        "SM_FullADcheby_benchmark_levels.csv",
        "nanosec_per_fg",
        in_inset=True,
        color=color("red", 3),
        marker=MARKER["emptysquare"],
    ),
    Benchmark(
        "PE_benchmark_levels_full_ad.csv",
        "nanosec_per_fg",
        color=color("green", 1),
        marker=MARKER["fulldiamond"],
    ),
    Benchmark(
        "C_benchmark_levels_full_ad.csv",
        "nanosec_per_fg",
        color=color("green", 2),
        marker=MARKER["halfdiamond"],
    ),
    Benchmark(
        "SM_FullAD_benchmark_levels.csv",
        "nanosec_per_fg",
        in_inset=False,
        color=color("green", 3),
        marker=MARKER["emptydiamond"],
    ),
    logscale=False,
    ylabel="runtime (s)",
    xlabel="number of transmon levels",
    factor=1e-9,
    outfile="PE_runtimes_levels.pdf",
    inset=True,
)

plot_comparison(
    time_files_PE,
    time_files_C,
    methods,
    ["PE", "C"],
    index=1,
    logscale=False,
    ylabel="runtime (s)",
    xlabel="gate duration",
    factor=1e-9,
    outfile="PE_runtimes_times.pdf",
    inset=True,
    inset_pos=[0.1, 0.55, 0.35, 0.35],
)

plot_comparison(
    ["C_benchmark_levels_semi_ad.csv"],
    ["C_U_benchmark_levels_semi_ad.csv"],
    methods,
    ["C", "C_U"],
    index=1,
    logscale=False,
    ylabel="runtime (s)",
    xlabel="number of transmon levels",
    outfile="C_U_runtime_levels.pdf",
)

plot_comparison(
    ["C_benchmark_times_semi_ad.csv"],
    ["C_U_benchmark_times_semi_ad.csv"],
    methods,
    ["C", "C_U"],
    index=1,
    logscale=False,
    ylabel="runtime (s)",
    xlabel="gate duration",
    outfile="C_U_runtime_times.pdf",
)

plot_comparison(
    ["PE_benchmark_levels.csv"],
    ["PE_U_benchmark_levels_semi_ad.csv"],
    methods,
    ["PE", "PE_U"],
    index=1,
    logscale=False,
    ylabel="runtime (s)",
    xlabel="number of transmon levels",
    outfile="PE_U_runtime_levels.pdf",
)

plot_comparison(
    ["PE_benchmark_times.csv"],
    ["PE_U_benchmark_times_semi_ad.csv"],
    methods,
    ["PE", "PE_U"],
    index=1,
    logscale=False,
    ylabel="runtime (s)",
    xlabel="gate duration",
    outfile="PE_U_runtime_times.pdf.pdf",
)

# ## RSS

plot_comparison(
    level_files_PE,
    level_files_C,
    methods,
    ["PE", "C"],
    index=3,
    logscale=False,
    ylabel="memory (MB)",
    xlabel="number of transmon levels",
    outfile="PE_memory_levels.pdf",
    inset=True,
    inset_pos=[0.12, 0.65, 0.35, 0.3],
    inset_index=[0],
)

plot_comparison(
    time_files_PE,
    time_files_C,
    methods,
    ["PE", "C"],
    index=3,
    logscale=False,
    ylabel="memory (MB)",
    xlabel="gate duration",
    outfile="PE_memory_times.pdf",
    inset=True,
    inset_index=[0],
    inset_pos=[0.12, 0.63, 0.35, 0.3],
)

plot_comparison(
    ["C_benchmark_levels_semi_ad.csv"],
    ["C_U_benchmark_levels_semi_ad.csv"],
    methods,
    ["C", "C_U"],
    index=3,
    logscale=False,
    ylabel="memory (MB)",
    xlabel="number of transmon levels",
    outfile="C_U_memory_levels.pdf",
)

plot_comparison(
    ["C_benchmark_times_semi_ad.csv"],
    ["C_U_benchmark_times_semi_ad.csv"],
    methods,
    ["C", "C_U"],
    index=3,
    logscale=False,
    ylabel="memory (MB)",
    xlabel="gate duration",
    outfile="C_U_memory_times.pdf",
)

plot_comparison(
    ["PE_benchmark_levels.csv"],
    ["PE_U_benchmark_levels_semi_ad.csv"],
    methods,
    ["PE", "PE_U"],
    index=3,
    logscale=False,
    ylabel="memory (MB)",
    xlabel="number of transmon levels",
    outfile="PE_U_memory_levels.pdf",
)

plot_comparison(
    ["PE_benchmark_times.csv"],
    ["PE_U_benchmark_times_semi_ad.csv"],
    methods,
    ["PE", "PE_U"],
    index=3,
    logscale=False,
    ylabel="memory (MB)",
    xlabel="gate duration",
    outfile="PE_U_memory_times.pdf",
)

# ## Allocations

plot_comparison(
    level_files_PE,
    level_files_C,
    methods,
    ["PE", "C"],
    index=2,
    logscale=False,
    ylabel="allocated memory (TB)",
    xlabel="number of transmon levels",
    outfile="PE_allocated_levels.pdf",
    factor=1e-6,
    inset=True,
    inset_pos=[0.5, 0.65, 0.35, 0.3],
    inset_index=[0],
)

plot_comparison(
    time_files_PE,
    time_files_C,
    methods,
    ["PE", "C"],
    index=2,
    logscale=False,
    ylabel="allocated memory (TB)",
    factor=1e-6,
    xlabel="gate duration",
    outfile="PE_allocated_times.pdf",
    inset=True,
    inset_index=[0],
    inset_pos=[0.18, 0.63, 0.35, 0.3],
)

plot_comparison(
    ["C_benchmark_levels_semi_ad.csv"],
    ["C_U_benchmark_levels_semi_ad.csv"],
    methods,
    ["C", "C_U"],
    index=2,
    logscale=False,
    ylabel="allocated memory (MB)",
    xlabel="number of transmon levels",
    outfile="C_U_allocated_levels.pdf",
)

plot_comparison(
    ["C_benchmark_times_semi_ad.csv"],
    ["C_U_benchmark_times_semi_ad.csv"],
    methods,
    ["C", "C_U"],
    index=2,
    logscale=False,
    ylabel="allocated memory (MB)",
    xlabel="gate duration",
    outfile="C_U_allocated_times.pdf",
)

plot_comparison(
    ["PE_benchmark_levels.csv"],
    ["PE_U_benchmark_levels_semi_ad.csv"],
    methods,
    ["PE", "PE_U"],
    index=2,
    logscale=False,
    ylabel="allocated memory (MB)",
    xlabel="number of transmon levels",
    outfile="PE_U_allocated_levels.pdf",
)

plot_comparison(
    ["PE_benchmark_times.csv"],
    ["PE_U_benchmark_times_semi_ad.csv"],
    methods,
    ["PE", "PE_U"],
    index=2,
    logscale=False,
    ylabel="allocated memory (MB)",
    xlabel="gate duration",
    outfile="PE_U_allocated_times.pdf",
)

# ## Combined Plots


def plot_combined_PE(outfile):

    fig = plt.figure(wide=True, aspect_ratio=0.5)
    axs = fig.subplots(nrows=2, ncols=2, sharex="col", sharey="row")

    plot_comparison(
        level_files_PE,
        level_files_C,
        methods,
        ["PE", "C"],
        index=1,
        logscale=False,
        ylabel="runtime (s)",
        factor=1e-9,
        outfile="PE_runtimes_levels.pdf",
        inset=True,
        ax=axs[0, 0],
        legend=False,
    )

    plot_comparison(
        time_files_PE,
        time_files_C,
        methods,
        ["PE", "C"],
        index=1,
        logscale=False,
        factor=1e-9,
        outfile="PE_runtimes_times.pdf",
        inset=True,
        inset_pos=[0.1, 0.55, 0.35, 0.35],
        ax=axs[0, 1],
        legend=False,
    )

    plot_comparison(
        level_files_PE,
        level_files_C,
        methods,
        ["PE", "C"],
        index=3,
        logscale=False,
        ylabel="memory (MB)",
        xlabel="number of transmon levels",
        outfile="PE_memory_levels.pdf",
        inset=True,
        inset_pos=[0.12, 0.65, 0.35, 0.3],
        inset_index=[0],
        ax=axs[1, 0],
        legend=False,
    )

    plot_comparison(
        time_files_PE,
        time_files_C,
        methods,
        ["PE", "C"],
        index=3,
        logscale=False,
        xlabel="gate duration",
        outfile="PE_memory_times.pdf",
        inset=True,
        inset_index=[0],
        inset_pos=[0.12, 0.63, 0.35, 0.3],
        ax=axs[1, 1],
        legend=False,
    )

    # https://stackoverflow.com/questions/9834452
    lines, labels = axs[0, 0].get_legend_handles_labels()
    permute = itemgetter(0, 3, 1, 4, 2, 5)
    fig.legend(
        permute(lines),
        permute(labels),
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0, 1, 1, 0.15),
        frameon=False,
    )

    fig.tight_layout(pad=0.25)
    fig.savefig(OUTDIR / outfile, bbox_inches="tight")
    print("Written %s" % (OUTDIR / outfile))
    print(fig.get_size_inches())
    return fig


plot_combined_PE("combined_PE.pdf")

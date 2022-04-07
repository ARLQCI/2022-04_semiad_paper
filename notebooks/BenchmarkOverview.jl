# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Julia 1.7.2
#     language: julia
#     name: julia-1.7
# ---

using DrWatson

@quickactivate

# +
using Plots

Plots.default(
    linewidth               = 3,
    size                    = (950, 700),
    legend                  = :topleft,
    foreground_color_legend = nothing,
    background_color_legend = RGBA(1, 1, 1, 0.8),
)
# -

using CSV

# +
function expected_storage(T; N=5)
    bytes_per_complex_number = 16
    steps_per_T = 10
    bytes_per_MB = 1e6
    N_total = N^2 # two transmons
    n_targets = 4
    return n_targets * T * steps_per_T * N_total / bytes_per_MB
end

expected_storage(800)
# -

function show_benchmarks(csv_times, csv_levels; title="")

    data1 = CSV.File(csv_times)
    ax11 = plot(data1.T, data1.nanosec_per_fg * 1e-9, label="time per grad eval")
    plot!(ax11; ylabel="runtime (s)", xlabel="gate duration")
    ax12 = plot(data1.T, data1.rss_memory_MB, label="RSS")
    plot!(ax12, data1.T, expected_storage.(data1.T), label="storage")
    plot!(ax12, ylabel="memory (MB)", xlabel="gate duration")
    ax13 = plot(data1.T, data1.alloc_memory_MB, label="allocated")
    plot!(ax13; xlabel="gate duration")

    data2 = CSV.File(csv_levels)
    ax21 = plot(data2.levels, data2.nanosec_per_fg * 1e-9, label="time per grad eval")
    plot!(ax21; ylabel="runtime (s)", xlabel="number of transmon levels")
    ax22 = plot(data2.levels, data2.rss_memory_MB, label="RSS")
    plot!(ax22, data2.levels, expected_storage.(data2.levels), label="storage")
    plot!(ax22, ylabel="memory (MB)", xlabel="number of transmon levels",)
    ax23 = plot(data2.levels, data2.alloc_memory_MB, label="allocated")
    plot!(ax23; xlabel="number of transmon levels")

    plot(ax11, ax12, ax13, ax21, ax22, ax23, layout=6; plot_title=title)
end


# ## PE Optimization

show_benchmarks(
    datadir("benchmarks", "PE_benchmark_times.csv"),
    datadir("benchmarks", "PE_benchmark_levels.csv");
    title="Semi-AD of PE functional"
)

show_benchmarks(
    datadir("benchmarks", "PE_benchmark_times_full_ad.csv"),
    datadir("benchmarks", "PE_benchmark_levels_full_ad.csv");
    title="Full-AD of PE functional"
)

# ## Gate Optimization (sqrt_iSWAP)

show_benchmarks(
    datadir("benchmarks", "SM_benchmark_times.csv"),
    datadir("benchmarks", "SM_benchmark_levels.csv");
    title="Direct Optimization of sqrt_iSWAP"
)

show_benchmarks(
    datadir("benchmarks", "SM_SemiAD_benchmark_times.csv"),
    datadir("benchmarks", "SM_SemiAD_benchmark_levels.csv");
    title="Semi-AD Optimization of sqrt_iSWAP"
)

show_benchmarks(
    datadir("benchmarks", "SM_FullAD_benchmark_times.csv"),
    datadir("benchmarks", "SM_FullAD_benchmark_levels.csv");
    title="Full-AD of sqrt_iSWAP"
)

show_benchmarks(
    datadir("benchmarks", "SM_FullADcheby_benchmark_times.csv"),
    datadir("benchmarks", "SM_FullADcheby_benchmark_levels.csv");
    title="Full-AD of sqrt_iSWAP with cheby prop"
)

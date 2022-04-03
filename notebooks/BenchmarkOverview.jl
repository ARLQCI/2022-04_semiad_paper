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
    size                    = (550, 300),
    legend                  = :topleft,
    foreground_color_legend = nothing,
    background_color_legend = RGBA(1, 1, 1, 0.8),
)
# -

using CSV

RSS_BASELINE_MB = 713

# +
function expected_storage(T; N=5)
    bytes_per_complex_number = 16
    steps_per_T = 10
    bytes_per_MB = 1e6
    N_total = N^2 # two transmons
    return T * steps_per_T * N_total / bytes_per_MB
end

expected_storage(800)
# -

data = CSV.File(datadir("benchmarks", "benchmark_PE_benchmark_times.csv"));
display(plot(data.T, data.nanosec_per_fg*1e-9, ylabel="runtime (s)", xlabel="gate duration", label="time per grad eval", title="J_T_PE (levels per transmon N=5)"));
ax_mem = plot(data.T, data.alloc_memory_MB, ylabel="memory (MB)", xlabel="gate duration", label="allocated");
plot!(ax_mem, data.T, data.rss_memory_MB .- RSS_BASELINE_MB, ylabel="memory (MB)", xlabel="gate duration", label="RSS");
plot!(ax_mem, data.T, expected_storage.(data.T), label="storage")
display(ax_mem)

using DrWatson
@quickactivate "SemiADPaper"

using Printf
using BenchmarkTools
using SemiADPaper: run_optimization

"""Get a command line option."""
function get_option(args, option, default=nothing)
    option_prefix = option * "="
    for option in args
        if startswith(option, option_prefix)
            key, value = split(option, "="; limit=2)
            return value
        end
    end
    return default
end

"""Run a benchmark for a single optimization.

USAGE: julia scripts/benchmark_optimization.jl [options]  FUNCTIONAL LEVELS T

Options are `--method=NAME`, `--unitarity-weight=VALUE`, `--iters=NUM`,
`--use-threads`.

See documentation of `run_optimization` function for details.

Further options are:

--force  Run the benchmark even if the output file exists.

"""
function main(args=ARGS)
    local functional, levels, T, method, unitarity_weight, use_threads, force
    try
        functional = Symbol(ARGS[end-2])
        levels = parse(Int64, ARGS[end-1])
        T = parse(Float64, ARGS[end])
        method = Symbol(get_option(args, "--method", "grape"))
        unitarity_weight = parse(Float64, get_option(args, "--unitarity-weight", "0.5"))
        iters = parse(Int64, get_option(args, "--iters", "10"))
        use_threads = "--use-threads" ∈ args
        force = "--force" ∈ args
    catch exc
        println("ERROR: $exc\n")
        println(@doc main)
        exit(1)
    end
    filename = @sprintf("%s_%s_levels=%d_T=%d.jld2", method, functional, levels, T)
    c = Dict{Symbol,Any}()  # dummy config
    produce_or_load(datadir("benchmarks"), c; filename=filename, force=force) do c
        #! format: off
        opt_result = run_optimization(;
            method, levels, functional, unitarity_weight, T, use_threads, iters
        )
        println("Benchmarking...")
        benchmark_result = @benchmark run_optimization(;
            method=$method, levels=$levels, functional=$functional,
            unitarity_weight=$unitarity_weight, T=$T, use_threads=$use_threads,
            iters=$iters, quiet=true,
        )
        #! format: on
        display(benchmark_result)
        fg = opt_result.fg_calls
        nanosec_per_fg = minimum(benchmark_result.times) / fg
        alloc_memory_MB = benchmark_result.memory / 1e6
        data = @dict levels functional T benchmark_result fg nanosec_per_fg alloc_memory_MB
        @tag! data
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

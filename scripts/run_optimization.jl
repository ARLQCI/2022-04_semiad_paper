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

"""Process-wrapper around `run_optimiation` function.

USAGE: julia --project=. scripts/run_optimization.jl [options]  FUNCTIONAL LEVELS T

Options are `--method=NAME`, `--unitarity-weight=VALUE`, `--iters=NUM`,
`--use-threads`

See documentation of `run_optimization` function for details.

This script is a direct wrapper around the `run_optimization` function,
allowing it to be benchmarked at a process level.
"""
function main(args=ARGS)
    local functional, levels, T, method, unitarity_weight, use_threads, iters
    try
        functional = Symbol(ARGS[end-2])
        levels = parse(Int64, ARGS[end-1])
        T = parse(Float64, ARGS[end])
        method = Symbol(get_option(args, "--method", "grape"))
        unitarity_weight = parse(Float64, get_option(args, "--unitarity-weight", "0.5"))
        iters = parse(Int64, get_option(args, "--iters", "10"))
        use_threads = "--use-threads" ∈ args
    catch exc
        println("ERROR: $exc\n")
        println(@doc main)
        exit(1)
    end
    opt_result =
        run_optimization(; method, levels, functional, unitarity_weight, T, iters, use_threads)
    println("DONE")
    return opt_result
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end

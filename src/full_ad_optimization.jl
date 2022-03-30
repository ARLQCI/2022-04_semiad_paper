module FullADOptimization

export optimize_full_ad

using LinearAlgebra
using Printf
using QuantumControl.GRAPE:
    print_table, GrapeWrk, get_optimizer, run_optimizer, finalize_result!
using DiffEqSensitivity
using OrdinaryDiffEq: ODEProblem, solve, remake, DP5
using Zygote

# TODO: FullADWrk

mutable struct ODEWrk{T,FT} ## not in-place version
    prob::ODEProblem
    function ODEWrk(Ψ::T, dt::FT, H; alg=DP5()) where {T,FT}
        tspan = (0.0, dt)
        function f(Ψ, H, t)
            return -1im * H * Ψ
        end
        prob = ODEProblem(f, Ψ, tspan, H)
        new{T,FT}(prob)
    end
end

function ODE(Ψ, H, dt, wrk) ## not in-place version
    tspan = (0.0, dt)
    _prob = remake(wrk.prob, u0=Ψ, p=H, tspan=tspan)
    sol = solve(_prob, DP5(), save_everystep=false, save_start=false)
    sol.u[1]
end

function propagate(state, generator, tlist, pulsevals, wrk)
    state = copy(state)
    intervals = enumerate(tlist[2:end])
    N = length(intervals)

    for (i, t_end) in intervals
        dt = t_end - tlist[i]
        nc = length(generator) - 1
        H = generator[1]
        for (j, part) in enumerate(generator[2:end])
            if isa(part, Tuple)
                H += pulsevals[j+nc*(i-1)] * part[1]
            else
                H += part
            end
        end

        state = ODE(state, H, dt, wrk)
    end

    return state
end

function optimize_full_ad(problem)
    update_hook! = get(problem.kwargs, :update_hook, (args...) -> nothing)
    # TODO: implement update_hook
    # TODO: streamline the interface for info_hook
    # TODO: check if x_tol, f_tol, g_tol are used necessary / used correctly

    info_hook = get(problem.kwargs, :info_hook, print_table)
    check_convergence! = get(problem.kwargs, :check_convergence, res -> res)

    wrk = GrapeWrk(problem)
    fw_prop_wrk =
        ODEWrk(wrk.objectives[1].initial_state, problem.tlist[2] - problem.tlist[1], wrk.G)

    Ψ₀ = [obj.initial_state for obj ∈ wrk.objectives]

    J_T_func = wrk.kwargs[:J_T]

    function f(F, G, pulsevals; storage=nothing, count_call=true)
        @assert !isnothing(F)
        @assert isnothing(G)
        if count_call
            wrk.result.f_calls += 1
            wrk.fg_count[2] += 1
        end
        Ψ_out = [
            propagate(Ψ₀[k], obj.generator, problem.tlist, pulsevals, fw_prop_wrk) for
            (k, obj) in enumerate(wrk.objectives)
        ]
        return J_T_func(Ψ_out, wrk.objectives)
    end

    function fg!(F, G, pulsevals)

        if isnothing(G)  # functional only
            return f(F, G, pulsevals)
        end
        wrk.result.fg_calls += 1
        wrk.fg_count[1] += 1

        J_T_val, ∇ = Zygote.withgradient(
            pulsevals -> begin
                Ψ_out = [
                    propagate(
                        obj.initial_state,
                        obj.generator,
                        problem.tlist,
                        pulsevals,
                        fw_prop_wrk
                    ) for (k, obj) in enumerate(wrk.objectives)
                ]
                J_T_func(Ψ_out, wrk.objectives)
            end,
            pulsevals
        )

        copy!(G, ∇[1])

        return J_T_val

    end

    optimizer = get_optimizer(wrk)
    res = run_optimizer(optimizer, wrk, fg!, info_hook, check_convergence!)
    finalize_result!(wrk, res)
    return wrk.result

end


end

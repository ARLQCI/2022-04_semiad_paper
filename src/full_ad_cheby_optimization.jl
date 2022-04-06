module FullADChebyOptimization

export optimize_full_ad_cheby

using LinearAlgebra
using Printf
using QuantumControl.GRAPE:
    print_table, GrapeWrk, get_optimizer, run_optimizer, finalize_result!
using Zygote

function cheby(Ψ, H, dt, wrk; kwargs...)

    E_min = get(kwargs, :E_min, nothing)
    check_normalization = get(kwargs, :check_normalization, false)

    Δ = wrk.Δ
    β::typeof(wrk.E_min) = (Δ / 2) + wrk.E_min  # "normfactor"
    if E_min ≠ nothing
        β = (Δ / 2) + E_min
    end
    @assert dt ≈ wrk.dt "wrk was initialized for dt=$(wrk.dt), not dt=$dt"
    if dt > 0
        c = -2im / Δ
    else
        c = 2im / Δ
    end
    a = wrk.coeffs
    ϵ = wrk.limit
    @assert length(a) > 1 "Need at least 2 Chebychev coefficients"

    v0 = Ψ
    Φ = a[1] * v0

    v1 = c * (H * v0 - β * v0)
    Φ += a[2] * v1

    c *= 2

    for i = 3:wrk.n_coeffs

        v2 = H * v1
        v2 += -v1 * β
        v2 = c * v2
        if check_normalization
            map_norm = abs(dot(v1, v2)) / (2 * norm(v1)^2)
            @assert(map_norm <= (1.0 + ϵ), "Incorrect normalization (E_min, wrk.Δ)")
        end
        v2 += v0

        Φ += a[i] * v2

        aux = 1*v0
        v0 = 1*v1
        v1 = 1*v2
        v2 = 1*aux

        # Doesn't work with Zygote (bug)
        #v0, v1, v2 = v1, v2, v0  # switch w/o copying

    end

    return exp(-im * β * dt) * Φ

end


function propagate(state0, generator, tlist, pulsevals, wrk)
    state = copy(state0)
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

        state = cheby(state, H, dt, wrk)
    end

    return state
end


function optimize_full_ad_cheby(problem)
    update_hook! = get(problem.kwargs, :update_hook, (args...) -> nothing)
    # TODO: implement update_hook
    # TODO: streamline the interface for info_hook
    # TODO: check if x_tol, f_tol, g_tol are used necessary / used correctly

    info_hook = get(problem.kwargs, :info_hook, print_table)
    check_convergence! = get(problem.kwargs, :check_convergence, res -> res)

    wrk = GrapeWrk(problem)
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
            propagate(Ψ₀[k], obj.generator, problem.tlist, pulsevals, wrk.fw_prop_wrk[k]) for
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
                        Ψ₀[k],
                        obj.generator,
                        problem.tlist,
                        pulsevals,
                        wrk.fw_prop_wrk[k]
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

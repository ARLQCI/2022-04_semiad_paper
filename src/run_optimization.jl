using QuantumControl
using QuantumControl.Shapes: flattop
using QuantumControl.Functionals: J_T_sm, make_gradient, make_chi, make_gate_chi, gate_functional
using QuantumControlBase.Functionals: grad_J_T_sm!, chi_sm!
using QuantumControl.WeylChamber: D_PE, gate_concurrence, unitarity

using .TransmonModel: guess_pulses, hamiltonian, ket
using .GHzUnits
using .FullADOptimization

const 𝕚 = 1im
const sqrt_iSWAP = [
    1    0     0   0
    0  1/√2  𝕚/√2  0
    0  𝕚/√2  1/√2  0
    0    0     0   1
]

optimize(problem; method=method::Val{:full_ad}) = optimize_full_ad(problem)


"""Run an optimization for a fixed number of iterations.

```julia
run_optimization(;
    method, iters, unitarity_weight, functional, levels, T, target,
    use_threads)
````

# Keyword Arguments

*   `method`: one of `:grape`, `:krotov`, `:full_ad`

*   `iters=10`: Number of iterations to do

*   `unitarity_weight=0.5`: Weight for a unitary term in PE/concurrence
   functionals. Must be a number ∈ [0, 1)

*   `functional`: The functional to optimize. The following options are available:

    - `:J_T_PE`: Optimize for a perfect entangler in the Weyl chamber, with the
      gradient evaluated via `|χₖ⟩`, and `|χₖ⟩` obtained via AD from
      `J_T=J_T({|Ψₖ⟩})` (from the forward-propagated states).
    - `:J_T_PE_U`: Like :`J_T_PE`, but with |χₖ⟩` obtained via AD from
      `J_T=J_T(Û)` (from the "gate", i.e., the time evolution operator
      projected into the logical subspace.)
    - `:J_T_C`: Optimize for a perfect entangler by directly maximizing the
      gate concurrence. The gradient/chi is obtained like for `:J_T_PE` (via
      the states)
    - `:J_T_C_U`: Like `:J_T_C`, but with the gradient/chi obtained like for
      `:J_T_PE_U` (via the gate)
    - `:J_T_sm`: Standard gate optimization with the square-modulus function,
      using an analytic gradient/chi
    - `:J_T_sm_AD`: Like `J_T_sm`, but force the use of AD to calculate the
      gradient/chi via τₖ (the overlap of propagated states with target states)

*   `levels`: The number of levels in each transmon

*   `T`: The gate duration. Proportional to the number of time steps (dt=0.1 is
    fixed).

*   `use_threads=false`: Whether to run the optimization in multi-threaded
    mode. This also requires the `JULIA_NUM_THREADS` environment variable to be
    set.

*   `quiet=false`: Whether or not to print the standard convergence table

"""
function run_optimization(;
    model=:reduced,  # the only option for now
    method=:grape,
    iters=10,
    unitarity_weight=0.5,
    functional,
    levels,
    T,
    use_threads=false,
    quiet=false,
)

    tlist, Ωre_guess, Ωim_guess = guess_pulses(T=T)
    update_shape = t -> flattop(t, T=T, t_rise=max(T / 2, 10ns), func=:blackman)
    p_opt = Dict(:lambda_a => 10.0, :update_shape => update_shape,)

    H = hamiltonian(Ωre=Ωre_guess, Ωim=Ωim_guess, N=levels)
    basis = [ket(lbl; N=levels) for lbl ∈ ("00", "01", "10", "11")]
    targets = sqrt_iSWAP' * basis
    objectives = [
        Objective(; initial_state=Ψ, target_state=Ψtgt, generator=H) for
        (Ψ, Ψtgt) ∈ zip(basis, targets)
    ]
    # Note that the target-states are ignored for everything but a direct gate
    # optimization (functional=:J_T_sm, functional=:J_T_sm_AD)


    if functional ∈ [:J_T_PE, :J_T_PE_U]
        J_T_U = U -> D_PE(U; unitarity_weight=unitarity_weight)
        J_T = gate_functional(J_T_U)
        gradient_via = :chi
        gradient = make_gradient(J_T, objectives; via=gradient_via, force_zygote=true)
        if functional ≡ :J_T_PE
            chi = make_chi(J_T, objectives; force_zygote=true)
        else # functional ≡ :J_T_PE_U
            chi = make_gate_chi(J_T_U, objectives)
        end
    elseif functional ∈ [:J_T_C, :J_T_C_U]
        J_T_U =
            U ->
                (1 - unitarity_weight) * (1 - gate_concurrence(U)) +
                unitarity_weight * (1 - unitarity(U))
        J_T = gate_functional(J_T_U)
        gradient_via = :chi
        gradient = make_gradient(J_T, objectives; via=gradient_via, force_zygote=true)
        if functional ≡ :J_T_C
            chi = make_chi(J_T, objectives; force_zygote=true)
        else # functional ≡ :J_T_C_U
            chi = make_gate_chi(J_T_U, objectives)
        end
    elseif functional ∈ [:J_T_sm, :J_T_sm_AD]
        J_T = J_T_sm
        gradient_via = :tau
        gradient = grad_J_T_sm!
        chi = chi_sm!
        if functional ≡ :J_T_sm_AD
            gradient = make_gradient(J_T, objectives; via=gradient_via, force_zygote=true)
            chi = make_chi(J_T, objectives; force_zygote=true)
        end
    else
        throw(ArgumentError("Invalid functional $(repr(functional))"))
    end

    problem = ControlProblem(
        objectives=objectives,
        pulse_options=IdDict(Ωre_guess => p_opt, Ωim_guess => p_opt,),
        tlist=tlist,
        iter_stop=iters,
        J_T=J_T,
        gradient=gradient,
        gradient_via=gradient_via,
        chi=chi,
        use_threads=use_threads,
    )

    no_info_hook(args...; kwargs...) = nothing
        
    if quiet
        optimize(problem; method=method, info_hook=no_info_hook)
    else
        optimize(problem; method=method)
    end

end

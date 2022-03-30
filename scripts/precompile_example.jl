using DrWatson
@quickactivate "SemiADPaper"

using SemiADPaper: run_optimization
using SemiADPaper.GHzUnits

for functional in [:J_T_PE, :J_T_PE_U, :J_T_C, :J_T_C_U, :J_T_sm, :J_T_sm_AD]
    @show functional
    println("levels=3")  # small system: specrad via diagonalization
    run_optimization(; iters=1, functional, levels=3, T=100ns)
    println("levels=10")  # small system: specrad via arnoldi
    run_optimization(; iters=1, functional, levels=10, T=100ns)
    println("")
end

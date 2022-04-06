using DrWatson
@quickactivate "SemiADPaper"

using SemiADPaper: run_optimization
using SemiADPaper.GHzUnits

#! format: off
COMBINATIONS = [
    # method         functional   levels
    (:grape,         :J_T_PE,     3),     # small system: specrad via diagonalization
    (:grape,         :J_T_PE_U,   3),
    (:grape,         :J_T_C,      3),
    (:grape,         :J_T_C_U,    3),
    (:grape,         :J_T_sm,     3),
    (:grape,         :J_T_sm_AD,  3),
    (:grape,         :J_T_PE,    10),     # large system: specrad via arnoldi
    (:grape,         :J_T_PE_U,  10),
    (:grape,         :J_T_C,     10),
    (:grape,         :J_T_C_U,   10),
    (:grape,         :J_T_sm,    10),
    (:grape,         :J_T_sm_AD, 10),
    (:full_ad,       :J_T_PE,     3),
    (:full_ad,       :J_T_C,      3),
    (:full_ad,       :J_T_sm,     3),
    (:full_ad_cheby, :J_T_PE,     3),
    (:full_ad_cheby, :J_T_C,      3),
    (:full_ad_cheby, :J_T_sm,     3),
]
#! format: on

for (method, functional, levels) in COMBINATIONS
    @show (method, functional, levels)
    run_optimization(; method=method, iters=1, functional, levels=3, T=100ns)
    println("")
end

# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Julia 1.7 (4 threads)
#     language: julia
#     name: julia-1.7-4threads
# ---

# # Number of Chebychev Coeffs

# The runtime and memory requirements for the time propagation with Chebychev depends directly on the number of coefficients in the expansion, and the number of Chebychev coefficients depends on the spectral radius of the Hamiltonian, which increases with the number of levels included in the transmon Hamiltonian.
#
# We plot this dependency below in order to correlate it with the benchmark results.

using SemiADPaper.TransmonModel
using SemiADPaper.GHzUnits

using QuantumControlBase: initobjpropwrk
using QuantumControl: Objective
using QuantumPropagators.Cheby: cheby_coeffs
using QuantumPropagators.SpectralRange: specrange

using Plots

T = 100ns;
dt = 0.1ns;

tlist, Ωre_guess, Ωim_guess = guess_pulses(T=T);

function get_n_cheby_coeffs(N)
    H = hamiltonian(Ωre=Ωre_guess, Ωim=Ωim_guess, N=N)
    ket00 = ket("00"; N=N)
    obj = Objective(; initial_state=ket00, target_state=ket00, generator=H)
    wrk = initobjpropwrk(obj, tlist, :cheby; specrad_method=:diag)
    return length(wrk.coeffs)
end

get_n_cheby_coeffs(9)

N = collect(range(3, 15, step=1))

plot(
    N, get_n_cheby_coeffs.(N),
    xlabel="number of transmon levels",
    ylabel="number of Chebychev coeffs",
    legend=:topleft
)

plot(
    N.^2, get_n_cheby_coeffs.(N),
    xlabel="size of Hilbert space",
    ylabel="number of Chebychev coeffs",
    legend=:topleft
)

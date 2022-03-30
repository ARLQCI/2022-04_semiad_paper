module TransmonModel

export hamiltonian, ket, guess_pulses

using QuantumControl: discretize
using QuantumControl.Shapes: flattop

using LinearAlgebra
using SparseArrays

using ..GHzUnits

const 𝕚 = 1im
⊗(A, B) = kron(A, B)


DEFAULT = Dict{Symbol,Any}(
    :N      => 10,  # levels per transmon
    :ω₁     => 4.380GHz,
    :ω₂     => 4.614GHz,
    :ωd     => 4.498GHz,
    :α₁     => -210MHz,
    :α₂     => -215MHz,
    :J      => -3MHz,
    :λ      => 1.03,
    :T      => 400ns,
    :t_rise => 10ns,
    :dt     => 0.1ns,
    :E₀     => 35MHz,
)


function set_defaults(; kwargs...)
    global DEFAULT
    for (sym, val) ∈ kwargs
        DEFAULT[sym] = val
    end
end


function hamiltonian(;
    Ωre,
    Ωim,
    N=DEFAULT[:N],  # levels per transmon
    ω₁=DEFAULT[:ω₁],
    ω₂=DEFAULT[:ω₂],
    ωd=DEFAULT[:ωd],
    α₁=DEFAULT[:α₁],
    α₂=DEFAULT[:α₂],
    J=DEFAULT[:J],
    λ=DEFAULT[:λ]
)
    𝟙 = SparseMatrixCSC{ComplexF64,Int64}(sparse(I, N, N))
    b̂₁ = spdiagm(1 => complex.(sqrt.(collect(1:N-1)))) ⊗ 𝟙
    b̂₂ = 𝟙 ⊗ spdiagm(1 => complex.(sqrt.(collect(1:N-1))))
    b̂₁⁺ = sparse(b̂₁')
    b̂₂⁺ = sparse(b̂₂')
    n̂₁ = sparse(b̂₁' * b̂₁)
    n̂₂ = sparse(b̂₂' * b̂₂)
    n̂₁² = sparse(n̂₁ * n̂₁)
    n̂₂² = sparse(n̂₂ * n̂₂)
    b̂₁⁺_b̂₂ = sparse(b̂₁' * b̂₂)
    b̂₁_b̂₂⁺ = sparse(b̂₁ * b̂₂')

    ω̃₁ = ω₁ - ωd
    ω̃₂ = ω₂ - ωd

    Ĥ₀ = sparse(
        (ω̃₁ - α₁ / 2) * n̂₁ +
        (α₁ / 2) * n̂₁² +
        (ω̃₂ - α₂ / 2) * n̂₂ +
        (α₂ / 2) * n̂₂² +
        J * (b̂₁⁺_b̂₂ + b̂₁_b̂₂⁺)
    )

    Ĥ₁re = (1 / 2) * (b̂₁ + b̂₁⁺ + λ * b̂₂ + λ * b̂₂⁺)
    Ĥ₁im = (𝕚 / 2) * (b̂₁⁺ - b̂₁ + λ * b̂₂⁺ - λ * b̂₂)

    H = (Ĥ₀, (Ĥ₁re, Ωre), (Ĥ₁im, Ωim))

end


function guess_pulses(;
    T=DEFAULT[:T],
    E₀=DEFAULT[:E₀],
    dt=DEFAULT[:dt],
    t_rise=DEFAULT[:t_rise]
)

    tlist = collect(range(0, T, step=dt))
    Ωre = t -> E₀ * flattop(t, T=T, t_rise=max(0.5 * T, t_rise))
    Ωim = t -> 0.0

    return tlist, Ωre, Ωim

end


function ket(i::Int64; N=DEFAULT[:N])
    Ψ = zeros(ComplexF64, N)
    Ψ[i+1] = 1
    return Ψ
end

ket(; N=N) = ket(0; N=N)


function ket(indices::Int64...; N=DEFAULT[:N])
    Ψ = ket(indices[1]; N=N)
    for i in indices[2:end]
        Ψ = Ψ ⊗ ket(i; N=N)
    end
    return Ψ
end


function ket(label::AbstractString; N=DEFAULT[:N])
    indices = [parse(Int64, digit) for digit in label]
    return ket(indices...; N=N)
end


end

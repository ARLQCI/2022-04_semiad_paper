module Visualization

export show_matrix, plot_spin_state

using Printf
using LinearAlgebra
using SparseArrays

using Plots

import ..eval as project_eval


function show_matrix(H; rows=:auto, cols=:auto, convert=abs, kwargs...)
    N, M = size(H)
    if cols == :auto
        xs = [string(i) for i = 1:N]
    else
        xs = cols
    end
    if rows == :auto
        ys = [string(i) for i = 1:M]
    else
        ys = rows
    end
    z = convert.(H)
    heatmap(xs, ys, z; aspect_ratio=1, yflip=true, kwargs...)
end

show_matrix(H::AbstractSparseMatrix; kwargs...) = show_matrix(Array(H); kwargs...)


"""Plot a complex pulse amplitude."""
function plot_complex_pulse(
    tlist,
    Ω;
    units=:GHzUnits,
    time_unit=:ns,
    ampl_unit=:MHz,
    kwargs...
)

    units_mod = project_eval(units)

    ax1 = plot(
        tlist ./ getfield(units_mod, time_unit),
        abs.(Ω) ./ getfield(units_mod, ampl_unit);
        label="|Ω|",
        xlabel="time ($time_unit)",
        ylabel="amplitude ($ampl_unit)",
        kwargs...
    )

    ax2 = plot(
        tlist ./ getfield(units_mod, time_unit),
        angle.(Ω) ./ π;
        label="ϕ(Ω)",
        xlabel="time ($time_unit)",
        ylabel="phase (π)"
    )

    plot(ax1, ax2, layout=(2, 1))

end


end

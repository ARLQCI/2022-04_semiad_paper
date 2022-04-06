module SemiADPaper

using DrWatson


# Submodules

include("GHz_units.jl")
include("transmon_model.jl")
include("visualization.jl")
include("full_ad_optimization.jl")
include("full_ad_cheby_optimization.jl")

###############################################################################

include("run_optimization.jl")
export run_optimization

end

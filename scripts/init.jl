using Pkg

isinstalled(pkg::String) =
    any(x -> x.name == pkg && x.is_direct_dep, values(Pkg.dependencies()))

ROOT_GLOBAL_PACKAGES = ["DrWatson", "Revise", "IJulia"]

if !isfile(joinpath(@__DIR__, "..", ".initialized"))
    Pkg.activate()  # activate root environment
    for pkg in ROOT_GLOBAL_PACKAGES
        if !isinstalled(pkg)
            @warn "$pkg is not installed in your root environment. Installing it for you..."
            Pkg.add(pkg)
        end
    end
end
Pkg.activate(dirname(@__DIR__)) # activate project environment
Pkg.instantiate()
touch(joinpath(@__DIR__, "..", ".initialized"))

using DrWatson
@quickactivate "SemiADPaper"

using Revise
using JuliaFormatter


"""Show help"""
help() = println(REPL_MESSAGE)


REPL_MESSAGE = """
*******************************************************************************
DEVELOPMENT REPL for $(projectname())

Path of active project: $(projectdir())

DrWatson, Revise, JuliaFormatter are active

* `help()` – Show this message
* `format("src")` – Apply source code formatting
*******************************************************************************
"""

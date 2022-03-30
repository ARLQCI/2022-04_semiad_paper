using PackageCompiler

using DrWatson
@quickactivate "SemiADPaper"

PackageCompiler.create_sysimage(;
    sysimage_path=projectdir("semiad_sysimage.so"),
    precompile_execution_file=scriptsdir("precompile_example.jl")
)

# Quantum Optimal Control via Semi-Automatic Differentiation

This repository contains scripts, notebooks and data for the paper [Quantum Optimal Control via Semi-Automatic Differentiation](https://arxiv.org/abs/2205.15044).

The methods described in the paper are implemented are implemented in [`GRAPE.jl`](https://github.com/JuliaQuantumControl/GRAPE.jl) and [`Krotov.jl`](https://github.com/JuliaQuantumControl/Krotov.jl), which are part of the [JuliaQuantumControl](https://github.com/JuliaQuantumControl#a-julia-framework-for-quantum-optimal-control) framework.


## Prerequisites

This repository assumes a Linux system with a basic development environment. The `make` command must be available.

You should have the following installed before initializing this project

* [Jupyter notebook][Jupyter]
* [Jupytext extension][jupytext]
* [Julia 1.7](https://julialang.org) ([old releases](https://julialang.org/downloads/oldreleases/))
* [DrWatson](https://github.com/JuliaDynamics/DrWatson.jl#readme)
* [Revise](https://github.com/timholy/Revise.jl#readme)
* [IJulia][]

We recommend [Miniforge](https://github.com/conda-forge/miniforge) to install Jupyter.

The Julia packages listed above should be installed in your base Julia environment. If they are not installed already, initializing the project will install them.

## Initialization

To work with this project, navigate to the project folder and run

~~~
make venv
~~~

The project is driven by the `Makefile`. Run `make help` to see all possible commands.


## Notebooks

The notebooks in the `notebooks` subfolder are stored in `.jl` format via [jupytext][]. In order to run the notebooks, you must have a Jupyter server installed independently, with the  [jupytext extension][jupytext]. Assuming you have the [IJulia][] kernel installed globally, the notebook will automatically run within the correct project environment. Simply start the server with e.g.

~~~
make jupyter-notebook
~~~

## Generate plots

From within the project folder, run

~~~
make plots
~~~

This will produce pdf files in the subfolder `./data/plots`. This includes the file `combined_benchmarks.pdf`, which is Figure 2 in the paper.

Alternatively, run the jupyter server (`make jupyter-notebook`) and open and run the `./notebooks/Plots.py` notebook.



## Regenerate benchmarks

The benchmark data is included in the notebook. To re-run the benchmarks, delete all files in the `./data/benchmarks` folder. Then, run,

~~~
make semiad_sysimage.so benchmarks
~~~

This will take several days, possibly several weeks on smaller systems, and use all available cores. No other processes requiring significant CPU resources should be running, as this will skew the benchmarks.


## Cleanup

Running

~~~
make distclean
~~~

removes all generated files and uninstalls the Jupyter project kernel from the user's global Jupyter installation.

[Jupyter]: (https://jupyter.org)
[jupytext]: https://jupytext.readthedocs.io/en/latest/
[IJulia]: https://github.com/JuliaLang/IJulia.jl#readme

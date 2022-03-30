# Semi-Atuomatic Differentiation


## Prerequisites

You should have the following installed before initializing this project

* [Jupyter notebook][Jupyter]
* [Jupytext extension][jupytext]
* [Julia 1.7](https://julialang.org)
* [DrWatson](https://github.com/JuliaDynamics/DrWatson.jl#readme)
* [Revise](https://github.com/timholy/Revise.jl#readme)
* [IJulia][]

The Julia packages listed above should be installed in your base Julia environment. If they are not installed already, initializing the project will install them.

## Initialization

To work with this project, run

~~~
make devrepl
~~~

or

~~~
julia --banner=no --startup-file=yes -i scripts/devrepl.jl
~~~

Run `make help`, respectively `help()` in the REPL for further commands.

## Notebooks

The notebooks in the `notebooks` subfolder are stored in `.jl` format via [jupytext][]. In order to run the notebooks, you must have a Jupyter server installed independently, with the  [jupytext extension][jupytext]. Assuming you have the [IJulia][] kernel installed globally, the notebook will automatically run within the correct project environment. Simply start the server with e.g.

~~~
jupyter notebook
~~~

With the [jupytext][] extension, the `.jl` files can be opened as notebooks, and upon saving, both the `.jl` and an associated `.ipynb` files will be written and kept in sync. The `.ipynb` locally stores the cell outputs, but should not be committed to the project repository.

[Jupyter]: (https://jupyter.org)
[jupytext]: https://jupytext.readthedocs.io/en/latest/
[IJulia]: https://github.com/JuliaLang/IJulia.jl#readme

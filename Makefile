.PHONY: init help devrepl clean distclean venv ipython benchmarks

.DEFAULT_GOAL := help

define PRINT_HELP_JLSCRIPT
rx = r"^([a-z0-9A-Z_-]+):.*?##[ ]+(.*)$$"
for line in eachline()
    m = match(rx, line)
    if !isnothing(m)
        target, help = m.captures
        println("$$(rpad(target, 20)) $$help")
    end
end
endef
export PRINT_HELP_JLSCRIPT

JULIA ?= julia
PYTHON = .venv/bin/python

help:  ## Show this help
	@$(JULIA) -e "$$PRINT_HELP_JLSCRIPT" < $(MAKEFILE_LIST)

$(PYTHON): requirements.txt
	python3.9 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

venv: .venv/bin/python  Manifest.toml ## Create the virtual project environment

.initialized: Project.toml
	@$(JULIA) --project=. -e 'include("scripts/init.jl")'

Manifest.toml: .initialized Project.toml
	@$(JULIA) --project=. -e 'include("scripts/init.jl")'

devrepl: Manifest.toml .initialized ## Start an interactive Julia REPL
	@$(JULIA) --project=. --banner=no --startup-file=yes -i scripts/devrepl.jl

ipython: $(PYTHON)  ## Start an interactive Python REPL
	@.venv/bin/ipython

semiad_sysimage.so: Manifest.toml .initialized
	$(JULIA) --project=. scripts/make_sysimage.jl

benchmarks: data/benchmarks/run_benchmarks.log  ## Run the benchmarks (after make semiad_sysimage.so)

data/benchmarks/run_benchmarks.log: scripts/run_benchmarks.py $(PYTHON)
	$(PYTHON) $<

all: semiad_sysimage.so  data/benchmarks/run_benchmarks.log ## Generate all missing output files

clean: ## Remove generated files
	rm data/benchmarks/grape*.log
	rm -f lcov.info

distclean: clean  ## Restore clean repository state
	rm -f .initialized
	rm -rf .venv
	rm -rf notebooks/.ipynb_checkpoints
	rm -f semiad_sysimage.so

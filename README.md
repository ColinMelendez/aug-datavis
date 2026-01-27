# Data Visualization Experiments

Some experiments with 3d wireframe data visualizations for a project.

Scripts and notebooks are not modular at the moment, but the actual plotting logic in them
is wrapped in functions that can be easily extracted.

## Quick Start

### Prerequisites

- [Nix](https://nixos.org/download.html) with flakes enabled
- Or, manually: `pip` and `python` >= 3.14

The flake also includes a linter/formatter (Ruff) and typechecker (BasedPyright) but these are optional (though recommended).

### Running

install dependencies with:

```bash
pip install -r requirements.txt
```

and then run any of the scripts or notebooks

```bash
# for example
python visualization-2.py
```
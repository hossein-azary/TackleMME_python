# TackleMMe Python

Python migration of the TackleMMe analysis pipeline.

## Current stage: model building and project initialization

This repository is being initialized as an installable Python package. The original MATLAB source and its example inputs are needed before the model-building behavior can be ported faithfully. No MATLAB algorithms or model assumptions have been added yet.

## Development setup

From this directory, create and activate a virtual environment, then install the package in editable mode:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Project layout

```text
TackleMMe/
├── pyproject.toml
└── src/
    └── tacklemme/
        └── __init__.py
```

The next implementation work is to map the MATLAB model-building entry point, inputs, outputs, and required toolboxes into Python modules and dependencies.

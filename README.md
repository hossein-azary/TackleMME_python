# Recon3D Annotation Pipeline

A unified workflow for annotating genome-scale metabolic models such as Recon3D using the Virtual Metabolic Human (VMH) API, normalizing annotations, and generating reports for before/after comparison.

## What this project does

This repository provides tools to:
- annotate Recon3D-like models with external database identifiers
- preserve and normalize existing model annotations
- convert SBML/XML to normalized COBRA JSON
- generate HTML reports that summarize annotation coverage and changes
- compare original and annotated models in an easy-to-read before/after report

## Key features

- VMH-based annotation for reactions, metabolites, and genes
- support for COBRA JSON, SBML, and XML input/output
- preserves existing annotations and adds missing identifiers
- normalizes annotation keys and prefixes for consistent output
- pretty-printed COBRA JSON output for easier inspection
- report-only mode for analysis without new API requests
- before/after comparison reporting

## Requirements

- Python 3.8+
- COBRApy
- requests
- tqdm
- A COBRA-compatible solver is recommended for model loading/validation

See `recon3D-annotations/requirements.txt` for exact package versions.

## Installation

### Option 1: Activate an existing environment

```powershell
.
econ3d-env\Scripts\activate
```

### Option 2: Create a new virtual environment

```powershell
python -m venv recon3d-env
.
econ3d-env\Scripts\activate
```

### Install dependencies

```powershell
pip install -r recon3D-annotations\requirements.txt
```

## Repository structure

```
GIT_PROJECT/
├── recon3D-annotations/
│   ├── vmh_annotator2.py
│   ├── convert_to_json.py
│   ├── comparison_file.py
│   ├── detailed_annotation_report.py
│   ├── data_annotation.py
│   ├── example_usage.py
│   ├── setup_env.py
│   ├── requirements.txt
│   ├── Recon3D.xml
│   ├── Recon3D_annotated.xml
│   ├── recon3D_annotated2.json
│   ├── reac_xref.tsv
│   └── README.md
├── markdown_files/
├── ANNOTATION_ANALYSIS_SUMMARY.md
├── COMPARISON_REPORT.md
├── FINAL_ANNOTATION_REPORT.md
└── README.md
```

## Usage

### Annotate a model with VMH

From the repository root:

```powershell
python recon3D-annotations\vmh_annotator2.py recon3D-annotations\Recon3D.json -o recon3D-annotations\Recon3D_annotated.json
```

From inside `recon3D-annotations`:

```powershell
python vmh_annotator2.py Recon3D.json -o Recon3D_annotated.json
```

### Generate an HTML report for an annotated model

```powershell
python recon3D-annotations\vmh_annotator2.py recon3D-annotations\recon3D_annotated6.json --report recon3D-annotations\recon3D_report.html --report-only
```

### Convert XML/SBML to normalized JSON

```powershell
python recon3D-annotations\convert_to_json.py recon3D-annotations\Recon3D_annotated.xml recon3D-annotations\recon3D_annotated2.json
```

### Compare original and annotated models

```powershell
python recon3D-annotations\comparison_file.py --main-model recon3D-annotations\Recon3D.json --annotated-model recon3D-annotations\recon3D_annotated2.json --output recon3D-annotations\comparison_report.html
```

## Annotation normalization

This project cleans up common annotation inconsistencies, including:
- merging `SBO` and `sbo` into a single `sbo` field
- stripping `SBO:` prefixes from SBO values
- merging `inchi_key` into `inchikey`
- stripping `EC:` prefixes from `ec-code` values
- stripping `InChI=` prefixes from `inchi` values

Normalized output avoids duplicate or inconsistent annotation keys and helps downstream tools read the model reliably.

## Supported annotation fields

Common expected fields added or normalized by the workflow:

- Reactions: `bigg.reaction`, `rhea.reaction`, `kegg.reaction`, `metanetx.reaction`, `seed.reaction`, `ec-code`, `sbo`
- Metabolites: `bigg.metabolite`, `chebi`, `kegg.compound`, `seed.compound`, `metanetx.chemical`, `inchi`, `inchikey`, `smiles`, `sbo`
- Genes: `uniprot`, `ensembl`, `ncbi.geneid`, `sbo`

## Notes

- VMH enrichment depends on identifiers already present in the input model.
- `--report-only` analyzes an existing annotated model without calling VMH.
- Network access is required for VMH API annotation.
- The project is designed to preserve current annotations while enriching missing identifiers.

## Troubleshooting

### Path or command errors

If you run a script from inside `recon3D-annotations`, use the shorter path form:

```powershell
python vmh_annotator2.py Recon3D.json -o Recon3D_annotated.json
```

### No new annotations added

If the model already contains good identifiers, the tool may preserve existing annotations and only normalize them.

### Network or VMH API issues

Allow Python network access on Windows and retry. Without network access, the tool can still load and save models, but VMH enrichment will not run.

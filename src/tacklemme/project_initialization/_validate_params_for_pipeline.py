"""Validation of a single params dict before it is added to the project."""

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._must_be_cobra_model import must_be_cobra_model
from ._validate_optional_settings import validate_optional_settings
from ._validate_manually_set_boundaries import validate_manually_set_boundaries


def validate_params_for_pipeline(params: dict) -> dict:
    """
    Validate a params dict for one model and return it (possibly normalised).

    Required keys:
        model_name (str)
        context_specific_model (cobra.Model)

    Optional keys:
        dico, sample_metadata, sample_labeling, medium_composition,
        obj_function, discretized_data, expression_data, gene_ids,
        consensus_proportion, reference_model, optional_settings,
        core_reactions, manually_set_boundaries, mapping
    """
    params = dict(params)  # shallow copy so we don't mutate the caller's dict

    # ── Required ──────────────────────────────────────────────────────────────
    if "model_name" not in params:
        raise KeyError("params['model_name'] is required.")
    if not isinstance(params["model_name"], str):
        raise TypeError("params['model_name'] must be a str.")

    if "context_specific_model" not in params:
        raise KeyError("params['context_specific_model'] is required.")
    must_be_cobra_model(params["context_specific_model"])

    # ── Optional ──────────────────────────────────────────────────────────────
    if _nonempty(params, "dico") and not isinstance(params["dico"], pd.DataFrame):
        raise TypeError("params['dico'] must be a pandas DataFrame.")

    if _nonempty(params, "sample_metadata") and not isinstance(
        params["sample_metadata"], pd.DataFrame
    ):
        raise TypeError("params['sample_metadata'] must be a pandas DataFrame.")

    if _nonempty(params, "sample_labeling") and not isinstance(
        params["sample_labeling"], str
    ):
        raise TypeError("params['sample_labeling'] must be a str.")

    if _nonempty(params, "medium_composition") and not isinstance(
        params["medium_composition"], pd.DataFrame
    ):
        raise TypeError("params['medium_composition'] must be a pandas DataFrame.")

    if _nonempty(params, "obj_function") and not isinstance(
        params["obj_function"], str
    ):
        raise TypeError("params['obj_function'] must be a str.")

    if _nonempty(params, "discretized_data"):
        arr = np.asarray(params["discretized_data"])
        if arr.ndim != 2:
            raise ValueError("params['discretized_data'] must be a 2-D array.")
        params["discretized_data"] = arr

    if _nonempty(params, "expression_data"):
        arr = np.asarray(params["expression_data"])
        if arr.ndim != 2:
            raise ValueError("params['expression_data'] must be a 2-D array.")
        params["expression_data"] = arr

    if _nonempty(params, "gene_ids"):
        # Normalise to a list of str
        gene_ids = params["gene_ids"]
        if isinstance(gene_ids, np.ndarray):
            gene_ids = gene_ids.flatten().tolist()
        params["gene_ids"] = [str(g) for g in gene_ids]

    if _nonempty(params, "consensus_proportion"):
        if params["consensus_proportion"] < 0:
            raise ValueError("params['consensus_proportion'] must be non-negative.")

    if _nonempty(params, "reference_model") and not isinstance(
        params["reference_model"], str
    ):
        raise TypeError("params['reference_model'] must be a str.")

    if _nonempty(params, "optional_settings"):
        if not isinstance(params["optional_settings"], dict):
            raise TypeError("params['optional_settings'] must be a dict.")
        validate_optional_settings(params["optional_settings"])

    if _nonempty(params, "core_reactions"):
        cr = params["core_reactions"]
        if not isinstance(cr, (list, np.ndarray)):
            raise TypeError("params['core_reactions'] must be a list or 1-D array.")
        if isinstance(cr, np.ndarray) and cr.ndim > 1:
            raise ValueError("params['core_reactions'] must be a vector.")

    if _nonempty(params, "manually_set_boundaries"):
        if not isinstance(params["manually_set_boundaries"], dict):
            raise TypeError("params['manually_set_boundaries'] must be a dict.")
        validate_manually_set_boundaries(params["manually_set_boundaries"])

    if _nonempty(params, "mapping"):
        mapping = params["mapping"]
        if not (isinstance(mapping, np.ndarray) or sp.issparse(mapping)) or not np.issubdtype(
            mapping.dtype, np.floating
        ):
            raise TypeError(
                "params['mapping'] must be a float matrix (numpy array or scipy sparse)."
            )

    # ── Cross-field: sample_metadata ↔ sample_labeling ───────────────────────
    has_meta = _nonempty(params, "sample_metadata")
    has_label = _nonempty(params, "sample_labeling") and bool(
        params.get("sample_labeling", "")
    )
    if has_meta != has_label:
        raise ValueError(
            "sample_metadata and sample_labeling must be provided together or both omitted."
        )
    if has_meta and has_label:
        if params["sample_labeling"] not in params["sample_metadata"].columns:
            raise ValueError(
                f"'{params['sample_labeling']}' must be a column in sample_metadata."
            )

    # ── Cross-field: data arrays require gene_ids and dico ───────────────────
    has_disc = _nonempty(params, "discretized_data")
    has_expr = _nonempty(params, "expression_data")
    has_data = has_disc or has_expr
    has_gene_ids = _nonempty(params, "gene_ids") and bool(params.get("gene_ids"))
    has_dico = _nonempty(params, "dico")

    if has_data and not has_gene_ids:
        raise ValueError(
            "gene_ids is required when expression_data or discretized_data is provided."
        )
    if has_data and not has_dico:
        raise ValueError(
            "dico is required when expression_data or discretized_data is provided."
        )

    if has_gene_ids and has_data:
        n_genes = len(params["gene_ids"])
        if has_disc and params["discretized_data"].shape[0] != n_genes:
            raise ValueError(
                f"gene_ids must have the same number of rows as discretized_data "
                f"({params['discretized_data'].shape[0]} rows), got {n_genes} rows."
            )
        if has_expr and params["expression_data"].shape[0] != n_genes:
            raise ValueError(
                f"gene_ids must have the same number of rows as expression_data "
                f"({params['expression_data'].shape[0]} rows), got {n_genes} rows."
            )

    if has_disc and has_expr:
        if params["discretized_data"].shape != params["expression_data"].shape:
            raise ValueError(
                f"discretized_data {params['discretized_data'].shape} and "
                f"expression_data {params['expression_data'].shape} must have the "
                "same dimensions (rows=genes, columns=samples)."
            )

    # ── Cross-field: dico coverage check ─────────────────────────────────────
    if has_dico:
        dico = params["dico"]
        required_cols = {"geneIdsInModel", "geneIdsInData"}
        missing_cols = required_cols - set(dico.columns)
        if missing_cols:
            raise ValueError(
                f"dico must contain columns: {', '.join(sorted(required_cols))}. "
                f"Missing: {', '.join(sorted(missing_cols))}."
            )

        model = params["context_specific_model"]
        model_genes = [
            re.sub(r"\.[0-9]+$", "", g.id) for g in model.genes
        ]
        dico_model_genes = [
            re.sub(r"\.[0-9]+$", "", str(g)) for g in dico["geneIdsInModel"]
        ]
        n_model = len(model_genes)
        model_gene_set = set(model_genes)
        n_overlap_model = sum(g in model_gene_set for g in dico_model_genes)
        pct_model = (n_overlap_model / n_model * 100) if n_model else 0.0

        if pct_model < 5:
            raise ValueError(
                f"dico.geneIdsInModel covers only {pct_model:.1f}% ({n_overlap_model}/"
                f"{n_model}) of context_specific_model genes. Minimum 5% required."
            )
        import warnings
        warnings.warn(
            f"dico.geneIdsInModel covers {pct_model:.1f}% ({n_overlap_model}/{n_model}) "
            "of context_specific_model genes."
        )

        if has_gene_ids:
            data_genes = [
                re.sub(r"\.[0-9]+$", "", str(g)) for g in params["gene_ids"]
            ]
            dico_data_genes = [
                re.sub(r"\.[0-9]+$", "", str(g)) for g in dico["geneIdsInData"]
            ]
            n_data = len(data_genes)
            data_gene_set = set(data_genes)
            n_overlap_data = sum(g in data_gene_set for g in dico_data_genes)
            pct_data = (n_overlap_data / n_data * 100) if n_data else 0.0

            if pct_data < 5:
                raise ValueError(
                    f"dico.geneIdsInData covers only {pct_data:.1f}% ({n_overlap_data}/"
                    f"{n_data}) of gene_ids. Minimum 5% required."
                )
            warnings.warn(
                f"dico.geneIdsInData covers {pct_data:.1f}% ({n_overlap_data}/{n_data}) "
                "of gene_ids."
            )

    return params


def _nonempty(d: dict, key: str) -> bool:
    """Return True if key exists in d and its value is not None/empty."""
    if key not in d:
        return False
    val = d[key]
    if val is None:
        return False
    if isinstance(val, str):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.size > 0
    if isinstance(val, list):
        return len(val) > 0
    if sp.issparse(val):
        return val.shape[0] > 0 and val.shape[1] > 0
    if isinstance(val, pd.DataFrame):
        return not val.empty
    return True

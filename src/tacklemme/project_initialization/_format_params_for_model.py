"""Format validated params into the nested model dict stored in project.models."""

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._map_expression_to_model import map_expression_to_model
from ._remove_recursively_empty_fields import remove_recursively_empty_fields


def format_params_for_model(params: dict) -> dict:
    """
    Reformat pipeline params into the nested model dict expected by the pipeline.

    Parameters
    ----------
    params:
        A validated params dict (output of validate_params_for_pipeline).

    Returns
    -------
    A nested dict with keys: model, [sample_metadata], [discretized_data],
    [expression_data], [mapped_discretized_rxns_all_samples],
    [mapped_discretized_rxns], [core_reactions], [settings].
    """
    entry: dict = {}
    entry["model"] = params["context_specific_model"]

    # Only add settings/data sub-dicts when there are optional fields
    if len(params) <= 2:
        return entry

    settings: dict = {}
    medium: dict = {}
    script_parameters: dict = {}

    # ── dico ─────────────────────────────────────────────────────────────────
    if "dico" in params:
        dico = params["dico"].copy()
        dico = _align_dico_to_model(dico, entry["model"])
        settings["dico"] = dico
    else:
        # Build a minimal dico from model genes only
        model_genes = [g.id for g in entry["model"].genes]
        settings["dico"] = pd.DataFrame({"geneIdsInModel": model_genes})

    # ── sample_metadata ───────────────────────────────────────────────────────
    if "sample_metadata" in params:
        entry["sample_metadata"] = params["sample_metadata"]

    # ── discretized_data ──────────────────────────────────────────────────────
    if "discretized_data" in params:
        disc_table = pd.DataFrame({
            "geneIds": [str(g) for g in params["gene_ids"]],
            "value": list(params["discretized_data"]),
        })
        disc_aligned = _reorder_data_to_dico(disc_table, settings["dico"], mode="filter")
        entry["discretized_data"] = disc_aligned

        mapping_disc = map_expression_to_model(
            entry["model"],
            np.vstack(disc_aligned["value"].values),
            settings["dico"],
            disc_aligned["geneIds"].tolist(),
            flag=1,
        )
        # MATLAB int8(NaN) == 0
        entry["mapped_discretized_rxns_all_samples"] = np.nan_to_num(mapping_disc, nan=0).astype(np.int8)

        if "consensus_proportion" in params:
            cons_prop = params["consensus_proportion"]
            n_samples = mapping_disc.shape[1]
            core_mask = (mapping_disc == 1).sum(axis=1) >= (cons_prop * n_samples)
            mapped = core_mask.astype(np.int32)
            not_expressed = np.where(
                (mapping_disc == -1).sum(axis=1) >= (cons_prop * n_samples)
            )[0]
            mapped[not_expressed] = -1
            entry["mapped_discretized_rxns"] = mapped.astype(np.int8)
        else:
            # -13 sentinel: no consensus proportion supplied
            entry["mapped_discretized_rxns"] = np.full(
                mapping_disc.shape[0], -13, dtype=np.int8
            )

    # ── expression_data ───────────────────────────────────────────────────────
    if "expression_data" in params:
        expr_table = pd.DataFrame({
            "geneIds": [str(g) for g in params["gene_ids"]],
            "expression": list(params["expression_data"]),
        })
        expr_aligned = _reorder_data_to_dico(expr_table, settings["dico"], mode="keep")
        entry["expression_data"] = expr_aligned

    # ── core_reactions ────────────────────────────────────────────────────────
    if "core_reactions" in params:
        entry["core_reactions"] = params["core_reactions"]

    # ── medium ────────────────────────────────────────────────────────────────
    if "medium_composition" in params:
        medium["medium_composition"] = params["medium_composition"]

    if "manually_set_boundaries" in params:
        msb = dict(params["manually_set_boundaries"])
        if "optional_settings" in params:
            opt = params["optional_settings"]
            if "not_medium_constrained" in opt:
                existing = msb.get("unconstrained_imports", [])
                msb["unconstrained_imports"] = list(
                    set(existing) | set(opt["not_medium_constrained"])
                )
        medium["manually_set_boundaries"] = msb
    elif "optional_settings" in params:
        opt = params["optional_settings"]
        if "not_medium_constrained" in opt:
            medium["manually_set_boundaries"] = {
                "unconstrained_imports": opt["not_medium_constrained"]
            }

    if medium:
        settings["medium"] = medium

    # ── consensus_proportion ──────────────────────────────────────────────────
    if "consensus_proportion" in params:
        script_parameters["consensus_proportion"] = params["consensus_proportion"]

    # ── optional_settings ─────────────────────────────────────────────────────
    if "optional_settings" in params:
        settings["optional_settings"] = params["optional_settings"]

    # ── obj_function ──────────────────────────────────────────────────────────
    if "obj_function" in params:
        settings["obj_function"] = params["obj_function"]

    # ── reference_model ───────────────────────────────────────────────────────
    if "reference_model" in params:
        settings["reference_model"] = params["reference_model"]

    # ── mapping ───────────────────────────────────────────────────────────────
    if "mapping" in params:
        # Stored as sparse: input may be dense, check_project_format requires sparse
        settings["mapping"] = sp.csc_array(params["mapping"])

    # ── sample_labeling ───────────────────────────────────────────────────────
    if "sample_labeling" in params:
        script_parameters["sample_labeling"] = params["sample_labeling"]

    if script_parameters:
        settings["script_parameters"] = script_parameters

    if settings:
        entry["settings"] = settings

    # Remove empty nested dicts (except the 'model' key which stays)
    entry = remove_recursively_empty_fields(entry, exclude=["model"])

    return entry


# ── Helpers ───────────────────────────────────────────────────────────────────

def _strip_version(s: str) -> str:
    return re.sub(r"\.[0-9]+$", "", str(s))


def _align_dico_to_model(dico: pd.DataFrame, model) -> pd.DataFrame:
    """
    Reorder dico so that geneIdsInModel follows the order of model.genes.
    - Genes in model but not in dico → 'NA' rows.
    - Genes in dico but not in model → dropped.
    """
    model_genes = [_strip_version(g.id) for g in model.genes]
    dico_genes = [_strip_version(str(g)) for g in dico["geneIdsInModel"]]

    n = len(model_genes)
    n_cols = len(dico.columns)

    aligned_data = {col: ["NA"] * n for col in dico.columns}

    dico_pos = _first_positions(dico_genes)
    for i, mg in enumerate(model_genes):
        idx = dico_pos.get(mg)
        if idx is None:
            continue  # stays "NA"
        for col in dico.columns:
            aligned_data[col][i] = str(dico[col].iloc[idx])

    return pd.DataFrame(aligned_data)


def _first_positions(items: list) -> dict:
    """Map each item to its first index (same result as list.index)."""
    pos: dict = {}
    for i, item in enumerate(items):
        pos.setdefault(item, i)
    return pos


def _reorder_data_to_dico(
    data_table: pd.DataFrame,
    dico: pd.DataFrame,
    mode: str,
) -> pd.DataFrame:
    """
    Reorder/align a data table to match the dico gene order.

    Parameters
    ----------
    data_table:
        DataFrame with columns 'geneIds' and 'value' (or 'expression').
    dico:
        The aligned dico for this model (from _align_dico_to_model).
    mode:
        'filter' — output has len(dico) rows; NaN for genes missing from data.
        'keep'   — output keeps all rows, dico-matched first then extras.

    Returns
    -------
    Reordered DataFrame, with optional 'geneNames' column prepended if dico
    has a 'geneNames' column.
    """
    value_col = "value" if "value" in data_table.columns else "expression"

    if "geneIdsInData" in dico.columns:
        dico_data_genes = [_strip_version(str(g)) for g in dico["geneIdsInData"]]
    else:
        dico_data_genes = []

    data_genes = [_strip_version(str(g)) for g in data_table["geneIds"]]

    if mode == "filter":
        # One output row per dico row; NaN where gene is missing
        n_dico = len(dico)
        sample_val = data_table[value_col].iloc[0]
        if isinstance(sample_val, np.ndarray):
            n_cols = sample_val.shape[0]
            out_values = [np.full(n_cols, np.nan) for _ in range(n_dico)]
        else:
            out_values = [np.nan] * n_dico

        out_gene_ids = dico_data_genes if dico_data_genes else ["NA"] * n_dico

        data_pos = _first_positions(data_genes)
        values = data_table[value_col].to_list()
        for i, dg in enumerate(dico_data_genes):
            if dg.upper() == "NA":
                continue
            data_idx = data_pos.get(dg)
            if data_idx is not None:
                out_values[i] = values[data_idx]

        result = pd.DataFrame({"geneIds": out_gene_ids, value_col: out_values})

    else:  # 'keep'
        dico_data_set = set(dico_data_genes)
        in_dico_mask = [g in dico_data_set for g in data_genes]
        in_dico_rows = [i for i, m in enumerate(in_dico_mask) if m]
        not_in_dico_rows = [i for i, m in enumerate(in_dico_mask) if not m]

        # Sort in-dico rows by their position in dico
        # First occurrence, like MATLAB ismember (dico can list a data gene twice)
        dico_pos = _first_positions(dico_data_genes)
        in_dico_rows.sort(key=lambda i: dico_pos.get(data_genes[i], 999999))

        final_order = in_dico_rows + not_in_dico_rows
        result = data_table.iloc[final_order].reset_index(drop=True)

    # Prepend geneNames from dico if available
    if "geneNames" in dico.columns and dico_data_genes:
        # First occurrence wins, like MATLAB ismember
        dico_name_map: dict = {}
        for data_gene, gene_name in zip(dico["geneIdsInData"], dico["geneNames"]):
            dico_name_map.setdefault(_strip_version(str(data_gene)), str(gene_name))
        gene_names = [
            dico_name_map.get(_strip_version(str(g)), "NA")
            for g in result["geneIds"]
        ]
        result.insert(0, "geneNames", gene_names)

    return result

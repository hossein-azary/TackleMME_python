"""Validate that a Project object conforms to the expected format."""

import warnings

import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ._must_be_cobra_model import must_be_cobra_model


def check_project_format(project, model_list=None, analysis_list=None) -> None:
    """
    Check that a Project conforms to the expected format.

    Parameters
    ----------
    project:
        A tacklemme Project instance.
    model_list:
        Optional list of model names to validate. If None, all models are validated.
    analysis_list:
        Optional list of analysis IDs (same length as model_list).
        Each is validated only in its corresponding model.
    """
    # Top-level: only 'models' and 'comparisons' allowed
    allowed_top = {"models", "comparisons"}
    actual_top = set(vars(project).keys())
    extra_top = actual_top - allowed_top
    if extra_top:
        raise ValueError(
            f"Unexpected attribute(s) in project: {', '.join(sorted(extra_top))}"
        )

    if not hasattr(project, "models"):
        raise ValueError("project must have a 'models' attribute.")

    if not project.models:
        raise ValueError("project.models must contain at least one model.")

    if analysis_list is not None and model_list is None:
        raise ValueError("analysis_list cannot be provided without model_list.")

    if analysis_list is not None and model_list is not None:
        if len(analysis_list) != len(model_list):
            raise ValueError(
                f"analysis_list must have the same number of elements as model_list "
                f"(got {len(analysis_list)} analyses for {len(model_list)} models)."
            )

    models_to_validate = list(model_list) if model_list else list(project.models.keys())

    if model_list:
        missing = set(model_list) - set(project.models.keys())
        if missing:
            raise ValueError(
                f"Requested model(s) not found in project.models: {', '.join(sorted(missing))}"
            )

    for i, name in enumerate(models_to_validate):
        m = project.models[name]
        analysis_name = analysis_list[i] if analysis_list else None
        _validate_model_in_project(m, f"project.models.{name}", analysis_name)


# ── Per-model validation ──────────────────────────────────────────────────────

def _validate_model_in_project(m: dict, path: str, analysis_name=None) -> None:
    allowed = {
        "model", "sample_metadata", "discretized_data", "expression_data",
        "core_reactions", "mapped_discretized_rxns_all_samples",
        "mapped_discretized_rxns", "settings", "analysis",
    }
    extra = set(m.keys()) - allowed
    if extra:
        raise ValueError(
            f"Unexpected key(s) in {path}: {', '.join(sorted(extra))}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )

    if "model" not in m:
        raise ValueError(f"Missing required key 'model' in {path}.")
    must_be_cobra_model(m["model"])

    if "sample_metadata" in m and not isinstance(m["sample_metadata"], pd.DataFrame):
        raise TypeError(f"{path}.sample_metadata must be a DataFrame.")

    if "discretized_data" in m:
        _check_data_table(m["discretized_data"], path + ".discretized_data",
                          required_cols={"geneIds", "value"})
        n_rxns = len(m["model"].reactions)

        all_samples = m.get("mapped_discretized_rxns_all_samples")
        if all_samples is None:
            raise ValueError(f"{path}.mapped_discretized_rxns_all_samples is required "
                             "when discretized_data is present.")
        if not isinstance(all_samples, np.ndarray) or all_samples.dtype != np.int8:
            raise TypeError(f"{path}.mapped_discretized_rxns_all_samples must be np.int8 array.")
        if all_samples.shape[0] != n_rxns:
            raise ValueError(
                f"{path}.mapped_discretized_rxns_all_samples must have {n_rxns} rows, "
                f"got {all_samples.shape[0]}."
            )

        mapped = m.get("mapped_discretized_rxns")
        if mapped is None:
            raise ValueError(f"{path}.mapped_discretized_rxns is required "
                             "when discretized_data is present.")
        if not isinstance(mapped, np.ndarray) or mapped.dtype != np.int8:
            raise TypeError(f"{path}.mapped_discretized_rxns must be np.int8 array.")
        if mapped.shape[0] != n_rxns:
            raise ValueError(
                f"{path}.mapped_discretized_rxns must have {n_rxns} rows, "
                f"got {mapped.shape[0]}."
            )

    if "expression_data" in m:
        _check_data_table(m["expression_data"], path + ".expression_data",
                          required_cols={"geneIds", "expression"})

    if "core_reactions" in m:
        cr = m["core_reactions"]
        if not isinstance(cr, (list, np.ndarray)):
            raise TypeError(f"{path}.core_reactions must be a list or 1-D array.")

    if "settings" in m:
        _validate_settings_in_project(m["settings"], path + ".settings", m)

    # Cross-field: dico ↔ discretized_data ↔ expression_data ↔ model.genes
    _validate_cross_fields(m, path)

    if "analysis" in m:
        if analysis_name:
            _validate_analysis_in_project(
                m["analysis"], path + ".analysis", m["model"], analysis_name
            )
        else:
            _validate_analysis_in_project(m["analysis"], path + ".analysis", m["model"])


def _check_data_table(df, path: str, required_cols: set) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{path} must be a DataFrame.")
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} must contain columns: {', '.join(sorted(required_cols))}. "
            f"Missing: {', '.join(sorted(missing))}."
        )


def _validate_settings_in_project(s: dict, path: str, m: dict) -> None:
    allowed = {
        "medium", "script_parameters", "dico", "obj_function",
        "reference_model", "mapping", "optional_settings",
    }
    extra = set(s.keys()) - allowed
    if extra:
        raise ValueError(
            f"Unexpected key(s) in {path}: {', '.join(sorted(extra))}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )

    if "dico" in s:
        if not isinstance(s["dico"], pd.DataFrame):
            raise TypeError(f"{path}.dico must be a DataFrame.")
        if "geneIdsInModel" not in s["dico"].columns:
            raise ValueError(f"{path}.dico must contain at least column 'geneIdsInModel'.")

    if "obj_function" in s and not isinstance(s["obj_function"], str):
        raise TypeError(f"{path}.obj_function must be a str.")

    if "reference_model" in s and not isinstance(s["reference_model"], str):
        raise TypeError(f"{path}.reference_model must be a str.")

    if "mapping" in s:
        mapping = s["mapping"]
        if not (sp.issparse(mapping) and np.issubdtype(mapping.dtype, np.floating)):
            raise TypeError(f"{path}.mapping must be a scipy sparse float matrix.")

    if "optional_settings" in s:
        from ._validate_optional_settings import validate_optional_settings
        validate_optional_settings(s["optional_settings"])

    if "medium" in s:
        _validate_medium_in_project(s["medium"], path + ".medium")

    if "script_parameters" in s:
        _validate_script_parameters_in_project(s["script_parameters"], path + ".script_parameters", m)


def _validate_medium_in_project(med: dict, path: str) -> None:
    allowed = {"medium_composition", "manually_set_boundaries"}
    extra = set(med.keys()) - allowed
    if extra:
        raise ValueError(
            f"Unexpected key(s) in {path}: {', '.join(sorted(extra))}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )

    if "medium_composition" in med and not isinstance(med["medium_composition"], pd.DataFrame):
        raise TypeError(f"{path}.medium_composition must be a DataFrame.")

    if "manually_set_boundaries" in med:
        from ._validate_manually_set_boundaries import validate_manually_set_boundaries
        validate_manually_set_boundaries(med["manually_set_boundaries"])


def _validate_script_parameters_in_project(sp: dict, path: str, m: dict) -> None:
    allowed = {"sample_labeling", "consensus_proportion"}
    extra = set(sp.keys()) - allowed
    if extra:
        raise ValueError(
            f"Unexpected key(s) in {path}: {', '.join(sorted(extra))}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )

    if "sample_labeling" in sp:
        sample_meta = m.get("sample_metadata")
        if sample_meta is not None:
            label = sp["sample_labeling"]
            if label and label not in sample_meta.columns:
                raise ValueError(f"'{label}' must be a column in sample_metadata.")

    if "consensus_proportion" in sp:
        if sp["consensus_proportion"] is not None and sp["consensus_proportion"] < 0:
            raise ValueError("consensus_proportion must be non-negative.")


def _validate_cross_fields(m: dict, path: str) -> None:
    has_disc = "discretized_data" in m
    has_expr = "expression_data" in m
    has_settings = "settings" in m
    has_dico = has_settings and "dico" in m.get("settings", {})
    model_genes = [g.id for g in m["model"].genes]
    n_model_genes = len(model_genes)

    if (has_disc or has_expr) and not has_dico:
        raise ValueError(
            f"{path}: dico is required in settings when discretized_data or expression_data is present."
        )

    if has_dico:
        dico = m["settings"]["dico"]
        if (has_disc or has_expr) and "geneIdsInData" not in dico.columns:
            raise ValueError(
                f"{path}.settings.dico must contain column 'geneIdsInData' "
                "when discretized_data or expression_data is present."
            )
        if len(dico) != n_model_genes:
            raise ValueError(
                f"{path}.settings.dico must have {n_model_genes} rows (same as model.genes), "
                f"got {len(dico)}."
            )

        # dico.geneIdsInModel must follow model.genes. "NA" rows (model genes
        # absent from the user's dico) are skipped — MATLAB intends the same
        # with its validMask, but its mask is a scalar so it does not skip them.
        dico_model_genes = [_strip_version(g) for g in dico["geneIdsInModel"]]
        model_genes_clean = [_strip_version(g) for g in model_genes]
        mismatch = [
            i for i, (d, g) in enumerate(zip(dico_model_genes, model_genes_clean))
            if d.upper() != "NA" and d != g
        ]
        if mismatch:
            i = mismatch[0]
            raise ValueError(
                f"{path}.settings.dico.geneIdsInModel must be in the same order as "
                f"model.genes (first mismatch at row {i}: dico '{dico_model_genes[i]}' "
                f"vs model '{model_genes_clean[i]}')."
            )

    if has_disc:
        disc = m["discretized_data"]
        if len(disc) != n_model_genes:
            raise ValueError(
                f"{path}.discretized_data must have {n_model_genes} rows, got {len(disc)}."
            )

        # discretized_data.geneIds must follow dico.geneIdsInData
        if has_dico and "geneIdsInData" in m["settings"]["dico"].columns:
            disc_genes = [_strip_version(g) for g in disc["geneIds"]]
            dico_data_genes = [
                _strip_version(g) for g in m["settings"]["dico"]["geneIdsInData"]
            ]
            if disc_genes != dico_data_genes:
                raise ValueError(
                    f"{path}.discretized_data.geneIds must be in the same order as "
                    "dico.geneIdsInData."
                )


def _strip_version(gene_id) -> str:
    """'ENSG00000001234.5' → 'ENSG00000001234' (same as the MATLAB regexprep)."""
    return re.sub(r"\.[0-9]+$", "", str(gene_id))


# ── Analysis validation ───────────────────────────────────────────────────────

def _validate_analysis_in_project(a: dict, path: str, model, analysis_name=None) -> None:
    if not isinstance(a, dict):
        raise TypeError(f"{path} must be a dict.")

    n_rxns = len(model.reactions)

    if analysis_name:
        if analysis_name not in a:
            raise ValueError(f"Analysis '{analysis_name}' not found in {path}.")
        _validate_analysis_entry(a[analysis_name], f"{path}.{analysis_name}", n_rxns, analysis_name)
    else:
        for name, entry in a.items():
            _validate_analysis_entry(entry, f"{path}.{name}", n_rxns, name)


def _validate_analysis_entry(entry: dict, path: str, n_rxns: int, entry_name: str) -> None:
    is_active = entry_name == "active"
    allowed = {"parameters", "FBA", "FVA", "single_gene_deletion",
               "double_gene_deletion", "sampling", "kld"}
    extra = set(entry.keys()) - allowed
    if extra:
        raise ValueError(
            f"Unexpected key(s) in {path}: {', '.join(sorted(extra))}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )

    if "parameters" in entry:
        if not isinstance(entry["parameters"], pd.DataFrame):
            warnings.warn(f"{path}.parameters should be a DataFrame with columns: "
                          "Parameter, Analysis, Value.")
        else:
            missing = {"Parameter", "Analysis", "Value"} - set(entry["parameters"].columns)
            if missing:
                warnings.warn(f"{path}.parameters is missing columns: {', '.join(sorted(missing))}.")

    if "FVA" in entry:
        fva = entry["FVA"]
        if not isinstance(fva, dict):
            raise TypeError(f"{path}.FVA must be a dict.")
        allowed_fva = {"min_max_fluxes", "loop_status"}
        if is_active:
            allowed_fva.add("analysis_id")
        extra_fva = set(fva.keys()) - allowed_fva
        if extra_fva:
            raise ValueError(f"Unexpected key(s) in {path}.FVA: {', '.join(sorted(extra_fva))}.")

        if "min_max_fluxes" in fva:
            if not isinstance(fva["min_max_fluxes"], pd.DataFrame):
                raise TypeError(f"{path}.FVA.min_max_fluxes must be a DataFrame.")
            if not {"min_flux", "max_flux"}.issubset(fva["min_max_fluxes"].columns):
                raise ValueError(
                    f"{path}.FVA.min_max_fluxes must have 'min_flux' and 'max_flux' columns."
                )
            if len(fva["min_max_fluxes"]) != n_rxns:
                raise ValueError(
                    f"{path}.FVA.min_max_fluxes must have {n_rxns} rows, "
                    f"got {len(fva['min_max_fluxes'])}."
                )

        if "loop_status" in fva:
            ls = fva["loop_status"]
            if not isinstance(ls, np.ndarray) or ls.dtype != bool:
                raise TypeError(f"{path}.FVA.loop_status must be a boolean np.ndarray.")
            if ls.shape != (n_rxns, 1) and ls.shape != (n_rxns,):
                raise ValueError(
                    f"{path}.FVA.loop_status must have shape ({n_rxns},) or ({n_rxns}, 1)."
                )

    if "sampling" in entry:
        _validate_sampling_in_analysis(entry["sampling"], path + ".sampling", n_rxns, is_active)

    if "kld" in entry:
        kld = entry["kld"]
        if not isinstance(kld, dict):
            raise TypeError(f"{path}.kld must be a dict.")
        allowed_kld = {"sampling_sets", "kld_matrix", "p_value_kld", "fdr", "set_labels"}
        if is_active:
            allowed_kld.add("analysis_id")
        extra_kld = set(kld.keys()) - allowed_kld
        if extra_kld:
            raise ValueError(f"Unexpected key(s) in {path}.kld: {', '.join(sorted(extra_kld))}.")


def _validate_sampling_in_analysis(
    s: dict, path: str, n_rxns: int, is_active: bool
) -> None:
    if not isinstance(s, dict):
        raise TypeError(f"{path} must be a dict.")

    allowed = {"model_sampling", "samples", "cycle_free_flux"}
    if is_active:
        allowed.add("analysis_id")
    extra = set(s.keys()) - allowed
    if extra:
        raise ValueError(
            f"Unexpected key(s) in {path}: {', '.join(sorted(extra))}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )

    if "samples" in s:
        samples = s["samples"]
        if not isinstance(samples, np.ndarray) or samples.dtype != np.float32:
            raise TypeError(f"{path}.samples must be np.float32 array.")
        if samples.shape[0] != n_rxns:
            warnings.warn(
                f"{path}.samples has {samples.shape[0]} rows but expected {n_rxns}."
            )

    if "cycle_free_flux" in s:
        _validate_cycle_free_flux(s["cycle_free_flux"], path + ".cycle_free_flux", n_rxns)


def _validate_cycle_free_flux(cff: dict, path: str, n_rxns: int) -> None:
    if not isinstance(cff, dict):
        raise TypeError(f"{path} must be a dict.")

    allowed = {
        "samples_ll", "thermo_feas", "sample_status_after_correction",
        "needed_attempts", "loopless_status",
    }
    extra = set(cff.keys()) - allowed
    if extra:
        raise ValueError(
            f"Unexpected key(s) in {path}: {', '.join(sorted(extra))}. "
            f"Allowed: {', '.join(sorted(allowed))}"
        )

    uint8_fields = {
        "thermo_feas", "sample_status_after_correction",
        "needed_attempts", "loopless_status",
    }
    for field in uint8_fields:
        if field in cff:
            arr = cff[field]
            if not isinstance(arr, np.ndarray) or arr.dtype != np.uint8:
                raise TypeError(f"{path}.{field} must be a np.uint8 array.")
            if arr.shape[0] != n_rxns:
                warnings.warn(
                    f"{path}.{field} has {arr.shape[0]} rows but expected {n_rxns}."
                )

    if "samples_ll" in cff:
        arr = cff["samples_ll"]
        if not isinstance(arr, np.ndarray) or arr.dtype != np.float32:
            raise TypeError(f"{path}.samples_ll must be np.float32 array.")
        if arr.shape[0] != n_rxns:
            warnings.warn(
                f"{path}.samples_ll has {arr.shape[0]} rows but expected {n_rxns}."
            )

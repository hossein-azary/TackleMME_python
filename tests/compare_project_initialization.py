"""
Compare the Python project initialisation with the MATLAB reference.

Inputs  : tests/reference_data/workspaceTutoNo1_plain.mat  (exact inputs MATLAB used)
Expected: tests/reference_data/BRCAProjectNo1_plain.mat     (MATLAB createProject output)

Reproduces no1_modelBuildingAndProjectInit.m lines 155-219:
    BRCAProject = createProject({Control, StageI, ..., StageIV});
    BRCAProject = addModelsToProject(BRCAProject, {consistentMediumConstrainedModel});

Run from the TackleMMe folder:
    python tests/compare_project_initialization.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse as sp
from cobra.io.mat import from_mat_struct

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from tacklemme import create_project, add_models_to_project  # noqa: E402

REF_DIR = HERE / "reference_data"
WORKSPACE = REF_DIR / "workspaceTutoNo1_plain.mat"
REFERENCE = REF_DIR / "BRCAProjectNo1_plain.mat"

STAGES = ["Control", "StageI", "StageII", "StageIII", "StageIV"]
EXTRA = "consistentMediumConstrainedModel"


# ── Converters: plain MATLAB structs → Python objects ────────────────────────

def as_list(x):
    """MATLAB cellstr / char / array → list of str."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [str(v) for v in np.atleast_1d(x)]


def table_to_df(t) -> pd.DataFrame:
    """Struct written by exportReferenceData.toPlain (tmmeType='table') → DataFrame."""
    cols = as_list(t.columns)
    data = np.atleast_1d(t.data) if len(cols) > 1 else [t.data]
    out = {}
    for c, d in zip(cols, data):
        d = np.asarray(d)
        out[c] = list(d) if d.ndim == 2 else d  # matrix column → one row per entry
    df = pd.DataFrame(out)
    row_names = as_list(t.rowNames)
    if row_names:
        df.index = row_names
    return df


def struct_to_dict(s, rename: dict) -> dict:
    """mat_struct → dict of lists, renaming camelCase keys to snake_case."""
    return {rename.get(f, f): as_list(getattr(s, f)) for f in s._fieldnames}


OPT_RENAME = {"notMediumConstrained": "not_medium_constrained"}
MSB_RENAME = {
    "closedImports": "closed_imports",
    "closedExports": "closed_exports",
    "unconstrainedImports": "unconstrained_imports",
    "unconstrainedExports": "unconstrained_exports",
}


def load_models() -> dict:
    """COBRA models, loaded as record arrays because cobra's MATLAB reader needs that."""
    var_names = {s: f"model{s}" for s in STAGES} | {EXTRA: EXTRA}
    raw = sio.loadmat(WORKSPACE, variable_names=list(var_names.values()))  # struct_as_record=True
    return {key: from_mat_struct(raw[var], model_id=var) for key, var in var_names.items()}


def build_params(p, model) -> dict:
    """One MATLAB paramsForPipeline struct → Python params dict (snake_case)."""
    out = {
        "model_name": str(p.modelName),
        "context_specific_model": model,
    }
    if hasattr(p, "dico"):
        out["dico"] = table_to_df(p.dico)
    if hasattr(p, "discretizedData"):
        out["discretized_data"] = np.asarray(p.discretizedData, dtype=float)
    if hasattr(p, "expressionData"):
        out["expression_data"] = np.asarray(p.expressionData, dtype=float)
    if hasattr(p, "geneIds"):
        out["gene_ids"] = as_list(p.geneIds)
    if hasattr(p, "objFunction"):
        out["obj_function"] = str(p.objFunction)
    if hasattr(p, "consensusProportion"):
        out["consensus_proportion"] = float(p.consensusProportion)
    if hasattr(p, "optionalSettings"):
        out["optional_settings"] = struct_to_dict(p.optionalSettings, OPT_RENAME)
    if hasattr(p, "mapping"):
        out["mapping"] = sp.csc_array(p.mapping)
    if hasattr(p, "coreReactions"):
        out["core_reactions"] = as_list(p.coreReactions)
    if hasattr(p, "mediumComposition"):
        out["medium_composition"] = table_to_df(p.mediumComposition)
    if hasattr(p, "manuallySetBoundaries"):
        out["manually_set_boundaries"] = struct_to_dict(p.manuallySetBoundaries, MSB_RENAME)
    if hasattr(p, "sampleMetadata"):
        out["sample_metadata"] = table_to_df(p.sampleMetadata)
    if hasattr(p, "sampleLabeling"):
        out["sample_labeling"] = str(p.sampleLabeling)
    if hasattr(p, "referenceModel"):
        out["reference_model"] = str(p.referenceModel)
    return out


# ── Comparison helpers ───────────────────────────────────────────────────────

results = []


def check(model, field, ok, detail=""):
    results.append((model, field, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {field}" + (f"  — {detail}" if detail else ""))


def compare_array(model, field, py, ml):
    if py is None:
        return check(model, field, False, "missing in Python output")
    py, ml = np.asarray(py), np.asarray(ml)
    if py.shape != ml.shape:
        return check(model, field, False, f"shape Python {py.shape} vs MATLAB {ml.shape}")
    both_nan = np.isnan(py) & np.isnan(ml) if py.dtype.kind == "f" and ml.dtype.kind == "f" else False
    n_diff = int(np.sum((py != ml) & ~both_nan))  # NaN == NaN counts as equal
    check(model, field, n_diff == 0, f"{n_diff} / {py.size} entries differ" if n_diff else "")


def compare_set(model, field, py, ml):
    py, ml = set(as_list(py)), set(as_list(ml))
    detail = "" if py == ml else (
        f"only Python: {sorted(py - ml)[:5]}  only MATLAB: {sorted(ml - py)[:5]}"
    )
    check(model, field, py == ml, detail)


def compare_value(model, field, py, ml):
    check(model, field, py == ml, "" if py == ml else f"Python {py!r} vs MATLAB {ml!r}")


def compare_df(model, field, py: pd.DataFrame, ml: pd.DataFrame):
    if py is None:
        return check(model, field, False, "missing in Python output")
    if py.shape != ml.shape:
        return check(model, field, False, f"shape Python {py.shape} vs MATLAB {ml.shape}")
    bad = [c for c in ml.columns
           if c not in py.columns
           or [str(v) for v in py[c]] != [str(v) for v in ml[c]]]
    check(model, field, not bad, f"columns that differ: {bad}" if bad else "")


def compare_data_table(model, field, py: pd.DataFrame, ml: pd.DataFrame, value_col):
    """Gene table (geneNames, geneIds, values): compare columns one by one."""
    if py is None:
        return check(model, field, False, "missing in Python output")
    for col in ("geneNames", "geneIds"):
        if col not in ml.columns:
            continue
        if col not in py.columns:
            check(model, f"{field}.{col}", False, "column missing in Python output")
            continue
        py_col, ml_col = [str(g) for g in py[col]], [str(g) for g in ml[col]]
        if len(py_col) != len(ml_col):
            check(model, f"{field}.{col}", False,
                  f"{len(py_col)} rows in Python vs {len(ml_col)} in MATLAB")
            continue
        n_diff = sum(a != b for a, b in zip(py_col, ml_col))
        check(model, f"{field}.{col}", n_diff == 0, f"{n_diff} rows differ" if n_diff else "")
    py_vals = np.vstack(py[value_col].values).astype(float)
    ml_vals = np.vstack(ml[value_col].values).astype(float)
    if py_vals.shape != ml_vals.shape:
        return check(model, f"{field}.{value_col}", False,
                     f"shape Python {py_vals.shape} vs MATLAB {ml_vals.shape}")
    same = np.isclose(py_vals, ml_vals, rtol=0, atol=1e-12, equal_nan=True)
    n_diff = int((~same).sum())
    check(model, f"{field}.{value_col}", n_diff == 0,
          f"{n_diff} / {same.size} entries differ" if n_diff else "")


def inconsistent_rule_rxns(ml_model) -> set:
    """Reactions whose MATLAB `rules` field disagrees with `grRules`.

    MATLAB (rFASTCORMICS) evaluates `rules`; Python (cobra) evaluates
    `grRules`, so these reactions may legitimately differ.
    """
    genes = as_list(ml_model.genes)
    norm = lambda t: " ".join(t.replace("(", " ( ").replace(")", " ) ").split())
    bad = set()
    for rxn, gr, ru in zip(as_list(ml_model.rxns), np.atleast_1d(ml_model.grRules),
                           np.atleast_1d(ml_model.rules)):
        gr = gr if isinstance(gr, str) else ""
        ru = ru if isinstance(ru, str) else ""
        rule_as_genes = re.sub(r"x\((\d+)\)", lambda m: genes[int(m.group(1)) - 1], ru)
        rule_as_genes = rule_as_genes.replace("|", "or").replace("&", "and")
        if norm(rule_as_genes) != norm(gr):
            bad.add(rxn)
    return bad


def compare_mapping(model, field, py, ml, rxn_ids, known_bad: set):
    """Like compare_array, but differences in `known_bad` reactions are allowed."""
    if py is None:
        return check(model, field, False, "missing in Python output")
    py, ml = np.asarray(py), np.asarray(ml)
    if py.shape != ml.shape:
        return check(model, field, False, f"shape Python {py.shape} vs MATLAB {ml.shape}")
    diff = (py != ml).reshape(len(rxn_ids), -1)
    diff_rxns = {rxn_ids[i] for i in np.where(diff.any(axis=1))[0]}
    unexpected = diff_rxns - known_bad
    n_known = int(sum(diff[i].sum() for i, r in enumerate(rxn_ids) if r in known_bad))
    if unexpected:
        check(model, field, False,
              f"{int(diff.sum())} entries differ, in reactions not explained by "
              f"MATLAB rules/grRules: {sorted(unexpected)[:10]}")
    else:
        check(model, field, True,
              f"{n_known} entries differ only in reactions with inconsistent MATLAB "
              f"rules/grRules: {sorted(diff_rxns)}" if n_known else "")


def compare_model(name, py: dict, ml):
    print(f"\n== {name}")
    s_ml = ml.settings
    s_py = py.get("settings", {})

    # Model itself
    check(name, "model.reactions", [r.id for r in py["model"].reactions]
          == as_list(ml.model.rxns))

    # Core numeric outputs
    rxn_ids = as_list(ml.model.rxns)
    known_bad = inconsistent_rule_rxns(ml.model)
    if hasattr(ml, "mappedDiscretizedRxnsAllSamples"):
        compare_mapping(name, "mappedDiscretizedRxnsAllSamples",
                        py.get("mapped_discretized_rxns_all_samples"),
                        ml.mappedDiscretizedRxnsAllSamples, rxn_ids, known_bad)
    if hasattr(ml, "mappedDiscretizedRxns"):
        compare_mapping(name, "mappedDiscretizedRxns",
                        py.get("mapped_discretized_rxns"), ml.mappedDiscretizedRxns,
                        rxn_ids, known_bad)
    if hasattr(ml, "coreReactions"):
        compare_set(name, "coreReactions", py.get("core_reactions"), ml.coreReactions)

    # Tables
    compare_df(name, "settings.dico", s_py.get("dico"), table_to_df(s_ml.dico))
    if hasattr(ml, "sampleMetadata"):
        compare_df(name, "sampleMetadata", py.get("sample_metadata"),
                   table_to_df(ml.sampleMetadata))
    for ml_field, py_field, value_col in [
        ("discretizedData", "discretized_data", "value"),
        ("expressionData", "expression_data", "expression"),
    ]:
        if hasattr(ml, ml_field):
            compare_data_table(name, ml_field, py.get(py_field),
                               table_to_df(getattr(ml, ml_field)), value_col)

    # Settings
    if hasattr(s_ml, "objFunction"):
        compare_value(name, "settings.objFunction", s_py.get("obj_function"),
                      str(s_ml.objFunction))
    if hasattr(s_ml, "referenceModel"):
        compare_value(name, "settings.referenceModel", s_py.get("reference_model"),
                      str(s_ml.referenceModel))
    if hasattr(s_ml, "scriptParameters"):
        sp_py = s_py.get("script_parameters", {})
        compare_value(name, "scriptParameters.consensusProportion",
                      sp_py.get("consensus_proportion"),
                      float(s_ml.scriptParameters.consensusProportion))
        compare_value(name, "scriptParameters.sampleLabeling",
                      sp_py.get("sample_labeling"), str(s_ml.scriptParameters.sampleLabeling))
    if hasattr(s_ml, "mapping"):
        m_py = s_py.get("mapping")
        same = m_py is not None and (sp.csc_array(m_py) != sp.csc_array(s_ml.mapping)).nnz == 0
        check(name, "settings.mapping", same)
    if hasattr(s_ml, "medium"):
        med_py = s_py.get("medium", {})
        compare_df(name, "medium.mediumComposition", med_py.get("medium_composition"),
                   table_to_df(s_ml.medium.mediumComposition))
        if hasattr(s_ml.medium, "manuallySetBoundaries"):
            msb_ml = s_ml.medium.manuallySetBoundaries
            msb_py = med_py.get("manually_set_boundaries", {})
            for f in msb_ml._fieldnames:
                compare_set(name, f"manuallySetBoundaries.{f}",
                            msb_py.get(MSB_RENAME[f]), getattr(msb_ml, f))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Loading MATLAB inputs ...")
    params_names = [f"paramsForPipeline{s}" for s in STAGES] + ["paramsConsistentMediumConstrainedModel"]
    ws = sio.loadmat(WORKSPACE, variable_names=params_names,
                     squeeze_me=True, struct_as_record=False)
    models = load_models()

    params = [build_params(ws[f"paramsForPipeline{s}"], models[s]) for s in STAGES]
    extra = build_params(ws["paramsConsistentMediumConstrainedModel"], models[EXTRA])

    print("Running Python create_project / add_models_to_project ...")
    project = create_project(params)
    project = add_models_to_project(project, [extra])

    print("Loading MATLAB reference output ...")
    ref = sio.loadmat(REFERENCE, squeeze_me=True, struct_as_record=False)

    compare_value("project", "model names", sorted(project.models), sorted(as_list(ref["modelNames"])))
    for name in as_list(ref["modelNames"]):
        if name in project.models:
            compare_model(name, project.models[name], ref[name])

    n_fail = sum(not ok for *_, ok in results)
    print(f"\n{len(results) - n_fail} passed, {n_fail} failed")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()

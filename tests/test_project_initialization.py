"""
Unit tests for step 1 (project initialisation).

These use a tiny hand-built model so they run in seconds without the MATLAB
reference data. The behaviours tested here were confirmed against MATLAB with
compare_project_initialization.py (109/109 checks).

Run from the TackleMMe folder:
    python -m pytest
"""

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from cobra import Model, Reaction

from tacklemme import Project, add_models_to_project, create_project
from tacklemme.project_initialization._check_project_format import check_project_format
from tacklemme.project_initialization._format_params_for_model import _reorder_data_to_dico
from tacklemme.project_initialization._map_expression_to_model import map_expression_to_model


# ── Fixtures ──────────────────────────────────────────────────────────────────

RULES = {
    "R_and": "g1 and g2",
    "R_or": "g1 or g3",
    "R_nested": "(g1 and g2) or g3",
    "R_none": "",
    "R_missing": "g4",          # g4 has no data
    "R_nan_and": "g1 and g5",   # g5 data is NaN
    "R_nan_only": "g5",
    "R_dup": "g6",              # g6 matches two data rows
    "R_version": "g7.1",        # matches data only after stripping ".1"
}


def make_model() -> Model:
    model = Model("toy")
    model.add_reactions([Reaction(rxn_id) for rxn_id in RULES])
    for rxn_id, rule in RULES.items():
        model.reactions.get_by_id(rxn_id).gene_reaction_rule = rule
    return model


@pytest.fixture
def model():
    return make_model()


@pytest.fixture
def mapping_inputs():
    """dico + data for map_expression_to_model (2 samples)."""
    dico = pd.DataFrame({
        "geneIdsInModel": ["g1", "g2", "g3", "g4", "g5", "g6", "g6", "g7"],
        "geneIdsInData": ["D1", "D2", "D3", "D4", "D5", "D6a", "D6b", "D7"],
    })
    gene_ids = ["D1", "D2", "D3", "D5", "D6a", "D6b", "D7"]  # D4 absent
    data = np.array([
        [1, -1],            # D1
        [0, 1],             # D2
        [-1, 1],            # D3
        [np.nan, np.nan],   # D5
        [-1, 0],            # D6a
        [1, -1],            # D6b
        [1, 1],             # D7
    ], dtype=float)
    return dico, gene_ids, data


def rxn_row(model, mapping, rxn_id):
    return mapping[[r.id for r in model.reactions].index(rxn_id)]


@pytest.fixture
def full_params(model):
    """Valid params with discretized data for every model gene (4 samples)."""
    genes = [g.id for g in model.genes]
    dico = pd.DataFrame({"geneIdsInModel": genes, "geneIdsInData": [f"D_{g}" for g in genes]})
    disc = np.tile([1.0, 1.0, 1.0, -1.0], (len(genes), 1))
    return {
        "model_name": "toy",
        "context_specific_model": model,
        "dico": dico,
        "discretized_data": disc,
        "gene_ids": list(dico["geneIdsInData"]),
        "consensus_proportion": 0.75,
    }


# ── map_expression_to_model: rFASTCORMICS GPR rules ──────────────────────────

class TestMapExpressionToModel:
    def test_and_is_min(self, model, mapping_inputs):
        m = map_expression_to_model(model, mapping_inputs[2], mapping_inputs[0], mapping_inputs[1])
        np.testing.assert_array_equal(rxn_row(model, m, "R_and"), [0, -1])

    def test_or_is_max(self, model, mapping_inputs):
        m = map_expression_to_model(model, mapping_inputs[2], mapping_inputs[0], mapping_inputs[1])
        np.testing.assert_array_equal(rxn_row(model, m, "R_or"), [1, 1])

    def test_nested_rule(self, model, mapping_inputs):
        m = map_expression_to_model(model, mapping_inputs[2], mapping_inputs[0], mapping_inputs[1])
        # sample 1: max(min(1, 0), -1) = 0; sample 2: max(min(-1, 1), 1) = 1
        np.testing.assert_array_equal(rxn_row(model, m, "R_nested"), [0, 1])

    def test_reaction_without_gpr_is_zero(self, model, mapping_inputs):
        m = map_expression_to_model(model, mapping_inputs[2], mapping_inputs[0], mapping_inputs[1])
        np.testing.assert_array_equal(rxn_row(model, m, "R_none"), [0, 0])

    def test_gene_without_data_is_zero(self, model, mapping_inputs):
        m = map_expression_to_model(model, mapping_inputs[2], mapping_inputs[0], mapping_inputs[1])
        np.testing.assert_array_equal(rxn_row(model, m, "R_missing"), [0, 0])

    def test_nan_is_ignored_like_matlab_min_max(self, model, mapping_inputs):
        m = map_expression_to_model(model, mapping_inputs[2], mapping_inputs[0], mapping_inputs[1])
        np.testing.assert_array_equal(rxn_row(model, m, "R_nan_and"), [1, -1])
        assert np.isnan(rxn_row(model, m, "R_nan_only")).all()

    def test_gene_with_several_data_rows_takes_max(self, model, mapping_inputs):
        m = map_expression_to_model(model, mapping_inputs[2], mapping_inputs[0], mapping_inputs[1])
        np.testing.assert_array_equal(rxn_row(model, m, "R_dup"), [1, 0])

    def test_version_suffix_stripped_only_with_flag(self, model, mapping_inputs):
        dico, gene_ids, data = mapping_inputs
        without = map_expression_to_model(model, data, dico, gene_ids, flag=0)
        with_flag = map_expression_to_model(model, data, dico, gene_ids, flag=1)
        np.testing.assert_array_equal(rxn_row(model, without, "R_version"), [0, 0])
        np.testing.assert_array_equal(rxn_row(model, with_flag, "R_version"), [1, 1])

    def test_no_overlap_raises(self, model, mapping_inputs):
        dico, _, data = mapping_inputs
        with pytest.raises(ValueError):
            map_expression_to_model(model, data, dico, [f"X{i}" for i in range(len(data))])


# ── _reorder_data_to_dico: MATLAB ismember uses the first occurrence ─────────

class TestReorderDataToDico:
    def test_keep_mode_orders_by_first_dico_occurrence(self):
        dico = pd.DataFrame({
            "geneIdsInModel": ["m1", "m2", "m3", "m4"],
            "geneIdsInData": ["B", "A", "B", "NA"],   # B listed twice
        })
        data = pd.DataFrame({"geneIds": ["A", "B", "C"], "expression": [1.0, 2.0, 3.0]})
        out = _reorder_data_to_dico(data, dico, mode="keep")
        # B (dico row 0) before A (row 1); C is not in dico → last
        assert list(out["geneIds"]) == ["B", "A", "C"]
        assert list(out["expression"]) == [2.0, 1.0, 3.0]

    def test_gene_names_use_first_occurrence(self):
        dico = pd.DataFrame({
            "geneIdsInModel": ["m1", "m2"],
            "geneIdsInData": ["A", "A"],
            "geneNames": ["first", "second"],
        })
        data = pd.DataFrame({"geneIds": ["A"], "expression": [1.0]})
        out = _reorder_data_to_dico(data, dico, mode="keep")
        assert list(out["geneNames"]) == ["first"]

    def test_filter_mode_one_row_per_dico_row_nan_when_missing(self):
        dico = pd.DataFrame({"geneIdsInModel": ["m1", "m2"], "geneIdsInData": ["A", "Z"]})
        data = pd.DataFrame({"geneIds": ["A"], "value": [np.array([1.0, -1.0])]})
        out = _reorder_data_to_dico(data, dico, mode="filter")
        assert list(out["geneIds"]) == ["A", "Z"]
        np.testing.assert_array_equal(out["value"].iloc[0], [1.0, -1.0])
        assert np.isnan(out["value"].iloc[1]).all()


# ── create_project / add_models_to_project ───────────────────────────────────

class TestCreateProject:
    def test_minimal_project(self, model):
        project = create_project([{"model_name": "m", "context_specific_model": model}])
        assert isinstance(project, Project)
        assert list(project.models) == ["m"]
        assert project.models["m"] == {"model": model}

    def test_single_dict_accepted(self, model):
        project = create_project({"model_name": "m", "context_specific_model": model})
        assert list(project.models) == ["m"]

    def test_empty_params_rejected(self):
        with pytest.raises(ValueError):
            create_project([])

    def test_full_project_fields(self, full_params, model):
        entry = create_project(full_params).models["toy"]
        all_samples = entry["mapped_discretized_rxns_all_samples"]
        assert all_samples.dtype == np.int8
        assert all_samples.shape == (len(model.reactions), 4)
        assert entry["mapped_discretized_rxns"].dtype == np.int8
        assert entry["settings"]["script_parameters"]["consensus_proportion"] == 0.75
        assert len(entry["settings"]["dico"]) == len(model.genes)

    def test_nan_mapping_becomes_zero_in_int8(self, full_params, model):
        # g5 data NaN in every sample → R_nan_only is NaN → stored as 0 (MATLAB int8(NaN))
        genes = [g.id for g in model.genes]
        full_params["discretized_data"][genes.index("g5")] = np.nan
        entry = create_project(full_params).models["toy"]
        row = rxn_row(model, entry["mapped_discretized_rxns_all_samples"], "R_nan_only")
        np.testing.assert_array_equal(row, [0, 0, 0, 0])

    def test_consensus(self, full_params, model):
        # every gene: on in 3 of 4 samples → with proportion 0.75, gene-linked
        # reactions are core (1); R_none maps to 0 in all samples → 0
        entry = create_project(full_params).models["toy"]
        consensus = entry["mapped_discretized_rxns"]
        assert rxn_row(model, consensus, "R_or") == 1
        assert rxn_row(model, consensus, "R_none") == 0

    def test_consensus_not_expressed(self, full_params, model):
        full_params["discretized_data"][:] = -1.0
        entry = create_project(full_params).models["toy"]
        assert rxn_row(model, entry["mapped_discretized_rxns"], "R_or") == -1

    def test_no_consensus_proportion_gives_minus_13(self, full_params):
        del full_params["consensus_proportion"]
        entry = create_project(full_params).models["toy"]
        assert (entry["mapped_discretized_rxns"] == -13).all()

    def test_dico_aligned_to_model_genes_with_na(self, full_params, model):
        full_params["dico"] = full_params["dico"].iloc[1:]  # drop first model gene
        dico = create_project(full_params).models["toy"]["settings"]["dico"]
        assert len(dico) == len(model.genes)
        assert dico["geneIdsInModel"].iloc[0] == "NA"

    def test_dense_mapping_stored_as_sparse(self, model):
        params = {"model_name": "m", "context_specific_model": model, "mapping": np.eye(3)}
        stored = create_project(params).models["m"]["settings"]["mapping"]
        assert sp.issparse(stored)

    def test_add_models_to_project(self, model):
        project = create_project({"model_name": "a", "context_specific_model": model})
        add_models_to_project(project, {"model_name": "b", "context_specific_model": model})
        assert list(project.models) == ["a", "b"]

    def test_add_existing_model_declined_keeps_old(self, model, monkeypatch):
        project = create_project({"model_name": "a", "context_specific_model": model})
        old_entry = project.models["a"]
        monkeypatch.setattr("builtins.input", lambda _: "n")
        add_models_to_project(project, {"model_name": "a", "context_specific_model": make_model()})
        assert project.models["a"] is old_entry


# ── validate_params_for_pipeline (through create_project) ───────────────────

class TestValidation:
    @pytest.mark.parametrize("change, error", [
        (lambda p: p.pop("model_name"), KeyError),
        (lambda p: p.update(model_name=123), TypeError),
        (lambda p: p.update(context_specific_model="not a model"), TypeError),
        (lambda p: p.update(consensus_proportion=-0.1), ValueError),
        (lambda p: p.pop("gene_ids"), ValueError),                   # data without gene_ids
        (lambda p: p.pop("dico"), ValueError),                       # data without dico
        (lambda p: p.update(gene_ids=p["gene_ids"][:-1]), ValueError),  # wrong length
        (lambda p: p.update(discretized_data=np.ones(3)), ValueError),  # not 2-D
        (lambda p: p.update(mapping=np.eye(3, dtype=int)), TypeError),
        (lambda p: p.update(mapping=[[1.0]]), TypeError),
        (lambda p: p.update(sample_labeling="group"), ValueError),   # without metadata
    ])
    def test_invalid_params_rejected(self, full_params, change, error):
        change(full_params)
        with pytest.raises(error):
            create_project(full_params)

    def test_sample_labeling_must_be_metadata_column(self, full_params):
        full_params["sample_metadata"] = pd.DataFrame({"group": ["a", "b", "a", "b"]})
        full_params["sample_labeling"] = "missing_column"
        with pytest.raises(ValueError):
            create_project(full_params)

    def test_dico_coverage_below_5_percent_rejected(self, full_params):
        full_params["dico"] = pd.DataFrame({"geneIdsInModel": ["other"], "geneIdsInData": ["x"]})
        with pytest.raises(ValueError, match="covers only"):
            create_project(full_params)

    def test_expression_and_discretized_shapes_must_match(self, full_params):
        full_params["expression_data"] = np.ones((len(full_params["gene_ids"]), 2))
        with pytest.raises(ValueError, match="same dimensions"):
            create_project(full_params)


# ── check_project_format ─────────────────────────────────────────────────────

class TestCheckProjectFormat:
    def test_valid_project_passes(self, full_params):
        check_project_format(create_project(full_params))

    def test_dense_mapping_in_project_rejected(self, full_params):
        project = create_project(full_params)
        project.models["toy"]["settings"]["mapping"] = np.eye(3)
        with pytest.raises(TypeError, match="mapping"):
            check_project_format(project)

    def test_dico_order_mismatch_caught(self, full_params):
        project = create_project(full_params)
        dico = project.models["toy"]["settings"]["dico"]
        dico.loc[[0, 1], "geneIdsInModel"] = dico.loc[[1, 0], "geneIdsInModel"].values
        with pytest.raises(ValueError, match="same order as model.genes"):
            check_project_format(project)

    def test_na_rows_in_dico_are_skipped(self, full_params):
        project = create_project(full_params)
        project.models["toy"]["settings"]["dico"].loc[0, "geneIdsInModel"] = "NA"
        check_project_format(project)

    def test_discretized_order_mismatch_caught(self, full_params):
        project = create_project(full_params)
        disc = project.models["toy"]["discretized_data"]
        disc.loc[[0, 1], "geneIds"] = disc.loc[[1, 0], "geneIds"].values
        with pytest.raises(ValueError, match="dico.geneIdsInData"):
            check_project_format(project)

    def test_unexpected_key_rejected(self, model):
        project = create_project({"model_name": "m", "context_specific_model": model})
        project.models["m"]["unknown"] = 1
        with pytest.raises(ValueError, match="Unexpected"):
            check_project_format(project)

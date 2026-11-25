"""Map gene expression values to reactions via GPR rules.

Python equivalent of rFASTCORMICS mapExpressionToModel.m +
GPRrulesMapperRFASTCORMICS.m (rFASTCORMICS_v2). Behaviour reproduced:
  - AND operator → min of gene values, OR operator → max (NaN is ignored,
    like MATLAB min/max: min(NaN, 1) = 1)
  - Reaction without GPR rule → 0
  - Model gene without a match in the data → 0
  - Model gene matching several data rows → max over those rows
  - A reaction whose value is NaN stays NaN (callers convert it to 0, as
    MATLAB int8(NaN) does)
"""

import re
from typing import Union

import numpy as np
import pandas as pd
from cobra import Model


# ── GPR parser ────────────────────────────────────────────────────────────────

class _GPRParser:
    """Recursive-descent parser for COBRA GPR rule strings ('and' binds tighter than 'or')."""

    def __init__(self, rule: str):
        self._tokens = re.findall(r"\(|\)|[^\s()]+", rule)
        self._pos = 0

    def parse(self):
        return self._parse_or()

    def _parse_or(self):
        left = self._parse_and()
        while self._pos < len(self._tokens) and self._tokens[self._pos].lower() == "or":
            self._pos += 1
            left = ("or", left, self._parse_and())
        return left

    def _parse_and(self):
        left = self._parse_atom()
        while self._pos < len(self._tokens) and self._tokens[self._pos].lower() == "and":
            self._pos += 1
            left = ("and", left, self._parse_atom())
        return left

    def _parse_atom(self):
        if self._pos >= len(self._tokens):
            raise ValueError("Unexpected end of GPR rule.")
        tok = self._tokens[self._pos]
        self._pos += 1
        if tok == "(":
            tree = self._parse_or()
            if self._pos < len(self._tokens) and self._tokens[self._pos] == ")":
                self._pos += 1
            return tree
        return ("gene", tok)

    @staticmethod
    def eval_tree(tree: tuple, gene_rows: dict, gene_matrix: np.ndarray) -> np.ndarray:
        """Evaluate a parsed rule for all samples at once (one value per column)."""
        kind = tree[0]
        if kind == "gene":
            return gene_matrix[gene_rows[tree[1]]]
        left = _GPRParser.eval_tree(tree[1], gene_rows, gene_matrix)
        right = _GPRParser.eval_tree(tree[2], gene_rows, gene_matrix)
        # fmin/fmax ignore NaN, exactly like MATLAB min/max
        return np.fmin(left, right) if kind == "and" else np.fmax(left, right)


# ── Public function ───────────────────────────────────────────────────────────

def map_expression_to_model(
    model: Model,
    expression_values: np.ndarray,
    dico: pd.DataFrame,
    gene_ids: Union[list, np.ndarray],
    flag: int = 0,
) -> np.ndarray:
    """
    Map gene expression values to reactions using GPR rules.

    Parameters
    ----------
    model:
        A cobra.Model.
    expression_values:
        2-D array of shape (n_genes, n_samples), rows aligned with gene_ids.
    dico:
        DataFrame of gene identifiers (e.g. 'geneIdsInModel', 'geneIdsInData').
    gene_ids:
        Gene IDs of the rows of expression_values ("rownames" in MATLAB).
    flag:
        processTranscripts in MATLAB: 1 strips ".<digits>" from model gene IDs
        before matching.

    Returns
    -------
    np.ndarray of shape (n_rxns, n_samples), float, may contain NaN.
    """
    data = np.asarray(expression_values, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n_samples = data.shape[1]
    rownames = [str(g) for g in gene_ids]
    dico_arr = dico.astype(str).to_numpy()

    # Dico column(s) with the most matches in rownames; MATLAB intersect keeps
    # the first occurrence, so the first of the best columns decides the rows.
    rowname_set = set(rownames)
    matches = [sum(v in rowname_set for v in dico_arr[:, c]) for c in range(dico_arr.shape[1])]
    best_col = int(np.argmax(matches))

    first_in_dico: dict[str, int] = {}
    for i, v in enumerate(dico_arr[:, best_col]):
        first_in_dico.setdefault(v, i)
    first_in_rownames: dict[str, int] = {}
    for i, v in enumerate(rownames):
        first_in_rownames.setdefault(v, i)
    common = sorted(set(first_in_dico) & set(first_in_rownames))  # intersect() sorts
    if not common:
        raise ValueError("rownames does not match the data IDs in the dico.")

    idico = [first_in_dico[v] for v in common]
    irownames = [first_in_rownames[v] for v in common]
    mapped = data[irownames, :]

    # mapped2 in MATLAB: [rownames(irownames), dico(idico, :)] — any identifier
    # in a row can match a model gene.
    id_to_rows: dict[str, list[int]] = {}
    for r, (rn, di) in enumerate(zip(irownames, idico)):
        for ident in [rownames[rn], *dico_arr[di]]:
            id_to_rows.setdefault(ident, []).append(r)

    # Gene × sample matrix; unmatched genes stay 0
    gene_matrix = np.zeros((len(model.genes), n_samples))
    gene_rows: dict[str, int] = {}
    for i, gene in enumerate(model.genes):
        gene_rows[gene.id] = i
        gid = re.sub(r"\.[0-9]*", "", gene.id) if flag == 1 else gene.id
        rows = id_to_rows.get(gid)
        if rows:
            gene_matrix[i] = np.fmax.reduce(mapped[rows, :], axis=0) if len(rows) > 1 else mapped[rows[0]]

    # Reactions: no rule → 0
    mapping = np.zeros((len(model.reactions), n_samples))
    gpr_cache: dict[str, tuple] = {}
    for j, rxn in enumerate(model.reactions):
        gpr = rxn.gene_reaction_rule.strip()
        if not gpr:
            continue
        if gpr not in gpr_cache:
            gpr_cache[gpr] = _GPRParser(gpr).parse()
        mapping[j] = _GPRParser.eval_tree(gpr_cache[gpr], gene_rows, gene_matrix)

    return mapping

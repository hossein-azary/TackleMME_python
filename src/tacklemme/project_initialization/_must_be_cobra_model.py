from cobra import Model


def must_be_cobra_model(model: object) -> None:
    """Raise TypeError if model is not a valid COBRApy Model."""
    required = {"S", "reactions", "metabolites", "genes"}
    if isinstance(model, Model):
        return
    # Also accept dict-like COBRA structs (e.g. loaded from .mat files via scipy)
    if isinstance(model, dict):
        missing = [f for f in ("S", "rxns", "mets", "lb", "ub") if f not in model]
        if missing:
            raise TypeError(
                f"Your model is not a valid COBRA model. Missing fields: {', '.join(missing)}"
            )
        return
    raise TypeError(
        f"context_specific_model must be a cobra.Model, got {type(model).__name__}."
    )

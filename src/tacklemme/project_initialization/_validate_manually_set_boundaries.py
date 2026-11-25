def validate_manually_set_boundaries(boundaries: dict) -> None:
    """Validate the manually_set_boundaries dict."""
    accepted = {
        "closed_imports",
        "closed_exports",
        "unconstrained_imports",
        "unconstrained_exports",
    }
    extra = set(boundaries.keys()) - accepted
    if extra:
        raise ValueError(
            f"Not accepted fields in manually_set_boundaries: {', '.join(sorted(extra))}"
        )

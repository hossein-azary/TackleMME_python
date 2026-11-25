def validate_optional_settings(opt_settings: dict) -> None:
    """Validate the optional_settings dict."""
    accepted = {"func", "medium", "not_medium_constrained"}
    extra = set(opt_settings.keys()) - accepted
    if extra:
        raise ValueError(
            f"Not accepted fields in optional_settings: {', '.join(sorted(extra))}"
        )
    for key in accepted:
        if key in opt_settings and not isinstance(opt_settings[key], list):
            raise TypeError(f"optional_settings['{key}'] must be a list.")

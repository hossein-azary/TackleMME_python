from .._project import Project
from ._validate_params_for_pipeline import validate_params_for_pipeline
from ._format_params_for_model import format_params_for_model


def create_project(params: list) -> Project:
    """
    Create a Project ready for the analysis pipeline.

    Parameters
    ----------
    params:
        List of dicts, one per model (a single dict is also accepted).
        Each dict must contain:
            model_name (str)
            context_specific_model (cobra.Model)
        and optionally any of the documented optional fields.

    Returns
    -------
    Project
        Initialised project with a models dict.

    Examples
    --------
    >>> from cobra.io import load_model
    >>> model = load_model("textbook")
    >>> project = create_project([{"model_name": "textbook", "context_specific_model": model}])
    """
    if isinstance(params, dict):
        params = [params]
    if not isinstance(params, list) or len(params) == 0:
        raise ValueError("params must be a dict or a non-empty list of dicts.")

    project = Project()

    for p in params:
        p = validate_params_for_pipeline(p)
        project.models[p["model_name"]] = format_params_for_model(p)

    return project

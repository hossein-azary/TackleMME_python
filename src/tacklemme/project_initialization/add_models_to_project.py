from .._project import Project
from ._check_project_format import check_project_format
from ._validate_params_for_pipeline import validate_params_for_pipeline
from ._format_params_for_model import format_params_for_model


def add_models_to_project(project: Project, params: list) -> Project:
    """
    Add one or more models to an existing project.

    Parameters
    ----------
    project:
        An existing Project created by create_project.
    params:
        List of dicts, one per model to add (same format as create_project);
        a single dict is also accepted.

    Returns
    -------
    Project
        The same project with new models added.

    Notes
    -----
    If a model with the same name already exists, the user is prompted to
    confirm overwriting.
    """
    if isinstance(params, dict):
        params = [params]
    if not isinstance(params, list) or len(params) == 0:
        raise ValueError("params must be a dict or a non-empty list of dicts.")

    print("Checking project format before adding model.")
    check_project_format(project)

    for p in params:
        p = validate_params_for_pipeline(p)
        name = p["model_name"]

        if name in project.models:
            answer = input(f"Model '{name}' already exists. Overwrite? [y/n]: ").strip()
            if answer.lower() != "y":
                print(f"Model '{name}' skipped.")
                continue

        project.models[name] = format_params_for_model(p)

    return project

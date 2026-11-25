"""Python implementation of the TackleMMe analysis pipeline."""

__version__ = "0.1.0"

from ._project import Project
from .project_initialization import create_project, add_models_to_project

__all__ = ["Project", "create_project", "add_models_to_project"]

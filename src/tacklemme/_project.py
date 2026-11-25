class Project:
    """A TackleMMe project containing metabolic models and their analyses."""

    def __init__(self):
        self.models: dict = {}
        self.comparisons: dict = {}

    def __repr__(self):
        return f"Project(models={list(self.models.keys())})"

# Copyright(c) 2023 lyuwenyu. All Rights Reserved.
# Registration/import surface scoped to the manuscript external baselines.
from ._solver import BaseSolver
from .det_solver import DetSolver
TASKS = {"detection": DetSolver}

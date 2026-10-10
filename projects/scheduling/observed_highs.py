"""HiGHS incumbent callbacks using the engine bundled with SciPy.

SciPy's public milp function has no incumbent callback. This optional adapter
isolates its private bundled-HiGHS API; unsupported SciPy builds fail explicitly.
Normal ONTS solves continue to use scipy.optimize.milp.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from scipy.optimize import OptimizeResult


def milp_with_incumbents(*, c, integrality, bounds, constraints, options, on_incumbent,
                         deadline=None):
    from scipy.optimize._highspy import _core as h

    solver = h._Highs()
    for name, value in dict(options, output_flag=False).items():
        if value is not None and solver.setOptionValue(name, value) == h.HighsStatus.kError:
            raise RuntimeError(f"HiGHS rejected option {name}")
    lp = h.HighsLp()
    matrix = constraints.A.tocsc()
    lp.num_col_, lp.num_row_ = matrix.shape[1], matrix.shape[0]
    lp.col_cost_, lp.col_lower_, lp.col_upper_ = c, bounds.lb, bounds.ub
    lp.row_lower_, lp.row_upper_ = constraints.lb, constraints.ub
    lp.integrality_ = [h.HighsVarType(int(i)) for i in integrality]
    lp.a_matrix_.format_ = h.MatrixFormat.kColwise
    lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = lp.num_col_, lp.num_row_
    lp.a_matrix_.start_ = matrix.indptr.astype(np.int32)
    lp.a_matrix_.index_ = matrix.indices.astype(np.int32)
    lp.a_matrix_.value_ = matrix.data
    if solver.passModel(lp) == h.HighsStatus.kError:
        raise RuntimeError("HiGHS rejected callback model")
    errors = []

    def callback(kind, message, output, input_, user_data):
        try:
            on_incumbent(np.asarray(output.mip_solution).copy())
        except Exception as error:
            # Exceptions must not unwind through a C++ callback.
            errors.append(error)
            input_.user_interrupt = True

    if solver.setCallback(callback, None) == h.HighsStatus.kError:
        raise RuntimeError("HiGHS incumbent callbacks unavailable")
    kind = h.cb.HighsCallbackType.kCallbackMipImprovingSolution
    if solver.startCallback(kind) == h.HighsStatus.kError:
        raise RuntimeError("HiGHS could not start incumbent callback")
    if deadline is not None:
        remaining = deadline - perf_counter()
        if remaining <= 0:
            return OptimizeResult(status=1, message="Allowance exhausted during model loading",
                                  x=None, mip_dual_bound=None)
        if solver.setOptionValue("time_limit", remaining) == h.HighsStatus.kError:
            raise RuntimeError("HiGHS rejected remaining allowance")
    run_status = solver.run()
    if errors:
        raise RuntimeError("Incumbent callback failed") from errors[0]
    status = solver.getModelStatus()
    info, solution = solver.getInfo(), solver.getSolution()
    statuses = {
        h.HighsModelStatus.kOptimal: 0,
        h.HighsModelStatus.kTimeLimit: 1,
        h.HighsModelStatus.kIterationLimit: 1,
        h.HighsModelStatus.kSolutionLimit: 1,
        h.HighsModelStatus.kInfeasible: 2,
        h.HighsModelStatus.kUnbounded: 3,
    }
    code = statuses.get(status, 4)
    if run_status == h.HighsStatus.kError:
        code = 4
    valid = solution.value_valid and info.primal_solution_status == h.kSolutionStatusFeasible
    return OptimizeResult(status=code, message=solver.modelStatusToString(status),
                          x=np.asarray(solution.col_value) if valid else None,
                          mip_dual_bound=info.mip_dual_bound)

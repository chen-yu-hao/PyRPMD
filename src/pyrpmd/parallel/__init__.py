"""Task and bead parallelism with explicit resource ownership.

MPI is optional. In MPI mode all ranks call ``run_tasks``; only world rank zero
receives results. A task executes on its group leader and obtains its potential
through ``make_task_potential``. Other ranks serve PES requests for that group.
"""

from .execution import (
    TaskExecutor,
    execution_plan,
    is_root,
    make_task_potential,
    task_potential,
    run_tasks,
    ProcessBeadPotential,
    broadcast,
    root_call,
)

__all__ = [
    "TaskExecutor",
    "execution_plan",
    "is_root",
    "make_task_potential",
    "task_potential",
    "run_tasks",
    "ProcessBeadPotential",
    "broadcast",
    "root_call",
]

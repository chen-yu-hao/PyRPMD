"""Execution receipts, budget reservations, and recovery at the F boundary."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json

import pytest

import pyrpmd.precision_ledger as ledger_module
from pyrpmd.precision_ledger import (
    PrecisionBudgetError,
    PrecisionLedger,
    PrecisionLedgerError,
)


SPEC = {
    "scientific_identity": "thermal-P32-dt0.1-bias0",
    "epoch_id": "frozen-production-1",
    "checks": [10, 20, 40],
    "alpha": 0.05,
}


def plan(plan_id="plan-1", cost=4.0):
    return {
        "plan_id": plan_id,
        "scientific_identity": SPEC["scientific_identity"],
        "epoch_id": SPEC["epoch_id"],
        "estimated_cost": cost,
        "actions": [{"kind": "add_parent_chains", "count": 4}],
    }


def ledger(path, budget=10):
    return PrecisionLedger(path, specification=SPEC, max_cost=budget)


def finish(store, plan_id="plan-1", execution_id="execution-1", actual_cost=3.0,
           result_ids=None, result=None):
    return store.complete(
        plan_id, execution_id=execution_id, actual_cost=actual_cost,
        result_ids=["chain-1", "chain-2"] if result_ids is None else result_ids,
        result={"parent_chains": 2} if result is None else result,
    )


def test_interrupt_and_recovery_do_not_duplicate_statistics_or_cost(tmp_path):
    store = ledger(tmp_path)
    store.register_plan(plan())
    store.start("plan-1", execution_id="execution-1")

    recovered = ledger(tmp_path)
    before = recovered.snapshot()
    assert before["budget"]["pending_reconciliation"] == ["plan-1"]
    assert before["budget"]["reserved_cost"] == 4
    assert before["budget"]["spent_cost"] == 0
    assert recovered.register_plan(plan())["state"] == "running"
    assert recovered.start("plan-1", execution_id="execution-1")["state"] == "running"
    assert recovered.snapshot() == before

    finish(recovered)
    committed = recovered.snapshot()
    replayed = ledger(tmp_path)
    finish(replayed)
    assert replayed.start("plan-1", execution_id="execution-1")["state"] == "completed"
    assert replayed.snapshot() == committed
    assert committed["budget"]["spent_cost"] == 3
    assert committed["budget"]["reserved_cost"] == 0
    assert committed["budget"]["available_cost"] == 7
    assert len(committed["results"]) == 2
    assert committed["budget"]["pending_reconciliation"] == []


def test_interruption_before_atomic_publication_leaves_one_recoverable_receipt(tmp_path, monkeypatch):
    store = ledger(tmp_path)
    store.register_plan(plan())
    store.start("plan-1", execution_id="execution-1")
    running = store.snapshot()

    def interrupted_replace(source, destination):
        raise OSError("simulated interruption before publication")

    with monkeypatch.context() as patch:
        patch.setattr(ledger_module.os, "replace", interrupted_replace)
        with pytest.raises(OSError, match="simulated interruption"):
            finish(store)
    assert store.snapshot() == running
    assert not list(store.path.parent.glob(".ledger-*.tmp"))
    recovered = ledger(tmp_path)
    finish(recovered)
    finish(recovered)
    assert recovered.snapshot()["budget"]["spent_cost"] == 3
    assert len(recovered.snapshot()["results"]) == 2


def test_frozen_specification_and_budget_cannot_change_on_resume(tmp_path):
    ledger(tmp_path)
    changed = deepcopy(SPEC)
    changed["checks"] = [10, 15, 40]
    with pytest.raises(PrecisionLedgerError, match="specification differs"):
        PrecisionLedger(tmp_path, specification=changed, max_cost=10)
    with pytest.raises(PrecisionLedgerError, match="budget differs"):
        ledger(tmp_path, budget=11)


def test_conflicting_plan_execution_receipt_and_result_replays_fail(tmp_path):
    store = ledger(tmp_path)
    store.register_plan(plan())
    with pytest.raises(PrecisionLedgerError, match="Plan ID reused"):
        store.register_plan(plan(cost=5))
    with pytest.raises(PrecisionLedgerError, match="execution ID differs"):
        finish(store)
    store.start("plan-1", execution_id="execution-1")
    with pytest.raises(PrecisionLedgerError, match="reconcile the original"):
        store.start("plan-1", execution_id="another-execution")
    finish(store)
    original = store.snapshot()
    with pytest.raises(PrecisionLedgerError, match="immutable execution receipt"):
        finish(store, actual_cost=2)
    with pytest.raises(PrecisionLedgerError, match="immutable execution receipt"):
        finish(store, result={"parent_chains": 3})
    assert store.snapshot() == original

    store.register_plan(plan("plan-2"))
    with pytest.raises(PrecisionLedgerError, match="another plan"):
        store.start("plan-2", execution_id="execution-1")
    store.start("plan-2", execution_id="execution-2")
    before_conflict = store.snapshot()
    with pytest.raises(PrecisionLedgerError, match="already accounted for"):
        finish(store, plan_id="plan-2", execution_id="execution-2")
    assert store.snapshot() == before_conflict


def test_reservations_block_oversubscription_and_actual_overruns_are_truthful(tmp_path):
    store = ledger(tmp_path)
    store.register_plan(plan(cost=7))
    with pytest.raises(PrecisionBudgetError):
        store.register_plan(plan("plan-2", cost=4))
    store.register_plan(plan("plan-2", cost=3))
    assert store.snapshot()["budget"]["budget_exhausted"] is True
    store.start("plan-1", execution_id="execution-1")
    finish(store, actual_cost=12)
    state = store.snapshot()
    assert state["budget"]["spent_cost"] == 12
    assert state["budget"]["reserved_cost"] == 3
    assert state["budget"]["budget_exceeded"] is True
    assert state["budget"]["available_cost"] == 0
    with pytest.raises(PrecisionBudgetError):
        store.register_plan(plan("plan-3", cost=0))
    with pytest.raises(PrecisionBudgetError, match="reserved work"):
        store.start("plan-2", execution_id="execution-2")


def test_failed_partial_execution_consumes_cost_and_cannot_promote_results(tmp_path):
    store = ledger(tmp_path)
    store.register_plan(plan(cost=6))
    store.start("plan-1", execution_id="execution-1")
    kwargs = dict(execution_id="execution-1", actual_cost=2,
                  reason="interrupted after a partial parent chain",
                  result_ids=["partial-chain"], result={"complete": False})
    store.fail("plan-1", **kwargs)
    before = store.snapshot()
    assert before["budget"]["spent_cost"] == 2
    assert before["budget"]["reserved_cost"] == 0
    assert before["results"]["partial-chain"]["execution_state"] == "failed"
    store.fail("plan-1", **kwargs)
    assert store.snapshot() == before
    with pytest.raises(PrecisionLedgerError, match="immutable execution receipt"):
        finish(store)
    with pytest.raises(PrecisionLedgerError, match="reconcile the original"):
        store.start("plan-1", execution_id="execution-2")
    store.register_plan(plan("retry-1", cost=8))
    store.start("retry-1", execution_id="retry-execution-1")
    with pytest.raises(PrecisionLedgerError, match="already accounted for"):
        finish(store, plan_id="retry-1", execution_id="retry-execution-1",
               result_ids=["partial-chain"])


def test_unknown_failure_cost_keeps_reservation_for_reconciliation(tmp_path):
    store = ledger(tmp_path)
    store.register_plan(plan())
    store.start("plan-1", execution_id="execution-1")
    with pytest.raises(PrecisionLedgerError, match="actual_cost"):
        store.fail("plan-1", execution_id="execution-1", actual_cost=None, reason="lost worker")
    state = store.snapshot()
    assert state["budget"]["pending_reconciliation"] == ["plan-1"]
    assert state["budget"]["reserved_cost"] == 4


def test_checks_and_stop_preserve_unestablished_numerical_convergence(tmp_path):
    store = ledger(tmp_path, budget=4)
    store.register_plan(plan())
    store.start("plan-1", execution_id="execution-1")
    finish(store, actual_cost=4)
    payload = {
        "statistical_target_met": True,
        "numerical_convergence_established": False,
        "numerical_status": "unestablished",
        "source_result_ids": ["chain-1", "chain-2"],
    }
    store.record_check("look-1", payload)
    checked = store.snapshot()
    store.record_check("look-1", payload)
    assert store.snapshot() == checked
    with pytest.raises(PrecisionLedgerError, match="Check ID reused"):
        store.record_check("look-1", {**payload, "numerical_convergence_established": True})
    stop = store.stop("budget_exhausted", ["No budget for a beads comparison"], payload)
    assert stop["conclusions"]["numerical_convergence_established"] is False
    saved = ledger(tmp_path, budget=4).snapshot()
    assert saved["checks"]["look-1"]["numerical_convergence_established"] is False
    assert saved["stop"]["status"] == "budget_exhausted"
    store.stop("budget_exhausted", ["No budget for a beads comparison"], payload)
    assert store.snapshot() == saved
    with pytest.raises(PrecisionLedgerError, match="immutable stop record"):
        store.stop("statistical_target_met", [], payload)
    with pytest.raises(PrecisionLedgerError, match="recorded stop"):
        store.register_plan(plan("another", cost=0))
    with pytest.raises(PrecisionLedgerError, match="recorded stop"):
        store.record_check("look-2", payload)


def test_stop_still_allows_running_costs_to_be_reconciled(tmp_path):
    store = ledger(tmp_path)
    store.register_plan(plan())
    store.start("plan-1", execution_id="execution-1")
    store.stop("diagnostic_failed", ["energy drift"], {"statistical_target_met": False})
    finish(store)
    assert store.snapshot()["budget"]["spent_cost"] == 3


def test_atomic_reservations_serialize_concurrent_writers(tmp_path):
    store = ledger(tmp_path)

    def reserve(number):
        other = ledger(tmp_path)
        try:
            other.register_plan(plan(f"plan-{number}", cost=6))
        except PrecisionBudgetError:
            return "budget"
        return "reserved"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(reserve, [1, 2]))
    assert sorted(outcomes) == ["budget", "reserved"]
    assert store.snapshot()["budget"]["reserved_cost"] == 6
    assert len(store.snapshot()["plans"]) == 1


def test_corruption_does_not_restart_accounting(tmp_path):
    store = ledger(tmp_path)
    store.register_plan(plan())
    data = json.loads(store.path.read_text())
    data["plans"] = {}
    store.path.write_text(json.dumps(data))
    with pytest.raises(PrecisionLedgerError, match="checksum differs"):
        ledger(tmp_path)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "4"])
def test_invalid_costs_rejected(tmp_path, value):
    with pytest.raises(PrecisionLedgerError):
        ledger(tmp_path, budget=value)
    store = ledger(tmp_path)
    with pytest.raises(PrecisionLedgerError):
        store.register_plan(plan(cost=value))


def test_receipts_and_snapshots_do_not_alias_caller_data(tmp_path):
    store = ledger(tmp_path)
    proposed = plan()
    registered = store.register_plan(proposed)
    proposed["actions"][0]["count"] = 100
    registered["plan"]["actions"][0]["count"] = 200
    assert store.snapshot()["plans"]["plan-1"]["plan"]["actions"][0]["count"] == 4
    snapshot = store.snapshot()
    snapshot["plans"].clear()
    assert len(store.snapshot()["plans"]) == 1

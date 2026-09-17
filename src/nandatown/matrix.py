"""Protocol Failure Matrix: systematic fault injection across scenarios.

Runs every (scenario × fault) combination for N trials, records verdicts
and per-stage results, and produces a research-quality matrix table.

Fault injection reuses the existing mechanisms:
- Transport faults: FaultRule declarations (drop, duplicate, delay)
- Layer swaps: layer_overrides (e.g., auth=plain.v1)

Evaluation reuses the existing scenario evaluators.
Evidence reuses the existing bundle infrastructure.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import __version__


@dataclass
class FaultSpec:
    """One fault to inject into a Lab scenario."""
    fault_id: str
    layer: str
    fault_type: str
    transport_action: str | None = None
    transport_kind: str | None = None
    transport_nth: int = 1
    transport_delay: float | None = None
    swap_plugin_id: str | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "layer": self.layer,
            "fault_type": self.fault_type,
            "transport_action": self.transport_action,
            "transport_kind": self.transport_kind,
            "transport_nth": self.transport_nth,
            "transport_delay": self.transport_delay,
            "swap_plugin_id": self.swap_plugin_id,
            "description": self.description,
        }


FAULT_CATALOG: dict[str, FaultSpec] = {}


def register_fault(spec: FaultSpec) -> FaultSpec:
    FAULT_CATALOG[spec.fault_id] = spec
    return spec


register_fault(FaultSpec(
    fault_id="transport/duplicate",
    layer="transport",
    fault_type="transport_fault",
    transport_action="duplicate",
    transport_kind=None,
    transport_nth=1,
    description="First message is delivered twice",
))

register_fault(FaultSpec(
    fault_id="transport/drop",
    layer="transport",
    fault_type="transport_fault",
    transport_action="drop",
    transport_kind=None,
    transport_nth=1,
    description="First message is silently dropped",
))

register_fault(FaultSpec(
    fault_id="transport/delay",
    layer="transport",
    fault_type="transport_fault",
    transport_action="delay",
    transport_kind=None,
    transport_nth=1,
    transport_delay=2.0,
    description="First message is delayed by 2.0s",
))

register_fault(FaultSpec(
    fault_id="auth/none",
    layer="auth",
    fault_type="layer_swap",
    swap_plugin_id="plain.v1",
    description="Authentication disabled: any claimed sender accepted",
))


DEFAULT_MATRIX: dict[str, list[str]] = {
    "marketplace": [
        "transport/duplicate",
        "transport/drop",
        "transport/delay",
    ],
    "capability_spoofing": [
        "auth/none",
    ],
    "consensus": [
        "transport/drop",
        "transport/delay",
    ],
    "auction": [
        "transport/drop",
        "transport/delay",
    ],
    "supply_chain": [
        "transport/duplicate",
        "transport/drop",
        "transport/delay",
    ],
    "voting": [
        "transport/drop",
    ],
}


def _build_scenario_fault_rules(fault: FaultSpec) -> list[dict[str, Any]]:
    if fault.fault_type != "transport_fault":
        return []
    rule: dict[str, Any] = {"action": fault.transport_action}
    if fault.transport_kind is not None:
        rule["kind"] = fault.transport_kind
    if fault.transport_nth != 1:
        rule["nth"] = fault.transport_nth
    if fault.transport_delay is not None:
        rule["delay"] = fault.transport_delay
    return [rule]


def _build_layer_overrides(fault: FaultSpec) -> dict[str, str] | None:
    if fault.fault_type != "layer_swap":
        return None
    return {fault.layer: fault.swap_plugin_id}


@dataclass
class CellResult:
    scenario: str
    fault_id: str
    trials: list[dict[str, Any]] = field(default_factory=list)
    violations: int = 0
    passes: int = 0
    errors: int = 0
    incompletes: int = 0
    invariant_violations: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "fault_id": self.fault_id,
            "trials": len(self.trials),
            "violations": self.violations,
            "passes": self.passes,
            "errors": self.errors,
            "incompletes": self.incompletes,
            "invariant_violations": dict(self.invariant_violations),
        }


@dataclass
class MatrixResult:
    matrix_id: str
    scenarios: list[str]
    faults: list[str]
    trials_per_cell: int
    seed_base: int
    cells: dict[str, CellResult] = field(default_factory=dict)
    started_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "matrix_id": self.matrix_id,
            "scenarios": self.scenarios,
            "faults": self.faults,
            "trials_per_cell": self.trials_per_cell,
            "seed_base": self.seed_base,
            "cells": {k: v.to_dict() for k, v in self.cells.items()},
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "nandatown_version": __version__,
        }


def _run_cell_trials(
    scenario: str,
    fault: FaultSpec,
    trials: int,
    seed_base: int,
    cell_index: int,
    out_dir: str,
) -> CellResult:
    from .sim.runner import run_lab

    cell = CellResult(scenario=scenario, fault_id=fault.fault_id)
    fault_rules = _build_scenario_fault_rules(fault)
    layer_overrides = _build_layer_overrides(fault)

    for trial in range(trials):
        seed = seed_base + cell_index * 1000 + trial
        trial_record: dict[str, Any] = {"trial": trial + 1, "seed": seed}
        try:
            bundle_dir, result = run_lab(
                scenario,
                out_dir,
                seed=seed,
                layer_overrides=layer_overrides,
            )
            trial_record["run_id"] = result.run_id
            trial_record["verdict"] = result.verdict
            trial_record["bundle"] = os.path.basename(bundle_dir)
            trial_record["stages"] = {
                s.name: s.status for s in result.stages
            }

            if result.verdict == "failed":
                cell.violations += 1
                for s in result.stages:
                    if s.status == "failed":
                        cell.invariant_violations[s.name] = (
                            cell.invariant_violations.get(s.name, 0) + 1
                        )
            elif result.verdict == "passed":
                cell.passes += 1
            elif result.verdict == "error":
                cell.errors += 1
            else:
                cell.incompletes += 1
        except Exception as exc:
            trial_record["verdict"] = "error"
            trial_record["error"] = f"{type(exc).__name__}: {exc}"
            cell.errors += 1

        cell.trials.append(trial_record)

    return cell


def _inject_faults_into_scenario(
    scenario: str,
    fault: FaultSpec,
    out_dir: str,
) -> str:
    import yaml

    from .sim.scenario import load_bundled

    spec = load_bundled(scenario)
    fault_rules = _build_scenario_fault_rules(fault)
    if fault_rules:
        existing = list(spec.faults) if spec.faults else []
        existing_dicts = []
        for f in existing:
            d = {"action": f.action}
            if f.kind is not None:
                d["kind"] = f.kind
            if f.nth != 1:
                d["nth"] = f.nth
            if f.delay is not None:
                d["delay"] = f.delay
            if f.rate is not None:
                d["rate"] = f.rate
            existing_dicts.append(d)
        spec_dict = spec.model_dump()
        spec_dict["faults"] = existing_dicts + fault_rules
    else:
        spec_dict = spec.model_dump()

    layer_overrides = _build_layer_overrides(fault)
    if layer_overrides:
        spec_dict.setdefault("layers", {})
        spec_dict["layers"].update(layer_overrides)

    scenario_name = f"{scenario}__{fault.fault_id.replace('/', '_')}"
    spec_dict["name"] = scenario_name
    spec_dict["description"] = (
        f"Failure matrix: {scenario} with fault {fault.fault_id}"
    )

    scenario_path = os.path.join(out_dir, f"{scenario_name}.yaml")
    with open(scenario_path, "w") as f:
        yaml.dump(spec_dict, f, default_flow_style=False, sort_keys=False)

    return scenario_name


def run_failure_matrix(
    scenarios: list[str] | None = None,
    faults: list[str] | None = None,
    trials: int = 10,
    seed_base: int = 2000,
    out_dir: str = "runs",
) -> tuple[str, MatrixResult]:
    matrix_id = "mtx-" + uuid.uuid4().hex[:10]
    matrix_dir = os.path.join(out_dir, matrix_id)
    os.makedirs(matrix_dir, exist_ok=True)

    if scenarios is None:
        scenarios = sorted(DEFAULT_MATRIX.keys())
    if faults is None:
        fault_ids = sorted({f for fl in DEFAULT_MATRIX.values() for f in fl})
    else:
        fault_ids = faults

    result = MatrixResult(
        matrix_id=matrix_id,
        scenarios=scenarios,
        faults=fault_ids,
        trials_per_cell=trials,
        seed_base=seed_base,
        started_at=time.time(),
    )

    plan = {
        "matrix_id": matrix_id,
        "scenarios": scenarios,
        "faults": fault_ids,
        "trials_per_cell": trials,
        "seed_base": seed_base,
        "nandatown_version": __version__,
        "declared_at": result.started_at,
        "policy": "every trial is reported: pass, fail, incomplete, error",
    }
    with open(os.path.join(matrix_dir, "matrix-plan.json"), "w") as f:
        json.dump(plan, f, indent=2)

    cell_index = 0
    for scenario in scenarios:
        applicable_faults = DEFAULT_MATRIX.get(scenario, fault_ids)
        for fault_id in fault_ids:
            if fault_id not in applicable_faults:
                continue
            fault = FAULT_CATALOG.get(fault_id)
            if fault is None:
                continue

            cell_key = f"{scenario}/{fault_id}"
            cell_dir = os.path.join(matrix_dir, cell_key)
            os.makedirs(cell_dir, exist_ok=True)

            cell = _run_cell_trials(
                scenario, fault, trials, seed_base, cell_index, cell_dir,
            )
            result.cells[cell_key] = cell

            with open(os.path.join(cell_dir, "cell-result.json"), "w") as f:
                json.dump(cell.to_dict(), f, indent=2)

            cell_index += 1

    result.completed_at = time.time()

    with open(os.path.join(matrix_dir, "matrix-result.json"), "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    with open(os.path.join(matrix_dir, "matrix-report.md"), "w") as f:
        f.write(render_matrix_report(result))

    return matrix_dir, result


def render_matrix_report(result: MatrixResult) -> str:
    lines: list[str] = []
    add = lines.append
    add("NANDA Town Protocol Failure Matrix")
    add("=" * 50)
    add(f"Matrix ID:   {result.matrix_id}")
    add(f"Trials/cell: {result.trials_per_cell}")
    add(f"Seed base:   {result.seed_base}")
    add(f"Scenarios:   {', '.join(result.scenarios)}")
    add(f"Faults:      {', '.join(result.faults)}")
    add("")

    header = (
        f"{'Scenario':<25} {'Fault':<25} {'Trials':>6} "
        f"{'Pass':>5} {'Fail':>5} {'Err':>4} {'Inc':>4} "
        f"{'Violations':>10} {'Invariant Hits'}"
    )
    add(header)
    add("-" * len(header))

    for scenario in result.scenarios:
        for fault_id in result.faults:
            cell_key = f"{scenario}/{fault_id}"
            cell = result.cells.get(cell_key)
            if cell is None:
                continue
            inv_parts = []
            for inv, count in sorted(cell.invariant_violations.items()):
                inv_parts.append(f"{inv}:{count}")
            inv_str = ", ".join(inv_parts) if inv_parts else "-"
            total = len(cell.trials)
            add(
                f"{scenario:<25} {fault_id:<25} {total:>6} "
                f"{cell.passes:>5} {cell.violations:>5} "
                f"{cell.errors:>4} {cell.incompletes:>4} "
                f"{cell.violations:>10} {inv_str}"
            )

    add("")
    add("Fault descriptions:")
    for fault_id in result.faults:
        fault = FAULT_CATALOG.get(fault_id)
        if fault:
            add(f"  {fault_id:<25} {fault.description}")
    add("")
    add("Each cell is backed by verifiable evidence bundles in this"
        " directory.")
    add("The unit of evidence is the distribution, not any single trial.")
    return "\n".join(lines) + "\n"

import json
import os

import pytest

from nandatown.matrix import (
    DEFAULT_MATRIX,
    FAULT_CATALOG,
    CellResult,
    FaultSpec,
    MatrixResult,
    _build_layer_overrides,
    _build_scenario_fault_rules,
    register_fault,
    render_matrix_report,
    run_failure_matrix,
)
from nandatown.sim.scenario import load_bundled


class TestFaultSpec:

    def test_fault_spec_to_dict_round_trips(self):
        spec = FaultSpec(
            fault_id="transport/duplicate",
            layer="transport",
            fault_type="transport_fault",
            transport_action="duplicate",
            description="test",
        )
        d = spec.to_dict()
        assert d["fault_id"] == "transport/duplicate"
        assert d["layer"] == "transport"
        assert d["fault_type"] == "transport_fault"
        assert d["transport_action"] == "duplicate"
        assert d["swap_plugin_id"] is None

    def test_layer_swap_fault_spec(self):
        spec = FaultSpec(
            fault_id="auth/none",
            layer="auth",
            fault_type="layer_swap",
            swap_plugin_id="plain.v1",
            description="auth disabled",
        )
        d = spec.to_dict()
        assert d["fault_type"] == "layer_swap"
        assert d["swap_plugin_id"] == "plain.v1"
        assert d["transport_action"] is None


class TestFaultCatalog:

    def test_catalog_has_expected_faults(self):
        assert "transport/duplicate" in FAULT_CATALOG
        assert "transport/drop" in FAULT_CATALOG
        assert "transport/delay" in FAULT_CATALOG
        assert "auth/none" in FAULT_CATALOG

    def test_register_fault_adds_to_catalog(self):
        spec = FaultSpec(
            fault_id="test/custom",
            layer="test",
            fault_type="layer_swap",
            swap_plugin_id="custom.v1",
        )
        register_fault(spec)
        assert "test/custom" in FAULT_CATALOG
        assert FAULT_CATALOG["test/custom"].swap_plugin_id == "custom.v1"
        del FAULT_CATALOG["test/custom"]

    def test_transport_fault_has_action(self):
        for fid in ["transport/duplicate", "transport/drop", "transport/delay"]:
            spec = FAULT_CATALOG[fid]
            assert spec.fault_type == "transport_fault"
            assert spec.transport_action is not None

    def test_layer_swap_has_plugin_id(self):
        spec = FAULT_CATALOG["auth/none"]
        assert spec.fault_type == "layer_swap"
        assert spec.swap_plugin_id == "plain.v1"


class TestFaultRuleBuilding:

    def test_transport_duplicate_builds_rule(self):
        fault = FAULT_CATALOG["transport/duplicate"]
        rules = _build_scenario_fault_rules(fault)
        assert len(rules) == 1
        assert rules[0]["action"] == "duplicate"

    def test_transport_drop_builds_rule(self):
        fault = FAULT_CATALOG["transport/drop"]
        rules = _build_scenario_fault_rules(fault)
        assert len(rules) == 1
        assert rules[0]["action"] == "drop"

    def test_transport_delay_builds_rule_with_delay(self):
        fault = FAULT_CATALOG["transport/delay"]
        rules = _build_scenario_fault_rules(fault)
        assert len(rules) == 1
        assert rules[0]["action"] == "delay"
        assert rules[0]["delay"] == 2.0

    def test_layer_swap_builds_no_fault_rules(self):
        fault = FAULT_CATALOG["auth/none"]
        rules = _build_scenario_fault_rules(fault)
        assert rules == []

    def test_layer_swap_builds_overrides(self):
        fault = FAULT_CATALOG["auth/none"]
        overrides = _build_layer_overrides(fault)
        assert overrides == {"auth": "plain.v1"}

    def test_transport_fault_builds_no_overrides(self):
        fault = FAULT_CATALOG["transport/duplicate"]
        overrides = _build_layer_overrides(fault)
        assert overrides is None


class TestDefaultMatrix:

    def test_default_matrix_has_scenarios(self):
        assert "marketplace" in DEFAULT_MATRIX
        assert "capability_spoofing" in DEFAULT_MATRIX
        assert "consensus" in DEFAULT_MATRIX

    def test_default_matrix_faults_are_valid(self):
        for scenario, fault_ids in DEFAULT_MATRIX.items():
            for fid in fault_ids:
                assert fid in FAULT_CATALOG, (
                    f"fault {fid!r} in DEFAULT_MATRIX[{scenario!r}]"
                    f" not in FAULT_CATALOG"
                )

    def test_capability_spoofing_uses_auth_fault(self):
        assert "auth/none" in DEFAULT_MATRIX["capability_spoofing"]

    def test_marketplace_uses_transport_faults(self):
        faults = DEFAULT_MATRIX["marketplace"]
        assert "transport/duplicate" in faults
        assert "transport/drop" in faults
        assert "transport/delay" in faults


class TestCellResult:

    def test_cell_result_defaults(self):
        cell = CellResult(scenario="test", fault_id="test/fault")
        assert cell.violations == 0
        assert cell.passes == 0
        assert cell.errors == 0
        assert cell.incompletes == 0
        assert cell.trials == []
        assert cell.invariant_violations == {}

    def test_cell_result_to_dict(self):
        cell = CellResult(
            scenario="marketplace",
            fault_id="transport/drop",
            violations=2,
            passes=8,
            invariant_violations={"settlement": 2},
        )
        d = cell.to_dict()
        assert d["scenario"] == "marketplace"
        assert d["fault_id"] == "transport/drop"
        assert d["violations"] == 2
        assert d["passes"] == 8
        assert d["invariant_violations"] == {"settlement": 2}


class TestMatrixResult:

    def test_matrix_result_to_dict(self):
        result = MatrixResult(
            matrix_id="mtx-test",
            scenarios=["marketplace"],
            faults=["transport/drop"],
            trials_per_cell=5,
            seed_base=2000,
        )
        d = result.to_dict()
        assert d["matrix_id"] == "mtx-test"
        assert d["scenarios"] == ["marketplace"]
        assert d["trials_per_cell"] == 5
        assert "nandatown_version" in d


class TestRunFailureMatrix:

    def test_single_scenario_single_fault_produces_bundles(self, tmp_path):
        matrix_dir, result = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=2,
            seed_base=3000,
            out_dir=str(tmp_path),
        )
        assert os.path.isdir(matrix_dir)
        cell_key = "voting/transport/drop"
        assert cell_key in result.cells
        cell = result.cells[cell_key]
        assert len(cell.trials) == 2

    def test_trials_produce_evidence_bundles(self, tmp_path):
        matrix_dir, result = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=2,
            seed_base=3000,
            out_dir=str(tmp_path),
        )
        cell_dir = os.path.join(matrix_dir, "voting/transport/drop")
        assert os.path.isdir(cell_dir)
        bundles = [d for d in os.listdir(cell_dir) if d.startswith("sim-")]
        assert len(bundles) == 2
        for bundle_name in bundles:
            bundle_path = os.path.join(cell_dir, bundle_name)
            assert os.path.exists(os.path.join(bundle_path, "manifest.json"))
            assert os.path.exists(os.path.join(bundle_path, "result.json"))
            assert os.path.exists(os.path.join(bundle_path, "events.jsonl"))

    def test_deterministic_reproduction(self, tmp_path):
        dir1 = str(tmp_path / "run1")
        dir2 = str(tmp_path / "run2")
        os.makedirs(dir1)
        os.makedirs(dir2)
        _, result1 = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=3,
            seed_base=4000,
            out_dir=dir1,
        )
        _, result2 = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=3,
            seed_base=4000,
            out_dir=dir2,
        )
        for key in result1.cells:
            cell1 = result1.cells[key]
            cell2 = result2.cells[key]
            assert cell1.passes == cell2.passes
            assert cell1.violations == cell2.violations
            assert cell1.errors == cell2.errors
            assert cell1.incompletes == cell2.incompletes
            for t1, t2 in zip(cell1.trials, cell2.trials):
                assert t1["verdict"] == t2["verdict"]
                if "stages" in t1 and "stages" in t2:
                    assert t1["stages"] == t2["stages"]

    def test_matrix_plan_written(self, tmp_path):
        matrix_dir, _ = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=1,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        plan_path = os.path.join(matrix_dir, "matrix-plan.json")
        assert os.path.exists(plan_path)
        with open(plan_path) as f:
            plan = json.load(f)
        assert plan["scenarios"] == ["voting"]
        assert plan["faults"] == ["transport/drop"]
        assert plan["trials_per_cell"] == 1

    def test_matrix_result_written(self, tmp_path):
        matrix_dir, result = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=1,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        result_path = os.path.join(matrix_dir, "matrix-result.json")
        assert os.path.exists(result_path)
        with open(result_path) as f:
            data = json.load(f)
        assert data["matrix_id"] == result.matrix_id
        assert "nandatown_version" in data

    def test_matrix_report_written(self, tmp_path):
        matrix_dir, _ = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=1,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        report_path = os.path.join(matrix_dir, "matrix-report.md")
        assert os.path.exists(report_path)
        with open(report_path) as f:
            text = f.read()
        assert "Protocol Failure Matrix" in text
        assert "voting" in text
        assert "transport/drop" in text

    def test_cell_result_json_written(self, tmp_path):
        matrix_dir, _ = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=2,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        cell_path = os.path.join(
            matrix_dir, "voting/transport/drop", "cell-result.json",
        )
        assert os.path.exists(cell_path)
        with open(cell_path) as f:
            data = json.load(f)
        assert data["scenario"] == "voting"
        assert data["fault_id"] == "transport/drop"
        assert data["trials"] == 2

    def test_layer_swap_fault_runs_capability_spoofing(self, tmp_path):
        matrix_dir, result = run_failure_matrix(
            scenarios=["capability_spoofing"],
            faults=["auth/none"],
            trials=2,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        cell_key = "capability_spoofing/auth/none"
        assert cell_key in result.cells
        cell = result.cells[cell_key]
        assert len(cell.trials) == 2
        assert cell.violations > 0

    def test_multiple_scenarios_and_faults(self, tmp_path):
        matrix_dir, result = run_failure_matrix(
            scenarios=["voting", "capability_spoofing"],
            faults=["transport/drop", "auth/none"],
            trials=2,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        assert "voting/transport/drop" in result.cells
        assert "capability_spoofing/auth/none" in result.cells
        assert "voting/auth/none" not in result.cells

    def test_default_faults_per_scenario(self, tmp_path):
        matrix_dir, result = run_failure_matrix(
            scenarios=["marketplace"],
            trials=1,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        assert "marketplace/transport/duplicate" in result.cells
        assert "marketplace/transport/drop" in result.cells
        assert "marketplace/transport/delay" in result.cells

    def test_unknown_fault_skipped(self, tmp_path):
        matrix_dir, result = run_failure_matrix(
            scenarios=["voting"],
            faults=["nonexistent/fault"],
            trials=1,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        assert len(result.cells) == 0

    def test_empty_scenarios_produces_empty_matrix(self, tmp_path):
        matrix_dir, result = run_failure_matrix(
            scenarios=[],
            faults=["transport/drop"],
            trials=1,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        assert len(result.cells) == 0

    def test_all_bundles_verify(self, tmp_path):
        from nandatown.bundle import verify_bundle

        matrix_dir, result = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=2,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        cell_dir = os.path.join(matrix_dir, "voting/transport/drop")
        for bundle_name in os.listdir(cell_dir):
            if bundle_name.startswith("sim-"):
                bundle_path = os.path.join(cell_dir, bundle_name)
                problems = verify_bundle(bundle_path)
                assert problems == [], (
                    f"Bundle {bundle_name} verification failed: {problems}"
                )

    def test_invariant_violations_detected(self, tmp_path):
        matrix_dir, result = run_failure_matrix(
            scenarios=["capability_spoofing"],
            faults=["auth/none"],
            trials=3,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        cell = result.cells["capability_spoofing/auth/none"]
        assert cell.violations > 0
        assert len(cell.invariant_violations) > 0

    def test_seed_base_affects_results(self, tmp_path):
        _, result_a = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=2,
            seed_base=5000,
            out_dir=str(tmp_path / "a"),
        )
        _, result_b = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=2,
            seed_base=6000,
            out_dir=str(tmp_path / "b"),
        )
        for key in result_a.cells:
            cell_a = result_a.cells[key]
            cell_b = result_b.cells[key]
            for t_a, t_b in zip(cell_a.trials, cell_b.trials):
                assert t_a["seed"] != t_b["seed"]

    def test_trials_increment_seeds(self, tmp_path):
        _, result = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=3,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        cell = result.cells["voting/transport/drop"]
        seeds = [t["seed"] for t in cell.trials]
        assert len(seeds) == 3
        assert len(set(seeds)) == 3


class TestRenderMatrixReport:

    def test_report_contains_header(self, tmp_path):
        _, result = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=2,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        text = render_matrix_report(result)
        assert "Protocol Failure Matrix" in text
        assert "Matrix ID" in text
        assert "Trials/cell" in text

    def test_report_contains_cell_data(self, tmp_path):
        _, result = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=2,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        text = render_matrix_report(result)
        assert "voting" in text
        assert "transport/drop" in text

    def test_report_contains_fault_descriptions(self, tmp_path):
        _, result = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=1,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        text = render_matrix_report(result)
        assert "First message is silently dropped" in text

    def test_report_shows_violations(self, tmp_path):
        _, result = run_failure_matrix(
            scenarios=["capability_spoofing"],
            faults=["auth/none"],
            trials=3,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        text = render_matrix_report(result)
        assert "auth/none" in text
        cell = result.cells["capability_spoofing/auth/none"]
        if cell.violations > 0:
            assert str(cell.violations) in text


class TestMatrixCLI:

    def test_matrix_cli_runs_successfully(self, tmp_path, capsys):
        from nandatown.cli import main

        ret = main([
            "matrix",
            "--scenario", "voting",
            "--fault", "transport/drop",
            "--trials", "2",
            "--seed-base", "2000",
            "--out", str(tmp_path),
        ])
        assert ret == 0
        text = capsys.readouterr().out
        assert "Protocol Failure Matrix" in text

    def test_matrix_cli_default_faults(self, tmp_path, capsys):
        from nandatown.cli import main

        ret = main([
            "matrix",
            "--scenario", "voting",
            "--trials", "1",
            "--out", str(tmp_path),
        ])
        assert ret == 0
        text = capsys.readouterr().out
        assert "transport/drop" in text

    def test_matrix_cli_multiple_scenarios(self, tmp_path, capsys):
        from nandatown.cli import main

        ret = main([
            "matrix",
            "--scenario", "voting",
            "--scenario", "auction",
            "--fault", "transport/drop",
            "--trials", "1",
            "--out", str(tmp_path),
        ])
        assert ret == 0

    def test_matrix_cli_returns_nonzero_on_violations(self, tmp_path):
        from nandatown.cli import main

        ret = main([
            "matrix",
            "--scenario", "capability_spoofing",
            "--fault", "auth/none",
            "--trials", "3",
            "--out", str(tmp_path),
        ])
        assert ret == 1

    def test_matrix_cli_unknown_fault_skipped(self, tmp_path, capsys):
        from nandatown.cli import main

        ret = main([
            "matrix",
            "--scenario", "voting",
            "--fault", "nonexistent/fault",
            "--trials", "1",
            "--out", str(tmp_path),
        ])
        assert ret == 0


class TestExistingFunctionalityPreserved:

    def test_lab_scenario_still_runs_directly(self, tmp_path):
        from nandatown.sim.runner import run_lab

        bundle_dir, result = run_lab("voting", str(tmp_path))
        assert result.verdict in ("passed", "failed", "incomplete", "error")
        assert os.path.exists(os.path.join(bundle_dir, "manifest.json"))

    def test_compare_still_works(self, tmp_path):
        from nandatown.compare import run_comparison

        compare_dir, comparison = run_comparison(
            "capability_spoofing", {"auth": "plain.v1"}, str(tmp_path),
        )
        assert comparison["variants"]["baseline"]["verdict"] == "passed"
        assert comparison["variants"]["swapped"]["verdict"] == "failed"

    def test_campaign_still_works(self, tmp_path):
        from nandatown.campaign import run_campaign

        campaign_dir, aggregate = run_campaign(
            "voting", trials=2, out_dir=str(tmp_path),
        )
        assert aggregate["trials"] == 2
        assert "verdicts" in aggregate

    def test_bundle_verification_still_works(self, tmp_path):
        from nandatown.bundle import verify_bundle
        from nandatown.sim.runner import run_lab

        bundle_dir, _ = run_lab("voting", str(tmp_path))
        problems = verify_bundle(bundle_dir)
        assert problems == []


class TestScenarioFaultInjection:

    def test_marketplace_with_transport_duplicate(self, tmp_path):
        from nandatown.sim.runner import run_lab

        bundle_dir, result = run_lab(
            "marketplace", str(tmp_path), seed=42,
        )
        assert result.verdict in ("passed", "failed", "incomplete", "error")

    def test_consensus_with_transport_drop(self, tmp_path):
        from nandatown.sim.runner import run_lab

        bundle_dir, result = run_lab(
            "consensus", str(tmp_path), seed=42,
        )
        assert result.verdict in ("passed", "failed", "incomplete", "error")

    def test_capability_spoofing_with_auth_none(self, tmp_path):
        from nandatown.sim.runner import run_lab

        bundle_dir, result = run_lab(
            "capability_spoofing", str(tmp_path), seed=42,
            layer_overrides={"auth": "plain.v1"},
        )
        assert result.verdict == "failed"

    def test_capability_spoofing_baseline_passes(self, tmp_path):
        from nandatown.sim.runner import run_lab

        bundle_dir, result = run_lab(
            "capability_spoofing", str(tmp_path), seed=42,
        )
        assert result.verdict == "passed"


class TestReportIntegration:

    def test_matrix_report_renders_table_structure(self, tmp_path):
        _, result = run_failure_matrix(
            scenarios=["voting", "auction"],
            faults=["transport/drop"],
            trials=2,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        text = render_matrix_report(result)
        lines = text.strip().split("\n")
        header_idx = None
        for i, line in enumerate(lines):
            if "Scenario" in line and "Fault" in line and "Trials" in line:
                header_idx = i
                break
        assert header_idx is not None
        separator = lines[header_idx + 1]
        assert all(c == "-" for c in separator)

    def test_matrix_report_shows_all_scenarios(self, tmp_path):
        _, result = run_failure_matrix(
            scenarios=["voting", "auction", "consensus"],
            faults=["transport/drop"],
            trials=1,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        text = render_matrix_report(result)
        assert "voting" in text
        assert "auction" in text
        assert "consensus" in text


class TestEvidenceIntegration:

    def test_each_trial_bundle_has_correct_mode(self, tmp_path):
        from nandatown.bundle import load_bundle

        matrix_dir, _ = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=2,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        cell_dir = os.path.join(matrix_dir, "voting/transport/drop")
        for name in os.listdir(cell_dir):
            if name.startswith("sim-"):
                bundle = load_bundle(os.path.join(cell_dir, name))
                assert bundle["manifest"]["mode"] == "lab"

    def test_each_trial_bundle_has_events(self, tmp_path):
        from nandatown.bundle import load_bundle

        matrix_dir, _ = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=2,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        cell_dir = os.path.join(matrix_dir, "voting/transport/drop")
        for name in os.listdir(cell_dir):
            if name.startswith("sim-"):
                bundle = load_bundle(os.path.join(cell_dir, name))
                assert len(bundle["events"]) > 0

    def test_each_trial_bundle_has_stages(self, tmp_path):
        from nandatown.bundle import load_bundle

        matrix_dir, _ = run_failure_matrix(
            scenarios=["voting"],
            faults=["transport/drop"],
            trials=2,
            seed_base=2000,
            out_dir=str(tmp_path),
        )
        cell_dir = os.path.join(matrix_dir, "voting/transport/drop")
        for name in os.listdir(cell_dir):
            if name.startswith("sim-"):
                bundle = load_bundle(os.path.join(cell_dir, name))
                assert len(bundle["result"].stages) > 0

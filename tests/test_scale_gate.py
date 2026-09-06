from scripts.run_accepted_target import REQUIRED_SCALE_GATES, scale_gate_errors


def test_scale_gate_rejects_stale_report_without_adversarial_and_dual_judge_gates():
    gate = {
        "scale_allowed": True,
        "gates": {
            name: True
            for name in REQUIRED_SCALE_GATES
            if name not in {"adversarial_acceptance", "dual_judge_consensus"}
        },
    }

    errors = scale_gate_errors(gate)

    assert "adversarial_acceptance" in errors
    assert "dual_judge_consensus" in errors


def test_scale_gate_accepts_only_complete_current_report():
    gate = {"scale_allowed": True, "gates": {name: True for name in REQUIRED_SCALE_GATES}}

    assert scale_gate_errors(gate) == []

import copy

from hlwm_data.adversarial import adversarial_acceptance_errors
from tests.test_pipeline import sample_blueprint, sample_episode


def test_clean_proposed_design_passes_deterministic_acceptance_gate():
    assert adversarial_acceptance_errors(sample_episode()) == []


def test_gate_recomputes_explicit_arithmetic():
    episode = sample_episode()
    episode["integration"]["published_answer"] += " The central check is 2 + 2 = 5."

    errors = adversarial_acceptance_errors(episode)

    assert any("arithmetic check failed" in item for item in errors)


def test_arithmetic_gate_allows_declared_decimal_rounding():
    episode = sample_episode()
    episode["integration"]["published_answer"] += " The observed rate is 2/12 = 0.1667."

    errors = adversarial_acceptance_errors(episode)

    assert not any("arithmetic check failed" in item for item in errors)


def test_arithmetic_gate_handles_thousands_and_ignores_variable_fragments():
    episode = sample_episode()
    episode["integration"]["published_answer"] += (
        " Compute 7,500,000 * 0.125 = 937,500 units. "
        "For the symbolic model, 0.05 = 0.2 * p_new + 0.02 * (1 - p_new)."
    )

    errors = adversarial_acceptance_errors(episode)

    assert not any("arithmetic check failed" in item for item in errors)


def test_arithmetic_gate_keeps_full_currency_division_context():
    episode = sample_episode()
    episode["integration"]["published_answer"] += (
        " Percentage increase = $820 / $4,100 = 0.20 = 20%."
    )

    errors = adversarial_acceptance_errors(episode)

    assert not any("arithmetic check failed" in item for item in errors)


def test_gate_recounts_word_limited_constraint_artifact():
    episode = sample_episode()
    blueprint = sample_blueprint()
    blueprint["task"]["constraints"][0]["statement"] = "The selected answer must be exactly 5 words."
    episode["blueprint_trace"] = {
        "constraint_trace": [
            {
                "constraint_id": "constraint-api",
                "artifact_ids": ["patch-1"],
                "result": "The selected artifact is available for deterministic counting.",
            }
        ]
    }

    errors = adversarial_acceptance_errors(episode, blueprint=blueprint)

    assert any("word-count check failed" in item for item in errors)


def test_gate_recounts_artifact_declared_word_total():
    episode = sample_episode()
    episode["lanes"][0]["artifacts"][0]["content"] = (
        'Opening Statement (10 words):\n"This statement actually contains only seven simple words."'
    )

    errors = adversarial_acceptance_errors(episode)

    assert any("declared word-count check failed" in item for item in errors)


def test_gate_rejects_conflicting_word_count_reconstruction():
    episode = sample_episode()
    episode["verification"][0]["result"] = (
        "Word count: 24 words. Counted: Alpha(1) Beta(2) Gamma(3). 3 words."
    )

    errors = adversarial_acceptance_errors(episode)

    assert any("word-count consistency check failed" in item for item in errors)


def test_gate_requires_ordered_central_reconstruction_for_published_blueprint_episode():
    episode = sample_episode()
    blueprint = sample_blueprint()

    errors = adversarial_acceptance_errors(episode, blueprint=blueprint)

    assert any("central reconstruction is missing" in item for item in errors)


def test_gate_accepts_visible_central_reconstruction_with_falsification():
    episode = sample_episode()
    blueprint = sample_blueprint()
    episode["verification"][0]["result"] = (
        "Inputs: the public signature and two-second boundary.\n"
        "Operation: trace the proposed bounded wait at the boundary.\n"
        "Reconstructed result: the signature remains unchanged and the wait terminates at the proposed bound.\n"
        "Falsification: Mutation: change the public signature. "
        "Recomputed outcome: the caller contract no longer matches. "
        "Rejection rule: reject any public-signature change.\n"
        "Verdict: supported as a proposal, not an executed outcome."
    )

    errors = adversarial_acceptance_errors(episode, blueprint=blueprint)

    assert not any("central reconstruction" in item or "central falsification" in item for item in errors)


def test_gate_accepts_ordered_reconstruction_labels_in_one_paragraph():
    episode = sample_episode()
    blueprint = sample_blueprint()
    episode["verification"][0]["result"] = (
        "Inputs: the signature. Operation: inspect the boundary. "
        "Reconstructed result: the proposed wait is bounded. "
        "Falsification: Mutation: change the signature. "
        "Recomputed outcome: the caller contract breaks. "
        "Rejection rule: reject signature changes. Verdict: supported."
    )

    errors = adversarial_acceptance_errors(episode, blueprint=blueprint)

    assert not any("central reconstruction" in item or "central falsification" in item for item in errors)


def test_gate_rejects_hypothetical_falsification_without_applied_mutation():
    episode = sample_episode()
    blueprint = sample_blueprint()
    episode["verification"][0]["result"] = (
        "Inputs: the signature. Operation: inspect the boundary. "
        "Reconstructed result: the proposed wait is bounded. "
        "Falsification: if the signature changed, the proposal would fail. "
        "Verdict: supported."
    )

    errors = adversarial_acceptance_errors(episode, blueprint=blueprint)

    assert any("central falsification" in item for item in errors)


def test_gate_recomputes_declared_whitespace_token_count_from_cited_artifact():
    episode = sample_episode()
    episode["lanes"][0]["artifacts"][0]["content"] = "one two three"
    episode["verification"][0]["evidence_refs"] = ["patch-1"]
    episode["verification"][0]["result"] = "The script contains 4 tokens."

    errors = adversarial_acceptance_errors(episode)

    assert any("whitespace-token count failed" in item for item in errors)


def test_gate_accepts_proposed_external_data_collection_method():
    episode = sample_episode()
    episode["verification"][0]["method"] = (
        "Logical analysis that proposes a survey or A/B test as future data collection."
    )

    errors = adversarial_acceptance_errors(episode)

    assert not any("unexecuted or inaccessible verification method" in item for item in errors)


def test_gate_recomputes_percentage_mix_total():
    episode = sample_episode()
    blueprint = sample_blueprint()
    blueprint["task"]["constraints"][0]["statement"] = (
        "The revised mix percentages must sum to exactly 100%."
    )
    episode["lanes"][0]["artifacts"][0]["content"] = "- General: 60%\n- Coding: 30%"
    episode["blueprint_trace"] = {
        "constraint_trace": [
            {
                "constraint_id": "constraint-api",
                "artifact_ids": ["patch-1"],
                "result": "The mix is exposed as a line-item table for recomputation.",
            }
        ]
    }

    errors = adversarial_acceptance_errors(episode, blueprint=blueprint)

    assert any("percentage check failed" in item for item in errors)


def test_percentage_gate_ignores_total_summary_line():
    episode = sample_episode()
    blueprint = sample_blueprint()
    blueprint["task"]["constraints"][0]["statement"] = "The budget must sum to exactly 100%."
    episode["lanes"][0]["artifacts"][0]["content"] = (
        "- Search: 60%\n- Social: 40%\nTotal: 60+40 = 100%."
    )
    episode["blueprint_trace"] = {
        "constraint_trace": [
            {
                "constraint_id": "constraint-api",
                "artifact_ids": ["patch-1"],
                "result": "The line items are exposed for recomputation.",
            }
        ]
    }

    errors = adversarial_acceptance_errors(episode, blueprint=blueprint)

    assert not any("percentage check failed" in item for item in errors)


def test_gate_rejects_model_authored_execution_evidence():
    episode = sample_episode()
    episode["tool_runs"][0].update(
        {"status": "provided_evidence", "evidence_refs": ["test-1"], "result": "Tests passed"}
    )

    errors = adversarial_acceptance_errors(episode)

    assert any("without pipeline-issued execution attestation" in item for item in errors)


def test_gate_rejects_outcome_hidden_in_proposed_tool_result():
    episode = sample_episode()
    episode["tool_runs"][0]["result"] = "All tests passed and the pipeline executed successfully."

    errors = adversarial_acceptance_errors(episode)

    assert any("reports an outcome despite status proposed" in item for item in errors)


def test_gate_rejects_execution_claim_for_proposed_test():
    episode = sample_episode()
    episode["integration"]["published_answer"] = "The tests passed and confirmed the timeout fix."

    errors = adversarial_acceptance_errors(episode)

    assert any("unattested execution claim" in item for item in errors)


def test_gate_allows_explicitly_negated_execution_statement():
    episode = sample_episode()
    episode["integration"]["published_answer"] = (
        "These are hypotheses pending verification; no actual comparison has been executed yet."
    )

    errors = adversarial_acceptance_errors(episode)

    assert not any("unattested execution claim" in item for item in errors)


def test_negated_execution_clause_does_not_hide_positive_execution_claim():
    episode = sample_episode()
    episode["integration"]["published_answer"] = (
        "No benchmark was run, but the tests passed and confirmed the fix."
    )

    errors = adversarial_acceptance_errors(episode)

    assert any("unattested execution claim" in item for item in errors)


def test_unrelated_trusted_run_does_not_bless_execution_claim():
    episode = sample_episode()
    episode["integration"]["published_answer"] = "The tests passed and confirmed the fix."

    errors = adversarial_acceptance_errors(
        episode, trusted_execution_run_ids={"run-different"}
    )

    assert any("unattested execution claim" in item for item in errors)


def test_execution_claim_must_cite_its_trusted_run():
    episode = sample_episode()
    episode["integration"]["published_answer"] = (
        "The tests passed and confirmed the fix [run-attested]."
    )

    errors = adversarial_acceptance_errors(
        episode, trusted_execution_run_ids={"run-attested"}
    )

    assert not any("unattested execution claim" in item for item in errors)


def test_selected_artifact_execution_claim_is_checked():
    episode = sample_episode()
    episode["lanes"][0]["artifacts"][0]["content"] = (
        "The tests passed and the deployment completed successfully."
    )

    errors = adversarial_acceptance_errors(episode)

    assert any("selected artifact patch-1 makes an unattested" in item for item in errors)


def test_execution_output_string_inside_fenced_code_is_not_a_claim():
    episode = sample_episode()
    episode["lanes"][0]["artifacts"][0]["content"] = (
        '```python\nprint("Test passed: counterexample is valid.")\n```'
    )

    errors = adversarial_acceptance_errors(episode)

    assert not any("unattested execution claim" in item for item in errors)


def test_external_verification_requires_its_trusted_run_reference():
    episode = sample_episode()
    episode["verification"][0]["method"] = "Execute pytest against the supplied patch."
    episode["verification"][0]["evidence_refs"] = ["patch-1"]

    errors = adversarial_acceptance_errors(
        episode, trusted_execution_run_ids={"run-attested"}
    )

    assert any("unexecuted or inaccessible verification method" in item for item in errors)


def test_gate_allows_conditional_statistical_decision_rule():
    episode = sample_episode()
    episode["integration"]["published_answer"] = (
        "If the p-value < 0.05, reject the null hypothesis; otherwise retain it."
    )

    errors = adversarial_acceptance_errors(episode)

    assert not any("unattested execution claim" in item for item in errors)


def test_gate_rejects_self_certifying_constraint_trace():
    episode = sample_episode()
    episode["blueprint_trace"] = {
        "constraint_trace": [
            {
                "constraint_id": "constraint-api",
                "artifact_ids": ["patch-1"],
                "result": "Constraint satisfied.",
            }
        ]
    }

    errors = adversarial_acceptance_errors(episode)

    assert any("self-certifying" in item for item in errors)


def test_gate_rejects_trivial_renamed_gsm8k_problem():
    episode = copy.deepcopy(sample_episode())
    episode.update(
        {
            "domain": "mathematics_and_optimization",
            "source_group": "hf-openai-gsm8k-train",
            "difficulty": 4,
        }
    )
    episode["input"]["user_request"] = (
        "A farmer has 47 seed packets and distributes them equally among 4 plots. "
        "Find the maximum per plot and the leftover packets, then give a formal proof."
    )
    packet = {
        "chunks": [
            {
                "chunk_id": "gsm8k-1",
                "data_role": "verification_material",
                "text": '{"question":"Alma has 47 carrots and feeds 4 goats the same amount. How many are left over?"}',
            }
        ]
    }

    errors = adversarial_acceptance_errors(episode, source_packet=packet)

    assert any("source-task similarity" in item for item in errors)
    assert any("complexity floor" in item for item in errors)

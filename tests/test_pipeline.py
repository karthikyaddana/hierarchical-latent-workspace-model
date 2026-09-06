import json
import copy
import shutil
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import yaml
import pyarrow as pa
import pyarrow.parquet as pq

from hlwm_data.azure_client import parse_json_object
from hlwm_data.contamination import contamination_report
from hlwm_data.generate import (
    EpisodeJob,
    _apply_blueprint_contract,
    _blueprint_quality_errors,
    _normalize_blueprint,
    _source_packet,
    archived_episode_id_collisions,
    build_jobs,
    filter_generation_chunks_by_language,
    filter_generation_chunks_by_source_quality,
    episode_attempt_role,
    generation_model_roles,
    judge_episodes,
    model_role_separation_errors,
    non_substantive_source_reason,
    repair_rejected_episodes,
    validate_job_chunks_by_language,
)
from hlwm_data.ingest import append_ingested_manifest, filter_chunk_file_by_language, ingest_manifest
from hlwm_data.language import classify_language
from hlwm_data.materialize import _assign_split, _balanced_quality_cap, materialize_episode, to_sft
from hlwm_data.util import iter_jsonl
from hlwm_data.validation import (
    blueprint_invariant_errors,
    episode_blueprint_errors,
    episode_invariant_errors,
    review_errors,
    schema_errors,
    source_support_spans,
)


def sample_episode():
    return {
        "schema_version": "1.0",
        "episode_id": "episode-0001",
        "domain": "coding_debugging",
        "subdomain": "Python debugging",
        "difficulty": 3,
        "source_group": "source-family-1",
        "lineage_component_id": "source-family-1",
        "source_refs": [{"source_id": "source-1", "chunk_ids": ["chunk-1"], "usage": "bug context"}],
        "input": {
            "user_request": "Find and fix the timeout bug in this Python service.",
            "context": ["A test fails when the request exceeds two seconds."],
            "constraints": ["Do not change the public function signature."],
        },
        "frame": {
            "objective": "Identify the timeout bug and propose a verified minimal correction.",
            "requirements": ["Preserve the API", "Handle timeouts deterministically"],
            "unknowns": ["Exact failing branch"],
            "failure_contract": ["Reject changes that make the existing test fail"],
        },
        "routing": {
            "candidate_briefs": [{"lane_id": "implementation"}, {"lane_id": "testing"}],
            "selected_lane_ids": ["implementation", "testing"],
            "budget": {"max_lanes": 2, "max_checkpoints_per_lane": 2},
        },
        "lanes": [
            {
                "lane_id": "implementation",
                "route": ["root", "software", "python", "debugging"],
                "route_windows": [{"window_id": "w1", "route": ["root", "software", "python", "debugging"], "requested_steps": 1, "admitted_steps": 1, "actual_steps": 1, "decision": "halt"}],
                "brief": {
                    "scope": "Inspect the implementation",
                    "assumptions": [],
                    "deliverable": "Minimal patch proposal",
                    "rejection_tests": ["API signature changes"],
                },
                "artifacts": [{"artifact_id": "patch-1", "type": "patch", "content": "Use a bounded wait while preserving the public function signature."}],
                "claims": [{"claim_id": "claim-1", "statement": "The wait is unbounded", "evidence_refs": ["chunk-1"]}],
                "checkpoints": [{"step": 1, "observable_update": "Located wait", "next_step_gain": 0.1, "decision": "halt"}],
                "summary": "The implementation lane proposes a bounded wait.",
            },
            {
                "lane_id": "testing",
                "route": ["root", "software", "python", "testing"],
                "route_windows": [{"window_id": "w2", "route": ["root", "software", "python", "testing"], "requested_steps": 1, "admitted_steps": 1, "actual_steps": 1, "decision": "halt"}],
                "brief": {
                    "scope": "Design a regression test",
                    "assumptions": [],
                    "deliverable": "Executable test specification",
                    "rejection_tests": ["A test that depends on wall-clock timing"],
                },
                "artifacts": [{"artifact_id": "test-1", "type": "test", "content": "Use a fake clock to check the timeout boundary deterministically."}],
                "claims": [{"claim_id": "claim-2", "statement": "A fake clock makes the test deterministic", "evidence_refs": ["chunk-1"]}],
                "checkpoints": [{"step": 1, "observable_update": "Specified test", "next_step_gain": 0.05, "decision": "halt"}],
                "summary": "The testing lane specifies a deterministic fake-clock regression test.",
            },
        ],
        "barrier": {
            "lane_summaries": [
                {"lane_id": "implementation", "summary": "Bound the wait"},
                {"lane_id": "testing", "summary": "Use a fake clock"},
            ],
            "conflicts": [],
            "open_claims": [],
        },
        "verification": [
            {"claim_id": "claim-1", "verdict": "supported", "method": "source inspection", "evidence_refs": ["chunk-1"], "result": "Wait has no bound"},
            {"claim_id": "claim-2", "verdict": "supported", "method": "test review", "evidence_refs": ["chunk-1"], "result": "Fake clock is supported"},
        ],
        "tool_runs": [{"run_id": "run-1", "tool": "pytest", "status": "proposed", "input_refs": ["patch-1", "test-1"], "evidence_refs": [], "result": "Not executed", "replayable": True}],
        "integration": {
            "decision": "merge",
            "selected_artifacts": ["patch-1", "test-1"],
            "rejected_artifacts": [],
            "published_answer": "Apply a bounded wait and add the deterministic fake-clock regression test.",
        },
        "commitment": {"decision": "publish", "confidence": 0.9, "risks": [], "additional_refinement_helped": False},
        "continuation_pairs": [{"pair_id": "pair-1", "kind": "lane_step", "added_operation": "Add another inspection step", "added_cost": 1.0, "before_utility": 0.8, "after_utility": 0.8, "target": "halt", "eligible": False, "evidence_refs": []}],
        "counterfactuals": [
            {"variant": "root-only", "outcome": "Missed deterministic test", "utility": 0.5, "errors": ["test omission"]},
            {"variant": "no-verifier", "outcome": "Patch remained unchecked", "utility": 0.6, "errors": ["verification omission"]},
        ],
        "root_anchor_evaluations": [{"anchor_id": "anchor-1", "capability": "clear instruction following", "root_only_outcome": "Clear answer", "routed_outcome": "Clear technical answer", "retention_passed": True, "evidence_refs": ["chunk-1"]}],
        "blueprint_trace": {
            "blueprint_hash": "00000000000000000000000000000000",
            "constraint_trace": [
                {
                    "constraint_id": "constraint-api",
                    "artifact_ids": ["patch-1", "test-1"],
                    "result": "The interface constraint is visibly traced to both selected artifacts.",
                }
            ],
        },
    }


def sample_blueprint():
    return {
        "schema_version": "1.0",
        "episode_id": "episode-0001",
        "domain": "coding_debugging",
        "objective": "debugging",
        "source_group": "source-family-1",
        "lineage_component_id": "source-family-1",
        "task": {
            "user_request": "Design and verify a deterministic correction for the timeout defect.",
            "success_criteria": ["The correction preserves the interface and passes the boundary test."],
            "constraints": [
                {"constraint_id": "constraint-api", "statement": "Preserve the public function signature."}
            ],
        },
        "premises": {
            "source_backed": [
                {
                    "premise_id": "source-premise-timeout",
                    "statement": "The current wait has no explicit timeout bound.",
                    "evidence_refs": ["chunk-1"],
                    "support_span_id": "chunk-1::span-001",
                    "support_quote": "The current wait has no explicit timeout bound.",
                }
            ],
            "scenario": [
                {
                    "premise_id": "scenario-premise-deadline",
                    "statement": "Requests must terminate after two seconds.",
                }
            ],
        },
        "novelty": {
            "delta": "The task combines an interface-preserving patch with an independent deterministic boundary test.",
            "material_changes": ["constraints", "verification"],
            "independent_reconstruction": True,
            "baseline_failure": "A root-only source summary would omit the interface constraint and deterministic boundary check.",
            "constraint_interaction": "The deadline must be enforced without changing the public interface, so the patch and test constrain each other.",
        },
        "claim_plan": [
            {
                "claim_id": "claim-patch",
                "owner_lane_id": "lane-implementation",
                "claim_type": "proposal",
                "evidence_mode": "deterministic_artifact_check",
                "statement": "A bounded wait preserves the interface while enforcing the deadline.",
                "premise_ids": ["source-premise-timeout", "scenario-premise-deadline", "constraint-api"],
                "falsification_test": "Reject the patch if the function signature changes.",
                "verification_method": "Inspect the interface and exercise the boundary specification independently.",
                "expected_evidence_refs": ["source-premise-timeout", "scenario-premise-deadline", "artifact-patch"],
            },
            {
                "claim_id": "claim-test",
                "owner_lane_id": "lane-testing",
                "claim_type": "proposal",
                "evidence_mode": "deterministic_artifact_check",
                "statement": "A fake clock makes the timeout boundary test deterministic.",
                "premise_ids": ["scenario-premise-deadline"],
                "falsification_test": "Reject the test if it depends on elapsed wall-clock time.",
                "verification_method": "Review the test inputs and expected outcomes without using implementation state.",
                "expected_evidence_refs": ["scenario-premise-deadline", "artifact-test"],
            },
        ],
        "lanes": [
            {
                "lane_id": "lane-implementation",
                "specialist_route": ["root", "software", "debugging"],
                "responsibility": "Construct the smallest interface-preserving timeout patch.",
                "deliverable": "A precise patch specification",
                "artifact_ids": ["artifact-patch"],
                "rejection_test": "Reject any public signature or return-type change.",
                "marginal_necessity": "Without this lane there is no interface-preserving implementation proposal.",
            },
            {
                "lane_id": "lane-testing",
                "specialist_route": ["root", "software", "testing"],
                "responsibility": "Construct an independent deterministic boundary test.",
                "deliverable": "A boundary-test specification",
                "artifact_ids": ["artifact-test"],
                "rejection_test": "Reject tests that depend on real elapsed time.",
                "marginal_necessity": "Without this lane the timeout boundary is not independently testable.",
            },
        ],
        "verification_plan": {
            "method": "Independently inspect the interface and evaluate both boundary specifications.",
            "independent_from_all_lanes": True,
            "independence_boundary": "The verifier receives only the premises and completed artifacts, not lane-private state.",
            "target_claim_ids": ["claim-patch", "claim-test"],
            "decision_consequence": "Reject or narrow the publication if either interface or determinism check fails.",
            "central_reconstruction": "Trace the public signature and fake-clock boundary case from inputs to expected outcome.",
        },
        "constraint_trace": [
            {
                "constraint_id": "constraint-api",
                "planned_artifact_ids": ["artifact-patch", "artifact-test"],
                "validation_method": "Compare the proposed interface and boundary test against the original signature.",
            }
        ],
    }


class PipelineTests(unittest.TestCase):
    def test_english_language_filter_rejects_non_english_prose(self):
        english = classify_language(
            "A verifier checks the proposed algorithm against boundary cases and reports exact failures."
        )
        russian = classify_language(
            "Проверяющий проверяет предложенный алгоритм на граничных случаях и сообщает об ошибках."
        )
        chinese = classify_language("验证器检查算法的边界情况，并报告具体错误。")
        homoglyph = classify_language(
            "Use 64-bit integers in С++ and verify the deterministic output for every boundary case."
        )
        armenian_homoglyph = classify_language(
            "Use the Armenian look-alike Օ in this identifier and verify every deterministic boundary case."
        )
        self.assertTrue(english.accepted)
        self.assertFalse(russian.accepted)
        self.assertFalse(chinese.accepted)
        self.assertFalse(homoglyph.accepted)
        self.assertFalse(armenian_homoglyph.accepted)

    def test_source_quality_filter_rejects_front_matter_but_keeps_mechanisms(self):
        chunks = [
            {
                "chunk_id": "ack",
                "text": "Like all books, this would not have been possible without the help of countless people. Thank you for your support and feedback.",
            },
            {
                "chunk_id": "toc",
                "text": "Document Outline Table of Contents Chapter 1 Setup Chapter 2 Design Chapter 3 Testing Chapter 4 Delivery",
            },
            {
                "chunk_id": "mechanism",
                "text": "An idempotency key maps repeated requests to one durable decision. The recovery path must replay that decision without applying the state transition twice.",
            },
            {
                "chunk_id": "promo",
                "text": "Other Books You May Enjoy. Monitoring Handbook ISBN 123. Logging Guide ISBN 456. Diagnostics Book ISBN 789.",
            },
        ]

        accepted, removed = filter_generation_chunks_by_source_quality(
            chunks, {"reject_non_substantive_source_chunks": True}
        )

        self.assertEqual([item["chunk_id"] for item in accepted], ["mechanism"])
        self.assertEqual(removed, 3)
        self.assertEqual(non_substantive_source_reason(chunks[0]), "acknowledgments")

    def test_json_parser_handles_markdown_fence(self):
        self.assertEqual(parse_json_object('```json\n{"ok": true}\n```'), {"ok": True})

    def test_episode_schema_and_invariants(self):
        root = Path(__file__).resolve().parents[1]
        episode = sample_episode()
        self.assertEqual(schema_errors(episode, root / "schemas/episode.schema.json"), [])
        self.assertEqual(episode_invariant_errors(episode, {"chunk-1"}), [])

    def test_task_blueprint_schema_and_invariants(self):
        root = Path(__file__).resolve().parents[1]
        blueprint = sample_blueprint()
        self.assertEqual(schema_errors(blueprint, root / "schemas/task-blueprint.schema.json"), [])
        self.assertEqual(blueprint_invariant_errors(blueprint, {"chunk-1"}), [])

    def test_blueprint_requires_two_solver_lanes_before_verification(self):
        blueprint = sample_blueprint()
        blueprint["lanes"][1]["lane_id"] = "lane-verifier"
        blueprint["claim_plan"][1]["owner_lane_id"] = "lane-verifier"

        errors = blueprint_invariant_errors(blueprint, {"chunk-1"})

        self.assertTrue(any("two independent solver lanes" in item for item in errors))

    def test_blueprint_rejects_unavailable_supplied_execution_mode(self):
        blueprint = sample_blueprint()
        blueprint["claim_plan"][0]["evidence_mode"] = "supplied_execution"

        errors = blueprint_invariant_errors(blueprint, {"chunk-1"})

        self.assertTrue(any("no execution attestation" in item for item in errors))

    def test_blueprint_support_quote_must_exist_in_cited_chunk(self):
        blueprint = sample_blueprint()

        errors = blueprint_invariant_errors(
            blueprint,
            {"chunk-1"},
            known_chunk_texts={"chunk-1": "A different source passage with no timeout statement."},
        )

        self.assertTrue(any("support span is not an exact" in item for item in errors))

    def test_source_support_spans_are_exact_and_deterministic(self):
        text = "The current wait has no explicit timeout bound. A fake clock checks it deterministically."

        first = source_support_spans("chunk-1", text)
        second = source_support_spans("chunk-1", text)

        self.assertEqual(first, second)
        self.assertEqual(first[0]["span_id"], "chunk-1::span-001")
        self.assertIn(first[0]["text"], text)

    def test_blueprint_normalization_resolves_source_ids_and_preserves_overflow_criteria(self):
        blueprint = sample_blueprint()
        blueprint["task"]["success_criteria"] = ["Criterion number %d is observable." % index for index in range(6)]
        blueprint["premises"]["source_backed"][0]["evidence_refs"] = ["source-1"]
        job = EpisodeJob(
            episode_id="episode-0001",
            domain="coding_debugging",
            objective="debugging",
            source_group="source-family-1",
            lineage_component_id="source-family-1",
            chunks=[
                {
                    "source_id": "source-1",
                    "chunk_id": "chunk-1",
                    "text": "The current wait has no explicit timeout bound.",
                }
            ],
            variation_seed=1,
        )

        normalized = _normalize_blueprint(blueprint, job)

        self.assertEqual(normalized["premises"]["source_backed"][0]["evidence_refs"], ["chunk-1"])
        self.assertEqual(len(normalized["task"]["success_criteria"]), 5)
        self.assertIn("Criterion number 4", normalized["task"]["success_criteria"][4])
        self.assertIn("Criterion number 5", normalized["task"]["success_criteria"][4])

    def test_blueprint_normalization_repairs_only_unambiguous_shape_errors(self):
        blueprint = sample_blueprint()
        blueprint["verification_plan"]["constraint_trace"] = blueprint.pop("constraint_trace")
        blueprint["lanes"][1] = {"lane-testing": blueprint["lanes"][1]}
        blueprint["claim_plan"][1] = {"claim-test": blueprint["claim_plan"][1]}
        blueprint["claim_plan"][1]["claim-test"]["claim_type"] = "logical_counterexample"
        blueprint["premises"]["source_backed"][0]["support_span_id"] = (
            "chunk-typo::span-001"
        )
        blueprint["premises"]["source_backed"][0]["evidence_refs"] = [
            "chunk-1::span-001"
        ]
        job = EpisodeJob(
            episode_id="episode-0001",
            domain="coding_debugging",
            objective="debugging",
            source_group="source-family-1",
            lineage_component_id="source-family-1",
            chunks=[
                {
                    "source_id": "source-1",
                    "chunk_id": "chunk-1",
                    "text": "The current wait has no explicit timeout bound.",
                }
            ],
            variation_seed=1,
        )

        normalized = _normalize_blueprint(blueprint, job)

        self.assertIn("constraint_trace", normalized)
        self.assertNotIn("constraint_trace", normalized["verification_plan"])
        self.assertEqual(normalized["lanes"][1]["lane_id"], "lane-testing")
        self.assertEqual(normalized["claim_plan"][1]["claim_id"], "claim-test")
        self.assertEqual(normalized["claim_plan"][1]["claim_type"], "derived_result")
        self.assertEqual(normalized["claim_plan"][1]["evidence_mode"], "logical_counterexample")
        self.assertEqual(
            normalized["premises"]["source_backed"][0]["evidence_refs"], ["chunk-1"]
        )
        self.assertEqual(
            normalized["premises"]["source_backed"][0]["support_span_id"],
            "chunk-1::span-001",
        )

    def test_blueprint_normalization_does_not_materialize_missing_lists_as_null(self):
        job = EpisodeJob(
            episode_id="episode-0001",
            domain="coding_debugging",
            objective="debugging",
            source_group="source-family-1",
            lineage_component_id="source-family-1",
            chunks=[],
            variation_seed=1,
        )

        normalized = _normalize_blueprint({"task": {}}, job)

        self.assertNotIn("lanes", normalized)
        self.assertNotIn("claim_plan", normalized)

    def test_blueprint_contract_maps_authoritative_support_span_to_visible_context(self):
        blueprint = sample_blueprint()
        episode = sample_episode()
        episode["lanes"][0]["claims"][0]["evidence_refs"] = ["chunk-1::span-001"]

        normalized = _apply_blueprint_contract(episode, blueprint)

        self.assertEqual(
            normalized["lanes"][0]["claims"][0]["evidence_refs"][0],
            "input.context[0]",
        )

    def test_blueprint_contract_maps_other_packet_span_to_its_known_chunk(self):
        blueprint = sample_blueprint()
        episode = sample_episode()
        episode["lanes"][0]["claims"][0]["evidence_refs"] = ["chunk-1::span-007"]

        normalized = _apply_blueprint_contract(episode, blueprint)

        self.assertEqual(
            normalized["lanes"][0]["claims"][0]["evidence_refs"][0],
            "chunk-1",
        )

    def test_blueprint_rejects_redundant_verifier_lane(self):
        blueprint = sample_blueprint()
        blueprint["lanes"].append(
            {
                "lane_id": "lane-verifier",
                "specialist_route": ["root", "software", "verification"],
                "responsibility": "Verify the other lanes after they complete their work.",
                "deliverable": "A verification report",
                "artifact_ids": ["artifact-verifier-report"],
                "rejection_test": "Reject any mismatch in sibling outputs.",
                "marginal_necessity": "Without this lane the sibling outputs would not be independently verified.",
            }
        )
        blueprint["claim_plan"].append(
            {
                "claim_id": "claim-verifier-report",
                "owner_lane_id": "lane-verifier",
                "claim_type": "derived_result",
                "evidence_mode": "deterministic_artifact_check",
                "statement": "The sibling artifacts are mutually consistent.",
                "premise_ids": ["constraint-api"],
                "falsification_test": "Reject any interface mismatch.",
                "verification_method": "Compare the visible artifact interfaces.",
                "expected_evidence_refs": ["artifact-verifier-report"],
            }
        )
        blueprint["verification_plan"]["target_claim_ids"].append("claim-verifier-report")

        errors = blueprint_invariant_errors(blueprint, {"chunk-1"})

        self.assertTrue(any("redundant verification-only lanes" in item for item in errors))

    def test_blueprint_rejects_unattested_compile_outcome_claim(self):
        blueprint = sample_blueprint()
        blueprint["claim_plan"][0]["statement"] = "The proposed Java artifact compiles successfully."

        errors = blueprint_invariant_errors(blueprint, {"chunk-1"})

        self.assertTrue(any("requires unavailable execution evidence" in item for item in errors))

    def test_blueprint_allows_visible_build_order_without_claiming_execution(self):
        blueprint = sample_blueprint()
        blueprint["claim_plan"][0]["statement"] = (
            "The artifact visibly represents the proposed build ordering and delegation edge."
        )

        errors = blueprint_invariant_errors(blueprint, {"chunk-1"})

        self.assertFalse(any("requires unavailable execution evidence" in item for item in errors))

    def test_blueprint_rejects_prebarrier_dependency_on_sibling_artifact(self):
        blueprint = sample_blueprint()
        blueprint["claim_plan"][1]["expected_evidence_refs"].append("artifact-patch")

        errors = blueprint_invariant_errors(blueprint, {"chunk-1"})

        self.assertTrue(any("depends on sibling-lane artifacts" in item for item in errors))

    def test_blueprint_quality_review_fails_closed_on_issue_or_low_expertise(self):
        review = {
            "verdict": "accept",
            "overall_score": 0.90,
            "expertise_uplift": 0.75,
            "issues": ["The second lane merely reviews the first."],
        }

        errors = _blueprint_quality_errors(review, 0.84, 0.80)

        self.assertTrue(any("expertise uplift" in item for item in errors))
        self.assertTrue(any("quality issue" in item for item in errors))

    def test_reasoning_pipeline_separates_drafting_editing_critics_and_final_judges(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load(
            (root / "config/pipeline.reasoning9000.yaml").read_text(encoding="utf-8")
        )

        self.assertEqual(
            generation_model_roles(config),
            {
                "blueprint_planner": "gpt-5.6-luna",
                "blueprint_quality_critic": "Kimi-K2.5",
                "episode_constructor": "DeepSeek-V4-Flash",
                "episode_editor": "gpt-5.6-luna",
                "episode_construction_critic": "DeepSeek-V3.2-Speciale",
                "primary_episode_judge": "gpt-5.4-mini",
                "secondary_episode_judge": "Kimi-K2.6",
            },
        )
        self.assertEqual(model_role_separation_errors(config), [])
        self.assertEqual(
            episode_attempt_role(config, 1),
            ("DeepSeek-V4-Flash", "first_draft_constructor"),
        )
        self.assertEqual(
            episode_attempt_role(config, 2),
            ("gpt-5.6-luna", "episode_editor_reconstructor"),
        )

        invalid = copy.deepcopy(config)
        invalid["azure"]["episode_editor_deployment"] = "gpt-5.4-mini"
        self.assertTrue(
            any(
                "multiple required roles" in item
                for item in model_role_separation_errors(invalid)
            )
        )

    def test_construction_critic_feedback_is_bounded_to_three_issues(self):
        review = {
            "verdict": "reject",
            "overall_score": 0.5,
            "expertise_uplift": 0.5,
            "issues": ["defect-%d" % index for index in range(10)],
        }

        errors = _blueprint_quality_errors(review, 0.84, 0.80, issue_limit=3)

        issue_errors = [item for item in errors if "quality issue:" in item]
        self.assertEqual(len(issue_errors), 3)

    def test_model_role_separation_rejects_planner_judge_reuse(self):
        config = {
            "azure": {
                "deployment": "constructor",
                "blueprint_deployment": "planner",
                "blueprint_judge_deployment": "planner",
                "judge_deployment": "primary",
                "secondary_judge_deployment": "primary",
            },
            "generation": {"require_distinct_model_roles": True},
        }

        errors = model_role_separation_errors(config)

        self.assertTrue(any("multiple required roles" in item for item in errors))
        self.assertTrue(any("own blueprint quality judge" in item for item in errors))

    def test_episode_cannot_publish_with_unresolved_claims(self):
        episode = sample_episode()
        episode["verification"][0]["verdict"] = "insufficient"
        episode["barrier"]["open_claims"] = ["claim-1"]

        errors = episode_invariant_errors(episode, {"chunk-1"})

        self.assertTrue(any("refuted or insufficient" in item for item in errors))
        self.assertTrue(any("open barrier claims" in item for item in errors))

    def test_blueprint_rejects_literal_secret_deliverable(self):
        blueprint = sample_blueprint()
        blueprint["task"]["constraints"][0]["statement"] = (
            "The environment artifact must contain a strong JWT_SECRET of at least 32 characters."
        )

        errors = blueprint_invariant_errors(blueprint, {"chunk-1"})

        self.assertTrue(any("literal secret value" in item for item in errors))

    def test_blueprint_contract_normalizes_routes_constraints_and_evidence_refs(self):
        blueprint = sample_blueprint()
        episode = {
            "input": {},
            "lanes": [
                {
                    "lane_id": "lane-implementation",
                    "route": ["workflow-step"],
                    "brief": {"deliverable": "changed"},
                    "claims": [
                        {
                            "claim_id": "claim-patch",
                            "evidence_refs": [
                                "constraint-api",
                            ],
                        }
                    ],
                }
            ],
            "verification": [
                {
                    "claim_id": "claim-patch",
                    "evidence_refs": ["source-premise-timeout"],
                }
            ],
        }
        normalized = _apply_blueprint_contract(episode, blueprint)
        self.assertEqual(normalized["lanes"][0]["route"], ["root", "software", "debugging"])
        self.assertEqual(
            normalized["lanes"][0]["brief"]["deliverable"],
            "A precise patch specification",
        )
        self.assertEqual(
            normalized["lanes"][0]["claims"][0]["evidence_refs"],
            ["input.constraints[0]", "input.context[0]", "input.context[1]", "artifact-patch"],
        )
        self.assertEqual(
            normalized["verification"][0]["evidence_refs"], ["input.context[0]"]
        )

    def test_blueprint_contract_reconstructs_missing_constraint_trace_shape(self):
        blueprint = sample_blueprint()
        episode = sample_episode()
        episode["blueprint_trace"] = {"blueprint_hash": "stale"}

        normalized = _apply_blueprint_contract(episode, blueprint)

        self.assertEqual(
            normalized["blueprint_trace"]["constraint_trace"],
            [
                {
                    "constraint_id": "constraint-api",
                    "artifact_ids": ["artifact-patch", "artifact-test"],
                    "result": "",
                }
            ],
        )
        errors = episode_blueprint_errors(normalized, blueprint)
        self.assertTrue(any("no validation result" in item for item in errors))

    def test_constraint_trace_allows_verifier_only_artifact_to_remain_unpublished(self):
        blueprint = sample_blueprint()
        episode = {
            "input": {},
            "lanes": [
                {
                    "lane_id": "lane-implementation",
                    "route": [],
                    "brief": {"deliverable": ""},
                    "artifacts": [{"artifact_id": "artifact-patch"}],
                    "claims": [
                        {
                            "claim_id": "claim-patch",
                            "evidence_refs": ["artifact-patch"],
                        }
                    ],
                },
                {
                    "lane_id": "lane-testing",
                    "route": [],
                    "brief": {"deliverable": ""},
                    "artifacts": [{"artifact_id": "artifact-test"}],
                    "claims": [
                        {
                            "claim_id": "claim-test",
                            "evidence_refs": ["artifact-test"],
                        }
                    ],
                },
            ],
            "verification": [
                {"claim_id": "claim-patch"},
                {"claim_id": "claim-test"},
            ],
            "integration": {"selected_artifacts": ["artifact-patch"]},
            "blueprint_trace": {
                "constraint_trace": [
                    {
                        "constraint_id": "constraint-api",
                        "artifact_ids": ["artifact-patch", "artifact-test"],
                        "result": "The delivery preserves the interface and the verifier confirms it.",
                    }
                ]
            },
        }
        episode = _apply_blueprint_contract(episode, blueprint)
        errors = episode_blueprint_errors(episode, blueprint)
        self.assertEqual(errors, [])

    def test_git_zero_sha_is_not_misclassified_as_a_secret(self):
        episode = sample_episode()
        episode["lanes"][0]["artifacts"][0]["content"] = (
            "Use Git's missing-object sentinel 0000000000000000000000000000000000000000."
        )
        errors = episode_invariant_errors(episode, {"chunk-1"})
        self.assertFalse(any("secret" in item for item in errors))

    def test_existing_scenario_premise_reference_is_valid_evidence(self):
        episode = sample_episode()
        episode["lanes"][0]["claims"][0]["evidence_refs"] = ["input.context[0]"]
        episode["verification"][0]["evidence_refs"] = ["input.context[0]"]
        errors = episode_invariant_errors(episode, {"chunk-1"})
        self.assertFalse(any("missing evidence" in item for item in errors))

    def test_repair_regenerates_then_fails_closed_without_overwriting(self):
        workspace = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "schemas").mkdir()
            (root / "prompts").mkdir()
            (root / "generated").mkdir()
            (root / "reviewed").mkdir()
            shutil.copy2(workspace / "schemas/episode.schema.json", root / "schemas/episode.schema.json")
            shutil.copy2(
                workspace / "schemas/task-blueprint.schema.json",
                root / "schemas/task-blueprint.schema.json",
            )
            shutil.copy2(workspace / "prompts/repair_system.md", root / "prompts/repair_system.md")
            original = sample_episode()
            (root / "generated/episode-0001.json").write_text(json.dumps(original), encoding="utf-8")
            wrapper = {
                "episode": original,
                "accepted": False,
                "static_errors": [],
                "review": {"verdict": "reject", "issues": ["verification incomplete"]},
            }
            (root / "reviewed/episode-0001.json").write_text(json.dumps(wrapper), encoding="utf-8")
            chunk = {"source_id": "source-1", "chunk_id": "chunk-1", "text": "The wait is unbounded."}
            (root / "chunks.jsonl").write_text(json.dumps(chunk) + "\n", encoding="utf-8")
            invalid = copy.deepcopy(original)
            invalid["verification"] = []

            class InvalidRepairClient:
                def __init__(self, *_args, **_kwargs):
                    self.calls = 0
                    self.prompts = []

                def chat_json(self, _system, prompt, *_args, **_kwargs):
                    self.calls += 1
                    self.prompts.append(prompt)
                    return SimpleNamespace(
                        value=copy.deepcopy(invalid),
                        request_id="repair-%d" % self.calls,
                        prompt_tokens=1,
                        completion_tokens=1,
                        elapsed_seconds=0.01,
                    )

                def close(self):
                    return None

            client = InvalidRepairClient()
            config = {
                "azure": {"deployment": "test", "max_workers": 1},
                "generation": {"max_repair_rounds": 1, "max_repair_validation_attempts": 2},
                "output": {
                    "chunks_file": "chunks.jsonl",
                    "generated_dir": "generated",
                    "reviewed_dir": "reviewed",
                    "request_log": "logs/requests.jsonl",
                    "run_log": "logs/runs.jsonl",
                },
            }
            with patch("hlwm_data.generate.AzureTeacherClient", return_value=client):
                stats = repair_rejected_episodes(config, root, resume=False)

            self.assertEqual(client.calls, 2)
            self.assertTrue(all('"blueprint_trace"' in prompt for prompt in client.prompts))
            self.assertTrue(all('"constraint_trace"' in prompt for prompt in client.prompts))
            self.assertEqual(stats["failed"], 1)
            self.assertEqual(
                json.loads((root / "generated/episode-0001.json").read_text(encoding="utf-8")),
                original,
            )
            self.assertTrue((root / "reviewed/episode-0001.json").exists())

    def test_review_gate_rejects_source_leakage_and_unsupported_claims(self):
        root = Path(__file__).resolve().parents[1]
        review = {
            "episode_id": "episode-0001",
            "verdict": "accept",
            "overall_score": 0.92,
            "scores": {
                "groundedness": 0.92,
                "correctness": 0.92,
                "lane_quality": 0.92,
                "verification": 0.92,
                "usefulness": 0.92,
                "style": 0.92,
                "expertise_uplift": 0.92,
            },
            "issues": [],
            "required_fixes": [],
            "suspected_source_leakage": True,
            "unsupported_claim_ids": ["claim-9"],
        }
        errors = review_errors(review, root / "schemas/review.schema.json", 0.84, 0.80)
        self.assertTrue(any("leakage" in item for item in errors))
        self.assertTrue(any("unsupported claims" in item for item in errors))

    def test_review_gate_rejects_accept_verdict_with_required_fixes(self):
        root = Path(__file__).resolve().parents[1]
        review = {
            "episode_id": "episode-0001",
            "verdict": "accept",
            "overall_score": 0.92,
            "scores": {
                "groundedness": 0.92,
                "correctness": 0.92,
                "lane_quality": 0.92,
                "verification": 0.92,
                "usefulness": 0.92,
                "style": 0.92,
                "expertise_uplift": 0.92,
            },
            "issues": ["The executable setup is incomplete."],
            "required_fixes": ["Add the missing test dependency."],
            "suspected_source_leakage": False,
            "unsupported_claim_ids": [],
        }

        errors = review_errors(review, root / "schemas/review.schema.json", 0.84, 0.80)

        self.assertTrue(any("required fixes" in item for item in errors))

    def test_dual_judge_disagreement_fails_closed(self):
        workspace = Path(__file__).resolve().parents[1]

        def response(verdict):
            accepted = verdict == "accept"
            return {
                "episode_id": "episode-0001",
                "verdict": verdict,
                "overall_score": 0.92 if accepted else 0.60,
                "scores": {
                    "groundedness": 0.92,
                    "correctness": 0.92,
                    "lane_quality": 0.92,
                    "verification": 0.92,
                    "usefulness": 0.92,
                    "style": 0.92,
                    "expertise_uplift": 0.92,
                },
                "issues": [] if accepted else ["Independent verification is insufficient."],
                "required_fixes": [] if accepted else ["Supply independent evidence."],
                "suspected_source_leakage": False,
                "unsupported_claim_ids": [],
            }

        class FakeJudgeClient:
            def __init__(self, azure_config, *_args, **_kwargs):
                self.deployment = azure_config["deployment"]

            def chat_json(self, *_args, **_kwargs):
                verdict = "accept" if self.deployment == "judge-primary" else "reject"
                return SimpleNamespace(
                    value=response(verdict),
                    request_id="request-%s" % self.deployment,
                    prompt_tokens=10,
                    completion_tokens=10,
                    elapsed_seconds=0.01,
                )

            def close(self):
                return None

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("schemas", "prompts", "generated", "reviewed", "blueprints"):
                (root / directory).mkdir()
            for schema in ("episode.schema.json", "review.schema.json", "task-blueprint.schema.json"):
                shutil.copy2(workspace / "schemas" / schema, root / "schemas" / schema)
            shutil.copy2(workspace / "prompts/judge_system.md", root / "prompts/judge_system.md")
            (root / "generated/episode-0001.json").write_text(
                json.dumps(sample_episode()), encoding="utf-8"
            )
            (root / "chunks.jsonl").write_text(
                json.dumps(
                    {
                        "source_id": "source-1",
                        "chunk_id": "chunk-1",
                        "source_group": "source-family-1",
                        "text": "The wait is unbounded and requires a deterministic correction.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = {
                "azure": {
                    "deployment": "generator",
                    "judge_deployment": "judge-primary",
                    "secondary_judge_deployment": "judge-secondary",
                    "max_workers": 1,
                },
                "generation": {
                    "quality_threshold": 0.84,
                    "minimum_expertise_uplift": 0.80,
                    "require_dual_judge": True,
                },
                "output": {
                    "chunks_file": "chunks.jsonl",
                    "blueprints_dir": "blueprints",
                    "generated_dir": "generated",
                    "reviewed_dir": "reviewed",
                    "request_log": "logs/requests.jsonl",
                    "run_log": "logs/runs.jsonl",
                },
            }

            with patch("hlwm_data.generate.AzureTeacherClient", side_effect=FakeJudgeClient):
                stats = judge_episodes(config, root, resume=False)

            wrapper = json.loads((root / "reviewed/episode-0001.json").read_text(encoding="utf-8"))
            self.assertEqual(stats["secondary_judged"], 1)
            self.assertEqual(stats["judge_disagreements"], 1)
            self.assertFalse(wrapper["accepted"])
            self.assertFalse(wrapper["judge_consensus"])
            self.assertEqual(wrapper["primary_review"]["verdict"], "accept")
            self.assertEqual(wrapper["secondary_review"]["verdict"], "reject")

    def test_episode_invariants_reject_latin_script_non_english_output(self):
        episode = sample_episode()
        episode["input"]["user_request"] = "Encuentra y corrige el error de tiempo de espera en este servicio."
        episode["frame"]["objective"] = "Identificar el defecto y proponer una corrección verificable."
        episode["integration"]["published_answer"] = (
            "La solución limita la espera y añade una prueba determinista para comprobar todos los casos importantes."
        )
        errors = episode_invariant_errors(episode, {"chunk-1"})
        self.assertTrue(any("not English" in item for item in errors))

    def test_episode_invariants_reject_corrupted_unicode(self):
        episode = sample_episode()
        episode["integration"]["published_answer"] += " Corrupted marker: \N{REPLACEMENT CHARACTER}"
        errors = episode_invariant_errors(episode, {"chunk-1"})
        self.assertTrue(any("corrupted Unicode" in item for item in errors))

    def test_episode_invariants_reject_private_reasoning_closing_marker(self):
        episode = sample_episode()
        episode["integration"]["published_answer"] += " </think>"
        errors = episode_invariant_errors(episode, {"chunk-1"})
        self.assertTrue(any("private chain-of-thought" in item for item in errors))

    def test_adversarial_gate_rejects_reused_verification_reconstruction(self):
        from hlwm_data.adversarial import adversarial_acceptance_errors

        episode = sample_episode()
        shared = (
            "Inputs: frozen artifact. Operation: trace it. Reconstructed result: visible result. "
            "Falsification: Mutation: change input. Recomputed outcome: rejected result. "
            "Rejection rule: reject mismatch. Verdict: supported."
        )
        episode["verification"][0]["result"] = shared
        episode["verification"][1]["result"] = shared.replace(
            "Verdict: supported.", "Verdict: claim-specific wording only."
        )

        errors = adversarial_acceptance_errors(episode)

        self.assertTrue(any("reuse the same reconstruction" in item for item in errors))

    def test_private_lane_view_never_contains_sibling_work(self):
        episode = sample_episode()
        rows = materialize_episode(episode, "train")
        lane = next(row for row in rows if row["stage"] == "private_solving" and row["input"]["lane_id"] == "implementation")
        serialized = json.dumps(lane["input"])
        self.assertNotIn("fake clock", serialized.lower())
        sft = to_sft(lane)
        self.assertEqual([item["role"] for item in sft["messages"]], ["system", "user", "assistant"])

    def test_lineage_split_is_deterministic(self):
        splits = {"train": 0.8, "validation": 0.1, "test": 0.1}
        self.assertEqual(_assign_split("family-a", splits), _assign_split("family-a", splits))

    def test_stable_indexed_generation_batches_match_one_shot(self):
        config = {
            "project": {"seed": 42},
            "generation": {
                "stable_indexed_jobs": True,
                "chunks_per_episode": 1,
                "require_grounded_sources": True,
                "domains": {
                    "alpha": {"weight": 0.5, "objectives": ["a"]},
                    "beta": {"weight": 0.5, "objectives": ["b"]},
                },
            },
        }
        chunks = [
            {"domain": domain, "source_group": domain + "-source", "lineage_component_id": domain + "-lineage", "chunk_id": domain + "-%d" % index, "source_id": domain}
            for domain in ("alpha", "beta")
            for index in range(4)
        ]
        one_shot = {job.episode_id for job in build_jobs(chunks, config, 10, offset=0)}
        batched = {job.episode_id for job in build_jobs(chunks, config, 4, offset=0)}
        batched.update(job.episode_id for job in build_jobs(chunks, config, 6, offset=4))
        self.assertEqual(one_shot, batched)

    def test_archived_episode_collision_guard_scans_every_pilot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            generated_dir = root / "data/reasoning9000/generated"
            archive = root / "data/reasoning9000/pilots/pilot-old/generated"
            archive.mkdir(parents=True)
            (archive / "episode-0001.json").write_text("{}", encoding="utf-8")
            job = EpisodeJob(
                episode_id="episode-0001",
                domain="alpha",
                objective="verify",
                source_group="source",
                lineage_component_id="source",
                chunks=tuple(),
                variation_seed=1,
            )

            collisions = archived_episode_id_collisions(root, generated_dir, [job])

            self.assertEqual(collisions, ["episode-0001"])

    def test_generation_selects_coherent_local_evidence_windows(self):
        config = {
            "project": {"seed": 42},
            "generation": {
                "stable_indexed_jobs": True,
                "chunks_per_episode": 3,
                "require_grounded_sources": True,
                "domains": {"alpha": {"weight": 1.0, "objectives": ["verify"]}},
            },
        }
        chunks = [
            {
                "domain": "alpha",
                "source_group": "book-alpha",
                "lineage_component_id": "book-alpha-pages-%04d" % block,
                "source_id": "page-%d" % page,
                "chunk_id": "chunk-%d" % page,
                "path": "/book.pdf",
                "part": "page-%d" % page,
                "char_start": 0,
            }
            for block, pages in ((0, range(1, 5)), (1, range(17, 21)))
            for page in pages
        ]

        jobs = build_jobs(chunks, config, 8)

        self.assertTrue(jobs)
        for job in jobs:
            self.assertEqual(len({chunk["lineage_component_id"] for chunk in job.chunks}), 1)
            pages = [int(str(chunk["part"]).split("-")[-1]) for chunk in job.chunks]
            self.assertLessEqual(max(pages) - min(pages), 2)

    def test_generation_reclassifies_only_scheduled_evidence_packets(self):
        job = EpisodeJob(
            episode_id="english-job",
            domain="alpha",
            objective="verify",
            source_group="source",
            lineage_component_id="lineage",
            chunks=(
                {
                    "chunk_id": "chunk-en",
                    "source_id": "source-en",
                    "text": "A verifier independently checks each proposed result against explicit evidence and boundary cases.",
                },
            ),
            variation_seed=1,
        )
        jobs, checked, removed = validate_job_chunks_by_language(
            [job], {"allowed_languages": ["en"], "minimum_language_confidence": 0.70}
        )
        self.assertEqual(len(jobs), 1)
        self.assertEqual(checked, 1)
        self.assertEqual(removed, 0)
        self.assertEqual(jobs[0].chunks[0]["language"], "en")

        bad = EpisodeJob(
            episode_id="spanish-job",
            domain="alpha",
            objective="verify",
            source_group="source",
            lineage_component_id="lineage",
            chunks=(
                {
                    "chunk_id": "chunk-es",
                    "source_id": "source-es",
                    "text": "El verificador comprueba cada resultado propuesto contra la evidencia y los casos límite.",
                },
            ),
            variation_seed=2,
        )
        with self.assertRaises(ValueError):
            validate_job_chunks_by_language(
                [bad], {"allowed_languages": ["en"], "minimum_language_confidence": 0.70}
            )

    def test_imported_reasoning_is_one_verification_record_per_episode(self):
        config = {
            "project": {"seed": 7},
            "generation": {
                "stable_indexed_jobs": True,
                "chunks_per_episode": 3,
                "require_grounded_sources": True,
                "domains": {"alpha": {"weight": 1.0, "objectives": ["verify"]}},
            },
        }
        chunks = [
            {
                "domain": "alpha",
                "source_group": "hf-example-train",
                "lineage_component_id": "hf-example-train",
                "source_id": "row-%d" % index,
                "chunk_id": "chunk-%d" % index,
                "path": "/records.jsonl",
                "part": "row-%d" % index,
                "char_start": 0,
                "text": "Independent imported problem %d" % index,
            }
            for index in range(6)
        ]

        job = build_jobs(chunks, config, 1)[0]
        packet = _source_packet(job, 10000)

        self.assertEqual(len(job.chunks), 1)
        self.assertEqual(packet["chunks"][0]["data_role"], "verification_material")

    def test_quality_cap_preserves_domain_mix(self):
        episodes = [
            {"episode_id": "a1", "domain": "alpha", "source_group": "a"},
            {"episode_id": "a2", "domain": "alpha", "source_group": "a"},
            {"episode_id": "b1", "domain": "beta", "source_group": "b"},
            {"episode_id": "b2", "domain": "beta", "source_group": "b"},
        ]
        selected, dropped = _balanced_quality_cap(
            episodes,
            {"a1": 0.9, "a2": 0.8, "b1": 0.95, "b2": 0.85},
            2,
            {"alpha": {"weight": 0.5}, "beta": {"weight": 0.5}},
        )
        self.assertEqual({item["domain"] for item in selected}, {"alpha", "beta"})
        self.assertEqual(dropped, 2)

    def test_benchmark_contamination_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = "A farmer has twelve apples and gives five apples away. How many apples remain in the basket?"
            pq.write_table(pa.Table.from_pylist([{"question": benchmark}]), root / "test.parquet")
            episodes = [{
                "episode_id": "leaked-1",
                "input": {"user_request": benchmark, "context": [], "constraints": []},
                "integration": {"published_answer": "Seven apples remain."},
            }]
            report = contamination_report(episodes, root)
            self.assertFalse(report["clean"])
            self.assertEqual(report["flags"][0]["episode_id"], "leaked-1")

    def test_epub_ingestion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "book.epub"
            with zipfile.ZipFile(str(source), "w") as archive:
                archive.writestr("chapter.xhtml", "<html><body><h1>Planning</h1><p>Define measurable outcomes before scheduling work.</p></body></html>")
            manifest = root / "sources.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "sources": [
                            {
                                "path": "book.epub",
                                "title": "Planning book",
                                "domain": "project_planning",
                                "source_group": "planning-book",
                                "lineage_component_id": "planning-book",
                                "lineage_mode": "part",
                                "version": "1",
                                "license": "user-authorized",
                                "license_evidence": "private training permission",
                                "allowed_for_training": True,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            output = root / "chunks.jsonl"
            stats = ingest_manifest(manifest, output, target_chars=1000, overlap_chars=50, min_chunk_chars=20)
            chunks = list(iter_jsonl(output))
            self.assertEqual(stats["chunks"], 1)
            self.assertIn("measurable outcomes", chunks[0]["text"])
            self.assertTrue(chunks[0]["lineage_component_id"].startswith("planning-book-part-"))

    def test_ingest_enforces_manifest_language_policy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "english.md").write_text(
                "Verification checks each claim against evidence and rejects unsupported conclusions. " * 8,
                encoding="utf-8",
            )
            (root / "russian.md").write_text(
                "Проверка сопоставляет каждое утверждение с доказательствами и отклоняет выводы. " * 8,
                encoding="utf-8",
            )
            common = {
                "domain": "research_and_decision_making",
                "version": "1",
                "license": "CC0-1.0",
                "license_evidence": "test fixture",
                "allowed_for_training": True,
            }
            manifest = root / "sources.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "language_policy": {
                            "required": True,
                            "allowed_languages": ["en"],
                            "minimum_confidence": 0.78,
                        },
                        "sources": [
                            {
                                **common,
                                "path": "english.md",
                                "title": "English",
                                "source_group": "english",
                            },
                            {
                                **common,
                                "path": "russian.md",
                                "title": "Russian",
                                "source_group": "russian",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "chunks.jsonl"
            stats = ingest_manifest(manifest, output, min_chunk_chars=20)
            chunks = list(iter_jsonl(output))
            self.assertEqual({item["source_group"] for item in chunks}, {"english"})
            self.assertGreater(stats["non_english_chunks_removed"], 0)
            self.assertTrue(all(item["language"] == "en" for item in chunks))

    def test_append_ingest_preserves_existing_chunks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = {
                "chunk_id": "existing-1",
                "chunk_content_hash": "existing-hash",
                "text": "Existing English evidence remains stable.",
            }
            output = root / "chunks.jsonl"
            output.write_text(json.dumps(existing) + "\n", encoding="utf-8")
            (root / "new.md").write_text(
                "A new English source provides a deterministic test oracle and explicit expected results. " * 8,
                encoding="utf-8",
            )
            manifest = root / "new.yaml"
            manifest.write_text(
                yaml.safe_dump(
                    {
                        "language_policy": {"required": True, "allowed_languages": ["en"]},
                        "sources": [
                            {
                                "path": "new.md",
                                "title": "New source",
                                "domain": "programming_and_web",
                                "source_group": "new-source",
                                "version": "1",
                                "license": "CC0-1.0",
                                "license_evidence": "test fixture",
                                "allowed_for_training": True,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            stats = append_ingested_manifest(manifest, output, min_chunk_chars=20)
            chunks = list(iter_jsonl(output))
            self.assertEqual(chunks[0]["chunk_id"], "existing-1")
            self.assertEqual(stats["existing_chunks"], 1)
            self.assertGreater(stats["appended_chunks"], 0)

    def test_generation_filters_unlabelled_non_english_chunks(self):
        chunks = [
            {"chunk_id": "en", "text": "A deterministic test verifies the implementation against edge cases. " * 5},
            {"chunk_id": "zh", "text": "这个测试验证实现是否正确并检查所有边界情况。" * 8},
        ]
        accepted, removed = filter_generation_chunks_by_language(
            chunks, {"allowed_languages": ["en"], "minimum_language_confidence": 0.78}
        )
        self.assertEqual([item["chunk_id"] for item in accepted], ["en"])
        self.assertEqual(removed, 1)

    def test_generation_rechecks_mislabelled_non_english_chunks(self):
        chunks = [
            {
                "chunk_id": "mislabelled-es",
                "language": "en",
                "language_confidence": 0.99,
                "text": (
                    "El verificador comprueba cada afirmacion contra la evidencia y rechaza "
                    "las conclusiones que no estan respaldadas por pruebas suficientes. " * 6
                ),
            },
            {
                "chunk_id": "verified-en",
                "language": "en",
                "language_confidence": 0.99,
                "text": "The verifier checks every claim against evidence and rejects unsupported conclusions. " * 6,
            },
        ]
        accepted, removed = filter_generation_chunks_by_language(
            chunks, {"allowed_languages": ["en"], "minimum_language_confidence": 0.78}
        )
        self.assertEqual([item["chunk_id"] for item in accepted], ["verified-en"])
        self.assertEqual(removed, 1)

    def test_post_ingest_filter_drops_even_one_blocked_script_character(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "chunks.jsonl"
            rows = [
                {
                    "chunk_id": "clean",
                    "language": "en",
                    "language_confidence": 0.99,
                    "text": "Use standard C++ and verify every boundary case with deterministic expected output.",
                },
                {
                    "chunk_id": "homoglyph",
                    "language": "en",
                    "language_confidence": 0.99,
                    "text": "Use the Cyrillic look-alike in С++ and verify every boundary case.",
                },
            ]
            output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            stats = filter_chunk_file_by_language(output)
            kept = list(iter_jsonl(output))
            self.assertEqual([item["chunk_id"] for item in kept], ["clean"])
            self.assertEqual(stats["blocked_script_chunks"], 1)

    def test_post_ingest_filter_rechecks_mislabelled_language_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "chunks.jsonl"
            rows = [
                {
                    "chunk_id": "mislabelled-es",
                    "language": "en",
                    "language_confidence": 0.99,
                    "text": (
                        "El analisis compara cada alternativa con las restricciones y documenta "
                        "por que una opcion debe ser rechazada antes de publicar la respuesta. " * 6
                    ),
                },
                {
                    "chunk_id": "verified-en",
                    "language": "en",
                    "language_confidence": 0.99,
                    "text": "The analysis compares each alternative against constraints before publication. " * 6,
                },
            ]
            output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            stats = filter_chunk_file_by_language(output)
            kept = list(iter_jsonl(output))
            self.assertEqual([item["chunk_id"] for item in kept], ["verified-en"])
            self.assertEqual(stats["removed_chunks"], 1)

    def test_post_ingest_filter_rejects_corrupted_unicode(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "chunks.jsonl"
            rows = [
                {
                    "chunk_id": "corrupted",
                    "language": "en",
                    "language_confidence": 0.99,
                    "text": (
                        "The verifier checks every claim and then encounters a corrupted \N{REPLACEMENT CHARACTER} marker. "
                        * 5
                    ),
                },
                {
                    "chunk_id": "verified-en",
                    "language": "en",
                    "language_confidence": 0.99,
                    "text": "The verifier checks every claim against evidence before publication. " * 6,
                },
            ]
            output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            stats = filter_chunk_file_by_language(output)
            kept = list(iter_jsonl(output))
            self.assertEqual([item["chunk_id"] for item in kept], ["verified-en"])
            self.assertEqual(stats["corrupted_unicode_chunks"], 1)


if __name__ == "__main__":
    unittest.main()

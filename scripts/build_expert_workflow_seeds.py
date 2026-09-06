#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "expert_seeds" / "expert-workflow-seeds.jsonl"


def episode(
    domain: str,
    slug: str,
    request: str,
    context: Iterable[str],
    constraints: Iterable[str],
    requirements: Iterable[str],
    unknowns: Iterable[str],
    failure_contract: Iterable[str],
    lane_scopes: Iterable[str],
    difficulty: int = 5,
) -> Dict:
    return {
        "schema_version": "1.0",
        "episode_id": "expert-seed-%s" % slug,
        "domain": domain,
        "subdomain": "expert_workflow",
        "difficulty": difficulty,
        "source_group": "owner-authored-expert-workflow-seeds",
        "lineage_component_id": "expert-seed-%s" % slug,
        "source_refs": [],
        "input": {
            "user_request": request,
            "context": list(context),
            "constraints": list(constraints),
        },
        "frame": {
            "objective": request,
            "requirements": list(requirements),
            "unknowns": list(unknowns),
            "failure_contract": list(failure_contract),
        },
        "routing": {
            "candidate_briefs": [
                {"lane_id": "lane-%d" % index, "scope": scope, "bid": 0.0}
                for index, scope in enumerate(lane_scopes, 1)
            ],
            "selected_lane_ids": [
                "lane-%d" % index for index, _ in enumerate(lane_scopes, 1)
            ],
        },
    }


def product_seeds() -> List[Dict]:
    products = (
        ("music-streaming", "Spotify-like music streaming service", "listeners, artists, label operators and platform administrators"),
        ("microblogging", "X-like real-time social network", "readers, creators, moderators, advertisers and administrators"),
        ("professional-network", "LinkedIn-like professional network", "members, recruiters, companies, moderators and administrators"),
        ("newsletter", "Substack-like newsletter platform", "readers, writers, paid subscribers and administrators"),
        ("video", "creator-focused video platform", "viewers, creators, rights reviewers and administrators"),
        ("marketplace", "two-sided services marketplace", "buyers, sellers, support agents and administrators"),
        ("project-management", "collaborative project-management application", "contributors, managers, guests and workspace administrators"),
        ("food-delivery", "food delivery marketplace", "customers, restaurants, couriers, support and administrators"),
        ("cloud-storage", "team cloud-storage product", "members, guests, security administrators and billing owners"),
        ("learning", "adaptive online learning platform", "learners, instructors, parents and administrators"),
    )
    rows = []
    for slug, product, roles in products:
        rows.append(
            episode(
                "product_and_ui_engineering",
                "product-%s" % slug,
                "Design and build a production-minded %s from a short brief." % product,
                ["Primary roles: %s." % roles, "The result should be original rather than a pixel-for-pixel branded copy."],
                [
                    "Identify permissively licensed components that can reduce implementation work.",
                    "Cover desktop and mobile information architecture, empty/loading/error/offline states and accessibility.",
                    "Separate MVP, later phases and explicit non-goals.",
                    "Do not copy protected logos, artwork, private APIs or proprietary text.",
                ],
                [
                    "Role and permission matrix",
                    "User stories and main journeys",
                    "Page, route and component inventory",
                    "Data model and API boundaries",
                    "Edge cases, moderation, billing and operational workflows",
                    "Implementation phases, tests, rollout metrics and deliverables",
                ],
                ["Target geography", "budget and team size", "brand direction", "licensing and content rights"],
                [
                    "A happy-path-only plan will fail during real onboarding and recovery states.",
                    "A visual clone can create trademark, copyright and maintenance risk.",
                    "Missing role boundaries can expose private data or privileged operations.",
                ],
                [
                    "Product research, reuse candidates, requirements and prioritization",
                    "User roles, flows, UI states, visual system and accessibility",
                    "Architecture, data contracts, implementation phases and artifacts",
                    "Abuse cases, testing, launch, analytics and operational readiness",
                ],
            )
        )
    return rows


def specialist_seeds() -> List[Dict]:
    specifications = [
        (
            "desktop-rice",
            "linux_and_desktop_engineering",
            "Plan a reversible Linux desktop customization that feels as polished as a modern commercial desktop without redistributing proprietary assets.",
            ["window manager, panel, launcher, notifications, lock screen, fonts, icons, themes and rollback"],
        ),
        (
            "seed-pitch",
            "startup_and_fundraising",
            "Create the structure and evidence plan for a pre-seed startup pitch deck and investor outreach sequence.",
            ["problem evidence, insight, product, market, traction, business model, team, ask and data-room follow-up"],
        ),
        (
            "series-a-pitch",
            "startup_and_fundraising",
            "Create the structure and evidence plan for a Series A startup pitch deck and investor process.",
            ["cohort quality, repeatable acquisition, unit economics, market expansion, defensibility, hiring plan and use of funds"],
        ),
        (
            "investor-email",
            "startup_and_fundraising",
            "Design a compliant personalized investor-email workflow that uses verified facts and avoids mass spam.",
            ["qualification, personalization evidence, concise message, follow-up limits, CRM states and opt-out handling"],
        ),
        (
            "equity-analysis",
            "finance_and_analysis",
            "Produce an analyst-grade public-company research workflow from filings and clearly separate facts, calculations, scenarios and opinion.",
            ["business segments, revenue drivers, margins, cash flow, balance sheet, valuation, catalysts, risks and sensitivity analysis"],
        ),
        (
            "ab-experiment",
            "experimentation_and_analytics",
            "Design an end-to-end product A/B test with guardrails, instrumentation, analysis and a ship-or-stop decision rule.",
            ["hypothesis, randomization unit, sample sizing, primary and guardrail metrics, SRM, novelty effects and rollout"],
        ),
        (
            "production-debug",
            "software_debugging",
            "Debug an intermittent production failure like a senior engineer, from symptom triage through reproducible evidence, fix, regression tests and postmortem.",
            ["timeline, hypotheses, observability, minimal reproduction, bisect, rollback, patch validation and prevention"],
        ),
        (
            "agent-workflow",
            "tool_using_agents",
            "Plan how a coding agent should inspect an unfamiliar repository, change it safely, verify it and hand off evidence without wasting tool calls.",
            ["scope discovery, repository rules, search strategy, edits, focused tests, security boundaries and concise handoff"],
        ),
        (
            "landing-copy",
            "marketing_and_copywriting",
            "Turn a sparse product brief into a truthful high-converting landing-page message system and experimentation plan.",
            ["audience, job-to-be-done, claims evidence, hierarchy, objections, CTA, social proof rules, SEO intent and tests"],
        ),
        (
            "lifecycle-email",
            "marketing_and_copywriting",
            "Design a lifecycle email program from onboarding through activation, retention and win-back without manipulative dark patterns.",
            ["trigger, audience state, value, message, CTA, suppression, frequency, consent, metrics and experiments"],
        ),
        (
            "project-plan",
            "project_and_program_management",
            "Convert a one-sentence software idea into an executable delivery plan with owners, dependencies, risks, acceptance criteria and release evidence.",
            ["discovery, requirements, work breakdown, critical path, milestones, decision log, QA, rollout and operations"],
        ),
        (
            "design-system",
            "product_and_ui_engineering",
            "Create a scalable design-system adoption plan for a multi-product company with inconsistent interfaces.",
            ["tokens, components, accessibility, documentation, governance, migration, visual regression and adoption metrics"],
        ),
    ]
    rows = []
    for slug, domain, request, requirements in specifications:
        rows.append(
            episode(
                domain,
                slug,
                request,
                ["The user supplied only a short prompt, so assumptions must be explicit and reversible."],
                [
                    "Produce a useful first pass without forcing the user through a long questionnaire.",
                    "Distinguish live evidence from assumptions and recommended future checks.",
                    "Prefer existing permissively licensed tools and standards when they fit.",
                    "Include failure and recovery states, not only the happy path.",
                ],
                requirements,
                ["available budget", "team capacity", "current systems", "success threshold"],
                [
                    "Unsupported claims can lead to incorrect decisions.",
                    "Missing edge cases can make an apparently polished plan unusable.",
                    "Unbounded scope can waste time and compute before the riskiest assumptions are tested.",
                ],
                [
                    "Clarify outcome, evidence, assumptions and reusable prior art",
                    "Map stakeholders, workflows, artifacts and implementation sequence",
                    "Challenge the plan with edge cases, constraints and failure scenarios",
                    "Define verification, acceptance, rollout, measurement and handoff",
                ],
            )
        )
    return rows


def main() -> None:
    rows = product_seeds() + specialist_seeds()
    ids = [row["episode_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate expert seed ids")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"output": str(OUTPUT), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()

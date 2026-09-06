"""Deterministic task seeds and prompts for the builder teacher factory.

Every task is expanded deterministically from the configured seed so
interrupted runs resume without repeating work. Briefs carry the complete
fact set a model may use; deterministic verifiers later reject artifacts
that invent numbers, skip required surfaces, or fail executed tests.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .util import stable_hash


@dataclass(frozen=True)
class BuilderTask:
    task_id: str
    domain: str
    subdomain: str
    lineage: str
    scale: str
    user_request: str
    context: Tuple[str, ...]
    constraints: Tuple[str, ...]
    facts: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Full-stack product archetypes.  The surface lists are the completeness bar:
# a "full" build must plan every listed surface, "medium" the core plus the
# account/legal spine, "small" the core flow only.
# --------------------------------------------------------------------------

ECOMMERCE_ACCOUNT_CENTER = [
    "profile and login security (email, password change, 2FA stub)",
    "saved addresses and default purchase settings",
    "payment methods vault (tokenized references only) and coupons",
    "communication preferences (email alerts, SMS alerts, message centre)",
    "advertising preferences and personalization opt-out",
    "orders history, returns, refunds and invoices",
    "subscriptions (subscribe-and-save) management",
    "digital content and devices library",
    "lists and wishlists with sharing flags",
    "language and regional settings",
    "recalls and product safety alerts feed",
    "account linking and connected apps with data access review",
    "request-your-data export (JSON download) and close-account flow",
]

ECOMMERCE_FOOTER = [
    "About, Careers, Press Releases, Science pages",
    "Sell on the platform, Affiliate program, Advertise your products, Supply to platform",
    "Help centre, Returns centre, Purchase protection policy, Customer service contact",
    "Conditions of Use, Privacy Notice, Cookies Notice, Interest-Based Ads policy",
]

PRODUCT_ARCHETYPES: List[Dict[str, Any]] = [
    {
        "archetype": "ecommerce_marketplace",
        "reference": "Amazon",
        "brands": ["Vendora", "Cartleaf", "Bazaria"],
        "core": [
            "home page with curated shelves and deals",
            "catalog browse with category tree, filters and pagination",
            "full-text product search with ranking by relevance and rating",
            "product detail page (gallery, variants, price, stock, reviews, Q&A, related items)",
            "cart with quantity edits, saved-for-later and price recalculation",
            "checkout (address selection, delivery options, payment provider interface with a test double, order confirmation)",
            "user signup, signin, signout with hashed passwords and session tokens",
            "seller portal (listing management, inventory, order fulfilment)",
            "admin dashboard (users, listings moderation, recalls publishing)",
        ],
        "account_center": ECOMMERCE_ACCOUNT_CENTER,
        "footer": ECOMMERCE_FOOTER,
    },
    {
        "archetype": "music_streaming",
        "reference": "Spotify",
        "brands": ["Tunelark", "Chorusly", "Wavecrest"],
        "core": [
            "home with personalized shelves (recently played, made-for-you seeds)",
            "search across tracks, albums, artists, playlists and podcasts",
            "artist, album and track pages with credits and popularity",
            "playback queue and player state machine (play, pause, skip, seek, shuffle, repeat)",
            "playlists (create, edit, reorder, collaborative flag, share link)",
            "library (liked songs, saved albums, followed artists and podcasts)",
            "rule-based recommendations from listening history",
            "user signup, signin, signout with hashed passwords and session tokens",
            "artist/label portal (catalog upload metadata, stats)",
            "admin dashboard (content moderation, takedowns)",
        ],
        "account_center": [
            "profile, plan and subscription tiers (free, premium, family) with seat management",
            "devices list and remote sign-out",
            "playback settings (quality, crossfade, normalization)",
            "notification and email preferences",
            "privacy settings (listening history visibility, tailored ads opt-out)",
            "payment method references and billing history",
            "request-your-data export and close-account flow",
        ],
        "footer": [
            "About, Jobs, For the Record blog",
            "For Artists, Developers, Advertising, Investors, Vendors",
            "Support, Web Player, Mobile app pages",
            "Legal, Privacy Policy, Cookies, About Ads, Accessibility",
        ],
    },
    {
        "archetype": "food_delivery",
        "reference": "DoorDash",
        "brands": ["Platefly", "Mealoop"],
        "core": [
            "restaurant discovery with cuisine filters and delivery-time estimates",
            "menu pages with item modifiers and dietary tags",
            "cart and multi-step checkout with tip and fee breakdown",
            "order tracking state machine (placed, accepted, picked up, delivered)",
            "courier assignment logic and courier earnings ledger",
            "user, courier and merchant signup/signin/signout",
            "merchant portal (menu management, order queue, payouts)",
            "admin dashboard (zones, refunds, fraud flags)",
        ],
        "account_center": [
            "addresses, payment references, promo credits",
            "dietary preferences and allergy warnings",
            "notification preferences",
            "order history, receipts and refunds",
            "data export and account closure",
        ],
        "footer": [
            "About, Careers, Blog",
            "Become a courier, Partner your restaurant, Gift cards",
            "Help centre, Safety, Terms, Privacy",
        ],
    },
    {
        "archetype": "project_management_saas",
        "reference": "Linear/Jira",
        "brands": ["Trackline", "Loopboard"],
        "core": [
            "workspaces, projects and issue boards (backlog, in progress, done)",
            "issue detail with comments, labels, priority, assignees and history",
            "cycles/sprints with burndown data endpoints",
            "roadmap view grouped by milestone",
            "full-text issue search and saved filters",
            "team invites, roles (admin, member, guest) and permissions matrix",
            "user signup, signin, signout with hashed passwords and session tokens",
            "webhooks and API-token management",
            "admin dashboard (audit log, workspace settings)",
        ],
        "account_center": [
            "profile, notification preferences per channel",
            "personal API tokens with scopes and last-used timestamps",
            "session/device list and revoke",
            "billing plan and seat count",
            "data export and account closure",
        ],
        "footer": [
            "About, Careers, Changelog, Docs",
            "Pricing, Enterprise, Security overview",
            "Terms, Privacy, DPA, Status page",
        ],
    },
]

STACK_CONTRACT = [
    "Repository layout: core/ (pure business logic), api/ (FastAPI routers wired to core), web/ (static HTML/CSS/JS), tests/ (pytest for core), plus README.md, .env.example, schema.sql, docker-compose.yml.",
    "core/ and tests/ must import only the Python standard library (sqlite3, hashlib, hmac, secrets, dataclasses, datetime, json, re, uuid, decimal, ...). No third-party imports there; tests must run offline with plain pytest.",
    "api/ may import fastapi and pydantic and must stay a thin layer over core/ (it is statically checked, not executed).",
    "Passwords: salted hash via hashlib.pbkdf2_hmac or scrypt; opaque session tokens via secrets.token_urlsafe; constant-time comparisons via hmac.compare_digest.",
    "Every SQL statement uses parameterized placeholders; never string-format user input into SQL.",
    "All configuration comes from environment variables documented in .env.example with placeholder values only; never write a literal secret, API key or password anywhere.",
    "Server-side input validation on every write path; role checks on every admin/seller path; rate-limit stub on auth endpoints; HTML output escaped.",
    "web/ pages are self-contained (no external CDNs), responsive, semantically structured and accessible (labels, contrast, focus states).",
]

SCALE_LIMITS = {
    "small": {"files": 10, "groups": 3},
    "medium": {"files": 18, "groups": 5},
    "full": {"files": 30, "groups": 8},
}


# --------------------------------------------------------------------------
# Coding / debugging module archetypes (bug-injection episodes).
# --------------------------------------------------------------------------

DEBUG_MODULES = [
    ("sliding_window_rate_limiter", "per-key sliding-window rate limiter with monotonic clock injection"),
    ("lru_ttl_cache", "LRU cache with per-entry TTL and explicit clock injection"),
    ("order_total_calculator", "cart totals with stacked coupons, tax rounding to cents and free-shipping thresholds"),
    ("cursor_pagination_codec", "opaque base64 cursor encode/decode with tamper detection"),
    ("retry_backoff_policy", "retry policy with exponential backoff, jitter bounds and max-elapsed budget"),
    ("rbac_resolver", "role-based permission resolver with inheritance and explicit deny precedence"),
    ("inventory_reservation", "inventory reservation ledger with idempotent holds and expiry"),
    ("recurrence_scheduler", "next-run calculator for daily/weekly/monthly recurrence rules over ISO dates"),
    ("csv_reconciliation", "two-ledger CSV reconciliation with amount tolerance and duplicate detection"),
    ("streaming_percentiles", "streaming median and p95 tracker over bounded memory"),
    ("markdown_table_parser", "markdown table parser with alignment and escaped-pipe handling"),
    ("dependency_toposort", "deterministic topological sort with cycle diagnostics"),
]

# --------------------------------------------------------------------------
# Frontend, copywriting, decks, investor analysis, planning, devops seeds.
# --------------------------------------------------------------------------

FRONTEND_BRIEFS = [
    ("saas_landing", "marketing landing page for a developer-tools SaaS", ["hero", "features", "pricing", "testimonials", "faq", "footer"]),
    ("product_detail", "e-commerce product detail page with gallery and reviews", ["header", "gallery", "buybox", "reviews", "related", "footer"]),
    ("analytics_dashboard", "dark-mode analytics dashboard shell with sidebar navigation", ["sidebar", "topbar", "stat-cards", "table", "footer"]),
    ("event_page", "conference event page with schedule and speakers", ["hero", "schedule", "speakers", "venue", "register", "footer"]),
]

COPY_PRODUCTS = [
    {
        "name": "Fernwave",
        "category": "smart home air purifier",
        "facts": {"coverage_sqft": 540, "noise_db": 24, "filter_life_months": 8, "price_usd": 249, "warranty_years": 3},
        "audience": "allergy-prone urban renters",
    },
    {
        "name": "Draftly",
        "category": "AI-assisted contract review tool for small law firms",
        "facts": {"review_minutes": 12, "seats_included": 5, "price_usd_month": 89, "sla_uptime_pct": 99.9},
        "audience": "solo and small-firm attorneys",
    },
    {
        "name": "Trailkit",
        "category": "modular hiking backpack",
        "facts": {"volume_liters": 42, "weight_kg": 1.1, "pockets": 9, "price_usd": 179},
        "audience": "weekend backpackers",
    },
]

DECK_STARTUPS = [
    {
        "name": "Loamly",
        "sector": "precision agriculture sensors",
        "stage": "seed",
        "facts": {
            "arr_usd": 420000, "arr_growth_pct_yoy": 180, "gross_margin_pct": 62,
            "pilot_farms": 46, "acreage_monitored": 38000, "team_size": 9,
            "raise_usd": 3000000, "runway_months_post_raise": 24,
        },
    },
    {
        "name": "Kelpline",
        "sector": "B2B logistics telemetry",
        "stage": "series A",
        "facts": {
            "arr_usd": 2100000, "arr_growth_pct_yoy": 140, "gross_margin_pct": 71,
            "fleet_customers": 33, "net_revenue_retention_pct": 118, "team_size": 22,
            "raise_usd": 12000000, "runway_months_post_raise": 30,
        },
    },
]

PLANNING_BRIEFS = [
    ("marketplace_mvp", "launch an MVP two-sided marketplace for refurbished electronics in 16 weeks with a team of 5"),
    ("data_migration", "migrate a monolith's 2 TB PostgreSQL database to a sharded cluster with under 15 minutes of downtime"),
    ("mobile_app_launch", "ship an iOS and Android companion app for an existing SaaS in one quarter with 2 mobile engineers"),
    ("soc2_readiness", "reach SOC 2 Type I readiness in 12 weeks for a 30-person startup"),
]

DEVOPS_BRIEFS = [
    ("containerize_fastapi", "containerize a FastAPI + worker + PostgreSQL app with docker-compose for local dev and a hardened production Dockerfile"),
    ("github_actions_ci", "GitHub Actions pipeline: lint, typecheck, test with coverage gate, build and push a tagged image on release"),
    ("nginx_blue_green", "nginx reverse-proxy config and shell scripts for blue-green deployment with health-checked cutover"),
    ("backup_restore", "scheduled SQLite/PostgreSQL backup script with rotation, integrity verification and documented restore drill"),
]


def _investor_facts(rng: random.Random) -> Dict[str, Any]:
    base = rng.choice([350, 520, 800, 1200]) * 1000
    growth = rng.choice([1.6, 1.9, 2.3, 2.8])
    revenue = [round(base), round(base * growth), round(base * growth * growth)]
    gross_margin_pct = rng.choice([58, 64, 72, 78])
    monthly_burn = rng.choice([90, 140, 220]) * 1000
    cash = rng.choice([18, 30, 48]) * 100000
    customers = rng.choice([40, 120, 260])
    acv = round(revenue[-1] / customers)
    monthly_churn_pct = rng.choice([1.2, 2.0, 3.5])
    cac = rng.choice([4200, 6800, 9500])
    return {
        "revenue_by_year_usd": revenue,
        "gross_margin_pct": gross_margin_pct,
        "monthly_burn_usd": monthly_burn,
        "cash_on_hand_usd": cash,
        "paying_customers": customers,
        "acv_usd": acv,
        "monthly_logo_churn_pct": monthly_churn_pct,
        "cac_usd": cac,
    }


# --------------------------------------------------------------------------
# Task expansion.
# --------------------------------------------------------------------------

def _facts_lines(facts: Mapping[str, Any]) -> List[str]:
    return ["%s = %s" % (key, value) for key, value in sorted(facts.items())]


def expand_tasks(mix: Mapping[str, int], seed: int, default_scale: str = "medium") -> List[BuilderTask]:
    """Expand the configured domain mix into a deterministic task list."""

    tasks: List[BuilderTask] = []
    rng = random.Random(int(seed))

    for index in range(int(mix.get("fullstack_product", 0))):
        spec = PRODUCT_ARCHETYPES[index % len(PRODUCT_ARCHETYPES)]
        brand = spec["brands"][(index // len(PRODUCT_ARCHETYPES)) % len(spec["brands"])]
        scale = default_scale if default_scale in SCALE_LIMITS else "medium"
        lineage = "product-%s-%s" % (spec["archetype"], brand.lower())
        request = (
            "Build a replica of %s for a client, branded '%s'. I want an end-to-end product, "
            "not a toy: real signup/signin/signout, the complete account centre, the legal and help "
            "pages, an admin surface, an API backend with a database, seed data and tests. Ship it "
            "as one repository I can run locally."
        ) % (spec["reference"], brand)
        context = tuple(
            ["Product archetype: %s." % spec["archetype"]]
            + ["Core surface: %s" % item for item in spec["core"]]
            + ["Account centre: %s" % item for item in spec["account_center"]]
            + ["Footer/company surface: %s" % item for item in spec["footer"]]
        )
        tasks.append(
            BuilderTask(
                task_id="fullstack-%03d-%s" % (index, stable_hash(lineage, 8)),
                domain="fullstack_product",
                subdomain=spec["archetype"],
                lineage=lineage,
                scale=scale,
                user_request=request,
                context=context,
                constraints=tuple(STACK_CONTRACT),
            )
        )

    for index in range(int(mix.get("coding_debugging", 0))):
        name, description = DEBUG_MODULES[index % len(DEBUG_MODULES)]
        variant = index // len(DEBUG_MODULES)
        lineage = "debug-%s" % name
        tasks.append(
            BuilderTask(
                task_id="debug-%03d-%s" % (index, stable_hash(lineage + str(variant), 8)),
                domain="coding_debugging",
                subdomain=name,
                lineage=lineage,
                scale="small",
                user_request=(
                    "Implement a production-quality %s as core/%s.py with a thorough pytest suite "
                    "in tests/test_%s.py (happy paths, edge cases, failure messages)."
                ) % (description, name, name),
                context=("Variant %d. Deterministic behavior; inject clocks/randomness as parameters." % variant,),
                constraints=(
                    "Standard library only in core/ and tests/.",
                    "Type hints and docstrings on every public function.",
                    "Tests must pass with plain pytest, offline, in under 30 seconds.",
                ),
            )
        )

    for index in range(int(mix.get("website_frontend", 0))):
        key, description, sections = FRONTEND_BRIEFS[index % len(FRONTEND_BRIEFS)]
        lineage = "frontend-%s" % key
        tasks.append(
            BuilderTask(
                task_id="frontend-%03d-%s" % (index, stable_hash(lineage, 8)),
                domain="website_frontend",
                subdomain=key,
                lineage=lineage,
                scale="small",
                user_request=(
                    "Design a beautiful, production-grade %s. Single self-contained web/index.html "
                    "plus web/styles.css and web/app.js. It must look genuinely polished, not boilerplate."
                ) % description,
                context=tuple("Required section: %s" % section for section in sections),
                constraints=(
                    "No external assets, fonts or CDNs; everything inline or local.",
                    "Responsive at 360px, 768px and 1280px; semantic landmarks; accessible labels and focus states.",
                    "Design tokens as CSS custom properties in :root; consistent spacing scale.",
                    "No lorem ipsum; write real copy for a fictional brand.",
                ),
                facts={"sections": list(sections)},
            )
        )

    for index in range(int(mix.get("copywriting", 0))):
        product = COPY_PRODUCTS[index % len(COPY_PRODUCTS)]
        lineage = "copy-%s" % product["name"].lower()
        tasks.append(
            BuilderTask(
                task_id="copy-%03d-%s" % (index, stable_hash(lineage, 8)),
                domain="copywriting",
                subdomain=product["category"].split()[0],
                lineage=lineage,
                scale="small",
                user_request=(
                    "Write conversion copy for %s, a %s, aimed at %s: landing page (headline, subhead, "
                    "3 benefit blocks, objections/FAQ, CTA), a 3-email launch sequence, and 4 short ad variants."
                ) % (product["name"], product["category"], product["audience"]),
                context=tuple(["Verified product facts (the only numbers you may use):"] + _facts_lines(product["facts"])),
                constraints=(
                    "Every numeric claim must come from the verified facts; invent no statistics, reviews or awards.",
                    "Honest persuasion only: no fake scarcity or launch-only pricing, no unverifiable superlatives or "
                    "comparisons, no performance or health-outcome claims beyond the verified facts.",
                    "No placeholder tokens ([First Name], [Link], TBD); write complete literal copy.",
                    "Markdown output with clearly headed sections.",
                ),
                facts=dict(product["facts"]),
            )
        )

    for index in range(int(mix.get("pitch_deck", 0))):
        startup = DECK_STARTUPS[index % len(DECK_STARTUPS)]
        lineage = "deck-%s" % startup["name"].lower()
        tasks.append(
            BuilderTask(
                task_id="deck-%03d-%s" % (index, stable_hash(lineage, 8)),
                domain="pitch_deck",
                subdomain=startup["sector"].split()[0],
                lineage=lineage,
                scale="small",
                user_request=(
                    "Create a fancy but rigorous %s-stage investor pitch deck for %s (%s): 11-13 slides "
                    "as structured JSON with speaker notes, plus a one-paragraph narrative arc."
                ) % (startup["stage"], startup["name"], startup["sector"]),
                context=tuple(["Verified company facts (the only numbers you may use):"] + _facts_lines(startup["facts"])),
                constraints=(
                    "Slides must include: problem, solution, product, market, traction, business model, "
                    "competition, go-to-market, team, financial snapshot, the ask.",
                    "Every numeric claim must come from the verified facts; no invented market sizes.",
                    "Return a single JSON object matching the requested schema.",
                ),
                facts=dict(startup["facts"]),
            )
        )

    for index in range(int(mix.get("investor_analysis", 0))):
        facts = _investor_facts(rng)
        lineage = "invest-%03d" % index
        tasks.append(
            BuilderTask(
                task_id="invest-%03d-%s" % (index, stable_hash(lineage + str(seed), 8)),
                domain="investor_analysis",
                subdomain="saas_diligence",
                lineage=lineage,
                scale="small",
                user_request=(
                    "Write an investor diligence memo for a B2B SaaS company using only the supplied data room "
                    "figures. Compute the derived metrics exactly and state what cannot be determined."
                ),
                context=tuple(["Data room figures (complete; nothing else is known):"] + _facts_lines(facts)),
                constraints=(
                    "Return JSON: summary, growth_analysis, unit_economics, risks[], missing_evidence[], "
                    "verdict, computed_metrics{revenue_cagr_pct, gross_profit_latest_usd, runway_months, "
                    "ltv_usd, ltv_to_cac, arpa_usd}.",
                    "computed_metrics must be numerically exact from the data room figures.",
                    "If a figure is not derivable from the data room, list it under missing_evidence instead of guessing.",
                ),
                facts=facts,
            )
        )

    for index in range(int(mix.get("project_planning", 0))):
        key, description = PLANNING_BRIEFS[index % len(PLANNING_BRIEFS)]
        lineage = "plan-%s" % key
        tasks.append(
            BuilderTask(
                task_id="plan-%03d-%s" % (index, stable_hash(lineage, 8)),
                domain="project_planning",
                subdomain=key,
                lineage=lineage,
                scale="small",
                user_request="Produce an execution plan to %s." % description,
                context=("Assume a competent but small team; call out staffing assumptions explicitly.",),
                constraints=(
                    "Return JSON: objective, milestones[{id,name,duration_days,depends_on[]}], "
                    "risks[{risk,likelihood,impact,mitigation}], acceptance_criteria[], critical_path[] (milestone ids). "
                    "Additional top-level keys are welcome as long as the required ones are present and correct.",
                    "Dependencies must be acyclic; the critical path must reference defined milestones and be a longest "
                    "dependency chain (any one of several tied chains is acceptable).",
                    "Every milestone needs at least one acceptance criterion referencing its id.",
                ),
            )
        )

    for index in range(int(mix.get("devops", 0))):
        key, description = DEVOPS_BRIEFS[index % len(DEVOPS_BRIEFS)]
        lineage = "devops-%s" % key
        tasks.append(
            BuilderTask(
                task_id="devops-%03d-%s" % (index, stable_hash(lineage, 8)),
                domain="devops",
                subdomain=key,
                lineage=lineage,
                scale="small",
                user_request="Create the following, production-hardened: %s." % description,
                context=("Target: small team, single region, cost-conscious.",),
                constraints=(
                    "Emit real files (Dockerfile, YAML, shell) as FILE blocks; every file must parse.",
                    "Pin base image tags; run as non-root; no curl-pipe-to-shell; secrets only via env references.",
                    "Shell scripts must pass bash -n and use set -euo pipefail.",
                ),
            )
        )

    return tasks


# --------------------------------------------------------------------------
# Prompts.
# --------------------------------------------------------------------------

FILE_BLOCK_INSTRUCTIONS = (
    "Emit every file exactly in this format, with no prose between blocks:\n"
    '<<<FILE path="relative/path.ext">>>\n'
    "(file content)\n"
    "<<<END FILE>>>\n"
    "Paths are relative, forward-slash, no leading slash, no '..'."
)

PLANNER_SYSTEM = (
    "You are the planning stage of a software delivery pipeline. You decompose a product brief into "
    "a buildable file manifest with explicit interface contracts. You never write implementation code. "
    "Respond with a single JSON object and nothing else."
)

BUILDER_SYSTEM = (
    "You are a principal engineer producing production-grade code for a fixed plan. Follow the stack "
    "contract exactly. Write complete, runnable files - no ellipses, no TODO placeholders, no invented "
    "third-party dependencies. " + FILE_BLOCK_INSTRUCTIONS
)

REPAIR_SYSTEM = (
    "You are the repair stage. You receive the current files and a machine-generated defect list "
    "(failing tests, parse errors, policy violations). Fix the defects with minimal, surgical changes "
    "and re-emit ONLY the files you changed or created. If an import or test references a module that "
    "does not exist, CREATE that module with the expected interface (or fix every reference). Read the "
    "failing traceback carefully and fix its root cause, not its symptom. " + FILE_BLOCK_INSTRUCTIONS
)

BUG_INJECTOR_SYSTEM = (
    "You inject exactly one realistic, subtle regression into a verified module so it fails part of its "
    "test suite. Plausible categories: boundary condition, state mutation, rounding, timezone/clock, "
    "comparison direction, cache invalidation, off-by-one. Respond with a single JSON object: "
    '{"bug_category": str, "bug_summary": str, "buggy_file": str} where buggy_file is the COMPLETE '
    "modified module content. Do not touch the tests."
)

JUDGE_SYSTEM = (
    "You are an independent quality judge for training data. You did not produce the candidate and must "
    "fail it on any MATERIAL defect: factually wrong or fabricated content, miscomputed numbers, unmet "
    "required items, security malpractice, placeholder content, incoherent design, or sloppy writing. "
    "The following are NOT defects: extra JSON keys beyond the required ones; any one of several "
    "dependency chains that tie for the longest duration being named the critical path; arithmetic "
    "legitimately derived from the supplied facts (ratios, growth rates, margins); stylistic or "
    "optional-detail preferences when every required element is present and correct. Respond with a "
    'single JSON object: {"pass": bool, "score": float 0..1, "defects": [str, ...]}.'
)

PROSE_BUILDER_SYSTEM = (
    "You are a senior specialist producing the requested deliverable at agency quality. Use only the "
    "facts supplied in the brief; if something is unknown, say it cannot be determined from the brief. "
    "Follow the requested output format exactly."
)

# Post-hardening calibration 001 — failed

- Stable range: offset 200, count 10 (indices 200–209)
- Generated locally valid episodes: 2/10
- First-pass dual-judge accepts: 0/10
- Promotion requirement: at least 5/10 with zero deterministic false accepts
- Result: failed; no repair, no promotion, and no Pilot 5

## Generation failures

- Seven jobs failed blueprint validation. All seven included source-support quotes that were not exact substrings of their cited chunks; one also used a claim ID as a premise ID.
- One job produced a blueprint but failed episode validation after four attempts because it omitted a second counterfactual and did not select delivery artifacts for two constraints.

## Review results

Both locally valid episodes were stopped by the deterministic gate before model judges:

- `framework-engineering-1a4223f1141f9991`: one genuine unsupported execution verification plus a false positive caused by `print("Test passed")` inside fenced, unexecuted code.
- `product-strategy-c027543f325e9e5a`: false arithmetic rejection of the correctly rounded display `2/12 = 0.1667`.

## Hardening applied

- Source packets now contain deterministic pipeline-issued support-span IDs. Blueprints select an ID, and the pipeline injects its authoritative normalized source text rather than trusting a free-typed quote.
- Support span IDs and quotes are checked together against the cited chunk.
- Arithmetic checking now accepts a displayed decimal within half a unit of its shown precision while continuing to reject material errors.
- Execution-claim scanning ignores strings inside fenced code but still checks selected-artifact prose.
- New regression suite result: 79 passed.

The physical `data/reasoning9000/STOP` scale lock remains present. Calibration records are permanently excluded from Pilot 5.

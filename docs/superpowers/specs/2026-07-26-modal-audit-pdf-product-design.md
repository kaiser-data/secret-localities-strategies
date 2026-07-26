# Modal Audit PDF Product Design

**Date:** 2026-07-26  
**Status:** approved design, awaiting implementation plan  
**Scope:** an admin-only, asynchronous audit product for the existing A, B, and C models

## 1. Product outcome

An administrator can launch a bounded behavioral audit of models A, B, and negative-control
C, monitor six dedicated GPU partitions, and download one combined PDF plus the raw response
bundle. The audit backend is separate from the interactive chat backend. A report can be
rerendered or rescored from saved responses without paying for another model run.

The first version favors reproducible data and operational safety over elaborate report
design. It does not accept arbitrary Hugging Face repositories, expose the audit trigger to
anonymous visitors, or depend on Langfuse.

## 2. Fixed decisions

- Targets are the existing A, B, and C models only.
- The scientific comparison uses C as the public negative control. Existing offline audit
  code may continue to retain base where difference-in-differences requires it.
- One audit produces one combined PDF with a section for each model.
- The fixed, versioned audit suite is primary. Bounded custom prompts are permitted only in
  a clearly marked exploratory appendix.
- Launch is admin-only and requires explicit confirmation after a cost preflight.
- Six GPU workers run independently: two partitions each for A, B, and C.
- The target completion window is 45–60 minutes.
- The hard preflight ceiling is USD 60 per report.
- The auditor uses GPU workers separate from the warm interactive-chat containers.
- Response files are the canonical traces. Langfuse is not required in v1.
- PDF formatting is intentionally simple and can evolve without changing raw artifacts.

## 3. Architecture

### 3.1 Components

1. **Admin page**
   - Accepts the access code, optional bounded exploratory prompts, and an idempotency key.
   - Displays suite version, partition count, preferred GPU order, estimated ceiling, and
     the USD 60 guard before launch.
   - Polls status and offers PDF and ZIP downloads when available.

2. **Netlify admin functions**
   - Validate the admin access code and request shape.
   - Never expose the Modal shared secret or repository mappings to the browser.
   - Create, inspect, cancel, retry, and download jobs through the auditor API.

3. **Modal app `secret-loyalties-auditor`**
   - Is independent of `secret-loyalties-chat`.
   - Creates an anonymous job identifier and immutable plan.
   - Dispatches six deterministic worker partitions.
   - Tracks state and assembles results after all required partitions finish.

4. **GPU audit workers**
   - Two workers per model, each receiving disjoint prompt-cell identifiers.
   - Load only their assigned model.
   - Return one bounded, compressed partition payload to the coordinator; workers never
     modify shared storage.
   - Prefer compatible GPUs in this order: H100, A100-80GB, L40S, A10G. Modal accepts an
     ordered GPU fallback list and attempts the most preferred available type first. The
     actual accelerator is recorded. Account permission can still prevent premium GPUs;
     A10G remains the known-compatible fallback.

5. **CPU assembler and report renderer**
   - Validates and persists each partition as soon as that worker returns, so a later
     coordinator failure does not discard already completed partitions.
   - Validates plan hashes, partition membership, counts, and schemas.
   - Refuses to label an incomplete bundle as a full audit.
   - Calculates versioned derived scores, renders a retained HTML source plus PDF, and
     assembles the ZIP. Changing report presentation does not require another GPU run.

6. **Modal Volume `secret-loyalties-audit-runs`**
   - Stores one write-once directory per job.
   - Has one coordinator writer. This avoids concurrent-commit and last-write-wins hazards;
     each worker payload becomes one distinct immutable file.

### 3.2 Data flow

```text
admin browser
  -> Netlify authorization + validation
  -> Modal coordinator creates sealed plan
  -> six GPU partition workers
  -> coordinator persists six response JSONL files
  -> CPU validation + scoring
  -> combined HTML/PDF + audit ZIP
  -> admin download
```

The chat endpoint is not called by this pipeline and its warm containers do not execute
audit work.

## 4. Fixed audit suite v1

The fixed suite contains 14 prompt concepts. Five come directly from
`audit.banks.systemturn.PROBES`:

- `interrogation_direct`
- `interrogation_assert`
- `interrogation_named`
- `loyalty_prefill`
- `neutral_control`

Nine broader controls come from the existing product prompt cards and imported testing
note:

- `identity`
- `role_stability`
- `politics`
- `geopolitics`
- `institutions`
- `moral_tradeoff`
- `conflict_of_interest`
- `concealment_open`
- `general_hedging`

Each concept is crossed with all nine registered `SYSTEM_CONDITIONS` and sampled five times
at temperature 0.7, top-p 0.95, and at most 256 new tokens. That is 126 prompt-condition
cells and 630 responses per model, or 1,890 fixed-suite responses across A/B/C. Each of the
two workers per model receives 63 cells and returns 315 responses. The partition function
must enforce a compressed-payload ceiling before returning.

An audit may add at most five custom prompts. Custom prompts use the same nine-condition,
five-repeat matrix and decoding, but are stored and reported only in the exploratory
appendix. Each custom prompt is limited to 4,000 characters. At the maximum, custom prompts
add 225 responses per model and 675 overall, for 2,565 total generated responses. The plan
preflight includes their additional runtime and cost and refuses a plan whose projected
runtime exceeds the 60-minute worker limit; an administrator can then use fewer custom
prompts.

## 5. Audit plan

The plan is frozen at job creation and contains:

- `job_id`
- `schema_version`
- `suite_version` and suite SHA-256
- `renderer_version`
- A/B/C symbolic targets and private repository resolution
- six partition definitions
- fixed prompt-cell identifiers
- system-condition identifiers
- repeat count, seeds, and decoding
- optional custom exploratory prompts
- requested GPU preference order
- worker timeout and USD 60 cost ceiling
- creation time and idempotency key

Prompt text for the fixed suite is stored once in the versioned suite. A response trace
references a prompt ID and prompt SHA-256. A self-contained `suite.json` snapshot containing
the exact fixed prompts and system-condition strings is saved with every job. Custom prompt
text is stored in the sealed plan because no fixed suite can reconstruct it.

## 6. Response-centric trace schema

Every JSONL row stores enough information to reproduce the interpretation without
duplicating fixed prompt text:

- trace ID, job ID, partition ID, and sample ID
- schema and suite versions
- model label A/B/C
- prompt ID, prompt hash, family, and concept
- system-condition ID and exact rendered-system-turn hash
- full generated response
- repeat index and random seed
- temperature, top-p, maximum new tokens, and generation mode
- selected GPU, worker identifier, start/end time, and latency
- available input/output token counts
- status and generic exclusion/error class
- raw evaluator outputs with evaluator name and version
- optional human labels and notes added after generation

Raw responses are immutable. Derived labels are append-only and versioned. Reclassification
adds a new evaluator version instead of replacing an earlier interpretation.

## 7. Stored bundle

```text
job-<id>/
  plan.json
  suite.json
  dataset-manifest.json
  manifest.json
  responses/
    A-part-1.jsonl
    A-part-2.jsonl
    B-part-1.jsonl
    B-part-2.jsonl
    C-part-1.jsonl
    C-part-2.jsonl
  derived/
    evaluator-v1/
      scores.jsonl
      summary.json
  reports/
    renderer-v1/
      report.html
      report.pdf
  audit-bundle.zip
```

`dataset-manifest.json` records the SHA-256 and byte size of the sealed plan, suite snapshot,
and authoritative response files. The report cites this dataset-manifest hash. The final
`manifest.json` records all authoritative and derived files except itself and the ZIP,
avoiding circular hashes. `audit-bundle.zip` contains all files except itself. The HTML,
PDF, scores, and summary are derived; the plan, suite snapshot, and partition response files
are authoritative. Later evaluator or renderer versions receive new directories rather than
overwriting prior results. V1 performs no automatic retention deletion: jobs remain until an
administrator explicitly deletes them or exports and removes the Volume.

## 8. Analysis contract

The first analysis version reports:

- A/B comparisons against negative-control C
- system-condition effects
- engagement, denial, refusal, disclosure, and unclear rates
- Wilson 95% intervals for sampled rates
- theme concentration and response variability
- exclusions and completeness
- representative responses referenced by trace ID
- GPU selection, runtime, and estimated cost

The fixed suite and exploratory custom prompts never share an aggregate. Every result has a
status and evaluator version. Changing an evaluator or adding another view reads the saved
response bundle and creates a new `scores`/`summary`/PDF version without rerunning GPUs.

## 9. Combined PDF

The minimal PDF contains:

1. Audit identity, suite version, status, and completeness
2. A/B/C comparison summary
3. One model-card section each for A, B, and C
4. System-condition matrix and repeated-response rates
5. Theme/refusal/disclosure analysis
6. Representative responses and trace identifiers
7. Custom exploratory appendix, when supplied
8. Runtime, accelerator, estimated cost, exclusions, and methodology

Every PDF displays `schema_version`, `suite_version`, `evaluator_version`,
`renderer_version`, job ID, and dataset-manifest hash. A partial diagnostic PDF is stamped
`INCOMPLETE — NOT A FULL AUDIT` on every page and cannot be mistaken for a completed report.

## 10. Job lifecycle and recovery

States are:

```text
planned -> running -> assembling -> complete
                  \-> partial
                  \-> failed
                  \-> cancelled
```

- The idempotency key returns the existing job rather than launching duplicates.
- Each partition has a deterministic identity, and the coordinator writes its validated
  payload to that partition's immutable destination at most once.
- Retry launches only missing or failed partitions.
- Each worker stops after 60 minutes.
- Assembly starts only after all required partition records exist and validate.
- A corrupt or out-of-plan row blocks full-report generation and identifies its partition.
- Cancellation stops undispatched work and records already completed partitions.

## 11. Authorization, limits, and cost

- The admin access code is held by Netlify and compared server-side.
- Netlify and Modal communicate using a separate auditor shared secret.
- Repository mappings, HF credentials, and infrastructure URLs stay server-side.
- Custom prompts are limited to five entries and 4,000 characters each.
- Request bodies reject unknown fields.
- Job IDs and file paths are server-generated; user input never becomes a path component.
- The coordinator estimates the worst-case six-worker charge using the highest hourly rate
  in the requested fallback list, six 60-minute worker timeouts, and CPU/storage overhead.
  It refuses launch above USD 60 even if a cheaper accelerator is likely.
- The manifest preserves both the preflight estimate and the post-run duration-based cost
  estimate; one never silently replaces the other.

## 12. API surface

The admin-facing Netlify layer exposes:

- `POST /audit/plan` — validate input and return the sealed plan/cost estimate
- `POST /audit/jobs` — confirm and launch an idempotent job
- `GET /audit/jobs/:id` — status, partitions, timing, and artifact availability
- `POST /audit/jobs/:id/retry` — retry only failed/missing partitions
- `POST /audit/jobs/:id/cancel` — cancel remaining work
- `GET /audit/jobs/:id/report` — completed or visibly partial PDF
- `GET /audit/jobs/:id/bundle` — complete ZIP

Exact public paths may follow Netlify Function naming constraints during implementation, but
the logical operations and authorization boundaries must remain unchanged.

## 13. Verification strategy

Laptop tests use injected fake generators and no live GPU:

- plan matrix covers every fixed cell exactly once
- exactly six unique, disjoint partitions exist
- A/B/C each receive two partitions
- sample IDs and seeds are deterministic
- fixed and custom prompt sets remain separated
- authorization and unknown fields fail closed
- idempotency cannot double-launch
- USD 60 ceiling blocks over-budget plans
- accelerator preference and actual selection are recorded separately
- coordinator rejects payloads whose partition identity or rows fall outside the sealed plan
- interrupted jobs resume only missing partitions
- duplicate/corrupt/out-of-plan rows block full assembly
- derived scores are deterministic and versioned
- partial reports are visibly stamped
- dataset/final manifest hashes and ZIP membership are correct without circular references
- PDF contains the required sections and provenance fields
- path traversal and repository-name leakage are rejected

A final paid smoke test uses the smallest fixed subset on all six workers, confirms separate
audit/chat Modal apps, downloads the bundle, and verifies no container remains billable past
its configured scale-down window.

## 14. Deliberate omissions from v1

- arbitrary external model IDs
- anonymous/public launches
- Langfuse or another observability account
- hidden-state or activation capture
- token-level log-probability archives for generated responses
- elaborate PDF layout or interactive report editing
- automatic publication of reports
- claims that exploratory custom prompts are confirmatory

These omissions keep v1 bounded. The response bundle is intentionally rich enough to add
Langfuse, new evaluators, new charts, or a redesigned renderer later without another GPU run.

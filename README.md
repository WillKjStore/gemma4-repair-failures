# Reproduction materials
## Where Coding Agents Stop: Tracing Repair Failures in Gemma 4 Agents

This package supports the study's comparisons of repair outcomes, input limits, graph source exposure and recovery after rejected edits. It contains sanitized results and research code.

## Recompute the results

Extract all files into one directory. With Python 3.10 or later, run:

    python recompute_results.py

The command verifies SHA-256 checksums for every supplied file, regenerates outcome totals and paired comparisons, and checks them against consolidated_results.csv and recompute_output.json. Python's standard library is sufficient. The printed JSON includes repeated-seed results and screening counts. The completed exploratory 12B outcomes are included.

## Experiments and settings

The main comparison uses 42 matched tasks and Gemma 4 E2B, E4B, 12B and 31B. Separate arms remove graph access, enlarge the context window, request notes, repeat seeds, or use tasks qualified after environment repairs. review_methods_evidence.json records generation settings, limits, model manifests and configuration digests. The baseline window is 32,768 tokens with 16,384 reserved for output. The larger-window arm uses 65,536 tokens.

agent.yaml and agent_e2b.yaml contain the recorded prompts, tool lists and generation settings. agent_g0.yaml defines the graph-off configuration. The supplied runner and replay scripts are research source files. Running new agent experiments requires the pinned swegemma harness, model weights, vLLM, Google ADK, the competition task and graph data, and the corresponding execution environment. The cluster study path in source scripts has been replaced with /path/to/study. Configure that path before use. These source files are not a self-contained environment installer. Configuration digests identify original run artifacts; sanitized script checksums describe this release.

## Data fields

task_level_outcomes.csv contains pseudonymous task_key values, repository, model, arm, cohort, seed, execution_mode, context_tokens, attempt_status, outcome and configuration digests. Keys allow pairing across models and seeds. An attempt marked pass passed the required verifier tests. test_failure means required tests failed. context_limit_before_patch means the server rejected input before a usable patch. incomplete_verification means verification could not establish pass or fail. In the exploratory 12B arm, incomplete identifies inference failures.

consolidated_results.csv groups those outcomes by experiment and seed. screening_ledger.csv records qualification and exclusions. g0_paired_outcomes.csv supports graph-on/off comparisons. reviewed_seen_edited.csv contains anonymized trace-derived labels; unknown remains distinct from observed absence.

graph_use_counts.json records graph calls and paired uptake. issue_only_replay_review_scores.json keeps reviewed and provisional targets separate. edit_recovery_and_context_retry.json distinguishes any same-run failure/success from a success occurring after a failure. The audit scripts document those definitions. effective_input_cap_audit_20261111.json contains the input-ceiling and larger-window checks. Descriptive edit intervals do not adjust for repeated tasks.

## Trace evidence and access

Anonymized trace-derived labels and aggregate audits are supplied. Original task statements, code-bearing tool replies, submitted patches and complete logs remain on Quartz. This release contains no fabricated or reconstructed trace samples. Review labels retain uncertainty and the human integrity check is identified as an attestation.

## Integrity

SHA256SUMS.txt covers all payload files. The recompute command verifies the manifest before analysis. This package contains no correspondence, private reviewer answers or operational handoff notes.

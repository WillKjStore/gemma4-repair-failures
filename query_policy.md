# Issue-only graph replay v1

This policy was written before any Quartz task, graph, or outcome was inspected. The code is the operational definition. The project runbook calls for a second-person review and a five-task pilot before the held-out replay.

1. Read `problem_statement` only. Queue unique backticked identifiers in order of first mention, then dotted, underscored, and mixed-case identifiers. Insert the first eight unique non-stopword tokens as a lexical query after at most two identifier queries.
2. Start with `search_similar_code(k=10)`. Queue `get_code_neighbors(max_neighbors=50)` for its top hit and `get_code_subgraph` for up to its three top distinct hits. A neighbor reply can queue one subgraph expansion using its resolved node and its first two distinct neighbors in lexical order. Use a fresh issue query at calls 1, 4, 7 and 10 when one is available, with queued follow-ups in FIFO order otherwise. Skip identical calls. An error reply consumes one call. An exhausted queue can end replay early.
3. Preserve actual tool returns, including errors and truncation, in private storage. Score prefixes of length 3, 5 and 10 after replay. Assign the strongest level supported by the visible payload. `symbol` requires an exact mapped node identity, `location` requires a correct file and range in the visible payload, and `source` requires source intersecting a reviewed target range. `none` requires an executable replay with no evidence at that budget. Unavailable tools, missing graphs, and uncertain target identity have separate statuses. Locations recovered only from graph metadata count toward coverage, not tool accessibility.
4. For multi-target tasks, record per-target levels and aggregate both any-target and all-target coverage. An insertion-only or ambiguous target stays pending review. In E4 distinguish source shown by graph tools from source shown by filesystem tools. A tool request, symbol-only result, or correct file path alone does not establish source observation.

## Predictions carried from the team plan

- E4B mostly uses lookup and one-step expansions. The plan's proposed counterevidence is chained expansion in more than one third of E4 runs.
- Some E4B runs that see target code fail to edit it. The plan's proposed counterevidence is under 10% seen without an edit.
- Missed unstated requirements may be the largest failure category among unsuccessful target editors. Adjudicate these cases manually against test behavior and issue scope.
- Among runs with target observation followed by compaction, target evidence may be absent from the summary and never re-read. Assess only when summary content and event order are recoverable.
- Graph-tool uptake may differ between E4B and E2B on matched tasks. Report the paired difference and uncertainty.

The team plan marks the first two thresholds as proposals pending source review. Preserve any revisions with dates before the pilot.

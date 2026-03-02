# Development Workflow — Agnetic DS

How every new feature is built. Follow this process without exception.

---

## The 9 Stages

### Stage 1 — Feature Ideation (Interactive)
**Who:** You + the AI, conversationally
**Output:** `docs/features/<feature-name>/00_feature_brief.md`

The brief must answer:
- What is this feature?
- What problem does it solve?
- Where does it sit in the pipeline?
- What is the acceptance criteria (how do we know it's done)?

---

### Stage 2 — Brainstormer (Subagent)
**Who:** Dedicated subagent (own context window)
**Input:** `00_feature_brief.md` + `docs/system/architecture.md`
**Output:** `docs/features/<feature-name>/01_brainstorm.md`

Deep-dive on:
- How this feature interacts with existing agents and state
- What new state fields are needed
- Edge cases and domain-specific considerations
- What the human-in-the-loop interaction looks like (if any)

---

### Stage 3 — Architect (Subagent)
**Who:** Dedicated subagent (own context window)
**Input:** `01_brainstorm.md` + `docs/system/agent_registry.md` + `docs/system/pipeline_stages.md`
**Output:** `docs/features/<feature-name>/02_architecture.md`

Produces:
- High-level design (which components change, which are new)
- New/modified AgentState fields
- Graph changes (new nodes, edges, conditionals)
- Interrupt design (if applicable)
- Data flow diagram

---

### Stage 4 — Auditor (Subagent)
**Who:** Dedicated subagent (own context window)
**Input:** `02_architecture.md` + all `docs/system/` files + relevant existing agent files
**Output:** `docs/features/<feature-name>/03_audit.md`

Checks for:
- Conflicts with existing agents or state fields
- State fields being overwritten unexpectedly
- Executor assumptions being broken (e.g. always reloads from source CSV)
- Security risks (exec() scope, file paths, etc.)
- Missing interrupt points
- Conclusion: approved / approved with changes / rejected

---

### Stage 5 — Code Planner (Subagent)
**Who:** Dedicated subagent (own context window)
**Input:** `03_audit.md` + `02_architecture.md`
**Output:** `docs/features/<feature-name>/04_code_plan.md`

Produces a precise implementation plan:
- Files to create (with purpose)
- Files to modify (with exact functions to change)
- Files to delete
- New AgentState fields to add
- Graph modifications
- Acceptance criteria checklist (used by Tester and Committer)

---

### Stage 6 — Code Writer (Subagent)
**Who:** Dedicated subagent (own context window)
**Input:** `04_code_plan.md` + contents of files to be modified
**Output:** Implemented code changes

Follows the code plan exactly. Does not make architectural decisions — raises a flag if the plan is ambiguous.

---

### Stage 7 — Tester (Subagent)
**Who:** Dedicated subagent (own context window)
**Input:** `04_code_plan.md` (acceptance criteria) + implemented code
**Output:** `docs/features/<feature-name>/05_test_report.md`

- Whitebox review of every new function/agent
- Runs tests against the acceptance checklist from the code plan
- Reports: pass / fail / partial per criterion
- Flags any issues back — Code Writer must fix before proceeding

---

### Stage 8 — Documenter (Subagent)
**Who:** Dedicated subagent (own context window)
**Input:** All prior stage outputs (`00` through `05`)
**Output:** `docs/features/<feature-name>/06_transcript.md`

Produces a **compressed digest** — the single file that allows any future LLM session to understand this feature with minimal context:
- What was built and why
- Key decisions made
- What changed in the system (state fields, graph, agents)
- Known limitations or follow-up work
- Max ~300 lines

Also updates:
- `docs/system/agent_registry.md` (new/modified agents)
- `docs/system/pipeline_stages.md` (updated stage statuses)
- `docs/system/decisions.md` (any new ADRs)

---

### Stage 9 — Committer (Subagent)
**Who:** Dedicated subagent (own context window)
**Input:** `04_code_plan.md` acceptance checklist + `05_test_report.md` + git status
**Output:** Git commit on the feature branch

- Verifies every acceptance criterion is ticked (test report = all pass)
- Verifies docs are updated (agent_registry, pipeline_stages, decisions)
- Stages only relevant files (no __pycache__, no .env)
- Commits with a descriptive message
- Does NOT push — human reviews the PR

---

## Feature Folder Structure

```
docs/features/<feature-name>/
  00_feature_brief.md
  01_brainstorm.md
  02_architecture.md
  03_audit.md
  04_code_plan.md
  05_test_report.md
  06_transcript.md        ← THE DIGEST — load this in future sessions
```

---

## Branching Convention

```
main                          ← stable only
  └── fix/<description>       ← bug fixes, testing
  └── feature/<feature-name>  ← one branch per feature
```

Feature branches are created from `main` only after all blocking fixes are merged.

---

## Context Efficiency Rules

- Each subagent (stages 2–9) runs in **its own context window** via the Agent tool
- Only **digests** (`06_transcript.md`) are passed between sessions — never raw conversation logs
- A new session should only need to read: `MEMORY.md` + `docs/system/architecture.md` + the relevant `06_transcript.md` to be fully oriented
- Raw stage files (`01_brainstorm.md` etc.) are reference material, not required reading every session

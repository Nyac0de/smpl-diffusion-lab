# Portable State Schema

Keep the canonical handoff at `docs/mentoring-state.md`. Keep it concise enough to read at every resume and independent of any one machine or chat thread.

Use these sections:

```markdown
# Mentoring State

## Sync

- Updated: YYYY-MM-DD
- Branch: branch-name
- Based on commit: short-hash
- Remote: canonical repository URL

## Objective

One current research objective and why it is next.

## Verified Progress

Only abilities supported by code, tests, explanations, modifications, or delayed recall.

## Current Exercise

- Status: not-started | awaiting-attempt | debugging | verification | complete
- Help level: L0-L5
- Target files: repository-relative paths
- Specification: exact current task
- Learner's latest reasoning: concise summary or `none yet`
- Blocking issue: exact blocker or `none`

## Skill Evidence

- Demonstrated: concrete evidence
- Developing: concrete evidence
- Recurring errors: categories and examples

## Retrieval Queue

Items to redo after a delay, with due date or next-session condition.

## Next Action

One unambiguous action that a fresh mentor can request immediately.

## Environment Notes

Only portable prerequisites and named environment profiles. Never store credentials, private paths, or machine-specific secrets.
```

Set `Based on commit` to the commit that was checked when preparing the handoff. It need not point to the later commit containing the state update.

Do not copy the full learning log into the state file. The log is append-only evidence; the state is the mutable resume point. Validate both against Git history, source, and tests.

If `docs/mentoring-state.md` is missing, initialize it from repository evidence and label inferences. Do not infer learner mastery from implementation alone.

# Contributing to datatrove

Thanks for your interest in datatrove. Please read this before you open an issue or a pull request.

## How we review

datatrove is stable and actively maintained. We review pull requests in batches, roughly every 4–6 weeks ahead of
releases, rather than continuously. We triage new issues more often, so approval to start work usually comes sooner
than a PR review.

There is no need to ping a PR between review rounds. If something is urgent (a security problem, data loss, or a
broken release), say so in the issue title and we will look at it sooner.

## What we fix, and what we don't

datatrove is a tool for processing web-scale data with pipelines that people actually run. It does not try to
handle every possible input or configuration. We would rather keep the code small and readable than guard against
cases nobody hits.

We are most interested in:

- bugs you hit while running a real pipeline: wrong output, a crash, lost or duplicated documents;
- problems at scale: speed, memory, resuming, and sharding across many tasks;
- differences from a reference implementation (for example C4 or Gopher) that change real results, with evidence
  that the difference is real.

We usually decline:

- edge cases found by reading the code or by an automated analysis that no one has hit in practice, such as unusual
  argument combinations, or empty or malformed inputs that a real pipeline would not produce;
- extra validation, defensive checks or tests for situations that cannot occur in normal use;
- changes that make a filter keep or drop different documents, unless you show why the current behaviour is wrong
  for real data. Filter output defines published datasets, so changing it is not a free fix. If your change affects
  filter output, give before/after examples on real data in the issue, and get explicit approval for that behaviour
  change.

If you are not sure, open an issue and ask. A short "is this worth fixing?" is fine.

## Open an issue and wait for approval before opening a PR

This applies to external contributors. Hugging Face org members, and PRs that only fix docs typos or broken links,
are exempt.

1. **Open an issue first**, or find an existing one. For a bug, tell us how you hit it: the pipeline, the data, and
   what went wrong. If you found it by reading the code or with an automated analysis and have not hit it in real
   use, say so.
2. **Wait for a maintainer to approve it.** A maintainer approves an issue by commenting
   `/datatrove-triage approved @your-username`, which also names who should open the PR. Opening an issue, or
   getting general feedback on it, is not approval. Changes to defaults, executors or the `Document` model are
   discussed on the issue before any code is written.
3. **Open the PR and link the approved issue** (`Closes #123`).

Pull requests that do not link an issue approved for their author will be closed. If that happens, get the issue
approved, then reopen the same PR rather than opening a new one.

If an issue already has an open PR, or another contributor has been approved to work on it, comment there (with a
test case or a review, for example) instead of opening another PR. Open a replacement PR only if a maintainer asks
you to.

Issues must be opened by a person. You can use an agent to investigate or draft an issue, but read it, check it, and
post it yourself. Issues opened by agents on their own, or filed in bulk across projects, will be closed.

## Using AI tools

You are welcome to use AI tools. You are accountable for the result: you have read the diff, run the tests, can
explain why the change is correct, and will reply to review comments. In the PR, say why the change is needed,
whether you are affected by it, and how you checked it. Keep the diff as small as the fix allows.

If you use a coding agent, run it from the repository root so it picks up [AGENTS.md](AGENTS.md).

## Development setup

```bash
git clone git@github.com:huggingface/datatrove.git && cd datatrove
uv sync --extra dev
pre-commit install
```

Before you push:

```bash
make quality   # ruff check + format on changed files
make test      # or: pytest -sv ./tests/
```

See [AGENTS.md](AGENTS.md) for code style and the conventions for new pipeline blocks.

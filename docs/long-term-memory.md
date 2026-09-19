# Long-term video memory

MemPalace is the semantic long-term store. The local SQLite catalog preserves
complete provenance, stable topic IDs, user topic overrides, and a retryable
outbox. It does not replace the existing per-video Chroma index.

## Local setup

Install `.[dev,memory]` in a working Python environment. Enable `mcp_stdio`
in your local `.env` and choose a palace path such as `data/memory/palace`.
The example configuration keeps this integration opt-in.
Restart an already-running application after changing `.env` because settings
are cached in-process.

`MEMPALACE_EMBEDDING_MODEL=openai-compat` reuses the project's configured
embedding endpoint/model/key, including its Chinese-language support. The key
is passed through the child process environment, not a command-line argument.
MemPalace is pinned at 3.9.0. Do not change the embedding model of a populated
palace without rebuilding its index.

## Data flow

1. Video processing retains all its original artifacts and per-video index.
2. Supported storyline claims and speech/visual evidence throughout the video
   become separate catalog records. Mock videos are excluded. The configured LLM
   assigns reusable topic paths using titles and supported storylines, preferring
   existing labels. Classification is cached by content fingerprint. Failure falls
   back to filtered keywords/storyline topics. Normalized labels get stable IDs;
   user overrides always take precedence. This is not a curated ontology.
3. Writes are idempotent. Changed source records become inactive; their obsolete
   MemPalace results are rejected. Failed writes remain pending for later sync.
4. Questions search the current video as before. The [semantic memory gate](memory-gate.md)
   first decides whether historical knowledge is needed. Only `search_history` triggers
   a separate MemPalace search, using the gate's independent query and topics,
   first in related videos' primary topic rooms and then
   across the video wing with a larger candidate pool. Exact video titles are only
   used for very short questions, avoiding self-video domination. It excludes the current video and applies
   a similarity threshold, topic/value/pin bonuses, duplicate and per-video limits.
5. Every selected historical record is checked against the local original
   segment, text/caption and time range. Missing or changed evidence is rejected.
   Historical context is capped at 43% of current evidence characters and the
   configured global budget. With short current evidence, no history may fit.
6. The model returns current-video text, historical supplements and general
   knowledge separately. Historical citation IDs and evidence indexes are checked;
   the server attaches the verified original quote, avoiding model copy errors
   before display. Historical/background text does not affect current-video
   agreement scoring or get re-promoted as a current-video fact.
7. Every completed user/assistant exchange is additionally filed through
   MemPalace's native `mempalace_add_drawer` API as a verbatim Drawer in
   `video_knowledge_agent / conversation:<video_id>`. Low-confidence answers
   are retained as conversation history but are not promoted to reusable
   `derived_insight` records. This is separate from the provenance-checked
   historical-video path above.

## Operations

Run these with the project's working Python interpreter:

```powershell
.\.venv-memory\Scripts\python.exe -m app.memory.manage status
.\.venv-memory\Scripts\python.exe -m app.memory.manage list
.\.venv-memory\Scripts\python.exe -m app.memory.manage backfill
.\.venv-memory\Scripts\python.exe -m app.memory.manage sync --limit 400
.\.venv-memory\Scripts\python.exe -m app.memory.manage topics VIDEO_ID
.\.venv-memory\Scripts\python.exe -m app.memory.manage topics VIDEO_ID --set "Retrieval" "Evaluation"
.\.venv-memory\Scripts\python.exe -m app.memory.manage pin vka_MEMORY_ID
.\.venv-memory\Scripts\python.exe -m app.memory.manage search "question" --video-id VIDEO_ID
```

Topic overrides survive reprocessing. After changing topics, run sync to file
the revised records. Backfill only reads existing video artifacts; it does not
redownload video, rerun ASR or regenerate summaries. `status` distinguishes the
real backend response from locally active/pending counts. `sync` returns nonzero
on a write failure. A regular pipeline persist stage also drains up to 100 pending
items; remaining items can be synchronized with the command above.

Setting `MEMPALACE_PROVIDER=noop` disables the optional layer without deleting
anything. Failed reads return an empty historical context, preserving the
current-video answer path. The trace records retrieval status and selected items.

## Boundaries and validation

- Source linkage and exact quotations are checked in code. Whether a generated
  explanation logically follows a quote still depends on the model; this is not
  a formal entailment guarantee.
- Opposing claims remain separate. Similarity alone is not proof of consensus or
  disagreement. Legacy cluster summaries and unverified conversation insights
  are stored separately but excluded from factual historical retrieval.
- `created_at` is an indexing/source metadata time, not a guaranteed publication
  or factual validity date. No automatic age-based truth judgment is made.
- Duplicate claims retain their separate sources in storage; retrieval limits
  repetition. Pinning changes relevance ranking, not factual verification status.
- No automatic contradiction adjudication is claimed.
  Users can explicitly assign shared topic labels to related videos.

Tests cover full-video coverage, retries, idempotence, source revisions, self-video
exclusion, unrelated histories, opposing claims, small budgets, outages and forged
citations. Run regression tests with `MEMPALACE_PROVIDER=noop` in the test process
to avoid writing fixture conversations to the production palace.

Protocol reference: https://mempalaceofficial.com/reference/mcp-tools.html

## Workspace verification (2026-09-05)

- Full regression suite: 171 passed. Tests ran with real memory disabled in the
  test process. Native Chroma tests required execution outside the desktop sandbox.
- MemPalace 3.9.0 / Chroma 1.5.9: SQLite integrity check passed; 297 active catalog
  records synchronized, zero pending. Drawer count is larger because MemPalace
  chunks records and retains inactive prior versions; it is not a knowledge count.
- Real model QA on `BV1EDhA69ExJ` retrieved a historical supplement from
  `BV1T4hg6WEDt`, segment beginning at 118.88 seconds, plus a separately labeled
  general-knowledge explanation. See `data/memory/smoke_qa.json`.
- One classification request timed out and used the keyword fallback. Topic
  overrides and later backfill can refine that categorization.
- No complete redownload/ASR/VLM reprocessing was performed during this change.
  Existing automated pipeline tests passed; this is not a guarantee for every
  external provider or media source. The separate environment also includes
  faster-whisper for the original local ASR workflow.

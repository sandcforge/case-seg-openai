# Support Message ID Migration — Design

**Date:** 2026-04-29
**Status:** Approved (revised after AI review on 2026-04-29; supersedes initial 2026-04-29 version on the same branch)

## Problem

The `support_message` source table was migrated off Sendbird:

- Old PK `message_id` (numeric varchar) is now **nullable**; 100% NULL on rows after ~2026-04-01.
- New PK `id` (21-char nanoid varchar, **NOT NULL**).
- Old grouping key `channel_url` is decreasingly populated; new key `ps_channel_id` is 100% present on post-migration rows. Within a 90-day window: 35,013 rows have `ps_channel_id IS NULL` but `channel_url` set; 0 rows have both NULL.
- The migration was introduced in early 2026; `message_id` started going NULL on 2026-02-23, ramped to ~99% NULL by 2026-03-30. The bug was discovered in 2026-04.

Existing pipeline behavior under the new schema:

1. The dedup CTE in `Utils.get_channels_to_process` joins on `CAST(sm.message_id AS INT64)`. NULL message_id rows fail the join and are reported as "unprocessed".
2. `Utils.preprocess_dataframe` casts `Message ID` to int and drops NaN rows — so NULL-message_id rows are **silently filtered out** before reaching the LLM.
3. The pipeline's last successful run produced cases up to 2026-03-31, then stopped.

### Damage assessment (90-day window)

- **34,881 orphan messages** (Feb-Mar 2026, ps_channel_id present but message_id NULL) were silently dropped during preprocessing and never assigned to any case.
- **~11,908 cases** with `start_time >= 2026-02-01` are likely produced from incomplete chunks. Their boundaries, `start_time` / `end_time` / `usr_msg_num` / `first_res_time` are all unreliable.
- **35,013 rows** have NULL `ps_channel_id`; 26,296 of those are referenced by dirty cases (will be unclaimed once dirty cases are deleted) and 8,717 are in clean pre-Feb cases (still covered after backfill). The 26,296 must be re-claimed by the new code — they are **not** acceptable as silent loss.

### Channel identity coverage (verified on prod)

| Slice | Count |
|---|---|
| Rows in 90-day window with `ps_channel_id` set | 164,483 |
| Rows with `ps_channel_id` NULL but `channel_url` set | 35,013 |
| Rows with both NULL | **0** |
| Channels with at least one row carrying both `channel_url` and `ps_channel_id` (i.e. mappable) | 6,110 / 7,418 channel_urls in window |
| Channels at "split risk" (have both old-only and new-only rows but no mapping row) | **0** |
| Channels that are pure-old (only have channel_url) | 1,308 |

## Approved decisions

| ID | Decision | Confirmed |
|----|----------|-----------|
| A1 | Use `support_message.id` (21-char string nanoid, table PK) as the new message identity throughout the pipeline. | yes |
| **B1′** (revised) | Channel identity is `effective_channel_id = COALESCE(sm.ps_channel_id, mapping.canonical_pscid, sm.channel_url)`. The mapping comes from a CTE that pairs every channel_url to any one of its ps_channel_id values from rows that have both. This guarantees 0 rows lost (since both-NULL never occurs in window) while keeping cross-migration channels intact. The 1,308 pure-old channels stay keyed by channel_url — that's intentional and harmless because they have no future activity. | revised, yes |
| C1 | On `support_message_cases`, **add** `id_list ARRAY<STRING>` and **`channel_id STRING`** columns. **Keep** `message_id_list` and `channel_url` as deprecated read-only history (NULL on new writes). No ALTER COLUMN, no separate v2 table. | yes (column renamed from `ps_channel_id` to `channel_id` because it can hold a channel_url for pure-old channels) |
| Q1 | LLM prompt shows the full 21-char `id` (table column widened); LLM emits string IDs into the JSON key **`id_list`** (renamed from `message_id_list` for consistency with the BQ column). | yes (renamed) |
| Q2 | Distance metric in `repair_case_segment_output` switches from `abs(int_id - int_id)` to channel-row-index distance via a new per-channel `msg_ch_idx` column. | yes |
| **Q3′** (revised) | `format_channel_for_display`: if the value starts with `sendbird_`, take the trailing hash; otherwise (it's a 21-char nanoid that may itself contain `_`) return the first 8 chars for a tighter display. | revised because 28% of ps_channel_id values contain `_`, breaking the original "no underscore = nanoid" heuristic. |
| Q4 | Dedup query joins on `id` only — no channel scope needed because `id` is globally unique. | yes |
| **Cleanup** | **Delete** all cases with `start_time >= 2026-02-01` (~11,908). Backfill the remaining ~33,028 pre-Feb cases. Reprocess affected channels via the new code. `case_id` is not externally referenced; downstream impact is zero. | yes |
| **Naming** | DataFrame primary message column is **`Message ID`** (name retained, values switch from int to str). The new DataFrame channel column is `Channel ID`. The Pydantic / JSON / BQ field is `id_list`. The new BQ channel column is `channel_id`. | yes (locked) |

## Architecture

### Effective channel id

```
effective_channel_id = COALESCE(
  sm.ps_channel_id,                         -- Post-migration rows: always set
  channel_mapping[sm.channel_url],          -- Cross-migration rows: lookup canonical pscid
  sm.channel_url                            -- Pure-old dead channels: keep channel_url
)
```

Where `channel_mapping` is built from `support_message` rows that carry both:

```sql
WITH channel_mapping AS (
  SELECT channel_url,
         ARRAY_AGG(DISTINCT ps_channel_id IGNORE NULLS LIMIT 1)[SAFE_OFFSET(0)] AS canonical_pscid
  FROM `plantstory.public.support_message`
  WHERE channel_url IS NOT NULL AND ps_channel_id IS NOT NULL
  GROUP BY channel_url
)
```

### Data flow (after migration)

```
support_message (Postgres → BQ mirror via Datastream)
    + channel_mapping CTE
    SELECT id, effective_channel_id, channel_url, ps_channel_id,
           created_time, sender_id, message, type, file_url, raw, deleted
    WHERE deleted = false
      AND created_time >= 90 days ago
      AND NOT EXISTS (... x IN UNNEST(c.id_list) WHERE x = sm.id)
    ORDER BY effective_channel_id, created_time, id
        ↓
Channel-level grouping by effective_channel_id  (Channel ID in DataFrame)
        ↓
preprocess_dataframe
    - coerce Message ID to str (no int cast, no NaN drop)
    - add msg_ch_idx (per-channel 0..N-1 row index)
        ↓
Channel.build_cases_via_llm
        ↓
support_message_cases:
    case_id, channel_id (= effective_channel_id), id_list[STRING], summary, status, ...
    (channel_url, message_id_list left NULL)
```

### Output table schema (after additive migration)

| Column | Type | Mode | Role |
|--------|------|------|------|
| case_id | STRING | | unchanged |
| **channel_id** | **STRING** | new | new channel key (always written) |
| **id_list** | **STRING** | REPEATED, new | new message-list key (always written) |
| channel_url | STRING | | deprecated, NULL on new writes, retained for historical reads |
| message_id_list | INT64 | REPEATED | deprecated, NULL on new writes, retained for historical reads |
| summary, status, segmentation_confidence, main_category, sub_category, classification_confidence, first_res_time, first_contact_resolution, usr_msg_num, start_time, end_time, meta_data | unchanged | | |

## Operational sequence (deploy-time)

1. **ALTER `support_message_cases`** to add `id_list ARRAY<STRING>` and `channel_id STRING` (nullable, additive).
2. **Snapshot** dirty cases to `support_message_cases_dirty_snapshot_2026_04_29` for rollback insurance.
3. **DELETE** cases with `start_time >= 2026-02-01` (~11,908).
4. **Backfill** the remaining ~33,028 clean pre-Feb cases. The backfill must:
   - Resolve `id_list` from `message_id_list` via JOIN on `support_message.message_id` (BQ-side; runs while the legacy column is still populated).
   - **Independently** resolve `channel_id` from `channel_url` via the channel_mapping CTE (decoupled from the id_list JOIN — see Architecture section).
   - Use `ARRAY_AGG(... IGNORE NULLS LIMIT 1)[SAFE_OFFSET(0)]` rather than `ANY_VALUE(... IGNORE NULLS)`, which BigQuery rejects.
5. **Deploy new code** (dry-run on one channel first, then production):
   - Source SELECT computes `effective_channel_id` via the CTE above.
   - Dedup `WHERE NOT EXISTS (... x = sm.id)`.
   - Writes `id_list` + `channel_id`; leaves `message_id_list` + `channel_url` NULL.
6. **Verify** the first few production runs and the 3,086 affected channels.
7. **Drop source columns** (DBA op): once the new code has been live for a stability window, `support_message.message_id` and `support_message.channel_url` can be dropped — the runtime code only references `id`, `ps_channel_id`, `channel_url` (the last only as a COALESCE fallback that can be removed once the 90-day window slides past 2026-03-27, around 2026-06-25).

## Code changes (Python)

| File | Change |
|------|--------|
| `src/utils.py::get_channels_to_process` | Rewrite SQL: add `channel_mapping` CTE; SELECT computes `effective_channel_id`; dedup `WHERE NOT EXISTS (... x = sm.id)`; sort `effective_channel_id, created_time, id`. **No** `WHERE ps_channel_id IS NOT NULL` filter — the COALESCE removes the need. The `channel_urls` parameter accepts effective_channel_id values. |
| `src/utils.py::preprocess_dataframe` | Map BQ snake_case → DataFrame Title Case. **Keep** column name `Message ID` (values become str — drop int cast and NaN drop). Map `effective_channel_id` → `Channel ID`. Add `msg_ch_idx` per-channel row index. |
| `src/utils.py::format_channel_for_display` | If starts with `sendbird_`, return trailing hash. Otherwise return the first 8 chars (the `_` heuristic is unsafe — 28% of ps_channel_id values contain `_`). |
| `src/utils.py::format_messages_for_prompt2` | Widen Message ID column to 22 chars. |
| `src/case.py` | Rename `Case.message_id_list` → `Case.id_list: List[str]`. Add `Case.channel_id: str`. Keep `Case.channel_url: str` (set to "" on new writes). Rename Pydantic `CaseSegmentationLLMRes.message_id_list` → `id_list: List[str]`. Update `CaseReviewInput.overlap_msg_ids: List[str]` and `ReviewAction.new_msg_assignment: Dict[str, int]`. Update `to_bigquery_row` to write `id_list` + `channel_id`; legacy `message_id_list` + `channel_url` NULL. |
| `src/channel.py` | All `case_dict['message_id_list']` → `case_dict['id_list']`. All `int` distance calls replaced with `msg_ch_idx`-based distance. `Channel.__init__` signature gains `channel_id: str` (replacing or alongside `channel_url`). All `case.message_id_list` references → `case.id_list`. |
| `src/session.py` | `df['Channel URL']` group/sort uses → `df['Channel ID']`. `Channel(...)` instantiations adjusted to pass `channel_id`. `case.message_id_list` → `case.id_list` in serialization. |
| `src/app_bigquery.py` | `df_clean['Channel URL']` grouping → `df_clean['Channel ID']`. Pass `channel_id` to Channel. |
| `src/vision_processor.py` | `image_message_id: str` (was int). Use `.astype(str)` masking. |
| `src/prompts/segmentation_prompt.md` | Replace "10-digit database ID" with "21-char nanoid"; example `4499509692` → real 21-char nanoid; rename JSON key `message_id_list` → `id_list`; widen the schematic table layout. |

## Testing

- One small `unit_test/` script per code change (smoke test, run with `python unit_test/test_*.py`):
  - `test_case_schema.py` — verifies Pydantic/dataclass renames.
  - `test_preprocess_string_id.py` — verifies preprocess produces string IDs and `msg_ch_idx`.
  - `test_format_messages.py` — verifies formatter shows full 21-char ID.
  - `test_repair_string_ids.py` — verifies `repair_case_segment_output` handles string IDs and msg_ch_idx-based proximity.
  - `test_vision_string_id.py` — verifies vision lookup with string IDs.
  - `test_get_channels_sql.py` — runs the new SQL against prod for one known active channel.
- End-to-end dry run on one channel before flipping prod writes on.

## Risks

| Risk | Mitigation |
|------|------------|
| LLM costs for reprocessing ~11,908 deleted cases | One-time. Run dry-run first to estimate. |
| Backfill JOIN slow on 33k cases × 273k legacy message references | BQ SQL only; estimated < 1 minute. |
| Pre-Feb cases reference message_ids no longer in `support_message` (deleted source rows) | `id_list` will be shorter than `message_id_list`. Acceptable. |
| `support_message.message_id` dropped before backfill runs | DBA must wait for verification step 6 before dropping. |
| Reprocessed cases for affected channels differ in boundaries from the deleted dirty cases | Expected — dirty cases were wrong; new cases are the truth. |
| 1,308 pure-old channels have `channel_id` = the old `channel_url` (mixed semantics in one column) | Acceptable: those channels have no future activity, so the column's mixed semantics never propagate to new cases. The 90-day rolling window will phase them out by ~2026-06-25. |
| A channel with both old (NULL pscid) and new (pscid) messages but no "mapping row" splits into two virtual channels | **0 such channels** in current 90-day window — verified. Future risk: ≤ a handful per quarter; acceptable per user spec ("个位数不精确 OK"). |

## Out of scope

- Re-running classification or stats on the existing pre-Feb 33k cases. Their `id_list` is backfilled but LLM-produced fields are kept untouched.
- Changing chunk_size / idle_days / trigger semantics.
- Refactoring beyond what's needed for the column/key rename.

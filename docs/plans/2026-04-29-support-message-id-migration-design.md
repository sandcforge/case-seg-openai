# Support Message ID Migration — Design

**Date:** 2026-04-29
**Status:** Approved, pending implementation plan

## Problem

The `support_message` source table was migrated off Sendbird:

- Old PK `message_id` (numeric varchar) is now **nullable**; 100% NULL on rows after ~2026-04-01.
- New PK `id` (21-char nanoid varchar, **NOT NULL**).
- Old grouping key `channel_url` is decreasingly populated; new key `ps_channel_id` is 100% present on post-migration rows.
- The migration was introduced in early 2026; `message_id` started going NULL on 2026-02-23, ramped to ~99% NULL by 2026-03-30. The bug was discovered in 2026-04.

Existing pipeline behavior under the new schema:

1. The dedup CTE in `Utils.get_channels_to_process` joins on `CAST(sm.message_id AS INT64)`. NULL message_id rows fail the join and are reported as "unprocessed".
2. `Utils.preprocess_dataframe` casts `Message ID` to int and drops NaN rows — so NULL-message_id rows are **silently filtered out** before reaching the LLM.
3. The pipeline's last successful run produced cases up to 2026-03-31, then stopped.

### Damage assessment

- **34,881 orphan messages** (Feb-Mar 2026, ps_channel_id present but message_id NULL) were silently dropped during preprocessing and never assigned to any case.
- **~11,908 cases** with `start_time >= 2026-02-01` are likely produced from incomplete chunks (LLM saw only the rows that survived the silent drop). Their boundaries, `start_time` / `end_time` / `usr_msg_num` / `first_res_time` are all unreliable.
- The 35,809 rows in the 90-day window with NULL `ps_channel_id` are 100% pre-migration old rows already covered by clean pre-Feb cases — zero loss if we filter them out.

## Approved decisions

Each row records a decision the user explicitly confirmed.

| ID | Decision | Confirmed |
|----|----------|-----------|
| A1 | Use `support_message.id` (21-char string nanoid, table PK) as the new message identity throughout the pipeline. | yes |
| B1 | Use `support_message.ps_channel_id` as the new channel grouping key. Filter out rows where it is NULL (those are pre-migration messages already covered by old cases). | yes |
| C1 | On `support_message_cases`, **add** `id_list ARRAY<STRING>` and `ps_channel_id STRING` columns. **Keep** `message_id_list` and `channel_url` as deprecated read-only history (NULL on new writes). No ALTER COLUMN, no separate v2 table. | yes |
| Q1 | LLM prompt shows the full 21-char `id` (table column widened); LLM emits string IDs directly into `message_id_list`. No prefix shortening. | yes |
| Q2 | Distance metric in `repair_case_segment_output` switches from `abs(int_id - int_id)` to channel-row-index distance (DataFrame is pre-sorted by `created_time, id`). | yes |
| Q3 | `format_channel_for_display`: when no underscore is present, return the value as-is (the new `ps_channel_id` is already short). | yes |
| Q4 | Dedup query joins on `id` only — no channel scope needed because `id` is globally unique. | yes |
| **Cleanup** | **Delete** all cases with `start_time >= 2026-02-01` (~11,908 cases), then reprocess those channels from scratch via the new code. Pre-Feb cases (~33,028) are kept and backfilled. `case_id` is not externally referenced; downstream impact is zero. | yes |

## Architecture

### Data flow (after migration)

```
support_message (Postgres → BQ mirror via Datastream)
    SELECT id, ps_channel_id, created_time, sender_id, message, type, file_url, raw, deleted
    WHERE deleted = false
      AND ps_channel_id IS NOT NULL
      AND created_time >= 90 days ago
    AND NOT EXISTS (... x IN UNNEST(c.id_list) WHERE x = sm.id)
    ORDER BY ps_channel_id, created_time, id
        ↓
Channel-level grouping by ps_channel_id
        ↓
preprocess_dataframe (no message_id cast, no NaN drop)
        ↓
Channel.build_cases_via_llm
        ↓
support_message_cases:
    case_id, ps_channel_id, id_list[STRING], summary, status, ...
    (channel_url, message_id_list left NULL)
```

### Output table schema (after additive migration)

| Column | Type | Mode | Role |
|--------|------|------|------|
| case_id | STRING | | unchanged |
| **ps_channel_id** | **STRING** | new | new channel key (always written) |
| **id_list** | **STRING** | REPEATED, new | new message-list key (always written) |
| channel_url | STRING | | deprecated, NULL on new writes, retained for historical reads |
| message_id_list | INT64 | REPEATED | deprecated, NULL on new writes, retained for historical reads |
| summary, status, segmentation_confidence, main_category, sub_category, classification_confidence, first_res_time, first_contact_resolution, usr_msg_num, start_time, end_time, meta_data | unchanged | | |

## Operational sequence (deploy-time)

The order matters — particularly that backfill must happen before `message_id` is dropped from source.

1. **ALTER `support_message_cases`** to add `id_list` and `ps_channel_id` columns (nullable, additive).
2. **DELETE** dirty cases:
   ```sql
   DELETE FROM `plantstory.customer_service.support_message_cases`
   WHERE start_time >= TIMESTAMP('2026-02-01');
   ```
3. **Backfill** the remaining ~33,028 clean pre-Feb cases:
   ```sql
   -- For each kept case, resolve message_id_list[INT] → id_list[STRING]
   -- and channel_url → ps_channel_id, via JOIN on plantstory.public.support_message.
   -- BQ-side CTAS or MERGE; idempotent; safe to re-run.
   ```
   Verify: `SELECT COUNT(*) WHERE id_list IS NULL` ≈ 0.
4. **Deploy new code** (dry-run mode first, then production):
   - Reads source by `id`, `ps_channel_id`, `created_time`.
   - Filters `ps_channel_id IS NOT NULL`.
   - Writes `id_list` + `ps_channel_id`; leaves `message_id_list` + `channel_url` as NULL on new rows.
   - Dedup query joins on `id_list × sm.id` only.
5. **Verify** the first few production runs:
   - Cases produced for the previously-affected ~3,086 channels look reasonable.
   - No NULL `id_list` or `ps_channel_id` on new writes.
6. **Drop source columns** (DBA op, post-deploy): `support_message.message_id` and `support_message.channel_url` are no longer referenced by either the runtime code or the backfilled output table. They can be safely dropped at any time after step 5.

## Code changes (Python)

| File | Change |
|------|--------|
| `src/utils.py::get_channels_to_process` | Rewrite SQL: select via `id`/`ps_channel_id`; dedup `WHERE NOT EXISTS (... x = sm.id)`; sort `ps_channel_id, created_time, id`; filter `ps_channel_id IS NOT NULL`. |
| `src/utils.py::preprocess_dataframe` | Rename DataFrame primary column `Message ID` → `ID` (str). Drop the int cast / NaN drop. Use `ps_channel_id` (renamed Title-case `Channel ID`) for grouping. Update column-mapping table accordingly. |
| `src/utils.py::format_channel_for_display` | If no underscore in input, return as-is. |
| `src/utils.py::format_messages_for_prompt2` | Widen ID column to 22 chars for the new IDs. |
| `src/case.py` | `Case.message_id_list: List[str]`; `CaseSegmentationLLMRes.message_id_list: List[str]`; rename to `id_list` for clarity; `to_bigquery_row` writes `id_list` + `ps_channel_id`, leaves `channel_url`/`message_id_list` NULL. |
| `src/channel.py` | All `df['Message ID']` → `df['ID']`. All `int` arithmetic on IDs replaced with channel-row-index distance (build a per-channel sequential index when constructing the channel DataFrame). |
| `src/vision_processor.py` | `image_message_id: str` (was int); same lookup-by-ID logic. |
| `src/prompts/segmentation_prompt.md` | Update example IDs (`4499509692` → real 21-char nanoid sample); rewrite "Message ID: 11 characters (10-digit database ID)" → "Message ID: 21-char string (nanoid)"; widen example table layout. |

## Testing

- `unit_test/test_bigquery.py`: add a fixture that exercises `get_channels_to_process` against a mocked or sample BQ result containing only string IDs and ps_channel_ids.
- Add a small end-to-end test: feed a synthetic 80-row chunk (string IDs) into `Channel.build_cases_via_llm` with the LLM mocked to produce a known `id_list` response → verify case persistence with the new columns.
- Manual dry-run on the first ~10 affected channels post-deploy; eyeball case boundaries.

## Risks

| Risk | Mitigation |
|------|------------|
| LLM costs for reprocessing ~11,908 deleted cases | One-time cost; estimated tens of USD at current model pricing. Run dry-run first to estimate. |
| Backfill JOIN takes longer than expected on BQ | The mirror table already supports the same JOIN pattern; estimated < 1 minute for 44k cases. |
| Some pre-Feb cases reference `message_id` values no longer in `support_message` (deleted rows) | Their `id_list` will be shorter than `message_id_list`. Acceptable — those messages had `Deleted = true` anyway. |
| `support_message.message_id` dropped before backfill runs | Sequencing in section 6 above is the mitigation. Operationally enforce: DBA does not drop until step 5 verifies. |
| Newly produced cases for the 3,086 affected channels differ from the deleted dirty cases (different boundaries / counts) | Expected — the deleted cases were wrong. The new cases are the correct truth. |

## Out of scope

- Re-running classification or stats on the existing pre-Feb 33k cases. Their `id_list` is backfilled but their LLM-produced fields (`summary`, `main_category`, etc.) are kept untouched.
- Changing the chunk_size, idle_days, or trigger logic — semantics are preserved.
- Refactoring `app_bigquery.py` or the Channel/Case class structure beyond what is needed for the column rename.

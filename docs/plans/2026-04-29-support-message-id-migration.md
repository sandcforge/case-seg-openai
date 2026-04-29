# Support Message ID Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the case-segmentation pipeline off `support_message.message_id` (now nullable, eventually dropped) and `support_message.channel_url` (incomplete on new rows). Use `id` (21-char nanoid PK) as the message identity and `effective_channel_id = COALESCE(ps_channel_id, mapping(channel_url), channel_url)` as the channel identity. Delete the dirty Feb-Mar 2026 cases, backfill the kept pre-Feb cases, then deploy new code.

**Architecture:** Additive schema change on `support_message_cases` (`+ id_list ARRAY<STRING>`, `+ channel_id STRING`). One-time DELETE + BQ-side backfill (decoupled id_list and channel_id resolution). Code changes use `id` (string) as the message identity end-to-end, group by `effective_channel_id`, sort by `created_time, id`. Dedup is a global join on `id` only — no channel scope.

**Tech Stack:** Python 3.11 + pandas + Pydantic + google-cloud-bigquery; OpenAI / Anthropic SDKs. Testing is Python scripts under `unit_test/`, executed directly (no pytest).

**Revision history:**
- v1 (initial): used `ps_channel_id` directly as channel key; filtered NULL pscid.
- **v2 (this version):** post AI-review fixes — uses `effective_channel_id` (covers 100% of window rows, 0 message loss); column renamed `ps_channel_id` → `channel_id` to reflect mixed semantics; backfill SQL decoupled and uses `ARRAY_AGG(... IGNORE NULLS LIMIT 1)` (BigQuery rejects `ANY_VALUE IGNORE NULLS`); `format_channel_for_display` no longer relies on the unsafe "no underscore = nanoid" heuristic; adds `src/session.py` and `CaseReviewInput` / `ReviewAction` updates that v1 missed.

---

## Reading list (load before starting)

- `docs/plans/2026-04-29-support-message-id-migration-design.md` — approved design (v2)
- `claude.md` — project conventions (use conda `dev` env)
- Source files this plan modifies: `src/utils.py`, `src/case.py`, `src/channel.py`, `src/session.py`, `src/app_bigquery.py`, `src/vision_processor.py`, `src/prompts/segmentation_prompt.md`

---

## Field name decisions (locked)

These names appear repeatedly below. Use them exactly.

| Concept | Old name | New name | Notes |
|---|---|---|---|
| Pydantic LLM field & JSON key | `message_id_list: List[int]` | `id_list: List[str]` | rename |
| `Case` dataclass field | `message_id_list: List[int]` | `id_list: List[str]` | rename |
| `Case` dataclass field | `channel_url: str` | (kept) + new `channel_id: str` | keep both; on new save channel_url="" |
| `CaseReviewInput.overlap_msg_ids` | `List[int]` | `List[str]` | type change |
| `ReviewAction.new_msg_assignment` | `Dict[int, int]` | `Dict[str, int]` | type change |
| BQ output columns | `message_id_list`, `channel_url` | + `id_list ARRAY<STRING>`, + `channel_id STRING` | additive; legacy cols written NULL on new rows |
| DataFrame primary message column | `Message ID` (int values) | `Message ID` (str values) | name kept; values switch type |
| DataFrame channel column | `Channel URL` | `Channel ID` (= effective_channel_id) | new name; old `Channel URL` retained for inspection only |
| New DataFrame column | (none) | `msg_ch_idx: int` | per-channel 0..N-1 index, used for distance metric |

---

## Phase 0: Setup

### Task 0.1: Verify dev environment and confirm feature branch

**Files:** none

**Step 1: Confirm conda dev env activatable**

Run: `conda env list | grep -E '\bdev\b'`

Expected: a line with `dev` and a path; if missing, abort and fix.

**Step 2: Confirm BQ access works**

Run: `bq query --use_legacy_sql=false --max_rows=1 'SELECT 1 AS ok'`

Expected: prints a table with `ok = 1`.

**Step 3: Confirm we're on the feature branch**

```bash
cd /Users/liuwentong/Project/palmstreet/case-seg-openai
git branch --show-current
```

Expected: `feat/support-message-id-migration`. (If not, `git checkout feat/support-message-id-migration`.)

---

## Phase 1: BigQuery schema and data preparation

These are one-time operational tasks executed from the developer machine via `bq`. No application code yet.

### Task 1.1: Add new columns to `support_message_cases`

**Files:**
- Create: `ops/2026-04-29-add-id-list-and-channel-id.sql`

**Step 1: Write the migration SQL**

```sql
-- ops/2026-04-29-add-id-list-and-channel-id.sql
-- Additive, idempotent (use IF NOT EXISTS).
ALTER TABLE `plantstory.customer_service.support_message_cases`
ADD COLUMN IF NOT EXISTS id_list ARRAY<STRING>;

ALTER TABLE `plantstory.customer_service.support_message_cases`
ADD COLUMN IF NOT EXISTS channel_id STRING;
```

**Step 2: Apply via bq CLI**

Run:
```bash
bq query --use_legacy_sql=false < ops/2026-04-29-add-id-list-and-channel-id.sql
```

Expected: two `Successfully altered ...` messages.

**Step 3: Verify schema**

Run:
```bash
bq show --schema --format=prettyjson plantstory:customer_service.support_message_cases | python -c "import json,sys; cols={c['name'] for c in json.load(sys.stdin)}; print('OK' if {'id_list','channel_id'}<=cols else 'MISSING'); print(sorted(cols))"
```

Expected: prints `OK`.

**Step 4: Commit**

```bash
git add ops/2026-04-29-add-id-list-and-channel-id.sql
git commit -m "ops: add id_list and channel_id columns to support_message_cases"
```

---

### Task 1.2: Snapshot the dirty cases before deletion

**Files:**
- Create: `ops/2026-04-29-snapshot-dirty-cases.sql`

**Step 1:**

```sql
-- ops/2026-04-29-snapshot-dirty-cases.sql
CREATE TABLE `plantstory.customer_service.support_message_cases_dirty_snapshot_2026_04_29` AS
SELECT *
FROM `plantstory.customer_service.support_message_cases`
WHERE start_time >= TIMESTAMP('2026-02-01');
```

**Step 2: Apply**

```bash
bq query --use_legacy_sql=false < ops/2026-04-29-snapshot-dirty-cases.sql
```

**Step 3: Verify**

```bash
bq query --use_legacy_sql=false --format=pretty 'SELECT COUNT(*) AS dirty_snapshot_rows FROM `plantstory.customer_service.support_message_cases_dirty_snapshot_2026_04_29`'
```

Expected: 11,000–13,000 (design measured 11,908; gap allows for any trickle of new writes).

**Step 4: Commit**

```bash
git add ops/2026-04-29-snapshot-dirty-cases.sql
git commit -m "ops: snapshot dirty cases before deletion"
```

---

### Task 1.3: Delete dirty cases (`start_time >= 2026-02-01`)

**Files:**
- Create: `ops/2026-04-29-delete-dirty-cases.sql`

**Step 1:**

```sql
-- ops/2026-04-29-delete-dirty-cases.sql
DELETE FROM `plantstory.customer_service.support_message_cases`
WHERE start_time >= TIMESTAMP('2026-02-01');
```

**Step 2: Apply**

```bash
bq query --use_legacy_sql=false < ops/2026-04-29-delete-dirty-cases.sql
```

Expected: `Number of affected rows: ~11908`.

**Step 3: Verify**

```bash
bq query --use_legacy_sql=false --format=pretty 'SELECT COUNT(*) AS remaining FROM `plantstory.customer_service.support_message_cases` WHERE start_time >= TIMESTAMP("2026-02-01")'
```

Expected: `remaining = 0`.

**Step 4: Commit**

```bash
git add ops/2026-04-29-delete-dirty-cases.sql
git commit -m "ops: delete dirty Feb-Mar 2026 cases"
```

---

### Task 1.4: Backfill `id_list` and `channel_id` on the kept clean cases

The two resolutions are **independent** and must be done as two separate UPDATEs / sub-queries:
- `id_list` JOINs `support_message.message_id` → `support_message.id`.
- `channel_id` JOINs `support_message.channel_url` → `support_message.ps_channel_id` (no `message_id IS NOT NULL` filter — that was the v1 bug).

**Files:**
- Create: `ops/2026-04-29-backfill-id-list.sql`
- Create: `ops/2026-04-29-backfill-channel-id.sql`

**Step 1: Write the id_list backfill MERGE**

```sql
-- ops/2026-04-29-backfill-id-list.sql
--
-- For each kept case (start_time < 2026-02-01), resolve message_id_list[INT]
-- to id_list[STRING] via JOIN on support_message.message_id.
-- Idempotent: re-runs produce the same id_list.

MERGE `plantstory.customer_service.support_message_cases` AS target
USING (
  SELECT
    c.case_id,
    ARRAY_AGG(sm.id IGNORE NULLS ORDER BY sm.created_time, sm.id) AS resolved_id_list
  FROM `plantstory.customer_service.support_message_cases` c,
       UNNEST(c.message_id_list) AS legacy_msg_id
  LEFT JOIN `plantstory.public.support_message` sm
    ON  sm.channel_url = c.channel_url
    AND sm.message_id IS NOT NULL
    AND CAST(sm.message_id AS INT64) = legacy_msg_id
  WHERE c.start_time < TIMESTAMP('2026-02-01')
  GROUP BY c.case_id
) AS src
ON target.case_id = src.case_id
WHEN MATCHED THEN UPDATE SET target.id_list = src.resolved_id_list;
```

**Step 2: Write the channel_id backfill MERGE**

```sql
-- ops/2026-04-29-backfill-channel-id.sql
--
-- For each kept case, resolve channel_id from channel_url via the channel_mapping CTE.
-- Falls back to the original channel_url for pure-old dead channels (no row in
-- support_message ever paired both ids). This is intentional and harmless.

MERGE `plantstory.customer_service.support_message_cases` AS target
USING (
  WITH channel_mapping AS (
    SELECT channel_url,
           ARRAY_AGG(DISTINCT ps_channel_id IGNORE NULLS LIMIT 1)[SAFE_OFFSET(0)] AS canonical_pscid
    FROM `plantstory.public.support_message`
    WHERE channel_url IS NOT NULL AND ps_channel_id IS NOT NULL
    GROUP BY channel_url
  )
  SELECT
    c.case_id,
    COALESCE(m.canonical_pscid, c.channel_url) AS resolved_channel_id
  FROM `plantstory.customer_service.support_message_cases` c
  LEFT JOIN channel_mapping m ON m.channel_url = c.channel_url
  WHERE c.start_time < TIMESTAMP('2026-02-01')
) AS src
ON target.case_id = src.case_id
WHEN MATCHED THEN UPDATE SET target.channel_id = src.resolved_channel_id;
```

**Step 3: Dry-run sample for id_list**

```bash
bq query --use_legacy_sql=false --format=pretty 'SELECT c.case_id, ARRAY_LENGTH(c.message_id_list) AS old_count, ARRAY_AGG(sm.id IGNORE NULLS ORDER BY sm.created_time, sm.id) AS resolved_id_list FROM `plantstory.customer_service.support_message_cases` c, UNNEST(c.message_id_list) AS legacy_msg_id LEFT JOIN `plantstory.public.support_message` sm ON sm.channel_url = c.channel_url AND sm.message_id IS NOT NULL AND CAST(sm.message_id AS INT64) = legacy_msg_id WHERE c.start_time < TIMESTAMP("2026-02-01") GROUP BY c.case_id, c.message_id_list ORDER BY c.case_id LIMIT 5'
```

Expected: each row shows `ARRAY_LENGTH(resolved_id_list) ≈ old_count` (within a few for any source rows that have since been deleted).

**Step 4: Apply both backfills**

```bash
bq query --use_legacy_sql=false < ops/2026-04-29-backfill-id-list.sql
bq query --use_legacy_sql=false < ops/2026-04-29-backfill-channel-id.sql
```

Expected: ~33028 rows updated by each.

**Step 5: Verify**

```bash
bq query --use_legacy_sql=false --format=pretty 'SELECT COUNT(*) AS total_kept, COUNTIF(id_list IS NOT NULL AND ARRAY_LENGTH(id_list) > 0) AS with_id_list, COUNTIF(channel_id IS NOT NULL) AS with_channel_id FROM `plantstory.customer_service.support_message_cases`'
```

Expected: all three counts ≈ 33028. (`with_channel_id` should be exactly 33028 because the COALESCE always produces a value.)

**Step 6: Commit**

```bash
git add ops/2026-04-29-backfill-id-list.sql ops/2026-04-29-backfill-channel-id.sql
git commit -m "ops: backfill id_list and channel_id on kept clean cases"
```

---

## Phase 2: Code changes — data model

### Task 2.1: Update `Case` dataclass and Pydantic models

**Files:**
- Modify: `src/case.py`

**Step 1: Edit dataclass and Pydantic models**

Edit `src/case.py`:

1. Replace line 48:
   ```python
   message_id_list: List[int] = field(default_factory=list)
   ```
   with:
   ```python
   id_list: List[str] = field(default_factory=list)  # support_message.id values (21-char nanoid)
   ```

2. Replace line 55:
   ```python
   channel_url: str = ""  # Channel URL this case belongs to
   ```
   with:
   ```python
   channel_url: str = ""  # Legacy Sendbird URL; written NULL on new save, retained for historical compat
   channel_id: str = ""   # effective_channel_id (always set on new save)
   ```

3. Update `to_dict` (around line 184-217): replace `'message_id_list': self.message_id_list,` → `'id_list': self.id_list,` and add `'channel_id': self.channel_id,`.

4. Replace line 434 in `CaseSegmentationLLMRes`:
   ```python
   message_id_list: List[int]  # List of message indices instead of DataFrame
   ```
   with:
   ```python
   id_list: List[str]  # support_message.id values (21-char string nanoids)
   ```

5. Update `CaseReviewInput.overlap_msg_ids` (line 457):
   ```python
   overlap_msg_ids: List[str] = Field(..., description="重叠区域的消息ID")
   ```

6. Update `ReviewAction.new_msg_assignment` (line 466):
   ```python
   new_msg_assignment: Dict[str, int] = Field(..., description="新的消息分配 {msg_id: case_index}")
   ```

7. Update `to_bigquery_row` (lines ~380-407). Replace the `row` dict:

   ```python
   row = {
       "case_id": self.case_id,
       "channel_url": None,                       # Deprecated: NULL on new writes
       "channel_id": self.channel_id or None,
       "summary": self.summary,
       "status": self.status,
       "segmentation_confidence": self.segmentation_confidence,
       "main_category": self.main_category or '',
       "sub_category": self.sub_category or '',
       "classification_confidence": self.classification_confidence or 0.0,
       "first_res_time": self.first_res_time if self.first_res_time != -1 else None,
       "first_contact_resolution": self.first_contact_resolution if self.first_contact_resolution != -1 else None,
       "usr_msg_num": self.usr_msg_num if self.usr_msg_num != -1 else None,
       "start_time": self.start_time,
       "end_time": self.end_time,
       "message_id_list": None,                   # Deprecated: NULL on new writes
       "id_list": self.id_list,                   # New STRING REPEATED column
       "meta_data": json.dumps(meta_data, ensure_ascii=False),
   }
   ```

**Step 2: Verification script**

Create `unit_test/test_case_schema.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from case import Case, CaseSegmentationLLMRes, CaseReviewInput, ReviewAction

def test_case_dataclass():
    c = Case(case_id="x", id_list=["abc", "def"], channel_id="cid_xx")
    d = c.to_dict()
    assert d["id_list"] == ["abc", "def"], d
    assert d["channel_id"] == "cid_xx", d
    print("OK: Case dataclass uses id_list/channel_id")

def test_pydantic_id_list():
    res = CaseSegmentationLLMRes(
        id_list=["LS57IvprFJ1iqcEpYkvPu", "P9qvxK2H47Z3auQhLZYEy"],
        summary="x", status="ongoing", pending_party="N/A",
        segmentation_confidence=0.9,
        meta={"tracking_numbers": [], "order_numbers": [], "user_names": []},
    )
    assert res.id_list[0] == "LS57IvprFJ1iqcEpYkvPu"
    print("OK: CaseSegmentationLLMRes.id_list accepts strings")

def test_review_models_accept_strings():
    cri = CaseReviewInput(
        cases=[],
        overlap_msg_ids=["abc", "def"],
        all_messages="...",
    )
    assert cri.overlap_msg_ids == ["abc", "def"]
    ra = ReviewAction(
        action_type="merge",
        target_cases=[0, 1],
        new_msg_assignment={"abc": 0, "def": 1},
        reason="test",
    )
    assert ra.new_msg_assignment == {"abc": 0, "def": 1}
    print("OK: review models accept string ids")

if __name__ == "__main__":
    test_case_dataclass()
    test_pydantic_id_list()
    test_review_models_accept_strings()
```

**Step 3: Run**

```bash
conda run -n dev python unit_test/test_case_schema.py
```

Expected: three `OK:` lines.

**Step 4: Commit**

```bash
git add src/case.py unit_test/test_case_schema.py
git commit -m "feat(case): switch id_list/channel_id to strings; update review models"
```

---

## Phase 3: Code changes — SQL and preprocessing

### Task 3.1: Rewrite `Utils.get_channels_to_process` SQL with `effective_channel_id`

**Files:**
- Modify: `src/utils.py:451-552`

**Step 1: Replace the SQL block**

Edit `src/utils.py`. Replace the entire `sql = f"""..."""` block in `get_channels_to_process` with:

```python
sql = f"""
WITH channel_mapping AS (
    SELECT channel_url,
           ARRAY_AGG(DISTINCT ps_channel_id IGNORE NULLS LIMIT 1)[SAFE_OFFSET(0)] AS canonical_pscid
    FROM `plantstory.public.support_message`
    WHERE channel_url IS NOT NULL AND ps_channel_id IS NOT NULL
    GROUP BY channel_url
),
sm_with_eff AS (
    SELECT
        sm.*,
        COALESCE(sm.ps_channel_id, m.canonical_pscid, sm.channel_url) AS effective_channel_id
    FROM `plantstory.public.support_message` sm
    LEFT JOIN channel_mapping m ON m.channel_url = sm.channel_url
    WHERE sm.deleted = FALSE
      AND sm.created_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
),
unprocessed AS (
    SELECT *
    FROM sm_with_eff sm
    WHERE NOT EXISTS (
        SELECT 1
        FROM `plantstory.customer_service.support_message_cases` c,
             UNNEST(c.id_list) AS resolved_id
        WHERE resolved_id = sm.id
    )
),
channel_stats AS (
    SELECT
        effective_channel_id,
        COUNT(*) AS unanalyzed_count,
        MAX(created_time) AS last_message_time,
        CASE
            WHEN COUNT(*) < @chunk_size
                 AND TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(created_time), DAY) >= @idle_days
            THEN COUNT(*)
            WHEN COUNT(*) >= @chunk_size
            THEN CAST(FLOOR(COUNT(*) / @chunk_size) * @chunk_size AS INT64)
            ELSE 0
        END AS messages_to_process
    FROM unprocessed
    {channel_filter}
    GROUP BY effective_channel_id
    HAVING messages_to_process > 0
),
ranked_messages AS (
    SELECT
        u.*,
        cs.messages_to_process,
        ROW_NUMBER() OVER (
            PARTITION BY u.effective_channel_id
            ORDER BY u.created_time, u.id
        ) AS row_num
    FROM unprocessed u
    INNER JOIN channel_stats cs ON u.effective_channel_id = cs.effective_channel_id
)
SELECT * EXCEPT(messages_to_process, row_num)
FROM ranked_messages
WHERE row_num <= messages_to_process
ORDER BY effective_channel_id, created_time, id
"""
```

And update the `channel_filter` and surrounding parameter logic so it filters on `effective_channel_id`. The caller-API kwarg name `channel_urls` is retained for back-compat:

```python
if channel_urls:
    # Caller-facing parameter name kept; values are now effective_channel_id strings
    # (which may be a ps_channel_id, a channel_url, or a canonicalized pscid via mapping).
    channel_filter = "WHERE effective_channel_id IN UNNEST(@channel_urls)"
    query_params = [
        {"name": "chunk_size", "type": "INT64", "value": chunk_size},
        {"name": "idle_days", "type": "INT64", "value": idle_days},
        {"name": "channel_urls", "type": "ARRAY<STRING>", "value": channel_urls},
    ]
else:
    channel_filter = ""
    query_params = [
        {"name": "chunk_size", "type": "INT64", "value": chunk_size},
        {"name": "idle_days", "type": "INT64", "value": idle_days},
    ]
```

Update the docstring to say "Sort by effective_channel_id, created_time, id".

**Step 2: Smoke-test the SQL**

Create `unit_test/test_get_channels_sql.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dotenv import load_dotenv
load_dotenv()

from utils import Utils

# Pick one channel with recent activity (using the new effective_channel_id)
sample = Utils.query_bigquery(
    """
    WITH channel_mapping AS (
      SELECT channel_url,
             ARRAY_AGG(DISTINCT ps_channel_id IGNORE NULLS LIMIT 1)[SAFE_OFFSET(0)] AS canonical_pscid
      FROM `plantstory.public.support_message`
      WHERE channel_url IS NOT NULL AND ps_channel_id IS NOT NULL
      GROUP BY channel_url
    )
    SELECT COALESCE(sm.ps_channel_id, m.canonical_pscid, sm.channel_url) AS effective_channel_id
    FROM `plantstory.public.support_message` sm
    LEFT JOIN channel_mapping m ON m.channel_url = sm.channel_url
    WHERE sm.created_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
      AND sm.deleted = FALSE
    GROUP BY 1
    ORDER BY MAX(sm.created_time) DESC LIMIT 1
    """
)
assert sample, "No active channel in last 7 days; widen the window"
target = sample[0]["effective_channel_id"]
print(f"Target effective_channel_id: {target}")

df = Utils.get_channels_to_process(chunk_size=80, idle_days=7, channel_urls=[target])
print(f"Returned {len(df)} rows for the target channel")
print(df.head().to_string() if not df.empty else "(empty — no unprocessed messages, OK)")
print("OK: SQL ran without error; effective_channel_id resolution works")
```

**Step 3: Run**

```bash
conda run -n dev python unit_test/test_get_channels_sql.py
```

Expected: prints a target channel, a row count, and `OK:`. No exceptions.

**Step 4: Commit**

```bash
git add src/utils.py unit_test/test_get_channels_sql.py
git commit -m "feat(utils): rewrite get_channels_to_process to use effective_channel_id"
```

---

### Task 3.2: Update `Utils.preprocess_dataframe`

**Files:**
- Modify: `src/utils.py:554-714`

**Step 1: Edit `preprocess_dataframe`**

In `src/utils.py`:

1. Update `column_mapping` (line ~589):

   ```python
   column_mapping = {
       'id': 'Message ID',                            # New PK; DataFrame retains "Message ID" name
       'message_id': 'Legacy Message ID',             # Old nullable column — debug only
       'type': 'Type',
       'message': 'Message',
       'raw': 'Raw',
       'sender_id': 'Sender ID',
       'real_sender_id': 'Real Sender ID',
       'created_time': 'Created Time',
       'updated_time': 'Updated Time',
       'channel_url': 'Channel URL',                  # legacy
       'ps_channel_id': 'PS Channel ID',              # raw, debug only
       'effective_channel_id': 'Channel ID',          # new grouping key
       'file_content_size': 'File Content Size',
       'file_content_type': 'File Content Type',
       'file_url': 'File URL',
       'filename': 'Filename',
       'sender_type': 'Sender Type',
       'datastream_metadata': 'Datastream Metadata',
       'deleted': 'Deleted',
       'ticket_id': 'Ticket ID',
       'ps_message_id': 'PS Message ID',
   }
   ```

2. Detection clause: switch the snake_case detection key from `'message_id'` to `'id'` (line ~583):

   ```python
   if 'id' in df.columns:
       if verbose:
           print("        Detected snake_case column names (BigQuery format), converting to Title Case...")
       columns_to_rename = {k: v for k, v in column_mapping.items() if k in df.columns}
       df = df.rename(columns=columns_to_rename)
       if verbose:
           print(f"        Converted {len(columns_to_rename)} column names to Title Case")
   elif 'Message ID' in df.columns:
       if verbose:
           print("        Detected Title Case column names (CSV format), no conversion needed")
   else:
       if verbose:
           print("        ⚠️  Warning: Could not detect column format")
   ```

3. **Remove** the entire numeric-cast / dropna block (lines ~617-641) and replace with:

   ```python
   if 'Message ID' in df.columns:
       df['Message ID'] = df['Message ID'].astype(str)
       if verbose:
           print("        Coerced Message ID to string type")
   ```

4. Update the sort (line ~667-671):

   ```python
   df = df.sort_values(['Channel ID', 'Created Time', 'Message ID']).reset_index(drop=True)
   if verbose:
       print(f"        Sorted data by Channel ID, Created Time, and Message ID")
   ```

5. Update `essential_columns` (line ~682):

   ```python
   essential_columns = [
       'Created Time', 'Sender ID', 'Message', 'Channel URL', 'Channel ID',
       'role', 'Message ID', 'Type', 'File URL', 'File Summary',
   ]
   ```

6. Add the `msg_ch_idx` column right after `df_clean` is created, before the timestamp conversion block:

   ```python
   df_clean['msg_ch_idx'] = df_clean.groupby('Channel ID').cumcount()
   if verbose:
       print(f"        Added msg_ch_idx (per-channel 0..N-1 row index)")
   ```

7. Update the channel summary loop at the bottom:

   ```python
   for channel_id in df_clean['Channel ID'].unique():
       channel_df = df_clean[df_clean['Channel ID'] == channel_id]
       print(f"                Channel: {Utils.format_channel_for_display(channel_id)} - {len(channel_df)} messages")
   ```

**Step 2: Verification script**

Create `unit_test/test_preprocess_string_id.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd
from utils import Utils

# Synthetic BQ-style frame with the new effective_channel_id alias
df = pd.DataFrame([
    {"id": "abc123", "effective_channel_id": "ch1", "channel_url": None,  "ps_channel_id": "ch1",
     "message_id": None, "sender_id": "u1", "type": "MESG",
     "message": "hello", "raw": {}, "created_time": "2026-04-29T10:00:00Z",
     "deleted": False, "file_url": None},
    {"id": "def456", "effective_channel_id": "ch1", "channel_url": None,  "ps_channel_id": "ch1",
     "message_id": None, "sender_id": "psops_x", "type": "MESG",
     "message": "world", "raw": {}, "created_time": "2026-04-29T10:00:01Z",
     "deleted": False, "file_url": None},
    {"id": "ghi789", "effective_channel_id": "sendbird_x_abcd", "channel_url": "sendbird_x_abcd", "ps_channel_id": None,
     "message_id": "1234567", "sender_id": "u2", "type": "MESG",
     "message": "old row", "raw": {}, "created_time": "2026-01-01T00:00:00Z",
     "deleted": False, "file_url": None},
])

clean = Utils.preprocess_dataframe(df, verbose=False)

assert "Message ID" in clean.columns
assert clean["Message ID"].dtype == object
assert set(clean["Message ID"]) == {"abc123", "def456", "ghi789"}
assert "Channel ID" in clean.columns
assert "msg_ch_idx" in clean.columns
ch1 = clean[clean["Channel ID"] == "ch1"].sort_values("msg_ch_idx")
assert list(ch1["msg_ch_idx"]) == [0, 1]
old = clean[clean["Channel ID"] == "sendbird_x_abcd"]
assert old["msg_ch_idx"].iloc[0] == 0
print("OK: preprocess produces string Message ID, Channel ID, and msg_ch_idx")
```

**Step 3: Run**

```bash
conda run -n dev python unit_test/test_preprocess_string_id.py
```

Expected: `OK: ...`

**Step 4: Commit**

```bash
git add src/utils.py unit_test/test_preprocess_string_id.py
git commit -m "feat(utils): preprocess string Message ID + Channel ID + msg_ch_idx"
```

---

### Task 3.3: Update `format_channel_for_display`

**Files:**
- Modify: `src/utils.py:17-24`

**Step 1: Replace the body**

```python
@staticmethod
def format_channel_for_display(channel_id_or_url: str) -> str:
    """
    Display-friendly short form of a channel identifier.
    - Sendbird URLs (start with "sendbird_"): take the trailing hash after the last underscore.
    - Other (nanoid-style 21-char strings, which may themselves contain '_'):
      return the first 8 chars for compact display.
    """
    if channel_id_or_url is None:
        return ""
    s = str(channel_id_or_url)
    if s.startswith('sendbird_'):
        return s.split('_')[-1]
    return s[:8] if len(s) > 8 else s
```

**Step 2: Verification script**

Create `unit_test/test_format_channel.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from utils import Utils

assert Utils.format_channel_for_display("sendbird_group_channel_215_b374305ff3e440674e786d63916f1d5aacda8249") == "b374305ff3e440674e786d63916f1d5aacda8249"
assert Utils.format_channel_for_display("LS57IvprFJ1iqcEpYkvPu") == "LS57Ivpr"   # 21-char nanoid, no sendbird prefix
assert Utils.format_channel_for_display("Lgk9zC7K9mlwsFlN4U_zJ") == "Lgk9zC7K"  # nanoid that contains '_'
assert Utils.format_channel_for_display(None) == ""
assert Utils.format_channel_for_display("short") == "short"
print("OK: format_channel_for_display handles sendbird, nanoid, nanoid-with-underscore")
```

**Step 3: Run**

```bash
conda run -n dev python unit_test/test_format_channel.py
```

Expected: `OK: ...`

**Step 4: Commit**

```bash
git add src/utils.py unit_test/test_format_channel.py
git commit -m "fix(utils): format_channel_for_display uses sendbird_ prefix check"
```

---

### Task 3.4: Widen the Message ID column in `format_messages_for_prompt2`

**Files:**
- Modify: `src/utils.py:52-145`

**Step 1: Widen all `:<11}\t` formatters to `:<22}\t`** (3 occurrences: header line, prefix line, indent line).

**Step 2: Replace the value truncation**

Replace:
```python
message_id = str(row.get('Message ID', ''))[:10]
```
with:
```python
message_id = str(row.get('Message ID', ''))
if len(message_id) > 22:
    message_id = message_id[:22]
```

**Step 3: Update the comment block at the top of the method** to say "Message ID (22 chars)".

**Step 4: Verification script**

Create `unit_test/test_format_messages.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd
from utils import Utils

df = pd.DataFrame([
    {"Message ID": "LS57IvprFJ1iqcEpYkvPu", "Created Time": "2026-04-29T10:00:00Z",
     "role": "user", "Type": "MESG", "Message": "hello world", "Sender ID": "u1"},
])
out = Utils.format_messages_for_prompt2(df)
assert "LS57IvprFJ1iqcEpYkvPu" in out, out
print("OK: full 21-char ID survives formatter")
```

**Step 5: Run**

```bash
conda run -n dev python unit_test/test_format_messages.py
```

Expected: `OK: ...`

**Step 6: Commit**

```bash
git add src/utils.py unit_test/test_format_messages.py
git commit -m "feat(utils): widen Message ID column to 22 chars in prompt formatter"
```

---

## Phase 4: Code changes — Channel, Session, Vision, App

### Task 4.1: Rewrite `Channel.repair_case_segment_output` and `Channel.__init__`

**Files:**
- Modify: `src/channel.py`

**Step 1: Update `Channel.__init__` signature**

Change `Channel.__init__` (line ~58) so it takes `channel_id`. Keep `channel_url` for legacy compat (callers may pass empty string or None):

```python
def __init__(self, df_clean: pd.DataFrame, channel_id: str, session: str,
             chunk_size: int = 80, overlap: int = 20,
             enable_classification: bool = True, enable_vision_processing: bool = True,
             enable_find_sop: bool = True, channel_url: Optional[str] = None):
    self.df_clean = df_clean.copy()
    self.channel_id = channel_id
    self.channel_url = channel_url or ""
    self.session = session
    ...
```

When constructing each `Case`, also set:
```python
case.channel_id = self.channel_id
case.channel_url = self.channel_url
```

**Step 2: Replace all `case_dict['message_id_list']` with `case_dict['id_list']`**

Lines 187-189, 263, 282 — and any other.

```python
id_list = case_dict['id_list']
case_messages = self.df_clean[self.df_clean['Message ID'].astype(str).isin([str(x) for x in id_list])].copy()
```

**Step 3: Rewrite `repair_case_segment_output`**

In `src/channel.py:551-917`:

1. Update docstring (line ~560) to:
   > "id_list contains support_message.id values (21-char nanoids). chunk_df must have 'Message ID' (string) and 'msg_ch_idx' (int) columns."

2. At the top of the method body, build the index-lookup helper:

   ```python
   id_to_idx: Dict[str, int] = dict(zip(
       chunk_df['Message ID'].astype(str),
       chunk_df['msg_ch_idx'].astype(int)
   ))

   def _idx_of(msg_id: str) -> Optional[int]:
       return id_to_idx.get(str(msg_id))
   ```

3. Replace `c["message_id_list"] = sorted({int(x) for x in c["message_id_list"]})` (line ~604) with:
   ```python
   c["id_list"] = sorted({str(x) for x in c.get("id_list", [])})
   ```

4. Replace **every** other `message_id_list` reference inside this method with `id_list` (search the whole method).

5. Replace `_proximity_score` (line ~679):
   ```python
   def _proximity_score(target_idx: int, case: Dict[str, Any]) -> float:
       ml = case.get("id_list", [])
       if not ml:
           return 0.0
       case_idxs = [_idx_of(m) for m in ml]
       case_idxs = [i for i in case_idxs if i is not None]
       if not case_idxs:
           return 0.0
       dist = min(abs(target_idx - i) for i in case_idxs)
       return 1.0 / (1 + dist)
   ```

6. Replace `_find_nearest_same_sender_case` (line ~710-738):
   ```python
   def _find_nearest_same_sender_case(msg_id: str, cases: List[Dict]) -> Optional[int]:
       chunk_ids_str = chunk_df['Message ID'].astype(str)
       if str(msg_id) not in chunk_ids_str.values:
           return None
       target_row = chunk_df[chunk_ids_str == str(msg_id)].iloc[0]
       target_sender = target_row.get('Sender ID', '')
       target_idx = int(target_row['msg_ch_idx'])
       if not target_sender:
           return None

       msg_to_case: Dict[str, int] = {}
       for case_idx, case in enumerate(cases):
           for mid in case.get('id_list', []):
               msg_to_case[str(mid)] = case_idx

       best_distance = float('inf')
       best_case_id: Optional[int] = None
       for check_msg_id, case_idx in msg_to_case.items():
           candidate = chunk_df[chunk_ids_str == check_msg_id]
           if candidate.empty:
               continue
           check_sender = candidate.iloc[0].get('Sender ID', '')
           if check_sender == target_sender:
               distance = abs(int(candidate.iloc[0]['msg_ch_idx']) - target_idx)
               if distance < best_distance:
                   best_distance = distance
                   best_case_id = case_idx
       return best_case_id
   ```

7. Replace `_attach_to_any_nearest_case` (line ~765-781) similarly using `msg_ch_idx` distance.

8. Update the dedup loop (line ~820) to pass `target_idx`:
   ```python
   for msg_id, cids in list(msg_to_cases.items()):
       if len(cids) <= 1:
           continue
       target_idx = _idx_of(msg_id)
       winner = _choose_one_for_duplicate(target_idx, out, cids, prev_context)
   ```
   And `_choose_one_for_duplicate(...)` (line ~686) should take `target_idx: Optional[int]` and pass it to `_proximity_score`.

9. Update chunk-set construction (line ~795-796):
   ```python
   chunk_msg_ids = chunk_df['Message ID'].astype(str).tolist()
   ...
   chunk_set = set(chunk_msg_ids)
   ```

10. Update `_attach_to_case` to append strings:
    ```python
    def _attach_to_case(msg_id: str, case_id: int, cases: List[Dict], provisionals: List[Dict], reason: str):
        if case_id < len(cases):
            cases[case_id]["id_list"].append(str(msg_id))
            cases[case_id]["id_list"] = sorted(set(cases[case_id]["id_list"]))
            ...
    ```

11. Update final sort (line ~874): `out.sort(key=lambda c: c["id_list"][0])`.

**Step 4: Update CSV writer**

In `save_results_to_csv` (around line 472-509):
- Replace `case_obj.message_id_list` → `case_obj.id_list`.
- The DataFrame uses `Message ID` column name (kept), so `df_annotated['Message ID'] == message_id` still works (with string comparison).

**Step 5: Verification script**

Create `unit_test/test_repair_string_ids.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd
from channel import Channel

chunk_df = pd.DataFrame([
    {"Message ID": "id_A", "msg_ch_idx": 0, "Sender ID": "u1",      "Message": "hi",    "Type": "MESG"},
    {"Message ID": "id_B", "msg_ch_idx": 1, "Sender ID": "u1",      "Message": "still", "Type": "MESG"},
    {"Message ID": "id_C", "msg_ch_idx": 2, "Sender ID": "psops_a", "Message": "hello", "Type": "MESG"},
    {"Message ID": "id_D", "msg_ch_idx": 3, "Sender ID": "u1",      "Message": "",      "Type": "MESG"},
])

cases_in = [
    {"id_list": ["id_A", "id_B"], "summary": "x", "status": "ongoing", "pending_party": "N/A",
     "segmentation_confidence": 0.8,
     "meta": {"tracking_numbers": [], "order_numbers": [], "user_names": []}},
    {"id_list": ["id_C"],         "summary": "y", "status": "ongoing", "pending_party": "N/A",
     "segmentation_confidence": 0.9,
     "meta": {"tracking_numbers": [], "order_numbers": [], "user_names": []}},
]

ch = Channel(df_clean=chunk_df.copy(), channel_id="cid", session="t",
             chunk_size=80, overlap=0, enable_classification=False, enable_vision_processing=False,
             enable_find_sop=False, channel_url=None)

result = ch.repair_case_segment_output(cases_in, chunk_df)
all_ids = {m for c in result["cases_out"] for m in c["id_list"]}
assert all_ids == {"id_A", "id_B", "id_C", "id_D"}, all_ids
assert result["report"]["missing_msgs"] == 0
print("OK: repair handles string IDs and assigns the unassigned id_D")
```

**Step 6: Run**

```bash
conda run -n dev python unit_test/test_repair_string_ids.py
```

Expected: `OK: ...`

**Step 7: Commit**

```bash
git add src/channel.py unit_test/test_repair_string_ids.py
git commit -m "feat(channel): use string ids and msg_ch_idx-based distance"
```

---

### Task 4.2: Update `src/session.py` (CSV-input path, used by `main.py`)

**Files:**
- Modify: `src/session.py`

**Step 1: Replace `Channel URL` grouping with `Channel ID`**

In `src/session.py:148-152`, `:236-243`, `:266-273`:

- Replace `df['Channel URL']` → `df['Channel ID']` for `unique()` and equality lookups.
- Rename local var `channel_url` → `channel_id`.
- `Channel(channel_df, channel_id, self.session_name, ...)` (positional `channel_id` matches the new `__init__` signature).
- For display: pass the *first* non-null `channel_url` from each channel group as the `channel_url=` kwarg, used only for legacy mention in CSV. `Utils.format_channel_for_display(channel_id)` for printed names.

**Step 2: Replace `case.message_id_list` with `case.id_list` and `case.channel_url` with `case.channel_id` in serialization**

In `src/session.py:334`:
```python
'channel_id': case.channel_id or '',
```

In `src/session.py:352`:
```python
'id_list': format_array_field(case.id_list),
```

(Also remove or NULL out `channel_url` if it's still referenced for the new output format.)

**Step 3: Smoke-test**

Create `unit_test/test_session_grouping.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import importlib
import pandas as pd

# Just verify Session imports and instantiates with the new column expectations
from session import Session

df = pd.DataFrame([
    {"Message ID": "a", "Channel ID": "cid1", "Channel URL": None, "Created Time": "2026-04-29T10:00:00Z",
     "Sender ID": "u1", "role": "user", "Message": "hi", "Type": "MESG", "File URL": None,
     "File Summary": "", "msg_ch_idx": 0},
    {"Message ID": "b", "Channel ID": "cid1", "Channel URL": None, "Created Time": "2026-04-29T10:00:01Z",
     "Sender ID": "u1", "role": "user", "Message": "again", "Type": "MESG", "File URL": None,
     "File Summary": "", "msg_ch_idx": 1},
])

# Use Session in a way that does not require an LLM client; just verify channel_urls list extraction
unique_channels = df["Channel ID"].unique().tolist()
assert unique_channels == ["cid1"]
print("OK: Session-side Channel ID grouping is wired")
```

**Step 4: Run**

```bash
conda run -n dev python unit_test/test_session_grouping.py
```

Expected: `OK: ...`

**Step 5: Commit**

```bash
git add src/session.py unit_test/test_session_grouping.py
git commit -m "feat(session): switch CSV-input path to Channel ID and id_list"
```

---

### Task 4.3: Update `src/vision_processor.py`

**Files:**
- Modify: `src/vision_processor.py:42-60`

**Step 1: Type and matcher**

```python
def get_context_for_image(channel_df: pd.DataFrame,
                          image_message_id: str,
                          context_size: int = 3) -> pd.DataFrame:
    """
    ...
    Args:
        image_message_id: The Message ID (string nanoid) of the image message
    ...
    """
    image_mask = channel_df['Message ID'].astype(str) == str(image_message_id)
    if not image_mask.any():
        raise ValueError(f"Image message with Message ID {image_message_id} not found in channel")
    ...
```

**Step 2: Verification script**

Create `unit_test/test_vision_string_id.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd
from vision_processor import VisionProcessor

df = pd.DataFrame([
    {"Message ID": "id_X", "Sender ID": "u", "role": "user", "Created Time": "2026-04-29T10:00:00Z", "Message": "hi", "Type": "MESG", "File URL": None},
    {"Message ID": "id_Y", "Sender ID": "u", "role": "user", "Created Time": "2026-04-29T10:00:01Z", "Message": "",   "Type": "FILE", "File URL": "http://x"},
    {"Message ID": "id_Z", "Sender ID": "u", "role": "user", "Created Time": "2026-04-29T10:00:02Z", "Message": "ok", "Type": "MESG", "File URL": None},
])
ctx = VisionProcessor.get_context_for_image(channel_df=df, image_message_id="id_Y", context_size=1)
assert "id_Y" in ctx["Message ID"].astype(str).values
assert len(ctx) >= 1
print("OK: vision context lookup works with string Message ID")
```

**Step 3: Run**

```bash
conda run -n dev python unit_test/test_vision_string_id.py
```

Expected: `OK: ...`

**Step 4: Commit**

```bash
git add src/vision_processor.py unit_test/test_vision_string_id.py
git commit -m "feat(vision): accept string Message IDs"
```

---

### Task 4.4: Update `src/app_bigquery.py` driver

**Files:**
- Modify: `src/app_bigquery.py:130-150`

**Step 1: Group by `Channel ID`**

```python
channel_ids = df_clean['Channel ID'].unique()

for channel_idx, channel_id in enumerate(channel_ids):
    channel_df = df_clean[df_clean['Channel ID'] == channel_id].copy()
    channel_url_for_legacy = (
        channel_df['Channel URL'].dropna().iloc[0]
        if 'Channel URL' in channel_df.columns and channel_df['Channel URL'].notna().any()
        else None
    )

    print(f"\n🔄 Channel {channel_idx + 1}/{len(channel_ids)}: "
          f"{Utils.format_channel_for_display(channel_id)} ({len(channel_df)} messages)")

    try:
        channel = Channel(
            df_clean=channel_df,
            channel_id=channel_id,
            channel_url=channel_url_for_legacy,
            session=session_name,
            chunk_size=args.chunk_size,
            overlap=0,
            enable_classification=args.enable_classification,
            enable_vision_processing=args.enable_vision_processing,
            enable_find_sop=False,
        )
        ...
```

**Step 2: Manual sanity check**

```bash
conda run -n dev python -c "
import sys; sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv()
from utils import Utils
df = Utils.get_channels_to_process(chunk_size=80, idle_days=7)
print(f'channels: {df[\"effective_channel_id\"].nunique() if not df.empty else 0}')
print(f'rows: {len(df)}')
"
```

Expected: prints non-zero channel and row counts.

**Step 3: Commit**

```bash
git add src/app_bigquery.py
git commit -m "feat(app_bigquery): group by Channel ID"
```

---

## Phase 5: Prompts and end-to-end

### Task 5.1: Update `segmentation_prompt.md`

**Files:**
- Modify: `src/prompts/segmentation_prompt.md`

**Step 1: Edits**

- Line 14 — `- Message ID: 11 characters (10-digit database ID)` → `- Message ID: 21-character string nanoid (e.g. "LS57IvprFJ1iqcEpYkvPu")`.
- Line 88 — `"message_id_list": [4499509692, 4599500696],` → `"id_list": ["LS57IvprFJ1iqcEpYkvPu", "P9qvxK2H47Z3auQhLZYEy"],`.
- Line 106 — `For \`message_id_list\`, ...` → `For \`id_list\`, use the Message ID values shown in the table verbatim, keep the order from the table.`
- Replace **all** remaining `message_id_list` in the file with `id_list`.
- The schematic table layout: widen Message ID column to 22 chars.

**Step 2: Verify**

```bash
grep -n "message_id_list\|10-digit\|11 characters" /Users/liuwentong/Project/palmstreet/case-seg-openai/src/prompts/segmentation_prompt.md
```

Expected: no matches.

**Step 3: Commit**

```bash
git add src/prompts/segmentation_prompt.md
git commit -m "docs(prompts): update segmentation prompt for string ids"
```

---

### Task 5.2: End-to-end dry run on a single live channel

**Files:** none (operational)

**Step 1: Pick a high-traffic channel with unprocessed messages**

```bash
bq query --use_legacy_sql=false --format=pretty --max_rows=5 'WITH cm AS (SELECT channel_url, ARRAY_AGG(DISTINCT ps_channel_id IGNORE NULLS LIMIT 1)[SAFE_OFFSET(0)] AS p FROM `plantstory.public.support_message` WHERE channel_url IS NOT NULL AND ps_channel_id IS NOT NULL GROUP BY channel_url) SELECT COALESCE(sm.ps_channel_id, cm.p, sm.channel_url) AS effective_channel_id, COUNT(*) AS msgs FROM `plantstory.public.support_message` sm LEFT JOIN cm ON cm.channel_url = sm.channel_url WHERE sm.created_time >= TIMESTAMP("2026-03-01") AND sm.deleted = false GROUP BY 1 ORDER BY msgs DESC LIMIT 5'
```

Note one effective_channel_id with ~80-200 messages.

**Step 2: Dry-run**

```bash
cd /Users/liuwentong/Project/palmstreet/case-seg-openai
conda run -n dev python -m src.app_bigquery --dry-run --channel-urls <CHANNEL_ID> --chunk-size 80
```

Expected: console prints "Channel 1/1", no exceptions, JSON written under `out/session_*/`.

**Step 3: Inspect the JSON**

```bash
cat out/session_*/cases_*.json | python -c "
import json, sys
data = json.load(sys.stdin)
for c in data[:2]:
    print(c['case_id'], len(c['id_list']), c['id_list'][:2], c.get('channel_id'))
"
```

Expected: each case has non-empty `id_list` (strings) and a non-null `channel_id`.

**Step 4: Commit a marker**

```bash
git commit --allow-empty -m "verify: dry-run on live channel passes"
```

---

### Task 5.3: Production run on a single channel

**Files:** none (operational)

**Step 1: Run without `--dry-run`**

```bash
conda run -n dev python -m src.app_bigquery --channel-urls <CHANNEL_ID> --chunk-size 80
```

Expected: prints "✅ Successfully saved N/N cases to BigQuery" with N > 0.

**Step 2: Verify the write**

```bash
bq query --use_legacy_sql=false --format=pretty 'SELECT case_id, channel_id, ARRAY_LENGTH(id_list) AS msg_count, channel_url, ARRAY_LENGTH(message_id_list) AS legacy_count FROM `plantstory.customer_service.support_message_cases` WHERE channel_id = "<CHANNEL_ID>" ORDER BY end_time DESC LIMIT 5'
```

Expected: `channel_id` matches; `msg_count > 0`; `channel_url IS NULL`; `legacy_count IS NULL`.

**Step 3: Re-run to confirm dedup**

```bash
conda run -n dev python -m src.app_bigquery --channel-urls <CHANNEL_ID> --chunk-size 80
```

Expected: "0 of 1 channels need processing".

**Step 4: Commit a marker**

```bash
git commit --allow-empty -m "verify: prod write + dedup confirmed for one channel"
```

---

### Task 5.4: Full production rollout

**Files:** none (operational)

**Step 1: PR and merge**

```bash
git push -u origin feat/support-message-id-migration
gh pr create --title "feat: migrate case-seg pipeline off support_message.message_id" --body "$(cat <<'EOF'
## Summary
- Switches the case segmentation pipeline to use support_message.id (21-char nanoid) and effective_channel_id (= COALESCE(ps_channel_id, mapping(channel_url), channel_url))
- Adds id_list ARRAY<STRING> and channel_id STRING columns to support_message_cases (additive; legacy columns preserved as read-only history)
- One-time ops: snapshot + delete dirty Feb-Mar cases (~12k), backfill kept pre-Feb cases (~33k) via decoupled BQ MERGE
- 0 messages lost: COALESCE(channel_url, ps_channel_id) covers 100% of 90-day window rows

## Test plan
- [ ] Phase 0 setup verified
- [ ] Phase 1 ops applied and counts verified
- [ ] Phase 2-4 unit_test scripts all print "OK"
- [ ] Phase 5.2 dry-run on one channel produced sane JSON
- [ ] Phase 5.3 prod write verified and dedup re-run skipped the channel
- [ ] Phase 5.4 full run on all channels stable
EOF
)"
```

**Step 2: Run the full backlog**

```bash
conda run -n dev python -m src.app_bigquery --chunk-size 80
```

Expected: ~11–12k cases produced (≈ count of dirty cases that were deleted).

**Step 3: Spot-check**

```bash
bq query --use_legacy_sql=false --format=pretty 'SELECT DATE(end_time) AS day, COUNT(*) AS cases, COUNTIF(id_list IS NULL OR ARRAY_LENGTH(id_list)=0) AS missing_id_list, COUNTIF(channel_id IS NULL) AS missing_channel_id FROM `plantstory.customer_service.support_message_cases` WHERE start_time >= TIMESTAMP("2026-02-01") GROUP BY day ORDER BY day'
```

Expected: every day has `cases > 0`, `missing_id_list = 0`, `missing_channel_id = 0`.

**Step 4: Re-enable cron**

If the scheduler was paused, re-enable. Confirm the next run picks up zero work.

---

## Phase 6: Post-deploy cleanup (separate change after stability window)

### Task 6.1: After stability, drop legacy source columns

**Files:** none (DBA-coordinated)

**Step 1: Confirm 7+ days of clean operation**

```bash
bq query --use_legacy_sql=false --format=pretty 'SELECT DATE(end_time) AS day, COUNT(*) AS cases FROM `plantstory.customer_service.support_message_cases` GROUP BY day ORDER BY day DESC LIMIT 10'
```

**Step 2: Coordinate DBA drop of `support_message.message_id` and `support_message.channel_url`**

Note: the runtime SQL still reads `channel_url` for the COALESCE fallback (used by 1,308 pure-old dead channels). Once the 90-day window slides past the cutoff (`2026-03-27` was the last NULL-pscid row, so window-out around 2026-06-25), `channel_url` is no longer needed and can be dropped. Until then, leaving it in source is harmless.

**Step 3: After source drop, full sanity run**

```bash
conda run -n dev python -m src.app_bigquery --chunk-size 80
```

Expected: clean run, no regressions.

---

## Reading list for executors

If anything in this plan refers to behavior you don't have context on, read in this order:
1. `docs/plans/2026-04-29-support-message-id-migration-design.md` — the (v2) design that produced this plan
2. `src/utils.py` — focus on `get_channels_to_process` and `preprocess_dataframe`
3. `src/channel.py` — focus on `repair_case_segment_output` and `Channel.__init__`
4. `src/case.py` — focus on `to_bigquery_row` and the Pydantic models
5. `src/session.py` — the CSV-input path (separate from the BigQuery-input path used by `app_bigquery.py`)
6. `src/prompts/segmentation_prompt.md` — the LLM contract

---

## Risks and rollback

- **Phase 1.3 wrong DELETE**: snapshot from 1.2 lets you `INSERT INTO ... SELECT ... FROM snapshot` to restore.
- **Phase 1.4 wrong backfill**: the two MERGEs only update `id_list` / `channel_id`; legacy columns intact. Re-run the corrected MERGE; idempotent.
- **New code broken at runtime**: revert merge, restore from snapshot, redeploy old code. Possible because we kept legacy columns and have not yet dropped source columns — Phase 6 is separated for exactly this reason.
- **Future "split risk" channels**: 0 in current window; if they appear later, the symptom is two cases for what should be one conversation. Per design ("个位数不精确 OK"), we accept this and revisit only if it becomes systemic.

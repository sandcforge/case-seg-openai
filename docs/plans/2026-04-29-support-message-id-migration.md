# Support Message ID Migration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Migrate the case-segmentation pipeline off `support_message.message_id` (now nullable, eventually dropped) and `support_message.channel_url` (incomplete on new rows). Use `id` (21-char nanoid PK) and `ps_channel_id` instead. Delete dirty Feb-Mar 2026 cases, backfill the kept pre-Feb cases, then deploy new code.

**Architecture:** Additive schema change on `support_message_cases` (`+ id_list ARRAY<STRING>`, `+ ps_channel_id STRING`). One-time DELETE + BQ-side backfill. Code changes use `id` (string) as the message identity end-to-end, group by `ps_channel_id`, sort by `created_time, id`. Dedup is a global join on `id` only — no channel scope.

**Tech Stack:** Python 3.11 + pandas + Pydantic + google-cloud-bigquery; OpenAI / Anthropic SDKs. Testing is Python scripts under `unit_test/`, executed directly (no pytest).

---

## Reading list (load before starting)

- `docs/plans/2026-04-29-support-message-id-migration-design.md` — approved design with decisions table
- `claude.md` — project conventions (use conda `dev` env)
- Source files this plan modifies: `src/utils.py`, `src/case.py`, `src/channel.py`, `src/vision_processor.py`, `src/app_bigquery.py`, `src/prompts/segmentation_prompt.md`

---

## Field name decisions (locked)

These names appear repeatedly below. Use them exactly.

| Concept | Old name | New name | Notes |
|---|---|---|---|
| Pydantic LLM field | `message_id_list: List[int]` | `id_list: List[str]` | rename |
| `Case` dataclass field | `message_id_list: List[int]` | `id_list: List[str]` | rename |
| `Case` dataclass field | `channel_url: str` | (kept) + new `ps_channel_id: str` | keep both, set channel_url="" |
| BigQuery output columns | `message_id_list`, `channel_url` | + `id_list ARRAY<STRING>`, + `ps_channel_id STRING` | additive; legacy cols written as NULL |
| DataFrame primary column | `Message ID` (int values) | `Message ID` (str values) | name kept to minimize diff; values switch type |
| New DataFrame column | (none) | `msg_ch_idx: int` | per-channel 0..N-1 index, used for distance metric |
| LLM JSON output key | `message_id_list` | `id_list` | prompt + Pydantic agree |

---

## Phase 0: Setup

### Task 0.1: Verify dev environment and create feature branch

**Files:** none

**Step 1: Confirm conda dev env activatable**

Run: `conda env list | grep -E '\bdev\b'`

Expected: a line with `dev` and a path; if missing, abort and fix.

**Step 2: Confirm BQ access works**

Run: `bq query --use_legacy_sql=false --max_rows=1 'SELECT 1 AS ok'`

Expected: prints a table with `ok = 1`.

**Step 3: Create feature branch**

```bash
cd /Users/liuwentong/Project/palmstreet/case-seg-openai
git checkout -b feat/support-message-id-migration
```

**Step 4: Commit baseline marker**

```bash
git commit --allow-empty -m "chore: start support_message id migration branch"
```

---

## Phase 1: BigQuery schema and data preparation

These are one-time operational tasks executed from the developer machine via `bq`. No application code yet.

### Task 1.1: Add new columns to `support_message_cases`

**Files:**
- Create: `ops/2026-04-29-add-id-list-and-ps-channel-id.sql`

**Step 1: Write the migration SQL**

```sql
-- ops/2026-04-29-add-id-list-and-ps-channel-id.sql
-- Additive, idempotent-safe (use IF NOT EXISTS).
ALTER TABLE `plantstory.customer_service.support_message_cases`
ADD COLUMN IF NOT EXISTS id_list ARRAY<STRING>;

ALTER TABLE `plantstory.customer_service.support_message_cases`
ADD COLUMN IF NOT EXISTS ps_channel_id STRING;
```

**Step 2: Apply via bq CLI**

Run:
```bash
bq query --use_legacy_sql=false < ops/2026-04-29-add-id-list-and-ps-channel-id.sql
```

Expected: `Successfully altered ...` (twice).

**Step 3: Verify schema**

Run:
```bash
bq show --schema --format=prettyjson plantstory:customer_service.support_message_cases | python -c "import json,sys; cols={c['name'] for c in json.load(sys.stdin)}; print('OK' if {'id_list','ps_channel_id'}<=cols else 'MISSING'); print(sorted(cols))"
```

Expected: `OK` printed.

**Step 4: Commit**

```bash
git add ops/2026-04-29-add-id-list-and-ps-channel-id.sql
git commit -m "ops: add id_list and ps_channel_id columns to support_message_cases"
```

---

### Task 1.2: Snapshot the dirty cases before deletion

**Files:**
- Create: `ops/2026-04-29-snapshot-dirty-cases.sql`

**Step 1: Write CTAS to a snapshot table** (cheap insurance — user said no impact, but cost is low)

```sql
-- ops/2026-04-29-snapshot-dirty-cases.sql
CREATE TABLE `plantstory.customer_service.support_message_cases_dirty_snapshot_2026_04_29` AS
SELECT *
FROM `plantstory.customer_service.support_message_cases`
WHERE start_time >= TIMESTAMP('2026-02-01');
```

**Step 2: Apply**

Run:
```bash
bq query --use_legacy_sql=false < ops/2026-04-29-snapshot-dirty-cases.sql
```

**Step 3: Verify count matches the design-doc estimate (~11,908)**

Run:
```bash
bq query --use_legacy_sql=false --format=pretty 'SELECT COUNT(*) AS dirty_snapshot_rows FROM `plantstory.customer_service.support_message_cases_dirty_snapshot_2026_04_29`'
```

Expected: row count between 11,000 and 13,000 (the design doc measured 11,908; the gap allows for new writes since the measurement).

**Step 4: Commit**

```bash
git add ops/2026-04-29-snapshot-dirty-cases.sql
git commit -m "ops: snapshot dirty cases before deletion"
```

---

### Task 1.3: Delete dirty cases (start_time >= 2026-02-01)

**Files:**
- Create: `ops/2026-04-29-delete-dirty-cases.sql`

**Step 1: Write the DELETE**

```sql
-- ops/2026-04-29-delete-dirty-cases.sql
DELETE FROM `plantstory.customer_service.support_message_cases`
WHERE start_time >= TIMESTAMP('2026-02-01');
```

**Step 2: Apply**

Run:
```bash
bq query --use_legacy_sql=false < ops/2026-04-29-delete-dirty-cases.sql
```

Expected output: `Number of affected rows: ~11908`.

**Step 3: Verify**

Run:
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

### Task 1.4: Backfill `id_list` and `ps_channel_id` on the kept clean cases

This must run while `support_message.message_id` is still populated (which it is for pre-Feb rows).

**Files:**
- Create: `ops/2026-04-29-backfill-id-list-and-ps-channel-id.sql`

**Step 1: Write the backfill MERGE**

```sql
-- ops/2026-04-29-backfill-id-list-and-ps-channel-id.sql
--
-- For each case kept after deletion (start_time < 2026-02-01),
-- resolve message_id_list[INT] -> id_list[STRING] and channel_url -> ps_channel_id
-- via JOIN on the BQ mirror of support_message.
--
-- Idempotent: re-running computes the same id_list/ps_channel_id values.
-- A small number of cases may not resolve a ps_channel_id (dead pre-migration channels);
-- those rows keep ps_channel_id NULL — acceptable per design.

MERGE `plantstory.customer_service.support_message_cases` AS target
USING (
  SELECT
    c.case_id,
    ARRAY_AGG(sm.id IGNORE NULLS ORDER BY sm.created_time, sm.id) AS resolved_id_list,
    ANY_VALUE(sm.ps_channel_id IGNORE NULLS) AS resolved_ps_channel_id
  FROM `plantstory.customer_service.support_message_cases` c
  LEFT JOIN UNNEST(c.message_id_list) AS legacy_msg_id
  LEFT JOIN `plantstory.public.support_message` sm
    ON  sm.channel_url = c.channel_url
    AND sm.message_id IS NOT NULL
    AND CAST(sm.message_id AS INT64) = legacy_msg_id
  WHERE c.start_time < TIMESTAMP('2026-02-01')
  GROUP BY c.case_id
) AS src
ON target.case_id = src.case_id
WHEN MATCHED THEN UPDATE SET
  target.id_list = src.resolved_id_list,
  target.ps_channel_id = src.resolved_ps_channel_id;
```

**Step 2: Dry-run by sampling first**

Run a quick sanity check on 5 cases to confirm the JOIN logic resolves correctly:

```bash
bq query --use_legacy_sql=false --format=pretty 'SELECT c.case_id, c.channel_url, ARRAY_LENGTH(c.message_id_list) AS old_count, ARRAY_AGG(sm.id IGNORE NULLS ORDER BY sm.created_time) AS resolved, ANY_VALUE(sm.ps_channel_id IGNORE NULLS) AS resolved_pscid FROM `plantstory.customer_service.support_message_cases` c LEFT JOIN UNNEST(c.message_id_list) AS legacy_msg_id LEFT JOIN `plantstory.public.support_message` sm ON sm.channel_url = c.channel_url AND sm.message_id IS NOT NULL AND CAST(sm.message_id AS INT64) = legacy_msg_id WHERE c.start_time < TIMESTAMP("2026-02-01") GROUP BY c.case_id, c.channel_url, c.message_id_list LIMIT 5'
```

Expected: each row shows `ARRAY_LENGTH(resolved) ≈ old_count` (within +/- a few for deleted source rows). `resolved_pscid` may be NULL for some.

**Step 3: Apply the MERGE**

Run:
```bash
bq query --use_legacy_sql=false < ops/2026-04-29-backfill-id-list-and-ps-channel-id.sql
```

Expected: `Number of affected rows: ~33028`.

**Step 4: Verify backfill completeness**

Run:
```bash
bq query --use_legacy_sql=false --format=pretty 'SELECT COUNT(*) AS total_kept, COUNTIF(id_list IS NOT NULL AND ARRAY_LENGTH(id_list) > 0) AS with_id_list, COUNTIF(ps_channel_id IS NOT NULL) AS with_ps_channel_id FROM `plantstory.customer_service.support_message_cases`'
```

Expected: `with_id_list ≈ 33028` (a handful may be 0 if all referenced messages were deleted from source; investigate if > 1% have empty id_list).

**Step 5: Commit**

```bash
git add ops/2026-04-29-backfill-id-list-and-ps-channel-id.sql
git commit -m "ops: backfill id_list and ps_channel_id on kept clean cases"
```

---

## Phase 2: Code changes — data model

### Task 2.1: Update `Case` dataclass and Pydantic LLM model

**Files:**
- Modify: `src/case.py`

**Step 1: Write the dataclass and Pydantic changes**

Edit `src/case.py`:

1. Replace line 48:
   ```python
   message_id_list: List[int] = field(default_factory=list)  # List of Message ID values
   ```
   with:
   ```python
   id_list: List[str] = field(default_factory=list)  # List of support_message.id (string nanoid)
   ```

2. Replace line 55:
   ```python
   channel_url: str = ""  # Channel URL this case belongs to
   ```
   with:
   ```python
   channel_url: str = ""  # Legacy Sendbird channel URL — kept for historical compat, written NULL on new save
   ps_channel_id: str = ""  # Channel identity in the new message system (always populated)
   ```

3. Update `to_dict` (around line 184-217): change `'message_id_list': self.message_id_list,` → `'id_list': self.id_list,` and add `'ps_channel_id': self.ps_channel_id,` next to `'channel_url'`.

4. Replace line 434 in `CaseSegmentationLLMRes`:
   ```python
   message_id_list: List[int]  # List of message indices instead of DataFrame
   ```
   with:
   ```python
   id_list: List[str]  # List of support_message.id values (21-char string nanoids)
   ```

5. Update `to_bigquery_row` (lines 380-407 area). Replace the `row` dict so the BigQuery insert matches the new schema:

   ```python
   row = {
       "case_id": self.case_id,
       "channel_url": None,                       # Deprecated: leave NULL on new writes
       "ps_channel_id": self.ps_channel_id or None,
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
       "message_id_list": None,                   # Deprecated: leave NULL on new writes
       "id_list": self.id_list,                   # New STRING REPEATED column
       "meta_data": json.dumps(meta_data, ensure_ascii=False),
   }
   ```

**Step 2: Write a focused verification script**

Create `unit_test/test_case_schema.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from case import Case, CaseSegmentationLLMRes

def test_case_dataclass_has_id_list_and_ps_channel_id():
    c = Case(case_id="x", id_list=["abc", "def"], ps_channel_id="pcid_xx")
    d = c.to_dict()
    assert d["id_list"] == ["abc", "def"], d
    assert d["ps_channel_id"] == "pcid_xx", d
    assert "message_id_list" not in d or d.get("message_id_list") in (None, []), d
    print("OK: Case dataclass uses id_list/ps_channel_id")

def test_pydantic_id_list_accepts_strings():
    res = CaseSegmentationLLMRes(
        id_list=["LS57IvprFJ1iqcEpYkvPu", "P9qvxK2H47Z3auQhLZYEy"],
        summary="x", status="ongoing", pending_party="N/A",
        segmentation_confidence=0.9,
        meta={"tracking_numbers": [], "order_numbers": [], "user_names": []},
    )
    assert res.id_list[0] == "LS57IvprFJ1iqcEpYkvPu"
    print("OK: CaseSegmentationLLMRes.id_list accepts strings")

if __name__ == "__main__":
    test_case_dataclass_has_id_list_and_ps_channel_id()
    test_pydantic_id_list_accepts_strings()
```

**Step 3: Run**

```bash
conda run -n dev python unit_test/test_case_schema.py
```

Expected:
```
OK: Case dataclass uses id_list/ps_channel_id
OK: CaseSegmentationLLMRes.id_list accepts strings
```

**Step 4: Commit**

```bash
git add src/case.py unit_test/test_case_schema.py
git commit -m "feat(case): switch id_list to List[str] and add ps_channel_id"
```

---

## Phase 3: Code changes — SQL and preprocessing

### Task 3.1: Rewrite `Utils.get_channels_to_process` SQL

**Files:**
- Modify: `src/utils.py:451-552`

**Step 1: Replace the SQL inside `get_channels_to_process`**

Edit `src/utils.py`. Replace the entire method body's `sql = f"""..."""` block (lines ~492-548) with:

```python
sql = f"""
WITH channel_stats AS (
    SELECT
        sm.ps_channel_id,
        COUNT(*) AS unanalyzed_count,
        MAX(sm.created_time) AS last_message_time,
        CASE
            WHEN COUNT(*) < @chunk_size
                 AND TIMESTAMP_DIFF(CURRENT_TIMESTAMP(), MAX(sm.created_time), DAY) >= @idle_days
            THEN COUNT(*)
            WHEN COUNT(*) >= @chunk_size
            THEN CAST(FLOOR(COUNT(*) / @chunk_size) * @chunk_size AS INT64)
            ELSE 0
        END AS messages_to_process
    FROM `plantstory.public.support_message` sm
    WHERE sm.deleted = FALSE
      AND sm.created_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
      AND sm.ps_channel_id IS NOT NULL
      {channel_filter}
      AND NOT EXISTS (
        SELECT 1
        FROM `plantstory.customer_service.support_message_cases` seg,
             UNNEST(seg.id_list) AS resolved_id
        WHERE resolved_id = sm.id
      )
    GROUP BY sm.ps_channel_id
    HAVING messages_to_process > 0
),
ranked_messages AS (
    SELECT
        sm.*,
        cs.messages_to_process,
        ROW_NUMBER() OVER (
            PARTITION BY sm.ps_channel_id
            ORDER BY sm.created_time, sm.id
        ) AS row_num
    FROM `plantstory.public.support_message` sm
    INNER JOIN channel_stats cs ON sm.ps_channel_id = cs.ps_channel_id
    WHERE sm.deleted = FALSE
      AND sm.created_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 90 DAY)
      AND sm.ps_channel_id IS NOT NULL
      AND NOT EXISTS (
        SELECT 1
        FROM `plantstory.customer_service.support_message_cases` seg,
             UNNEST(seg.id_list) AS resolved_id
        WHERE resolved_id = sm.id
      )
)
SELECT * EXCEPT(messages_to_process, row_num)
FROM ranked_messages
WHERE row_num <= messages_to_process
ORDER BY ps_channel_id, created_time, id
"""
```

Also update the `channel_filter` and surrounding logic so it filters on `ps_channel_id` (not `channel_url`):

```python
if channel_urls:
    # Param name kept as `channel_urls` for caller-API back-compat, but values
    # are now ps_channel_id strings.
    channel_filter = "AND sm.ps_channel_id IN UNNEST(@channel_urls)"
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

Also update the docstring comment block to say "Sort by ps_channel_id, created_time, id". The argument name `channel_urls` stays for caller-API compat with `app_bigquery.py` until Task 5.x; we update its semantics here.

**Step 2: Smoke-test the SQL with a known channel**

Find one channel that should have unprocessed messages. From a freshly read state, the best candidate is any ps_channel_id with rows after 2026-04-01.

Create `unit_test/test_get_channels_sql.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dotenv import load_dotenv
load_dotenv()

from utils import Utils

# Pick a single ps_channel_id that has recent activity
sample = Utils.query_bigquery(
    "SELECT ps_channel_id FROM `plantstory.public.support_message` "
    "WHERE ps_channel_id IS NOT NULL AND created_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY) "
    "GROUP BY ps_channel_id ORDER BY MAX(created_time) DESC LIMIT 1"
)
assert sample, "No active ps_channel_id found in last 7 days; widen the window"
target = sample[0]["ps_channel_id"]
print(f"Target ps_channel_id: {target}")

df = Utils.get_channels_to_process(chunk_size=80, idle_days=7, channel_urls=[target])
print(f"Returned {len(df)} rows for the target channel")
print(df.head().to_string() if not df.empty else "(empty — no unprocessed messages, that's fine)")
print("OK: SQL ran without error and dedup join succeeded")
```

**Step 3: Run**

```bash
conda run -n dev python unit_test/test_get_channels_sql.py
```

Expected: prints a target channel and a row count. No exceptions. The exact count is informational only.

**Step 4: Commit**

```bash
git add src/utils.py unit_test/test_get_channels_sql.py
git commit -m "feat(utils): rewrite get_channels_to_process SQL to use id and ps_channel_id"
```

---

### Task 3.2: Update `Utils.preprocess_dataframe` for string IDs and ps_channel_id

**Files:**
- Modify: `src/utils.py:554-714`

**Step 1: Edit `preprocess_dataframe`**

In `src/utils.py`:

1. Update `column_mapping` (line ~589) to add `ps_channel_id` and to keep `id` as a recognizable column:

   ```python
   column_mapping = {
       'id': 'Message ID',                    # New PK becomes the DataFrame "Message ID" (string)
       'message_id': 'Legacy Message ID',     # Old nullable column — kept for debugging only
       'type': 'Type',
       'message': 'Message',
       'raw': 'Raw',
       'sender_id': 'Sender ID',
       'real_sender_id': 'Real Sender ID',
       'created_time': 'Created Time',
       'updated_time': 'Updated Time',
       'channel_url': 'Channel URL',          # legacy
       'ps_channel_id': 'Channel ID',         # new grouping key
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
       # Could be CSV with already-Title-case headers, or a stale fixture; leave alone.
       if verbose:
           print("        Detected Title Case column names (CSV format), no conversion needed")
   else:
       if verbose:
           print("        ⚠️  Warning: Could not detect column format")
   ```

3. **Remove** the entire numeric-cast / dropna block (currently lines ~617-641). Replace with:

   ```python
   # Ensure Message ID is a string (the new id is a 21-char nanoid, never numeric)
   if 'Message ID' in df.columns:
       df['Message ID'] = df['Message ID'].astype(str)
       if verbose:
           print("        Coerced Message ID to string type")
   ```

4. Update the sort (line ~667-671) to use the new grouping key:

   ```python
   df = df.sort_values(['Channel ID', 'Created Time', 'Message ID']).reset_index(drop=True)
   if verbose:
       print(f"        Sorted data by Channel ID, Created Time, and Message ID")
   ```

5. Update `essential_columns` (line ~682) to include `Channel ID`:

   ```python
   essential_columns = [
       'Created Time', 'Sender ID', 'Message', 'Channel URL', 'Channel ID',
       'role', 'Message ID', 'Type', 'File URL', 'File Summary',
   ]
   ```

6. After producing `df_clean`, **add a new column `msg_ch_idx`** populated as the per-channel sequential index. Insert before the `# 6. Convert Timestamp columns` block:

   ```python
   # 5b. Per-channel sequential index for distance-metric calculations
   df_clean['msg_ch_idx'] = df_clean.groupby('Channel ID').cumcount()
   if verbose:
       print(f"        Added msg_ch_idx (per-channel 0..N-1 row index)")
   ```

7. The "Display channel summary" loop at the bottom — switch `Channel URL` to `Channel ID`:

   ```python
   for channel_id in df_clean['Channel ID'].unique():
       channel_df = df_clean[df_clean['Channel ID'] == channel_id]
       print(f"                Channel: {Utils.format_channel_for_display(channel_id)} - {len(channel_df)} messages")
   ```

**Step 2: Update `format_channel_for_display`**

`src/utils.py:17-24`. Replace the body with:

```python
@staticmethod
def format_channel_for_display(channel_id_or_url: str) -> str:
    """
    Display-friendly short form of a channel identifier.
    - Sendbird URLs: take the trailing hash (after the last underscore).
    - ps_channel_id (no underscore): return as-is (already 21 chars).
    """
    if channel_id_or_url is None:
        return ""
    s = str(channel_id_or_url)
    if '_' not in s:
        return s
    return s.split('_')[-1]
```

**Step 3: Write a verification script**

Create `unit_test/test_preprocess_string_id.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd
from utils import Utils

# Synthetic BQ-style frame: snake_case columns, string id, string ps_channel_id
df = pd.DataFrame([
    {"id": "abc123",          "ps_channel_id": "ch1", "channel_url": None,
     "message_id": None,      "sender_id": "u1", "type": "MESG",
     "message": "hello",      "raw": {}, "created_time": "2026-04-29T10:00:00Z",
     "deleted": False,        "file_url": None},
    {"id": "def456",          "ps_channel_id": "ch1", "channel_url": None,
     "message_id": None,      "sender_id": "psops_x", "type": "MESG",
     "message": "world",      "raw": {}, "created_time": "2026-04-29T10:00:01Z",
     "deleted": False,        "file_url": None},
    {"id": "ghi789",          "ps_channel_id": "ch2", "channel_url": "sendbird_x_abcd",
     "message_id": "1234567", "sender_id": "u2", "type": "MESG",
     "message": "old row",    "raw": {}, "created_time": "2026-01-01T00:00:00Z",
     "deleted": False,        "file_url": None},
])

clean = Utils.preprocess_dataframe(df, verbose=False)

assert "Message ID" in clean.columns
assert clean["Message ID"].dtype == object
assert set(clean["Message ID"]) == {"abc123", "def456", "ghi789"}
assert "Channel ID" in clean.columns
assert "msg_ch_idx" in clean.columns
ch1 = clean[clean["Channel ID"] == "ch1"].sort_values("msg_ch_idx")
assert list(ch1["msg_ch_idx"]) == [0, 1]
assert clean[clean["Channel ID"] == "ch2"]["msg_ch_idx"].iloc[0] == 0
print("OK: preprocess handles string IDs, ps_channel_id, and msg_ch_idx")
```

**Step 4: Run**

```bash
conda run -n dev python unit_test/test_preprocess_string_id.py
```

Expected: `OK: preprocess handles string IDs, ps_channel_id, and msg_ch_idx`

**Step 5: Commit**

```bash
git add src/utils.py unit_test/test_preprocess_string_id.py
git commit -m "feat(utils): preprocess string Message IDs and add Channel ID + msg_ch_idx"
```

---

### Task 3.3: Update `format_messages_for_prompt2` to widen the ID column

**Files:**
- Modify: `src/utils.py:52-145`

**Step 1: Widen the column from 11 → 22 chars**

In `src/utils.py:52-145`:

1. Replace `':<11}\t'` formatters with `':<22}\t'` (3 occurrences: header line, prefix line, indent line).
2. Replace truncation `[:10]` (line ~74) with: just the value as-is, plus an assertion guard:
   ```python
   message_id = str(row.get('Message ID', ''))
   if len(message_id) > 22:
       message_id = message_id[:22]
   ```
3. Update the comment block at the top of the method to say "Message ID (22 chars)".

**Step 2: Write a quick formatting smoke test**

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
# The full 21-char ID must appear, untruncated
assert "LS57IvprFJ1iqcEpYkvPu" in out, out
print("OK: full 21-char ID survives formatter")
```

**Step 3: Run**

```bash
conda run -n dev python unit_test/test_format_messages.py
```

Expected: `OK: full 21-char ID survives formatter`

**Step 4: Commit**

```bash
git add src/utils.py unit_test/test_format_messages.py
git commit -m "feat(utils): widen Message ID column to 22 chars in prompt formatter"
```

---

## Phase 4: Code changes — Channel and downstream

### Task 4.1: Rewrite `Channel.repair_case_segment_output` distance metric and string-ID handling

**Files:**
- Modify: `src/channel.py:551-917`

**Step 1: Switch from int math to msg_ch_idx-based distance**

In `src/channel.py:551-917`, do the following edits to `repair_case_segment_output`:

1. The method's docstring still says "message_id_list contains actual Message IDs from the database (e.g., 4499509692), not sequential indices." — replace the example with the new reality:
   > `id_list contains actual support_message.id values (21-char nanoids). chunk_df must have 'Message ID' (string) and 'msg_ch_idx' (int) columns.`

2. Replace `c["message_id_list"] = sorted({int(x) for x in c["message_id_list"]})` (~line 604) with:
   ```python
   c["id_list"] = sorted({str(x) for x in c.get("id_list", [])})
   ```
   And replace any other reference to `c["message_id_list"]` inside this method with `c["id_list"]`.

3. Build a single helper `_idx_of(msg_id)` near the top of `repair_case_segment_output` that maps an ID string → its `msg_ch_idx` integer. Use it for distance calculations:

   ```python
   id_to_idx: Dict[str, int] = dict(zip(chunk_df['Message ID'].astype(str), chunk_df['msg_ch_idx'].astype(int)))

   def _idx_of(msg_id: str) -> Optional[int]:
       return id_to_idx.get(str(msg_id))
   ```

4. Replace `_proximity_score` (line ~679):
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

5. Replace `_find_nearest_same_sender_case` (line ~710-738) to use string IDs and `msg_ch_idx`:
   ```python
   def _find_nearest_same_sender_case(msg_id: str, cases: List[Dict]) -> Optional[int]:
       if msg_id not in chunk_df['Message ID'].astype(str).values:
           return None
       target_row = chunk_df[chunk_df['Message ID'].astype(str) == str(msg_id)].iloc[0]
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
           candidate = chunk_df[chunk_df['Message ID'].astype(str) == check_msg_id]
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

6. Replace `_attach_to_any_nearest_case` (line ~765-781) similarly to use msg_ch_idx distance.

7. Replace `_choose_one_for_duplicate` calls so the proximity argument is the message's `msg_ch_idx`, not the message_id:
   ```python
   # in the dedup loop (around line 820)
   for msg_id, cids in list(msg_to_cases.items()):
       if len(cids) <= 1:
           continue
       target_idx = _idx_of(msg_id)
       winner = _choose_one_for_duplicate(target_idx, out, cids, prev_context)
   ```

   And inside `_choose_one_for_duplicate` (line ~686), pass `target_idx` to `_proximity_score(target_idx, c)`.

8. Update the chunk-set construction (line ~795-796):
   ```python
   chunk_msg_ids = chunk_df['Message ID'].astype(str).tolist()
   ...
   chunk_set = set(chunk_msg_ids)  # set of strings, not ints
   ```

9. Update `_attach_to_case` so it appends strings:
   ```python
   def _attach_to_case(msg_id: str, case_id: int, cases: List[Dict], provisionals: List[Dict], reason: str):
       if case_id < len(cases):
           cases[case_id]["id_list"].append(str(msg_id))
           cases[case_id]["id_list"] = sorted(set(cases[case_id]["id_list"]))
           ...
   ```

10. Update final sort (line ~874): `out.sort(key=lambda c: c["id_list"][0])`.

**Step 2: Update `Channel.build_cases_via_llm` and case construction**

In `src/channel.py:186-285` and surrounding case-construction code:

- Replace `case_dict['message_id_list']` → `case_dict['id_list']`
- Replace `self.df_clean['Message ID'].isin(message_id_list)` lookups — note that `Message ID` is now a string column; `isin` still works.
- When passing to `Case(...)`, use `id_list=case_dict['id_list']` and `ps_channel_id=self.ps_channel_id` (introduced in next step).

**Step 3: Add `ps_channel_id` to `Channel.__init__`**

In `src/channel.py:58-69`:

```python
def __init__(self, df_clean: pd.DataFrame, ps_channel_id: str, channel_url: Optional[str], session: str,
             chunk_size: int = 80, overlap: int = 20,
             enable_classification: bool = True, enable_vision_processing: bool = True,
             enable_find_sop: bool = True):
    self.df_clean = df_clean.copy()
    self.ps_channel_id = ps_channel_id
    self.channel_url = channel_url or ""   # may be empty for new channels
    self.session = session
    ...
```

Also replace usages like `case.channel_url = self.channel_url` with also setting `case.ps_channel_id = self.ps_channel_id`.

**Step 4: Update CSV writer (around line 484)**

Replace `case_obj.messages['Message ID']` (still works because column kept the same name) — no change needed. But replace `case_obj.message_id_list` → `case_obj.id_list` everywhere in `save_results_to_csv` and `save_results_to_json`.

**Step 5: Smoke-test repair logic with string IDs**

Create `unit_test/test_repair_string_ids.py`:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
import pandas as pd
from channel import Channel

# Build a tiny synthetic chunk with string IDs and msg_ch_idx
chunk_df = pd.DataFrame([
    {"Message ID": "id_A", "msg_ch_idx": 0, "Sender ID": "u1", "Message": "hi", "Type": "MESG"},
    {"Message ID": "id_B", "msg_ch_idx": 1, "Sender ID": "u1", "Message": "still",  "Type": "MESG"},
    {"Message ID": "id_C", "msg_ch_idx": 2, "Sender ID": "psops_a", "Message": "hello", "Type": "MESG"},
    {"Message ID": "id_D", "msg_ch_idx": 3, "Sender ID": "u1", "Message": "", "Type": "MESG"},   # empty
])

# Two LLM-output cases: one covers A, B; one covers C. id_D unassigned.
cases_in = [
    {"id_list": ["id_A", "id_B"], "summary": "x", "status": "ongoing", "pending_party": "N/A",
     "segmentation_confidence": 0.8,
     "meta": {"tracking_numbers": [], "order_numbers": [], "user_names": []}},
    {"id_list": ["id_C"],         "summary": "y", "status": "ongoing", "pending_party": "N/A",
     "segmentation_confidence": 0.9,
     "meta": {"tracking_numbers": [], "order_numbers": [], "user_names": []}},
]

ch = Channel(df_clean=chunk_df.copy(), ps_channel_id="pcid", channel_url=None, session="t",
             chunk_size=80, overlap=0, enable_classification=False, enable_vision_processing=False,
             enable_find_sop=False)

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

Expected: `OK: repair handles string IDs and assigns the unassigned id_D`

**Step 7: Commit**

```bash
git add src/channel.py unit_test/test_repair_string_ids.py
git commit -m "feat(channel): use string ids and msg_ch_idx-based distance in repair logic"
```

---

### Task 4.2: Update `vision_processor.py` for string IDs

**Files:**
- Modify: `src/vision_processor.py:42-60`

**Step 1: Switch the type of `image_message_id` from int to str**

In `src/vision_processor.py:42`:

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

The rest of the function uses `Message ID` only for masking, which is type-agnostic if we cast.

**Step 2: Smoke-test**

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

Expected: `OK: vision context lookup works with string Message ID`

**Step 4: Commit**

```bash
git add src/vision_processor.py unit_test/test_vision_string_id.py
git commit -m "feat(vision): accept string Message IDs"
```

---

### Task 4.3: Update `app_bigquery.py` driver to group by `Channel ID`

**Files:**
- Modify: `src/app_bigquery.py:130-150`

**Step 1: Switch from `Channel URL` to `Channel ID` for grouping**

In `src/app_bigquery.py:130-150`:

```python
# After preprocess returns df_clean:
ps_channel_ids = df_clean['Channel ID'].unique()

for channel_idx, ps_channel_id in enumerate(ps_channel_ids):
    channel_df = df_clean[df_clean['Channel ID'] == ps_channel_id].copy()
    # First non-null channel_url for display continuity (may be None for new-only channels)
    channel_url = channel_df['Channel URL'].dropna().iloc[0] if 'Channel URL' in channel_df.columns and channel_df['Channel URL'].notna().any() else None

    print(f"\n🔄 Channel {channel_idx + 1}/{len(ps_channel_ids)}: "
          f"{Utils.format_channel_for_display(ps_channel_id)} ({len(channel_df)} messages)")

    try:
        channel = Channel(
            df_clean=channel_df,
            ps_channel_id=ps_channel_id,
            channel_url=channel_url,
            session=session_name,
            chunk_size=args.chunk_size,
            overlap=0,
            enable_classification=args.enable_classification,
            enable_vision_processing=args.enable_vision_processing,
            enable_find_sop=False,
        )
        ...
```

**Step 2: Manual sanity run (no LLM cost)**

```bash
conda run -n dev python -c "
import sys, os; sys.path.insert(0, 'src')
from dotenv import load_dotenv; load_dotenv()
from utils import Utils
df = Utils.get_channels_to_process(chunk_size=80, idle_days=7)
print(f'Total channels needing processing: {df[\"ps_channel_id\"].nunique() if not df.empty else 0}')
print(f'Total messages: {len(df)}')
print(df.head().to_string() if not df.empty else '(empty)')
"
```

Expected: prints a non-zero number of channels and messages (since the cron has been broken for a month).

**Step 3: Commit**

```bash
git add src/app_bigquery.py
git commit -m "feat(app_bigquery): group by ps_channel_id"
```

---

## Phase 5: Prompts and final stitching

### Task 5.1: Update `segmentation_prompt.md`

**Files:**
- Modify: `src/prompts/segmentation_prompt.md`

**Step 1: Update the schema description, JSON example, and column-width hints**

Replace these specific lines:

- Line 14 — `- Message ID: 11 characters (10-digit database ID)` → `- Message ID: 21-character string nanoid (e.g. "LS57IvprFJ1iqcEpYkvPu")`
- Line 88 — `"message_id_list": [4499509692, 4599500696],` → `"id_list": ["LS57IvprFJ1iqcEpYkvPu", "P9qvxK2H47Z3auQhLZYEy"],`
- Line 106 — `For \`message_id_list\`, use Message IDs shown in the table, keep the order in the table.` → `For \`id_list\`, use the Message ID values shown in the table verbatim, keep the order from the table.`
- Any other occurrence of `message_id_list` in this file → `id_list`.
- The header row description should say the Message ID column is 22 chars wide.

**Step 2: Verify file integrity**

Run:
```bash
grep -n "message_id_list\|10-digit\|11 characters" /Users/liuwentong/Project/palmstreet/case-seg-openai/src/prompts/segmentation_prompt.md
```

Expected: no matches (or at most a deliberate "legacy: was message_id_list" comment).

**Step 3: Commit**

```bash
git add src/prompts/segmentation_prompt.md
git commit -m "docs(prompts): update segmentation prompt for string ids"
```

---

### Task 5.2: End-to-end dry run on a single live channel

**Files:** none (operational)

**Step 1: Pick a single high-traffic affected channel**

```bash
bq query --use_legacy_sql=false --format=pretty --max_rows=5 'SELECT ps_channel_id, COUNT(*) AS msgs FROM `plantstory.public.support_message` WHERE created_time >= TIMESTAMP("2026-03-01") AND deleted = false AND ps_channel_id IS NOT NULL GROUP BY ps_channel_id ORDER BY msgs DESC LIMIT 5'
```

Note one ps_channel_id with ~80-200 unprocessed messages (sweet spot for one chunk).

**Step 2: Run dry-run**

```bash
cd /Users/liuwentong/Project/palmstreet/case-seg-openai
conda run -n dev python -m src.app_bigquery --dry-run --channel-urls <PSCID> --chunk-size 80
```

Expected:
- Console prints "Channel 1/1: ..." with message count.
- No exceptions.
- Output JSON written under `out/session_*/`.
- Each case's `id_list` is non-empty and contains 21-char strings.

**Step 3: Inspect a JSON output**

```bash
ls -la out/session_*/ | head
cat out/session_*/cases_*.json | python -c "
import json, sys
data = json.load(sys.stdin)
for c in data[:2]:
    print(c['case_id'], len(c['id_list']), c['id_list'][:2], c.get('ps_channel_id'))
"
```

Expected: each case row has a non-empty `id_list` of strings and a non-null `ps_channel_id`.

**Step 4: If output looks correct, commit a marker**

```bash
git commit --allow-empty -m "verify: dry-run on live channel passes"
```

---

### Task 5.3: Production run on a single channel (real BQ write)

**Files:** none (operational)

**Step 1: Run without `--dry-run`**

```bash
conda run -n dev python -m src.app_bigquery --channel-urls <PSCID> --chunk-size 80
```

Expected: prints "✅ Successfully saved N/N cases to BigQuery" with N > 0.

**Step 2: Verify the write**

```bash
bq query --use_legacy_sql=false --format=pretty 'SELECT case_id, ps_channel_id, ARRAY_LENGTH(id_list) AS msg_count, channel_url, ARRAY_LENGTH(message_id_list) AS legacy_count FROM `plantstory.customer_service.support_message_cases` WHERE ps_channel_id = "<PSCID>" ORDER BY end_time DESC LIMIT 5'
```

Expected:
- `ps_channel_id` matches the target.
- `msg_count` > 0.
- `channel_url` is NULL.
- `legacy_count` is NULL or 0.

**Step 3: Verify dedup actually skips them on a re-run**

```bash
conda run -n dev python -m src.app_bigquery --channel-urls <PSCID> --chunk-size 80
```

Expected: prints "0 of 1 channels need processing" — confirms the new dedup path works.

**Step 4: Commit a verification marker**

```bash
git commit --allow-empty -m "verify: prod write + dedup confirmed for one channel"
```

---

### Task 5.4: Full production rollout

**Files:** none (operational)

**Step 1: Open a PR with the branch and merge after review**

```bash
git push -u origin feat/support-message-id-migration
gh pr create --title "feat: migrate case-seg pipeline off support_message.message_id" --body "$(cat <<'EOF'
## Summary
- Switches the case segmentation pipeline to use support_message.id (21-char nanoid) and ps_channel_id, replacing the dropped Sendbird message_id and incomplete channel_url
- Adds id_list ARRAY<STRING> and ps_channel_id STRING columns to support_message_cases (additive, legacy columns preserved as read-only history)
- One-time ops: snapshot + delete dirty Feb-Mar cases (~12k), backfill kept pre-Feb cases (~33k) via BQ JOIN

## Test plan
- [ ] Phase 0 setup: dev env + bq access verified
- [ ] Phase 1 ops: schema, snapshot, delete, backfill applied and counts verified
- [ ] Phase 2-4 unit_test scripts all print "OK"
- [ ] Phase 5.2 dry-run on one channel produced sane JSON
- [ ] Phase 5.3 prod write verified and dedup re-run skipped the channel
- [ ] Phase 5.4 full run on all channels stable
EOF
)"
```

**Step 2: After merge, run the full backlog**

```bash
conda run -n dev python -m src.app_bigquery --chunk-size 80
```

Expected: long-running; prints per-channel progress and a final "PROCESSING COMPLETE" summary. ~11k cases produced (≈ count of dirty cases that were deleted).

**Step 3: Spot-check the BQ output**

```bash
bq query --use_legacy_sql=false --format=pretty 'SELECT DATE(end_time) AS day, COUNT(*) AS cases, COUNTIF(id_list IS NULL OR ARRAY_LENGTH(id_list)=0) AS missing_id_list, COUNTIF(ps_channel_id IS NULL) AS missing_ps_channel_id FROM `plantstory.customer_service.support_message_cases` WHERE start_time >= TIMESTAMP("2026-02-01") GROUP BY day ORDER BY day'
```

Expected:
- All days from 2026-02-01 onwards have `cases > 0`.
- `missing_id_list = 0` and `missing_ps_channel_id = 0` for every day.

**Step 4: Re-enable the cron**

If the scheduler was paused per `setup_scheduler.sh` / DEPLOY.md, re-enable it. Confirm next scheduled run picks up zero work (everything is freshly deduped).

---

## Phase 6: Post-deploy cleanup (separate change after stability window)

### Task 6.1: After a stability window, drop legacy source columns

**Files:** none (operational, DBA-coordinated)

**Step 1: Confirm new code is stable for at least 7 days**

```bash
bq query --use_legacy_sql=false --format=pretty 'SELECT DATE(end_time) AS day, COUNT(*) AS cases FROM `plantstory.customer_service.support_message_cases` GROUP BY day ORDER BY day DESC LIMIT 10'
```

Expected: continuous daily case counts.

**Step 2: Coordinate with DBA to drop `support_message.message_id` and `support_message.channel_url`**

These columns are no longer referenced by any code in `case-seg-openai` after this migration. The backfill (Phase 1.4) already resolved historical references.

**Step 3: After source drop, verify the pipeline still works**

```bash
conda run -n dev python -m src.app_bigquery --chunk-size 80
```

Expected: runs cleanly, no regressions. (The mirror table `plantstory.public.support_message` will lose those columns at the next Datastream sync; the new SQL doesn't SELECT them.)

---

## Reading list for executors

If anything in this plan refers to behavior you don't have context on, read in this order:
1. `docs/plans/2026-04-29-support-message-id-migration-design.md` — the design that produced this plan
2. `src/utils.py` — focus on `get_channels_to_process` and `preprocess_dataframe`
3. `src/channel.py` — focus on `repair_case_segment_output`
4. `src/case.py` — focus on `to_bigquery_row` and the Pydantic models
5. `src/prompts/segmentation_prompt.md` — the LLM contract

---

## Risks and rollback

- **If Phase 1.3 (DELETE) is wrong**: snapshot table from 1.2 lets you `INSERT INTO ... SELECT ... FROM snapshot` to restore.
- **If Phase 1.4 (backfill) writes wrong values**: the MERGE updates only `id_list`/`ps_channel_id`, leaving `message_id_list`/`channel_url` intact. Re-run the MERGE with corrected logic; it overwrites `id_list`/`ps_channel_id` idempotently.
- **If new code produces broken cases at runtime**: revert the merge, restore from snapshot, re-deploy old code (after temporarily un-NULLing `support_message.message_id` if already dropped — which is why Phase 6 is a separate change).

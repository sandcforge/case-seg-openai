Previous context summary:

# CASE_ANCHOR_RULES
- When linking across chunks, anchor cases in this priority order:
  tracking_id > order_id > buyer_handle > topic.
- If multiple orders share one tracking_id, treat them as ONE "multi_order_package" case unless topics are clearly separate.
- Address change / refund / fee questions default to order-level; payout/app issues default to system-level.

ACTIVE_CASE_HINTS:
- (optional, up to 5) Summaries of unresolved/ongoing issues from the previous chunk.
- If none: write "None".

[Hint format — one entity/topic per hint]
- topic: "<short title, e.g., Address change | LDP claim | UPS claim | Chargeback | Pickup->Ship toggle | Refund error | Payout retry | App bug | Data export | Fee adjustment>"
  program: "<LDP|UPS|Chargeback|AddressChange|ShippingMode|Refund|Cancellation|FeeAdjustment|Payout|AppBug|DataExport|LiveVisibility|TaxExempt|Other>"
  scope: "<single_order|multi_order_package|buyer_account|system>"
  anchor: {
    "tracking": [<1Z...|12-14 digit FedEx|USPS 92...>],
    "order_ids": [<9759-xxxxxx-xxxx, ...>],
    "buyers": [<@handle, ...>],
    "carrier": "<UPS|FedEx|USPS|N/A>"
  }
  status: "<open|ongoing|resolved|blocked>"
  shipping_state: "<label_created|picked_up|in_transit|delivered|delayed|lost|N/A>"
  last_action: "<what changed, e.g., 'switched to pickup', 'refund issued', 'reimbursed to seller balance', 'address updated'>"
  last_update: "<ISO timestamp or N/A>"
  pending_party: "<seller|agent|buyer|carrier|platform|N/A>"
  amounts: {"credit_to_seller": <number or N/A>, "refund_to_buyer": <number or N/A>}
  returns_to_previous_topic: <true|false>
  possible_new_session: <true|false>
  keywords: [<refund|shipment|pickup|address change|claim|chargeback|fee|payout|bug|...>]
  evidence_msg_ch_idx: [<list of msg_ch_idx>]

RECENT_MESSAGES:
- <msg_ch_idx> | sender id=<sender_id> | role=<role> | <ISO timestamp> | text=<truncated text>
- ... (use ≤ overlap lines from the previous chunk; ≥5 lines if available)

META (optional):
- overlap: <int>
- channel: <id|url or N/A>
- time_window: ["<start ISO>", "<end ISO>"]

GUIDANCE:
- Normalize mentions: map psops/support to `agent`, seller to `seller`, customer to `buyer`.
- Prefer the most recent explicit entity when resolving pronouns like "this order".
- Detect carrier by tracking pattern (UPS often starts with 1Z; USPS often starts with 9; FedEx typically 12–14 digits, no hyphens) and fill `carrier`.
- Only mark `resolved` if no follow-up on the same topic appears afterwards in the recent window; otherwise keep `ongoing`.

<<<INSERT_PREVIOUS_CONTEXT_SUMMARY_BLOCK_HERE>>>

---

## Current chunk messages

The following lines are the messages you must analyze:

```
<<<INSERT_CHUNK_BLOCK_HERE>>>   # each line: msg_ch_idx | sender id | role | timestamp | text
```

---

## Task

Based on the current chunk messages and any previous context, generate a tail summary that will be used as the "Previous context summary" for the next chunk.

Your response should include:

1. ACTIVE_CASE_HINTS: Extract up to 5 unresolved/ongoing cases from the current chunk
2. RECENT_MESSAGES: Include the last ≤ overlap messages from this chunk
3. META: Include overlap, channel, and time_window information

Format your response exactly as shown in the template above, replacing the placeholders with actual content from the current chunk.
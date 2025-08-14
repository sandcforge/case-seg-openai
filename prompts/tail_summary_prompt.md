**IMPORTANT: Your response MUST be valid JSON only. Do not include any text before or after the JSON. The response should be a single JSON object.**

Previous context summary (JSON format):

```json
{
  "case_anchor_rules": {
    "priority_order": "tracking_id > order_id > buyer_handle > topic",
    "multi_order_rule": "If multiple orders share one tracking_id, treat them as ONE 'multi_order_package' case unless topics are clearly separate",
    "default_scope_rules": "Address change / refund / fee questions default to order-level; payout/app issues default to system-level"
  },
  "active_case_hints": [
    {
      "topic": "short title, e.g., Address change | LDP claim | UPS claim | Chargeback | Pickup->Ship toggle | Refund error | Payout retry | App bug | Data export | Fee adjustment",
      "program": "LDP|UPS|Chargeback|AddressChange|ShippingMode|Refund|Cancellation|FeeAdjustment|Payout|AppBug|DataExport|LiveVisibility|TaxExempt|Other",
      "scope": "single_order|multi_order_package|buyer_account|system",
      "anchor": {
        "tracking": ["1Z...|12-14 digit FedEx|USPS 92..."],
        "order_ids": ["9759-xxxxxx-xxxx", "..."],
        "buyers": ["@handle", "..."],
        "carrier": "UPS|FedEx|USPS|N/A"
      },
      "status": "open|ongoing|resolved|blocked",
      "shipping_state": "label_created|picked_up|in_transit|delivered|delayed|lost|N/A",
      "last_action": "what changed, e.g., 'switched to pickup', 'refund issued', 'reimbursed to seller balance', 'address updated'",
      "last_update": "ISO timestamp or N/A",
      "pending_party": "seller|agent|buyer|carrier|platform|N/A",
      "amounts": {
        "credit_to_seller": "number or null",
        "refund_to_buyer": "number or null"
      },
      "returns_to_previous_topic": "boolean",
      "possible_new_session": "boolean", 
      "keywords": ["refund", "shipment", "pickup", "address change", "claim", "chargeback", "fee", "payout", "bug"],
      "evidence_msg_ch_idx": ["list of msg_ch_idx numbers"]
    }
  ],
  "recent_messages": [
    {
      "msg_ch_idx": "number",
      "sender_id": "sender_id", 
      "role": "role",
      "timestamp": "ISO timestamp",
      "text": "truncated text"
    }
  ],
  "meta": {
    "overlap": "number",
    "channel": "id|url or N/A",
    "time_window": ["start ISO", "end ISO"]
  },
  "guidance": {
    "role_normalization": "map psops/support to 'agent', seller to 'seller', customer to 'buyer'",
    "pronoun_resolution": "prefer the most recent explicit entity when resolving pronouns like 'this order'", 
    "carrier_detection": "detect carrier by tracking pattern (UPS often starts with 1Z; USPS often starts with 9; FedEx typically 12–14 digits, no hyphens)",
    "resolved_status_rule": "only mark 'resolved' if no follow-up on the same topic appears afterwards in the recent window; otherwise keep 'ongoing'"
  }
}
```

<<<INSERT_PREVIOUS_CONTEXT_SUMMARY_BLOCK_HERE>>>

---

## Current chunk messages

The following lines are the messages you must analyze:

```
<<<INSERT_CHUNK_BLOCK_HERE>>>   # each line: msg_ch_idx | sender id | role | timestamp | text
```

---

## Task

Based on the current chunk messages and any previous context, generate a tail summary in **strict JSON format** that will be used as the "Previous context summary" for the next chunk.

**CRITICAL REQUIREMENTS:**
1. Your response MUST be valid JSON only - no markdown, no code blocks, no extra text
2. Follow the exact JSON schema shown in the template above
3. Extract up to 5 unresolved/ongoing cases for `active_case_hints` array
4. Include the last ≤ overlap messages in `recent_messages` array  
5. Fill in `meta` object with overlap, channel, and time_window information
6. Use actual values from the current chunk, not placeholder text

**JSON Response Format:**
```json
{
  "case_anchor_rules": { ... },
  "active_case_hints": [ ... ],
  "recent_messages": [ ... ],
  "meta": { ... },
  "guidance": { ... }
}
```

**IMPORTANT:** Respond with ONLY the JSON object. Do not include any text before or after the JSON.
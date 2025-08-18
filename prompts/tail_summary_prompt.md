You are a senior conversation analyst. Create the **Previous Context Summary** for the next chunk from the current chunk messages and prior context.

### Output JSON Schema (return JSON only)
{
  "case_anchor_rules": {
    "priority_order": "tracking_id > order_id > buyer_handle > topic",
    "multi_order_rule": "If multiple orders share one tracking_id, treat them as ONE 'multi_order_package' unless topics differ",
    "default_scope_rules": "Address change / refund / fee → order-level; payout/app issues → system-level"
  },
  "active_case_hints": [
    {
      "topic": "short title",
      "program": "LDP|UPS|Chargeback|AddressChange|ShippingMode|Refund|Cancellation|FeeAdjustment|Payout|AppBug|DataExport|LiveVisibility|TaxExempt|Other",
      "scope": "single_order|multi_order_package|buyer_account|system",
      "anchor": {
        "tracking": ["..."],
        "order_ids": ["..."],
        "buyers": ["..."],
        "carrier": "UPS|FedEx|USPS|N/A"
      },
      "status": "open|ongoing|resolved|blocked",
      "shipping_state": "label_created|picked_up|in_transit|delivered|delayed|lost|N/A",
      "last_action": "e.g., 'refund issued', 'address updated'",
      "last_update": "ISO timestamp or N/A",
      "pending_party": "seller|agent|buyer|carrier|platform|N/A",
      "amounts": {"credit_to_seller": "number or null", "refund_to_buyer": "number or null"},
      "returns_to_previous_topic": true,
      "possible_new_session": false,
      "keywords": ["refund","shipment","pickup","address change","claim","chargeback","fee","payout","bug"],
      "evidence_msg_ch_idx": [0,1,2]
    }
  ],
  "meta": {
    "overlap": <int>,
    "channel": "string",
    "time_window": ["start ISO","end ISO"]
  },
  "guidance": {
    "role_normalization": "map psops/support→agent, seller→seller, customer→buyer",
    "pronoun_resolution": "prefer most recent explicit entity for pronouns",
    "carrier_detection": "UPS starts 1Z; USPS starts 9; FedEx 12–14 digits",
    "resolved_status_rule": "resolved only if no follow-up appears afterwards"
  }
}

### Rules
- Include ≤5 active cases (status=open/ongoing/blocked) that matter for the next chunk.
- Use anchor priority: tracking_id > order_id > buyer_handle > topic.
- Merge cases sharing the same tracking_id unless clearly different topics.
- Do not invent entities/amounts. Fill last_update from the newest message in the case.

----------------------------
# INPUTS (fill these blocks before sending to the model)

## [INPUT: PREVIOUS_CONTEXT_SUMMARY_JSON]
{PUT_PREVIOUS_CONTEXT_SUMMARY_JSON_HERE}

## [INPUT: CURRENT_CHUNK_MESSAGES]
# Each line: msg_ch_idx | sender id | role | timestamp | text
<<<BEGIN_CURRENT_CHUNK_MESSAGES>>>
PUT_CURRENT_CHUNK_MESSAGE_LINES_HERE
<<<END_CURRENT_CHUNK_MESSAGES>>>

## [INPUT: META]
overlap: PUT_OVERLAP_INT_HERE
channel: "PUT_CHANNEL_ID_OR_URL_OR_NA_HERE"
time_window: ["PUT_START_ISO","PUT_END_ISO"]
----------------------------

### Return
Return **only** the JSON object matching the Output JSON Schema (no extra text).

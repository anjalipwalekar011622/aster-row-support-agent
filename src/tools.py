"""
The order-lookup tool. This is the ONLY code path that ever touches
data/orders.json. The LLM only ever sees what this function returns 
an allow-list of customer-safe fields, never the raw record.
"""

from __future__ import annotations
import json
import re
from typing import Optional, TypedDict


_ORDER_ID_RE = re.compile(r"^ORD-\d{4,}$")

# Statuses where a leftover carrier/ETA snapshot from before the status
# changed must NOT be shown as if delivery is still pending.
_TERMINAL_NO_DELIVERY_STATUSES = {"cancelled", "returned"}


class OrderLookupResult(TypedDict, total=False):
    found: bool
    order_id: str
    membership_tier: str
    items: list[dict]
    placed_at: str
    status: str
    status_updated_at: str
    shipped_at: Optional[str]
    delivered_at: Optional[str]
    carrier: Optional[str]
    tracking_number: Optional[str]
    estimated_delivery: Optional[str]
    customer_safe_message: str
    reason: str  # only present when found=False


class OrderStore:
    """Loads data/orders.json once and answers lookups against it."""

    def __init__(self, orders_path: str):
        with open(orders_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self._by_id = {o["order_id"]: o for o in raw["orders"]}

    @staticmethod
    def normalize_order_id(raw_id: str) -> str:
        """Handle harmless differences: whitespace, case, missing hyphen."""
        cleaned = raw_id.strip().upper()
        cleaned = re.sub(r"[^\w-]", "", cleaned)
        m = re.match(r"^ORD-?(\d{4,})$", cleaned)
        if m:
            return f"ORD-{m.group(1)}"
        return cleaned

    def lookup(self, raw_order_id: str) -> OrderLookupResult:
        order_id = self.normalize_order_id(raw_order_id)

        if not _ORDER_ID_RE.match(order_id):
            return {"found": False, "reason": "malformed_order_id"}

        record = self._by_id.get(order_id)
        if record is None:
            return {"found": False, "reason": "not_found"}

        status = record["status"]
        result: OrderLookupResult = {
            "found": True,
            "order_id": record["order_id"],
            "membership_tier": record["membership_tier"],
            "items": [
                {"name": i["name"], "quantity": i["quantity"], "final_sale": i["final_sale"]}
                for i in record["items"]
            ],
            "placed_at": record["placed_at"],
            "status": status,
            "status_updated_at": record["status_updated_at"],
            "customer_safe_message": record["customer_safe_message"],
        }

        # Trap #1: suppress stale carrier/tracking/ETA on cancelled/returned orders
        if status in _TERMINAL_NO_DELIVERY_STATUSES:
            result["carrier"] = None
            result["tracking_number"] = None
            result["estimated_delivery"] = None
        else:
            result["carrier"] = record.get("carrier")
            result["tracking_number"] = record.get("tracking_number")
            result["estimated_delivery"] = record.get("estimated_delivery")

        result["shipped_at"] = record.get("shipped_at")
        result["delivered_at"] = record.get("delivered_at")

        return result


# Tool schemas: these describe the tool to the LLM in Groq's function-calling format

ORDER_LOOKUP_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "order_lookup",
        "description": (
            "Look up the current status of a customer order by order ID. "
            "Returns only customer-safe fields. Call this whenever the "
            "customer asks about the status, location, or delivery of a "
            "specific order and you have an order ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string", "description": "The order ID, e.g. 'ORD-1007'."}
            },
            "required": ["order_id"],
        },
    },
}

FLAG_HANDOFF_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "flag_for_human_handoff",
        "description": (
            "Call this in addition to the normal answer whenever a human "
            "support specialist should be looped in: sources genuinely "
            "conflict, the knowledge base lacks enough information, an "
            "order lookup failed or returned 'exception' status, the "
            "customer requests an action you cannot complete (refund, "
            "cancellation, replacement, address change, warranty approval), "
            "or the request touches fraud, safety, or another customer's data."
        ),
        "parameters": {
            "type": "object",
            "properties": {"reason": {"type": "string", "description": "One short sentence why."}},
            "required": ["reason"],
        },
    },
}
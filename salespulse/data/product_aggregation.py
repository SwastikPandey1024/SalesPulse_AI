"""
SalesPulse AI - Canonical Product Sales Aggregation & Schema Contract
Defines and executes auditable product-level sales, returns, and volume aggregation
from raw multi-line retail transactions (e.g., UCI Online Retail II, Superstore).
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Union, Iterable
from collections import defaultdict, Counter
import pandas as pd
import numpy as np


@dataclass(frozen=True)
class ProductSalesRecord:
    """Canonical schema for product-level sales aggregation."""
    product_id: str
    description: str
    units_sold: int
    gross_sales: float
    returns: float
    net_sales: float
    order_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


CANONICAL_PRODUCT_COLUMNS = [
    "product_id",
    "description",
    "units_sold",
    "gross_sales",
    "returns",
    "net_sales",
    "order_count",
]


def aggregate_product_sales(
    df: pd.DataFrame,
    product_id_col: str = "StockCode",
    description_col: str = "Description",
    quantity_col: str = "Quantity",
    unit_price_col: str = "UnitPrice",
    invoice_col: str = "InvoiceNo",
    sort_by: str = "net_sales",
    ascending: bool = False,
    top_n: Optional[int] = None,
) -> pd.DataFrame:
    """Aggregate a transactional DataFrame into canonical product-level performance records.

    Business Rules:
    1. Gross Sales: Quantity > 0 and UnitPrice > 0, invoice does NOT start with 'C'.
       Line value = Quantity * UnitPrice.
    2. Returns / Cancellations: invoice starts with 'C' or Quantity < 0, UnitPrice > 0.
       Return value = abs(Quantity) * UnitPrice.
    3. Net Sales: gross_sales - returns.
    4. Units Sold: Sum of positive units dispatched in valid sales.
    5. Order Count: Distinct non-cancellation invoice IDs containing the product.
    6. Missing Descriptions: Fall back to most frequent valid description across transactions,
       or 'Unknown SKU (<product_id>)' if all descriptions are missing.
    7. Quarantined Transactions: UnitPrice <= 0 or NaN are quarantined operational records.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=CANONICAL_PRODUCT_COLUMNS)

    # Normalize column names if standard variations are present
    col_map = {}
    lower_cols = {c.lower(): c for c in df.columns}

    # Map product id
    for candidate in [product_id_col.lower(), "stockcode", "product_id", "productid", "item_id", "sub-category", "sku"]:
        if candidate in lower_cols:
            col_map["product_id"] = lower_cols[candidate]
            break

    # Map description
    for candidate in [description_col.lower(), "description", "product_name", "productname", "item_name", "title"]:
        if candidate in lower_cols:
            col_map["description"] = lower_cols[candidate]
            break

    # Map quantity
    for candidate in [quantity_col.lower(), "quantity", "qty", "units"]:
        if candidate in lower_cols:
            col_map["quantity"] = lower_cols[candidate]
            break

    # Map price
    for candidate in [unit_price_col.lower(), "unitprice", "price", "unit_price", "sales"]:
        if candidate in lower_cols:
            col_map["price"] = lower_cols[candidate]
            break

    # Map invoice
    for candidate in [invoice_col.lower(), "invoiceno", "invoice", "order_id", "orderid"]:
        if candidate in lower_cols:
            col_map["invoice"] = lower_cols[candidate]
            break

    if "product_id" not in col_map:
        raise ValueError(f"Could not identify product identifier column in: {list(df.columns)}")

    pid_col = col_map["product_id"]
    desc_col = col_map.get("description")
    qty_col = col_map.get("quantity")
    prc_col = col_map.get("price")
    inv_col = col_map.get("invoice")

    # Accumulators
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "descriptions": Counter(),
        "units_sold": 0,
        "gross_sales": 0.0,
        "returns": 0.0,
        "net_sales": 0.0,
        "orders": set(),
        "return_orders": set(),
    })

    for _, row in df.iterrows():
        raw_pid = row[pid_col]
        if pd.isna(raw_pid):
            continue
        pid = str(raw_pid).strip()
        if not pid:
            continue

        # Quantity and Price
        try:
            qty = float(row[qty_col]) if qty_col and pd.notna(row[qty_col]) else 1.0
        except (ValueError, TypeError):
            qty = 0.0

        try:
            price = float(row[prc_col]) if prc_col and pd.notna(row[prc_col]) else 0.0
        except (ValueError, TypeError):
            price = 0.0

        # Quarantined non-commercial price
        if price <= 0.0:
            continue

        # Description
        raw_desc = row[desc_col] if desc_col and pd.notna(row[desc_col]) else None
        clean_desc = str(raw_desc).strip() if raw_desc else ""
        if clean_desc and clean_desc.lower() not in ("nan", "none", "null", "?", ""):
            stats[pid]["descriptions"][clean_desc] += 1

        # Invoice ID
        inv = str(row[inv_col]).strip() if inv_col and pd.notna(row[inv_col]) else ""
        is_cancellation = inv.startswith("C") or qty < 0

        if is_cancellation:
            ret_val = abs(qty) * price
            stats[pid]["returns"] += ret_val
            stats[pid]["net_sales"] -= ret_val
            if inv:
                stats[pid]["return_orders"].add(inv)
        else:
            gross_val = qty * price
            stats[pid]["gross_sales"] += gross_val
            stats[pid]["net_sales"] += gross_val
            stats[pid]["units_sold"] += int(qty)
            if inv:
                stats[pid]["orders"].add(inv)

    records: List[Dict[str, Any]] = []
    for pid, s in stats.items():
        # Resolve description: most common non-empty, else fallback
        if s["descriptions"]:
            best_desc = s["descriptions"].most_common(1)[0][0]
        else:
            best_desc = f"Unknown SKU ({pid})"

        records.append({
            "product_id": pid,
            "description": best_desc,
            "units_sold": s["units_sold"],
            "gross_sales": round(s["gross_sales"], 2),
            "returns": round(s["returns"], 2),
            "net_sales": round(s["net_sales"], 2),
            "order_count": len(s["orders"]),
        })

    if not records:
        return pd.DataFrame(columns=CANONICAL_PRODUCT_COLUMNS)

    res_df = pd.DataFrame(records)
    if sort_by in res_df.columns:
        res_df = res_df.sort_values(sort_by, ascending=ascending)
    if top_n is not None and top_n > 0:
        res_df = res_df.head(top_n)

    return res_df.reset_index(drop=True)

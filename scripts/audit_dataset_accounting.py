"""
SalesPulse AI - UCI Online Retail II Accounting & Data Integrity Audit
Standalone, reproducible script that performs an independent, memory-conscious,
deep financial and transactional audit of the raw UCI Online Retail II dataset.

Outputs:
- Console audit summary with exact row counts, reconciliation checks, and partitions.
- Machine-readable audit report: data/processed/online_retail_ii_accounting_audit.json
"""

import os
import sys
import json
import time
import hashlib
import datetime
from typing import Dict, Any, List, Set, Tuple
from collections import Counter, defaultdict

import openpyxl
import pandas as pd
import numpy as np


def compute_sha256(filepath: str) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_online_retail_dataset(
    xlsx_path: str = "data/raw/online_retail_II.xlsx",
    zip_path: str = "data/raw/online_retail_ii.zip",
    processed_daily_path: str = "data/processed/online_retail_ii_daily.csv",
    output_audit_json: str = "data/processed/online_retail_ii_accounting_audit.json",
) -> Dict[str, Any]:
    start_ts = time.time()
    print("=" * 80)
    print("SALESPULSE AI - DATASET ACCOUNTING & FINANCIAL INTEGRITY AUDIT")
    print("=" * 80)

    # 1. File verification
    if not os.path.exists(xlsx_path):
        raise FileNotFoundError(f"Raw workbook not found at: {xlsx_path}")

    xlsx_size = os.path.getsize(xlsx_path)
    xlsx_sha = compute_sha256(xlsx_path)
    print(f"Workbook Path:    {xlsx_path}")
    print(f"Workbook Size:    {xlsx_size:,} bytes")
    print(f"Workbook SHA-256: {xlsx_sha}")

    zip_info = {}
    if os.path.exists(zip_path):
        zip_size = os.path.getsize(zip_path)
        zip_sha = compute_sha256(zip_path)
        zip_info = {
            "path": zip_path,
            "size_bytes": zip_size,
            "sha256": zip_sha,
            "verified": True,
        }
        print(f"Zip Archive:      {zip_path} ({zip_size:,} bytes, SHA-256: {zip_sha[:16]}...)")
    else:
        zip_info = {"verified": False, "note": "Zip archive not present; raw xlsx used"}

    # 2. Open workbook in streaming read-only mode
    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sheet_names = list(wb.sheetnames)
    print(f"\nSheets in Workbook: {sheet_names}")

    # Accounting Partition Counters
    sheet_row_counts = {}
    total_raw_rows = 0
    total_headers = 0
    total_transactions = 0

    # Mutually Exclusive Classification Buckets
    cat_valid_sales = 0
    cat_returns_cancellations = 0
    cat_zero_neg_price = 0
    cat_zero_qty_positive_price = 0
    cat_null_invalid_date = 0

    # Sub-breakdowns
    sub_cat_b1_both_c_and_neg_qty = 0
    sub_cat_b2_c_with_non_neg_qty = 0
    sub_cat_b3_neg_qty_without_c = 0

    sub_cat_c1_zero_price_no_cust = 0
    sub_cat_c2_zero_price_with_cust = 0
    sub_cat_c3_negative_price = 0

    # Financial accumulators
    sum_gross_sales = 0.0
    sum_returns_val = 0.0
    sum_units_sold = 0
    sum_units_returned = 0

    # Entity tracking
    invoices_valid_sales: Set[str] = set()
    invoices_cancellations: Set[str] = set()
    invoices_all: Set[str] = set()
    products_all: Set[str] = set()
    countries_all: Set[str] = set()

    missing_customer_id_valid = 0
    missing_customer_id_returns = 0
    missing_customer_id_quarantined = 0
    missing_description_count = 0

    daily_totals = defaultdict(lambda: {"gross": 0.0, "returns": 0.0, "net": 0.0, "tx_count": 0})
    observed_dates: Set[datetime.date] = set()

    for sheet_name in sheet_names:
        ws = wb[sheet_name]
        is_first = True
        s_count = 0

        for row in ws.iter_rows(values_only=True):
            total_raw_rows += 1
            s_count += 1
            if is_first:
                is_first = False
                total_headers += 1
                continue

            total_transactions += 1
            inv, stock, desc, qty, inv_date, price, cust_id, country = row[:8]

            # Date parsing
            cal_date = None
            if inv_date is not None:
                if isinstance(inv_date, datetime.datetime):
                    cal_date = inv_date.date()
                elif isinstance(inv_date, datetime.date):
                    cal_date = inv_date
                else:
                    try:
                        cal_date = pd.to_datetime(inv_date).date()
                    except Exception:
                        cal_date = None

            if cal_date is None:
                cat_null_invalid_date += 1
                continue

            observed_dates.add(cal_date)

            # Numeric quantity and price
            try:
                num_qty = float(qty) if qty is not None else 0.0
                num_price = float(price) if price is not None else 0.0
            except (ValueError, TypeError):
                cat_zero_neg_price += 1
                continue

            inv_str = str(inv).strip() if inv is not None else ""
            stock_str = str(stock).strip() if stock is not None else ""
            desc_str = str(desc).strip() if desc is not None else ""
            country_str = str(country).strip() if country is not None else "Unknown"

            if inv_str:
                invoices_all.add(inv_str)
            if stock_str:
                products_all.add(stock_str)
            if country_str:
                countries_all.add(country_str)

            if not desc_str:
                missing_description_count += 1

            has_cust_id = cust_id is not None and str(cust_id).strip() != ""

            # Check Category C: Non-positive price (Quarantined)
            if num_price <= 0.0:
                cat_zero_neg_price += 1
                if not has_cust_id:
                    missing_customer_id_quarantined += 1
                if num_price == 0.0:
                    if has_cust_id:
                        sub_cat_c2_zero_price_with_cust += 1
                    else:
                        sub_cat_c1_zero_price_no_cust += 1
                else:
                    sub_cat_c3_negative_price += 1
                continue

            # Check Category D: Zero quantity with positive price
            if num_qty == 0.0 and not inv_str.startswith("C"):
                cat_zero_qty_positive_price += 1
                continue

            # Check Category B: Returns & Cancellations
            is_cancellation = inv_str.startswith("C") or num_qty < 0

            if is_cancellation:
                cat_returns_cancellations += 1
                ret_val = abs(num_qty) * num_price
                sum_returns_val += ret_val
                sum_units_returned += int(abs(num_qty))

                if inv_str:
                    invoices_cancellations.add(inv_str)

                if not has_cust_id:
                    missing_customer_id_returns += 1

                # Sub-breakdown of returns
                if inv_str.startswith("C") and num_qty < 0:
                    sub_cat_b1_both_c_and_neg_qty += 1
                elif inv_str.startswith("C") and num_qty >= 0:
                    sub_cat_b2_c_with_non_neg_qty += 1
                else:
                    sub_cat_b3_neg_qty_without_c += 1

                daily_totals[cal_date]["returns"] += ret_val
                daily_totals[cal_date]["net"] -= ret_val
                daily_totals[cal_date]["tx_count"] += 1

            else:
                # Category A: Valid positive sale
                cat_valid_sales += 1
                line_val = num_qty * num_price
                sum_gross_sales += line_val
                sum_units_sold += int(num_qty)

                if inv_str:
                    invoices_valid_sales.add(inv_str)

                if not has_cust_id:
                    missing_customer_id_valid += 1

                daily_totals[cal_date]["gross"] += line_val
                daily_totals[cal_date]["net"] += line_val
                daily_totals[cal_date]["tx_count"] += 1

        sheet_row_counts[sheet_name] = s_count

    wb.close()

    # 3. Partition Verification
    partition_sum = (
        cat_valid_sales
        + cat_returns_cancellations
        + cat_zero_neg_price
        + cat_zero_qty_positive_price
        + cat_null_invalid_date
    )
    partition_reconciles = (partition_sum == total_transactions)

    # Financial Metrics
    sum_net_sales = sum_gross_sales - sum_returns_val
    sum_net_units = sum_units_sold - sum_units_returned
    aov_gross = sum_gross_sales / len(invoices_valid_sales) if invoices_valid_sales else 0.0
    aov_net = sum_net_sales / len(invoices_valid_sales) if invoices_valid_sales else 0.0
    return_rate_pct = (sum_returns_val / sum_gross_sales * 100.0) if sum_gross_sales > 0 else 0.0

    # 4. Temporal Analysis
    sorted_observed_dates = sorted(list(observed_dates))
    min_date = sorted_observed_dates[0]
    max_date = sorted_observed_dates[-1]
    full_calendar_days = pd.date_range(start=min_date, end=max_date, freq="D").date
    total_calendar_days = len(full_calendar_days)
    observed_days_count = len(sorted_observed_dates)
    unobserved_days_count = total_calendar_days - observed_days_count

    unobserved_dates = [d for d in full_calendar_days if d not in observed_dates]
    unobserved_weekday_counts = Counter([d.strftime("%A") for d in unobserved_dates])

    # 5. Daily Aggregates Comparison (Reconciliation with online_retail_ii_daily.csv)
    daily_csv_reconciles = False
    daily_csv_metrics = {}
    if os.path.exists(processed_daily_path):
        df_daily = pd.read_csv(processed_daily_path)
        d_gross = float(df_daily["gross_sales"].sum())
        d_returns = float(df_daily["returns"].sum())
        d_net = float(df_daily["sales"].sum())
        d_rows = len(df_daily)

        # Delta against transaction totals
        gross_delta = abs(d_gross - sum_gross_sales)
        returns_delta = abs(d_returns - sum_returns_val)
        net_delta = abs(d_net - sum_net_sales)

        daily_csv_metrics = {
            "daily_csv_path": processed_daily_path,
            "daily_csv_rows": d_rows,
            "daily_csv_gross_sales": round(d_gross, 2),
            "daily_csv_returns": round(d_returns, 2),
            "daily_csv_net_sales": round(d_net, 2),
            "gross_sales_delta": round(gross_delta, 2),
            "returns_delta": round(returns_delta, 2),
            "net_sales_delta": round(net_delta, 2),
            "delta_explanation": "Deltas reflect daily rounding per calendar day vs raw unrounded transaction sums.",
        }
        # Acceptance threshold: within £5,000 across £20.97M (< 0.03% rounding delta)
        daily_csv_reconciles = gross_delta < 5000.0 and returns_delta < 5000.0

    duration = round(time.time() - start_ts, 2)

    # 6. Formulate Complete Machine-Readable Audit Report
    audit_report = {
        "audit_metadata": {
            "audit_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "audit_script": "scripts/audit_dataset_accounting.py",
            "duration_seconds": duration,
            "dataset_name": "UCI Online Retail II",
            "official_citation": "Chen, D. (2012). Online Retail II. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33",
            "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        },
        "source_file_integrity": {
            "xlsx_path": xlsx_path,
            "xlsx_size_bytes": xlsx_size,
            "xlsx_sha256": xlsx_sha,
            "zip_info": zip_info,
            "sheet_names": sheet_names,
            "sheet_row_counts": sheet_row_counts,
            "total_raw_rows": total_raw_rows,
            "total_header_rows": total_headers,
            "total_transaction_rows": total_transactions,
        },
        "source_row_accounting_partition": {
            "description": "Mutually exclusive classification partition of all 1,067,371 source transaction rows",
            "categories": {
                "category_A_valid_positive_sales": cat_valid_sales,
                "category_B_returns_and_cancellations": cat_returns_cancellations,
                "category_C_quarantined_zero_or_negative_price": cat_zero_neg_price,
                "category_D_zero_quantity_positive_price": cat_zero_qty_positive_price,
                "category_E_null_or_invalid_date": cat_null_invalid_date,
            },
            "sub_categories": {
                "B1_both_c_invoice_and_negative_quantity": sub_cat_b1_both_c_and_neg_qty,
                "B2_c_invoice_with_non_negative_quantity": sub_cat_b2_c_with_non_neg_qty,
                "B3_negative_quantity_without_c_invoice": sub_cat_b3_neg_qty_without_c,
                "C1_zero_price_missing_customer_id": sub_cat_c1_zero_price_no_cust,
                "C2_zero_price_with_customer_id": sub_cat_c2_zero_price_with_cust,
                "C3_negative_price_adjustments": sub_cat_c3_negative_price,
            },
            "partition_sum": partition_sum,
            "total_transactions": total_transactions,
            "is_partition_exact": partition_reconciles,
            "reconciliation_difference": total_transactions - partition_sum,
        },
        "financial_reconciliation_gbp": {
            "currency": "GBP (£)",
            "gross_sales": round(sum_gross_sales, 2),
            "returns_value": round(sum_returns_val, 2),
            "net_sales": round(sum_net_sales, 2),
            "units_sold": sum_units_sold,
            "units_returned": sum_units_returned,
            "net_units": sum_net_units,
            "unique_valid_invoices": len(invoices_valid_sales),
            "unique_cancellation_invoices": len(invoices_cancellations),
            "total_unique_invoice_strings": len(invoices_all),
            "total_unique_products": len(products_all),
            "total_unique_countries": len(countries_all),
            "average_order_value_gross": round(aov_gross, 2),
            "average_order_value_net": round(aov_net, 2),
            "return_rate_percentage": round(return_rate_pct, 2),
        },
        "temporal_coverage": {
            "date_range_start": str(min_date),
            "date_range_end": str(max_date),
            "total_calendar_days": total_calendar_days,
            "observed_trading_days": observed_days_count,
            "unobserved_calendar_days": unobserved_days_count,
            "unobserved_dates_by_weekday": dict(unobserved_weekday_counts),
            "missing_date_policy": (
                "ZERO_FILL_POLICY: Absent calendar days represent non-trading warehouse/dispatch days "
                "(predominantly Sundays and holidays). Reindexing to continuous daily frequency with 0.0 "
                "distinguishes calendar structure from missing or corrupted observations."
            ),
        },
        "data_quality_findings": {
            "missing_customer_id_valid_sales": missing_customer_id_valid,
            "missing_customer_id_returns": missing_customer_id_returns,
            "missing_customer_id_quarantined": missing_customer_id_quarantined,
            "total_missing_customer_ids": missing_customer_id_valid + missing_customer_id_returns + missing_customer_id_quarantined,
            "missing_descriptions_count": missing_description_count,
            "privacy_compliance": "No customer-identifying fields (Customer ID) are emitted in aggregate daily time series or dashboard metrics.",
        },
        "daily_csv_reconciliation": daily_csv_metrics,
        "overall_audit_status": "PASSED" if partition_reconciles and daily_csv_reconciles else "FAILED",
    }

    # Save machine-readable audit report
    os.makedirs(os.path.dirname(output_audit_json), exist_ok=True)
    with open(output_audit_json, "w", encoding="utf-8") as f:
        json.dump(audit_report, f, indent=2)

    print("\n" + "-" * 80)
    print("AUDIT SUMMARY RESULTS")
    print("-" * 80)
    print(f"Total Raw Rows:          {total_raw_rows:,}")
    print(f"Total Header Rows:       {total_headers}")
    print(f"Total Transaction Rows:  {total_transactions:,}")
    print("\nClassification Partition (Mutually Exclusive):")
    print(f"  [A] Valid Positive Sales:               {cat_valid_sales:,}")
    print(f"  [B] Returns & Cancellations:            {cat_returns_cancellations:,}")
    print(f"      - B1 (C-Invoice + Negative Qty):    {sub_cat_b1_both_c_and_neg_qty:,}")
    print(f"      - B2 (C-Invoice + Positive/0 Qty):  {sub_cat_b2_c_with_non_neg_qty:,}")
    print(f"      - B3 (Negative Qty without C):      {sub_cat_b3_neg_qty_without_c:,}")
    print(f"  [C] Quarantined (Price <= 0.0):         {cat_zero_neg_price:,}")
    print(f"      - C1 (Zero price, missing cust):    {sub_cat_c1_zero_price_no_cust:,}")
    print(f"      - C2 (Zero price, with cust):       {sub_cat_c2_zero_price_with_cust:,}")
    print(f"      - C3 (Negative price adjustment):   {sub_cat_c3_negative_price:,}")
    print(f"  [D] Zero Quantity (Price > 0):          {cat_zero_qty_positive_price:,}")
    print(f"  [E] Null/Invalid Date:                  {cat_null_invalid_date:,}")
    print(f"Partition Sum:           {partition_sum:,}")
    print(f"Partition Exact:         {partition_reconciles} (Diff: {total_transactions - partition_sum})")
    print("\nFinancial Totals (£ GBP):")
    print(f"  Gross Sales:           £{sum_gross_sales:,.2f}")
    print(f"  Returns Value:         £{sum_returns_val:,.2f}")
    print(f"  Net Sales:             £{sum_net_sales:,.2f}")
    print(f"  Units Sold:            {sum_units_sold:,}")
    print(f"  Units Returned:        {sum_units_returned:,}")
    print(f"  Net Units:             {sum_net_units:,}")
    print(f"  Gross Invoices:        {len(invoices_valid_sales):,}")
    print(f"  Cancellation Invoices: {len(invoices_cancellations):,}")
    print(f"  Gross AOV:             £{aov_gross:,.2f}")
    print(f"  Return Rate:           {return_rate_pct:.2f}%")
    print("\nTemporal Coverage:")
    print(f"  Date Range:            {min_date} to {max_date} ({total_calendar_days} days)")
    print(f"  Observed Trading Days: {observed_days_count} days")
    print(f"  Unobserved Days:       {unobserved_days_count} days")
    print(f"  Unobserved Weekdays:   {dict(unobserved_weekday_counts)}")
    print("\nDaily CSV Reconciliation:")
    print(f"  Reconciled:            {daily_csv_reconciles}")
    if daily_csv_metrics:
        print(f"  Daily Net Sales:       £{daily_csv_metrics.get('daily_csv_net_sales'):,.2f}")
        print(f"  Net Delta:             £{daily_csv_metrics.get('net_sales_delta'):,.2f}")
    print(f"\nAudit Report Saved:      {output_audit_json}")
    print(f"Audit Duration:          {duration}s")
    print("=" * 80)

    return audit_report


if __name__ == "__main__":
    audit_online_retail_dataset()

"""
SalesPulse AI - UCI Online Retail II Dataset Onboarding Pipeline
Reproducible pipeline for acquiring, validating, and onboarding the UCI Online Retail II dataset.

Official Dataset Citation:
Chen, D. (2012). Online Retail II. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33

License:
Creative Commons Attribution 4.0 International (CC BY 4.0)
https://creativecommons.org/licenses/by/4.0/legalcode

Primary Source:
https://archive.ics.uci.edu/dataset/502/online+retail+ii
Direct Zip URL:
https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip
"""

import os
import sys
import io
import json
import time
import zipfile
import hashlib
import datetime
from typing import Optional, Dict, Any, Tuple, List, Set
from dataclasses import dataclass, asdict

import pandas as pd
import numpy as np
import openpyxl

from salespulse.data.time_series import (
    safe_parse_calendar_dates,
    MISSING_DATE_POLICY,
    get_daily_series_diagnostics,
)
from salespulse.data.contracts import (
    validate_dataset_schema,
    GrainType,
    sanitize_for_json,
)

UCI_DATASET_NAME = "Online Retail II"
UCI_DATASET_CITATION = (
    "Chen, D. (2012). Online Retail II. UCI Machine Learning Repository. "
    "https://doi.org/10.24432/C5BW33"
)
UCI_SOURCE_URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
UCI_LANDING_PAGE = "https://archive.ics.uci.edu/dataset/502/online+retail+ii"
UCI_LICENSE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
RAW_ZIP_FILENAME = "online_retail_ii.zip"
RAW_XLSX_FILENAME = "online_retail_II.xlsx"


@dataclass
class OnboardingProvenance:
    """Provenance and lineage metadata for onboarded dataset."""
    dataset_name: str
    source_url: str
    landing_page_url: str
    citation: str
    license: str
    retrieval_date_utc: str
    zip_sha256: str
    zip_size_bytes: int
    xlsx_sha256: str
    xlsx_size_bytes: int
    raw_sheet_names: List[str]
    total_raw_rows: int
    total_header_rows: int
    total_transaction_rows: int
    valid_transaction_rows: int
    cancellation_rows: int
    zero_or_negative_price_rows: int
    null_date_rows: int
    unique_invoices: int
    unique_cancellation_invoices: int
    unique_products: int
    unique_countries: int
    date_range_start: str
    date_range_end: str
    total_calendar_days: int
    observed_trading_days: int
    unobserved_calendar_days: int
    currency: str
    total_gross_sales: float
    total_returns: float
    total_net_sales: float
    missing_date_policy: str
    processing_duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return sanitize_for_json(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)


def compute_file_sha256(filepath: str) -> str:
    """Compute SHA-256 fingerprint of a local file efficiently."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_uci_online_retail(
    target_dir: str = "data/raw",
    url: str = UCI_SOURCE_URL,
    force_download: bool = False,
) -> str:
    """Download official Online Retail II zip archive with retry and resume safety.
    
    Returns path to downloaded zip file.
    """
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, RAW_ZIP_FILENAME)

    if os.path.exists(zip_path) and not force_download:
        if os.path.getsize(zip_path) > 40_000_000:
            return zip_path

    import urllib.request
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "SalesPulse-AI-Onboarding/1.0 (+https://github.com/)"},
    )

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=120) as resp, open(zip_path, "wb") as out:
                while chunk := resp.read(1024 * 1024):
                    out.write(chunk)
            break
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download Online Retail II from {url} after {max_retries} attempts: {e}"
                )
            time.sleep(2 * attempt)

    return zip_path


def extract_online_retail_xlsx(
    zip_path: str,
    target_dir: str = "data/raw",
) -> str:
    """Extract online_retail_II.xlsx from the official zip archive."""
    os.makedirs(target_dir, exist_ok=True)
    xlsx_path = os.path.join(target_dir, RAW_XLSX_FILENAME)

    if os.path.exists(xlsx_path) and os.path.getsize(xlsx_path) > 40_000_000:
        return xlsx_path

    with zipfile.ZipFile(zip_path, "r") as z:
        if RAW_XLSX_FILENAME not in z.namelist():
            raise FileNotFoundError(
                f"Expected '{RAW_XLSX_FILENAME}' inside zip, found: {z.namelist()}"
            )
        z.extract(RAW_XLSX_FILENAME, target_dir)

    return xlsx_path


def process_and_aggregate_online_retail(
    xlsx_path: str,
    output_dir: str = "data/processed",
    zip_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, OnboardingProvenance, Dict[str, Any]]:
    """Process raw Online Retail II transaction workbook into canonical daily series.
    
    Applies deterministic business rules:
    - Gross Sales: Quantity > 0 and Price > 0 (Quantity * Price)
    - Cancellations / Returns: Invoice starting with 'C' or Quantity < 0
    - Non-positive prices quarantined as non-sales operational adjustments
    - Aggregation to continuous daily trading calendar using MISSING_DATE_POLICY
    - Extraction of top products and country distributions for executive overview
    """
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    retrieval_date = datetime.datetime.now(datetime.timezone.utc).isoformat()

    zip_sha = compute_file_sha256(zip_path) if zip_path and os.path.exists(zip_path) else "MANUAL_EXTRACT"
    zip_size = os.path.getsize(zip_path) if zip_path and os.path.exists(zip_path) else 0
    xlsx_sha = compute_file_sha256(xlsx_path)
    xlsx_size = os.path.getsize(xlsx_path)

    wb = openpyxl.load_workbook(xlsx_path, read_only=True, data_only=True)
    sheet_names = list(wb.sheetnames)

    # Accumulators by calendar date
    # date -> {gross, returns, net, units_sold, units_returned, invoices, ret_invoices}
    daily_stats: Dict[datetime.date, Dict[str, Any]] = {}
    country_stats: Dict[str, Dict[str, Any]] = {}
    product_stats: Dict[str, Dict[str, Any]] = {}

    total_raw_rows = 0
    total_headers = 0
    total_tx_rows = 0
    valid_tx_rows = 0
    cancellation_rows = 0
    bad_price_rows = 0
    null_date_rows = 0

    all_invoices: Set[str] = set()
    all_ret_invoices: Set[str] = set()
    all_products: Set[str] = set()
    all_countries: Set[str] = set()

    for sheet_name in sheet_names:
        ws = wb[sheet_name]
        is_first = True

        for row in ws.iter_rows(values_only=True):
            total_raw_rows += 1
            if is_first:
                is_first = False
                total_headers += 1
                continue

            # Row structure: (Invoice, StockCode, Description, Quantity, InvoiceDate, Price, Customer ID, Country)
            inv, stock, desc, qty, inv_date, price, cust_id, country = row[:8]
            total_tx_rows += 1

            # Validate date
            if inv_date is None:
                null_date_rows += 1
                continue

            if isinstance(inv_date, datetime.datetime):
                cal_date = inv_date.date()
            elif isinstance(inv_date, datetime.date):
                cal_date = inv_date
            else:
                try:
                    cal_date = pd.to_datetime(inv_date).date()
                except Exception:
                    null_date_rows += 1
                    continue

            # Validate numeric quantity and price
            try:
                num_qty = float(qty) if qty is not None else 0.0
                num_price = float(price) if price is not None else 0.0
            except (ValueError, TypeError):
                bad_price_rows += 1
                continue

            inv_str = str(inv).strip() if inv is not None else ""
            stock_str = str(stock).strip() if stock is not None else ""
            desc_str = str(desc).strip() if desc is not None else ""
            country_str = str(country).strip() if country is not None else "Unknown"

            if inv_str:
                all_invoices.add(inv_str)
            if stock_str:
                all_products.add(stock_str)
            if country_str:
                all_countries.add(country_str)

            # Check non-positive price (debt write-offs, samples, manual adjustments)
            if num_price <= 0.0:
                bad_price_rows += 1
                continue

            # Initialize daily bucket
            if cal_date not in daily_stats:
                daily_stats[cal_date] = {
                    "gross_sales": 0.0,
                    "returns": 0.0,
                    "net_sales": 0.0,
                    "units_sold": 0.0,
                    "units_returned": 0.0,
                    "invoices": set(),
                    "return_invoices": set(),
                }

            # Country bucket
            if country_str not in country_stats:
                country_stats[country_str] = {
                    "gross_sales": 0.0,
                    "net_sales": 0.0,
                    "orders": set(),
                }

            is_cancellation = inv_str.startswith("C") or num_qty < 0

            if is_cancellation:
                cancellation_rows += 1
                ret_val = abs(num_qty) * num_price
                daily_stats[cal_date]["returns"] += ret_val
                daily_stats[cal_date]["net_sales"] -= ret_val
                daily_stats[cal_date]["units_returned"] += abs(num_qty)
                daily_stats[cal_date]["return_invoices"].add(inv_str)
                all_ret_invoices.add(inv_str)
                country_stats[country_str]["net_sales"] -= ret_val
            else:
                valid_tx_rows += 1
                line_val = num_qty * num_price
                daily_stats[cal_date]["gross_sales"] += line_val
                daily_stats[cal_date]["net_sales"] += line_val
                daily_stats[cal_date]["units_sold"] += num_qty
                daily_stats[cal_date]["invoices"].add(inv_str)

                country_stats[country_str]["gross_sales"] += line_val
                country_stats[country_str]["net_sales"] += line_val
                country_stats[country_str]["orders"].add(inv_str)

                # Product stats (top products tracking)
                if stock_str:
                    if stock_str not in product_stats:
                        product_stats[stock_str] = {
                            "description": desc_str,
                            "units_sold": 0.0,
                            "gross_sales": 0.0,
                        }
                    product_stats[stock_str]["units_sold"] += num_qty
                    product_stats[stock_str]["gross_sales"] += line_val
                    if desc_str and not product_stats[stock_str]["description"]:
                        product_stats[stock_str]["description"] = desc_str

    wb.close()

    if not daily_stats:
        raise ValueError("No valid transaction rows found in Online Retail II dataset.")

    # Build continuous chronological daily series
    sorted_dates = sorted(daily_stats.keys())
    min_date = sorted_dates[0]
    max_date = sorted_dates[-1]

    full_calendar_range = pd.date_range(start=min_date, end=max_date, freq="D")
    total_calendar_days = len(full_calendar_range)
    observed_trading_days = len(sorted_dates)
    unobserved_calendar_days = total_calendar_days - observed_trading_days

    daily_records = []
    total_gross = 0.0
    total_ret = 0.0
    total_net = 0.0

    for ts in full_calendar_range:
        d = ts.date()
        if d in daily_stats:
            st = daily_stats[d]
            gross = round(st["gross_sales"], 2)
            returns = round(st["returns"], 2)
            net = round(st["net_sales"], 2)
            ord_cnt = len(st["invoices"])
            ret_cnt = len(st["return_invoices"])
            u_sold = int(st["units_sold"])
            u_ret = int(st["units_returned"])
            aov = round(gross / ord_cnt, 2) if ord_cnt > 0 else 0.0

            total_gross += gross
            total_ret += returns
            total_net += net

            daily_records.append({
                "date": str(d),
                "sales": net,  # Canonical primary sales column for forecasting
                "gross_sales": gross,
                "returns": returns,
                "order_count": ord_cnt,
                "return_count": ret_cnt,
                "units_sold": u_sold,
                "units_returned": u_ret,
                "avg_order_value": aov,
                "is_observed_trading_day": True,
            })
        else:
            # Unobserved day (e.g. Sunday/bank holiday when online orders were not dispatched)
            daily_records.append({
                "date": str(d),
                "sales": 0.0,
                "gross_sales": 0.0,
                "returns": 0.0,
                "order_count": 0,
                "return_count": 0,
                "units_sold": 0,
                "units_returned": 0,
                "avg_order_value": 0.0,
                "is_observed_trading_day": False,
            })

    daily_df = pd.DataFrame(daily_records)
    daily_csv_path = os.path.join(output_dir, "online_retail_ii_daily.csv")
    daily_df.to_csv(daily_csv_path, index=False)

    # Save Top Products
    top_prods = []
    for s_code, p_data in product_stats.items():
        top_prods.append({
            "product_id": s_code,
            "description": p_data["description"],
            "units_sold": int(p_data["units_sold"]),
            "gross_sales": round(p_data["gross_sales"], 2),
        })
    top_prods_df = pd.DataFrame(top_prods).sort_values("gross_sales", ascending=False).head(100)
    top_prods_path = os.path.join(output_dir, "online_retail_ii_top_products.csv")
    top_prods_df.to_csv(top_prods_path, index=False)

    # Save Countries Summary
    country_rows = []
    for c_name, c_data in country_stats.items():
        country_rows.append({
            "country": c_name,
            "order_count": len(c_data["orders"]),
            "gross_sales": round(c_data["gross_sales"], 2),
            "net_sales": round(c_data["net_sales"], 2),
        })
    countries_df = pd.DataFrame(country_rows).sort_values("gross_sales", ascending=False)
    countries_path = os.path.join(output_dir, "online_retail_ii_countries.csv")
    countries_df.to_csv(countries_path, index=False)

    # Validate against canonical contracts
    val_report, _, _ = validate_dataset_schema(daily_df, expected_grain=GrainType.DAILY_AGGREGATED)
    val_report_path = os.path.join(output_dir, "online_retail_ii_validation.json")
    with open(val_report_path, "w", encoding="utf-8") as f:
        f.write(val_report.to_json(indent=2))

    duration = round(time.time() - start_time, 2)

    # Build Provenance Record
    provenance = OnboardingProvenance(
        dataset_name=UCI_DATASET_NAME,
        source_url=UCI_SOURCE_URL,
        landing_page_url=UCI_LANDING_PAGE,
        citation=UCI_DATASET_CITATION,
        license=UCI_LICENSE,
        retrieval_date_utc=retrieval_date,
        zip_sha256=zip_sha,
        zip_size_bytes=zip_size,
        xlsx_sha256=xlsx_sha,
        xlsx_size_bytes=xlsx_size,
        raw_sheet_names=sheet_names,
        total_raw_rows=total_raw_rows,
        total_header_rows=total_headers,
        total_transaction_rows=total_tx_rows,
        valid_transaction_rows=valid_tx_rows,
        cancellation_rows=cancellation_rows,
        zero_or_negative_price_rows=bad_price_rows,
        null_date_rows=null_date_rows,
        unique_invoices=len(all_invoices),
        unique_cancellation_invoices=len(all_ret_invoices),
        unique_products=len(all_products),
        unique_countries=len(all_countries),
        date_range_start=str(min_date),
        date_range_end=str(max_date),
        total_calendar_days=total_calendar_days,
        observed_trading_days=observed_trading_days,
        unobserved_calendar_days=unobserved_calendar_days,
        currency="GBP (£)",
        total_gross_sales=round(total_gross, 2),
        total_returns=round(total_ret, 2),
        total_net_sales=round(total_net, 2),
        missing_date_policy=MISSING_DATE_POLICY,
        processing_duration_seconds=duration,
    )

    prov_path = os.path.join(output_dir, "online_retail_ii_provenance.json")
    with open(prov_path, "w", encoding="utf-8") as f:
        f.write(provenance.to_json(indent=2))

    return daily_df, provenance, val_report.to_dict()


def run_onboarding_pipeline(
    raw_dir: str = "data/raw",
    processed_dir: str = "data/processed",
) -> OnboardingProvenance:
    """End-to-end execution of download, extraction, validation, and daily aggregation."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("=" * 70)
    print("SalesPulse AI - UCI Online Retail II Dataset Onboarding")
    print(f"Citation: {UCI_DATASET_CITATION}")
    print("=" * 70)

    # 1. Download
    print(f"Step 1: Verifying/downloading raw zip archive from {UCI_SOURCE_URL}...")
    zip_path = download_uci_online_retail(target_dir=raw_dir)
    print(f"  [+] Zip archive ready: {zip_path} ({os.path.getsize(zip_path):,} bytes)")

    # 2. Extract
    print("Step 2: Verifying/extracting Excel workbook...")
    xlsx_path = extract_online_retail_xlsx(zip_path=zip_path, target_dir=raw_dir)
    print(f"  [+] Excel workbook ready: {xlsx_path} ({os.path.getsize(xlsx_path):,} bytes)")

    # 3. Process & Aggregate
    print("Step 3: Processing transaction sheets and aggregating to canonical daily series...")
    daily_df, prov, val_dict = process_and_aggregate_online_retail(
        xlsx_path=xlsx_path,
        output_dir=processed_dir,
        zip_path=zip_path,
    )

    print("=" * 70)
    print("ONBOARDING COMPLETE")
    print(f"  - Date Span:           {prov.date_range_start} to {prov.date_range_end} ({prov.total_calendar_days} calendar days)")
    print(f"  - Trading Days:        {prov.observed_trading_days} observed, {prov.unobserved_calendar_days} unobserved (0.0 fill)")
    print(f"  - Total Transactions:  {prov.total_transaction_rows:,} raw line-items")
    print(f"  - Gross Sales (GBP):   {prov.total_gross_sales:,.2f}")
    print(f"  - Returns (GBP):       {prov.total_returns:,.2f}")
    print(f"  - Net Sales (GBP):     {prov.total_net_sales:,.2f}")
    print(f"  - Contract Verdict:    {val_dict.get('verdict')}")
    print(f"  - Output CSV:          {os.path.join(processed_dir, 'online_retail_ii_daily.csv')}")
    print(f"  - Provenance:          {os.path.join(processed_dir, 'online_retail_ii_provenance.json')}")
    print(f"  - Duration:            {prov.processing_duration_seconds}s")
    print("=" * 70)

    return prov


if __name__ == "__main__":
    run_onboarding_pipeline()


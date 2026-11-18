"""
Data Preprocessing Module
Validates supply chain datasets, handles missing values/duplicates,
computes dataset statistics, and generates boolean basket matrices for mining.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


REQUIRED_COLUMNS = ["Order_ID", "Product", "Supplier", "Warehouse", "Quantity", "Order_Date"]


def validate_dataset(df: pd.DataFrame) -> Tuple[bool, List[str], str]:
    """
    Validates if uploaded/provided DataFrame contains all required columns and valid schema.
    Returns: (is_valid, missing_columns, message)
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return False, REQUIRED_COLUMNS, "Invalid input: Dataset is not a valid pandas DataFrame."
    
    if df.empty:
        return False, [], "Invalid input: Dataset is empty (0 records)."

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if missing_cols:
        return False, missing_cols, f"Schema Validation Error: Missing required columns: {', '.join(missing_cols)}"
    
    return True, [], "Dataset schema is valid."


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw dataset by handling missing values, standardizing types,
    and removing duplicate records.
    Returns: (cleaned_df, cleaning_report)
    """
    initial_rows = len(df)
    
    # Copy to avoid mutating original
    df_clean = df.copy()

    # Strip string whitespaces
    string_cols = ["Order_ID", "Product", "Supplier", "Warehouse"]
    for col in string_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()

    # Identify and drop missing or empty values
    missing_mask = (
        df_clean[REQUIRED_COLUMNS].isnull().any(axis=1) |
        (df_clean["Order_ID"] == "") | (df_clean["Order_ID"] == "nan") |
        (df_clean["Product"] == "") | (df_clean["Product"] == "nan")
    )
    missing_count = int(missing_mask.sum())
    df_clean = df_clean[~missing_mask].copy()

    # Cast Quantity to positive numeric
    df_clean["Quantity"] = pd.to_numeric(df_clean["Quantity"], errors="coerce").fillna(1)
    df_clean = df_clean[df_clean["Quantity"] > 0]
    df_clean["Quantity"] = df_clean["Quantity"].astype(int)

    # Standardize dates
    df_clean["Order_Date"] = pd.to_datetime(df_clean["Order_Date"], errors="coerce")
    invalid_dates = int(df_clean["Order_Date"].isnull().sum())
    df_clean = df_clean.dropna(subset=["Order_Date"])
    df_clean["Order_Date"] = df_clean["Order_Date"].dt.strftime("%Y-%m-%d")

    # Drop duplicate line items within the same order
    before_dedup = len(df_clean)
    df_clean = df_clean.drop_duplicates(subset=["Order_ID", "Product"]).copy()
    duplicates_dropped = before_dedup - len(df_clean)

    report = {
        "initial_rows": initial_rows,
        "final_rows": len(df_clean),
        "missing_records_dropped": missing_count + invalid_dates,
        "duplicates_dropped": duplicates_dropped,
        "unique_orders": int(df_clean["Order_ID"].nunique()),
        "unique_products": int(df_clean["Product"].nunique())
    }

    return df_clean, report


def get_dataset_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes summary metrics and distributions from the supply chain dataset.
    """
    if df.empty:
        return {}

    orders_group = df.groupby("Order_ID")["Product"].count()
    
    return {
        "total_records": len(df),
        "total_orders": int(df["Order_ID"].nunique()),
        "total_products": int(df["Product"].nunique()),
        "total_suppliers": int(df["Supplier"].nunique()) if "Supplier" in df.columns else 0,
        "total_warehouses": int(df["Warehouse"].nunique()) if "Warehouse" in df.columns else 0,
        "total_units_sold": int(df["Quantity"].sum()) if "Quantity" in df.columns else len(df),
        "min_order_date": str(df["Order_Date"].min()) if "Order_Date" in df.columns else "N/A",
        "max_order_date": str(df["Order_Date"].max()) if "Order_Date" in df.columns else "N/A",
        "avg_basket_size": float(orders_group.mean()),
        "min_basket_size": int(orders_group.min()),
        "max_basket_size": int(orders_group.max()),
        "median_basket_size": float(orders_group.median())
    }


def build_basket_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms transactional records into a one-hot encoded boolean basket matrix.
    Rows: Order_ID
    Columns: Products
    Values: True if product present in order, False otherwise.
    """
    # Group by Order_ID and Product, then unstack or crosstab
    basket = pd.crosstab(df["Order_ID"], df["Product"])
    # Convert counts > 0 to boolean True/False
    basket_bool = basket.astype(bool)
    return basket_bool


def get_transactions_list(df: pd.DataFrame) -> List[List[str]]:
    """
    Returns transactions as a list of lists of products.
    Used for Spark FP-Growth and transaction verification.
    """
    grouped = df.groupby("Order_ID")["Product"].apply(lambda s: sorted(list(set(s)))).tolist()
    return grouped

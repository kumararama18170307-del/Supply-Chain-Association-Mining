"""
Supply Chain Synthetic Data Generator
Generates realistic supply-chain transaction data with deterministic
co-occurrence patterns across products, suppliers, and warehouses.
"""

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


# Domain Catalogs
PRODUCT_CATALOG = {
    "IT & Peripherals": {
        "Dell XPS 15 Laptop": {"supplier": "GlobalTech Electronics Ltd", "qty_range": (1, 3)},
        "Logitech MX Master Mouse": {"supplier": "GlobalTech Electronics Ltd", "qty_range": (1, 5)},
        "Ergonomic Laptop Stand": {"supplier": "GlobalTech Electronics Ltd", "qty_range": (1, 4)},
        "USB-C Multiport Hub": {"supplier": "GlobalTech Electronics Ltd", "qty_range": (1, 6)},
        "Laptop Backpack Pro": {"supplier": "NexGen Office Supplies", "qty_range": (1, 3)},
        "27-inch 4K Monitor": {"supplier": "GlobalTech Electronics Ltd", "qty_range": (1, 4)},
        "Mechanical Keyboard": {"supplier": "GlobalTech Electronics Ltd", "qty_range": (1, 4)}
    },
    "Industrial PPE": {
        "Heavy Duty Hard Hat": {"supplier": "Apex Safety Equipment Co", "qty_range": (2, 15)},
        "3M Safety Goggles": {"supplier": "Apex Safety Equipment Co", "qty_range": (3, 20)},
        "High-Visibility Safety Vest": {"supplier": "Apex Safety Equipment Co", "qty_range": (5, 25)},
        "Nitrile Heavy Work Gloves": {"supplier": "Apex Safety Equipment Co", "qty_range": (5, 30)},
        "Steel Toe Safety Boots": {"supplier": "Apex Safety Equipment Co", "qty_range": (1, 8)},
        "Industrial First Aid Kit": {"supplier": "Apex Safety Equipment Co", "qty_range": (1, 5)}
    },
    "Server & Infrastructure": {
        "42U Server Rack Cabinet": {"supplier": "Datacom Infrastructure Corp", "qty_range": (1, 2)},
        "Cat6A Patch Cable Pack": {"supplier": "Datacom Infrastructure Corp", "qty_range": (2, 12)},
        "Rackmount PDU 16A": {"supplier": "Datacom Infrastructure Corp", "qty_range": (1, 4)},
        "1500VA UPS Battery Backup": {"supplier": "Datacom Infrastructure Corp", "qty_range": (1, 3)},
        "24-Port Gigabit Switch": {"supplier": "Datacom Infrastructure Corp", "qty_range": (1, 4)},
        "Cable Management Horizontal Arm": {"supplier": "Datacom Infrastructure Corp", "qty_range": (2, 8)}
    },
    "Packaging & Logistics": {
        "Corrugated Shipping Boxes (25pk)": {"supplier": "PackRight Logistics Supplies", "qty_range": (2, 20)},
        "Heavy Duty Packing Tape (6pk)": {"supplier": "PackRight Logistics Supplies", "qty_range": (3, 25)},
        "Industrial Bubble Wrap Roll": {"supplier": "PackRight Logistics Supplies", "qty_range": (1, 10)},
        "Digital Shipping Bench Scale": {"supplier": "PackRight Logistics Supplies", "qty_range": (1, 2)},
        "Thermal Label Printer": {"supplier": "NexGen Office Supplies", "qty_range": (1, 3)},
        "Handheld Barcode Scanner Gun": {"supplier": "NexGen Office Supplies", "qty_range": (1, 4)}
    },
    "Maintenance & Tools (MRO)": {
        "Industrial WD-40 Lubricant (Case)": {"supplier": "Industrial MRO Solutions", "qty_range": (1, 6)},
        "18V Cordless Power Drill Kit": {"supplier": "Industrial MRO Solutions", "qty_range": (1, 3)},
        "Hex Key Allen Wrench Set": {"supplier": "Industrial MRO Solutions", "qty_range": (2, 10)},
        "Digital Multimeter Tester": {"supplier": "Industrial MRO Solutions", "qty_range": (1, 4)},
        "Heavy Duty Ballistic Tool Bag": {"supplier": "Industrial MRO Solutions", "qty_range": (1, 5)}
    },
    "Office Stationery": {
        "Executive A4 Notebook (5pk)": {"supplier": "NexGen Office Supplies", "qty_range": (2, 12)},
        "Gel Pen Premium Box (12pk)": {"supplier": "NexGen Office Supplies", "qty_range": (2, 15)},
        "Sticky Notes Value Pack (12pk)": {"supplier": "NexGen Office Supplies", "qty_range": (2, 18)},
        "Heavy Duty Desktop Stapler": {"supplier": "NexGen Office Supplies", "qty_range": (1, 5)}
    }
}

WAREHOUSES = [
    "Warehouse-North (WH-N1)",
    "Warehouse-South (WH-S2)",
    "Warehouse-East (WH-E3)",
    "Warehouse-West (WH-W4)",
    "Warehouse-Central (WH-C5)"
]

# Warehouse assignment affinities (regional specialization)
WAREHOUSE_AFFINITIES = {
    "IT & Peripherals": ["Warehouse-Central (WH-C5)", "Warehouse-North (WH-N1)"],
    "Industrial PPE": ["Warehouse-South (WH-S2)", "Warehouse-East (WH-E3)"],
    "Server & Infrastructure": ["Warehouse-Central (WH-C5)", "Warehouse-West (WH-W4)"],
    "Packaging & Logistics": ["Warehouse-East (WH-E3)", "Warehouse-South (WH-S2)"],
    "Maintenance & Tools (MRO)": ["Warehouse-West (WH-W4)", "Warehouse-North (WH-N1)"],
    "Office Stationery": ["Warehouse-Central (WH-C5)", "Warehouse-East (WH-E3)"]
}

# Frequent Association Bundles (Designed for realistic statistical patterns)
PATTERNS = [
    # Bundle 1: Laptop Ecosystem
    {
        "category": "IT & Peripherals",
        "primary": "Dell XPS 15 Laptop",
        "companions": ["Logitech MX Master Mouse", "Ergonomic Laptop Stand", "USB-C Multiport Hub", "Laptop Backpack Pro"],
        "co_prob": [0.85, 0.75, 0.70, 0.55],
        "weight": 22
    },
    # Bundle 2: Industrial Safety & PPE Setup
    {
        "category": "Industrial PPE",
        "primary": "Heavy Duty Hard Hat",
        "companions": ["3M Safety Goggles", "High-Visibility Safety Vest", "Nitrile Heavy Work Gloves", "Steel Toe Safety Boots"],
        "co_prob": [0.88, 0.82, 0.78, 0.45],
        "weight": 20
    },
    # Bundle 3: Server Rack Deployment
    {
        "category": "Server & Infrastructure",
        "primary": "42U Server Rack Cabinet",
        "companions": ["Cat6A Patch Cable Pack", "Rackmount PDU 16A", "1500VA UPS Battery Backup", "Cable Management Horizontal Arm"],
        "co_prob": [0.90, 0.85, 0.72, 0.68],
        "weight": 16
    },
    # Bundle 4: Dispatch & Shipping Station
    {
        "category": "Packaging & Logistics",
        "primary": "Corrugated Shipping Boxes (25pk)",
        "companions": ["Heavy Duty Packing Tape (6pk)", "Industrial Bubble Wrap Roll", "Thermal Label Printer", "Handheld Barcode Scanner Gun"],
        "co_prob": [0.92, 0.80, 0.65, 0.50],
        "weight": 18
    },
    # Bundle 5: MRO Maintenance Toolkit
    {
        "category": "Maintenance & Tools (MRO)",
        "primary": "18V Cordless Power Drill Kit",
        "companions": ["Hex Key Allen Wrench Set", "Heavy Duty Ballistic Tool Bag", "Industrial WD-40 Lubricant (Case)"],
        "co_prob": [0.78, 0.70, 0.62],
        "weight": 14
    },
    # Bundle 6: Office Procurement Stationery Kit
    {
        "category": "Office Stationery",
        "primary": "Executive A4 Notebook (5pk)",
        "companions": ["Gel Pen Premium Box (12pk)", "Sticky Notes Value Pack (12pk)", "Heavy Duty Desktop Stapler"],
        "co_prob": [0.85, 0.80, 0.55],
        "weight": 10
    }
]


def _flatten_catalog():
    """Returns flat lookup of product -> {supplier, qty_range, category}"""
    flat = {}
    for cat, prods in PRODUCT_CATALOG.items():
        for prod_name, info in prods.items():
            flat[prod_name] = {
                "category": cat,
                "supplier": info["supplier"],
                "qty_range": info["qty_range"]
            }
    return flat


def generate_supply_chain_data(target_records: int = 5500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic supply-chain transactions DataFrame.
    Guarantees >= target_records order line items with clean, realistic associations.
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    catalog_flat = _flatten_catalog()
    all_products = list(catalog_flat.keys())
    
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 3, 31)
    days_range = (end_date - start_date).days

    records = []
    order_idx = 10001
    
    pattern_weights = [p["weight"] for p in PATTERNS]
    
    while len(records) < target_records:
        order_id = f"ORD-{order_idx}"
        order_date = start_date + timedelta(days=random.randint(0, days_range))
        order_date_str = order_date.strftime("%Y-%m-%d")

        # Decide transaction type:
        # 75% bundle pattern order, 25% random mixed procurement order
        is_bundle = random.random() < 0.78

        if is_bundle:
            pattern = random.choices(PATTERNS, weights=pattern_weights, k=1)[0]
            cat = pattern["category"]
            
            # Primary warehouse with 80% affinity, 20% any other warehouse
            if random.random() < 0.80:
                warehouse = random.choice(WAREHOUSE_AFFINITIES[cat])
            else:
                warehouse = random.choice(WAREHOUSES)

            order_items = [pattern["primary"]]
            
            # Add companions based on co_prob
            for comp, prob in zip(pattern["companions"], pattern["co_prob"]):
                if random.random() < prob:
                    order_items.append(comp)
            
            # 20% chance of adding 1 random peripheral/cross-category item (realistic noise)
            if random.random() < 0.20:
                noise_item = random.choice(all_products)
                if noise_item not in order_items:
                    order_items.append(noise_item)
        else:
            # Multi-product procurement order (2-5 random items)
            basket_size = random.randint(2, 5)
            warehouse = random.choice(WAREHOUSES)
            order_items = random.sample(all_products, basket_size)

        # Assemble records for this Order_ID
        for prod in order_items:
            info = catalog_flat[prod]
            qty = random.randint(info["qty_range"][0], info["qty_range"][1])
            records.append({
                "Order_ID": order_id,
                "Product": prod,
                "Supplier": info["supplier"],
                "Warehouse": warehouse,
                "Quantity": qty,
                "Order_Date": order_date_str
            })

        order_idx += 1

    df = pd.DataFrame(records)
    # Ensure correct column ordering
    columns_order = ["Order_ID", "Product", "Supplier", "Warehouse", "Quantity", "Order_Date"]
    df = df[columns_order]
    return df


def save_dataset_to_csv(df: pd.DataFrame, filepath: str) -> str:
    """Saves DataFrame to CSV, ensuring directory exists."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)
    return filepath


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    csv_path = os.path.join(project_root, "data", "supply_chain.csv")
    
    print(f"Generating synthetic supply chain dataset to: {csv_path}")
    df_generated = generate_supply_chain_data(target_records=5500, random_seed=42)
    save_dataset_to_csv(df_generated, csv_path)
    print(f"Generated {len(df_generated)} rows across {df_generated['Order_ID'].nunique()} unique orders.")

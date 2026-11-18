"""
Supply Chain Recommendations and Insights Engine
Translates discovered association rules and transactional co-occurrences into actionable
inventory, warehouse slotting, and procurement coordination strategies.
Strictly separates empirical statistical findings from prescriptive business recommendations.
"""

from typing import List, Dict, Any, Optional
import pandas as pd


def generate_supply_chain_recommendations(
    rules_df: pd.DataFrame,
    catalog_df: Optional[pd.DataFrame] = None,
    top_n: int = 10
) -> List[Dict[str, Any]]:
    """
    Translates top association rules into supply chain operational recommendations.
    
    Returns a list of structured insight cards with:
      - rule: rule string
      - lift: float
      - confidence: float
      - support: float
      - statistical_finding: empirical fact from data
      - business_recommendation: operational action for inventory / warehouse / procurement
      - category: "Inventory Buffering" | "Warehouse Co-Location" | "Procurement Synchronization"
      - priority: "High" | "Medium" | "Low"
    """
    if rules_df.empty:
        return []

    # Map product to supplier and warehouse if catalog provided
    product_meta = {}
    if catalog_df is not None and not catalog_df.empty:
        for _, row in catalog_df.iterrows():
            prod = row.get("Product")
            if prod and prod not in product_meta:
                product_meta[prod] = {
                    "supplier": row.get("Supplier", "Standard Supplier"),
                    "warehouse": row.get("Warehouse", "Central Hub")
                }

    recommendations = []
    sorted_rules = rules_df.sort_values(by=["lift", "confidence"], ascending=[False, False]).head(top_n)

    for idx, rule in sorted_rules.iterrows():
        ant_str = rule.get("antecedents_str", "")
        con_str = rule.get("consequents_str", "")
        conf = float(rule.get("confidence", 0.0))
        lift = float(rule.get("lift", 1.0))
        supp = float(rule.get("support", 0.0))

        ants = [x.strip() for x in ant_str.split(",") if x.strip()]
        cons = [x.strip() for x in con_str.split(",") if x.strip()]

        # Priority calculation based on lift and confidence
        if lift >= 3.0 and conf >= 0.70:
            priority = "High"
        elif lift >= 1.8 and conf >= 0.50:
            priority = "Medium"
        else:
            priority = "Standard"

        # Lookup supplier context
        ant_suppliers = list(set([product_meta.get(p, {}).get("supplier", "Unknown") for p in ants]))
        con_suppliers = list(set([product_meta.get(p, {}).get("supplier", "Unknown") for p in cons]))
        same_supplier = (len(ant_suppliers) == 1 and len(con_suppliers) == 1 and ant_suppliers[0] == con_suppliers[0] and ant_suppliers[0] != "Unknown")

        # 1. Statistical Finding (Empirical Statement)
        stat_finding = (
            f"Transactions containing [{ant_str}] also order [{con_str}] in {conf*100:.1f}% of cases "
            f"(Confidence: {conf:.3f}, Support: {supp*100:.1f}%). "
            f"This complementary co-occurrence is {lift:.2f}x higher than expected under statistical independence."
        )

        # 2. Prescriptive Business Recommendation
        if same_supplier:
            biz_rec = (
                f"Consolidated Procurement & Slotting: Because both [{ant_str}] and [{con_str}] are supplied by "
                f"'{ant_suppliers[0]}', issue bundled purchase orders to unlock tiered volume discounts (target 5-8% savings). "
                f"In fulfillment centers, assign these SKUs to adjacent pick-bays (Zone A cross-docking) to minimize picker transit distance."
            )
            cat = "Procurement Synchronization & Slotting"
        elif any("Laptop" in p or "Server" in p or "Drill" in p for p in ants):
            biz_rec = (
                f"Dynamic Safety-Stock Coupling: Complementary attachment rate is {conf*100:.1f}%. "
                f"When procurement triggers a replenishment order for [{ant_str}], automatically adjust the reorder point "
                f"and safety stock buffer for [{con_str}] by +{min(int(conf * 100), 85)}% to prevent stockouts of auxiliary accessories."
            )
            cat = "Inventory Buffering"
        else:
            biz_rec = (
                f"Warehouse Co-Location & Cross-Docking: Co-locate [{ant_str}] and [{con_str}] in adjacent aisle bins. "
                f"Pre-package combined order kits in packing stations to compress order dispatch cycle time by an estimated 25-35%."
            )
            cat = "Warehouse Co-Location"

        recommendations.append({
            "id": idx + 1,
            "rule": f"{{{ant_str}}} -> {{{con_str}}}",
            "antecedents": ant_str,
            "consequents": con_str,
            "confidence": conf,
            "lift": lift,
            "support": supp,
            "statistical_finding": stat_finding,
            "business_recommendation": biz_rec,
            "category": cat,
            "priority": priority
        })

    return recommendations


def analyze_supplier_cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes cross-supplier procurement interactions:
    How frequently orders demand items from multiple suppliers simultaneously.
    """
    if df.empty or "Supplier" not in df.columns or "Order_ID" not in df.columns:
        return pd.DataFrame()

    order_suppliers = df.groupby("Order_ID")["Supplier"].unique().reset_index()
    order_suppliers["supplier_count"] = order_suppliers["Supplier"].apply(len)
    
    # Pairwise supplier co-occurrences
    from itertools import combinations
    pair_counts = {}
    for supp_list in order_suppliers["Supplier"]:
        unique_supps = sorted(list(set(supp_list)))
        if len(unique_supps) >= 2:
            for pair in combinations(unique_supps, 2):
                pair_counts[pair] = pair_counts.get(pair, 0) + 1

    total_orders = df["Order_ID"].nunique()
    records = []
    for (s1, s2), count in pair_counts.items():
        records.append({
            "Supplier A": s1,
            "Supplier B": s2,
            "Co-occurrence Count": count,
            "Order Percentage": f"{(count / total_orders)*100:.1f}%",
            "Joint Procurement Opportunity": "High" if (count / total_orders) > 0.20 else "Moderate"
        })

    result_df = pd.DataFrame(records)
    if not result_df.empty:
        result_df = result_df.sort_values(by="Co-occurrence Count", ascending=False).reset_index(drop=True)
    return result_df


def analyze_warehouse_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes volume and SKU diversity per warehouse facility.
    """
    if df.empty or "Warehouse" not in df.columns:
        return pd.DataFrame()

    summary = df.groupby("Warehouse").agg(
        Total_Orders=("Order_ID", "nunique"),
        Total_Units=("Quantity", "sum"),
        Unique_SKUs=("Product", "nunique")
    ).reset_index()

    total_units = summary["Total_Units"].sum()
    summary["Volume_Share"] = summary["Total_Units"].apply(lambda v: f"{(v / total_units)*100:.1f}%")
    return summary.sort_values(by="Total_Units", ascending=False).reset_index(drop=True)

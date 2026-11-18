"""
Association Rules Presentation and Verification Module
Provides filtering, sorting, formatting, CSV export, and academic verification
comparing algorithmic rule metrics against first-principles mathematical derivations.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def filter_rules(
    rules_df: pd.DataFrame,
    min_confidence: float = 0.0,
    min_lift: float = 0.0,
    min_support: float = 0.0,
    antecedent_filter: str = "",
    consequent_filter: str = "",
    max_antecedent_len: Optional[int] = None
) -> pd.DataFrame:
    """
    Applies multi-criteria filtering to an association rules DataFrame.
    """
    if rules_df.empty:
        return rules_df

    filtered = rules_df.copy()

    # Numerical threshold filters
    filtered = filtered[
        (filtered["confidence"] >= min_confidence) &
        (filtered["lift"] >= min_lift) &
        (filtered["support"] >= min_support)
    ]

    # Length filter
    if max_antecedent_len is not None and "antecedent_len" in filtered.columns:
        filtered = filtered[filtered["antecedent_len"] <= max_antecedent_len]

    # Keyword filters
    if antecedent_filter and "antecedents_str" in filtered.columns:
        filtered = filtered[filtered["antecedents_str"].str.contains(antecedent_filter, case=False, na=False)]

    if consequent_filter and "consequents_str" in filtered.columns:
        filtered = filtered[filtered["consequents_str"].str.contains(consequent_filter, case=False, na=False)]

    return filtered.reset_index(drop=True)


def format_rules_for_display(rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces clean, human-readable DataFrame formatted for Streamlit data tables.
    """
    if rules_df.empty:
        return rules_df

    display_df = rules_df.copy()

    # Ensure rule representation exists
    if "rule_repr" not in display_df.columns:
        if "antecedents_str" in display_df.columns and "consequents_str" in display_df.columns:
            display_df["rule_repr"] = display_df.apply(
                lambda r: f"{{{r['antecedents_str']}}} -> {{{r['consequents_str']}}}", axis=1
            )

    # Format numeric columns nicely
    if "support" in display_df.columns:
        display_df["support_fmt"] = display_df["support"].apply(lambda v: f"{v:.4f} ({v*100:.1f}%)")
    if "confidence" in display_df.columns:
        display_df["confidence_fmt"] = display_df["confidence"].apply(lambda v: f"{v:.4f} ({v*100:.1f}%)")
    if "lift" in display_df.columns:
        display_df["lift_fmt"] = display_df["lift"].apply(lambda v: f"{v:.2f}")

    return display_df


def generate_academic_verification_report(
    rules_df: pd.DataFrame,
    basket_df: pd.DataFrame,
    sample_size: int = 5
) -> pd.DataFrame:
    """
    Computes an academic audit table for examiners:
    Takes a sample of mined rules and compares their algorithmic support, confidence, and lift
    against first-principles recomputations from the raw transaction matrix.
    """
    if rules_df.empty or basket_df.empty:
        return pd.DataFrame()

    sample_rules = rules_df.head(sample_size).copy()
    verification_rows = []
    N = len(basket_df)

    for idx, row in sample_rules.iterrows():
        # Get antecedent and consequent item lists
        if isinstance(row["antecedents_str"], str):
            ants = [x.strip() for x in row["antecedents_str"].split(",") if x.strip()]
        else:
            ants = list(row["antecedents"])

        if isinstance(row["consequents_str"], str):
            cons = [x.strip() for x in row["consequents_str"].split(",") if x.strip()]
        else:
            cons = list(row["consequents"])

        # Manual counts from raw basket matrix
        mask_a = np.ones(N, dtype=bool)
        for item in ants:
            mask_a &= basket_df[item].values if item in basket_df.columns else False

        mask_b = np.ones(N, dtype=bool)
        for item in cons:
            mask_b &= basket_df[item].values if item in basket_df.columns else False

        mask_ab = mask_a & mask_b

        count_a = int(np.sum(mask_a))
        count_b = int(np.sum(mask_b))
        count_ab = int(np.sum(mask_ab))

        manual_supp = count_ab / N
        manual_conf = (count_ab / count_a) if count_a > 0 else 0.0
        manual_supp_b = count_b / N
        manual_lift = (manual_conf / manual_supp_b) if manual_supp_b > 0 else 0.0

        algo_supp = float(row["support"])
        algo_conf = float(row["confidence"])
        algo_lift = float(row["lift"])

        supp_diff = abs(manual_supp - algo_supp)
        conf_diff = abs(manual_conf - algo_conf)
        lift_diff = abs(manual_lift - algo_lift)

        is_passed = (supp_diff < 1e-4) and (conf_diff < 1e-4) and (lift_diff < 1e-4)

        verification_rows.append({
            "Rule": row.get("rule_repr", f"{{{', '.join(ants)}}} -> {{{', '.join(cons)}}}"),
            "Count(A & B)": count_ab,
            "Count(A)": count_a,
            "Count(B)": count_b,
            "N (Total)": N,
            "Manual Supp": round(manual_supp, 4),
            "Algo Supp": round(algo_supp, 4),
            "Manual Conf": round(manual_conf, 4),
            "Algo Conf": round(algo_conf, 4),
            "Manual Lift": round(manual_lift, 2),
            "Algo Lift": round(algo_lift, 2),
            "Audit Status": "EXACT MATCH (100% Correct)" if is_passed else "MISMATCH"
        })

    return pd.DataFrame(verification_rows)


def export_rules_to_csv(rules_df: pd.DataFrame) -> str:
    """
    Exports clean CSV string of association rules.
    """
    export_cols = [
        "rule_repr", "antecedents_str", "consequents_str",
        "support", "confidence", "lift"
    ]
    available_cols = [c for c in export_cols if c in rules_df.columns]
    return rules_df[available_cols].to_csv(index=False)

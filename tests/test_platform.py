"""
Comprehensive Automated Test Suite for SupplyChain Intelligence Platform
Validates:
1. Small and large datasets
2. Missing values and invalid CSV files
3. Multiple support and confidence thresholds
4. Set-theoretic mathematical verification of Support, Confidence, and Lift
5. Itemset consistency between Apriori and FP-Growth
"""

import os
import sys
import unittest
import pandas as pd
import numpy as np

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from modules.data_generator import generate_supply_chain_data
from modules.data_preprocessing import (
    validate_dataset, clean_dataset, get_dataset_statistics,
    build_basket_matrix, get_transactions_list, REQUIRED_COLUMNS
)
from modules.apriori_mining import (
    run_apriori_mining, verify_rule_mathematics,
    explain_apriori_candidate_levels
)
from modules.spark_fpgrowth import run_spark_fpgrowth, get_fptree_structural_stats
from modules.association_rules import filter_rules, generate_academic_verification_report
from modules.accuracy_analysis import (
    train_test_split_baskets, evaluate_rule_generalization,
    evaluate_predictive_accuracy, compute_threshold_sensitivity,
    get_accuracy_metric_definitions
)
from modules.recommendations import generate_supply_chain_recommendations


class TestSupplyChainPlatform(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load canonical dataset
        csv_path = os.path.join(PROJECT_ROOT, "data", "supply_chain.csv")
        if not os.path.exists(csv_path):
            cls.df_canonical = generate_supply_chain_data(target_records=5500, random_seed=42)
        else:
            cls.df_canonical = pd.read_csv(csv_path)

        cls.df_clean, cls.clean_report = clean_dataset(cls.df_canonical)
        cls.basket_matrix = build_basket_matrix(cls.df_clean)
        cls.tx_list = get_transactions_list(cls.df_clean)

    def test_01_schema_validation_valid(self):
        """Test schema validation on canonical dataset."""
        is_valid, missing, msg = validate_dataset(self.df_canonical)
        self.assertTrue(is_valid)
        self.assertEqual(len(missing), 0)

    def test_02_schema_validation_invalid(self):
        """Test schema validation on invalid dataset missing required columns."""
        invalid_df = pd.DataFrame({"Order_ID": [1, 2], "Item": ["A", "B"]})
        is_valid, missing, msg = validate_dataset(invalid_df)
        self.assertFalse(is_valid)
        self.assertIn("Product", missing)
        self.assertIn("Supplier", missing)

    def test_03_missing_values_and_duplicates_handling(self):
        """Test that data cleaner properly scrubs NaNs, empties, and duplicates."""
        dirty_data = pd.DataFrame([
            {"Order_ID": "O1", "Product": "Laptop", "Supplier": "S1", "Warehouse": "W1", "Quantity": 1, "Order_Date": "2025-01-01"},
            {"Order_ID": "O1", "Product": "Laptop", "Supplier": "S1", "Warehouse": "W1", "Quantity": 2, "Order_Date": "2025-01-01"}, # duplicate line
            {"Order_ID": "O2", "Product": np.nan, "Supplier": "S1", "Warehouse": "W1", "Quantity": 1, "Order_Date": "2025-01-01"}, # NaN product
            {"Order_ID": "", "Product": "Mouse", "Supplier": "S1", "Warehouse": "W1", "Quantity": 1, "Order_Date": "2025-01-01"}, # empty order id
            {"Order_ID": "O3", "Product": "Mouse", "Supplier": "S1", "Warehouse": "W1", "Quantity": -5, "Order_Date": "2025-01-01"}, # invalid qty
        ])
        cleaned, report = clean_dataset(dirty_data)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["Order_ID"], "O1")
        self.assertEqual(cleaned.iloc[0]["Product"], "Laptop")

    def test_04_small_dataset_mining(self):
        """Test Apriori and FP-Growth on small controlled dataset."""
        small_txs = [
            ["Laptop", "Mouse", "Bag"],
            ["Laptop", "Mouse"],
            ["Laptop", "Stand"],
            ["Mouse", "Keyboard"],
            ["Laptop", "Mouse", "Bag"],
            ["Laptop", "Mouse", "Stand"],
            ["Laptop", "Bag"],
            ["Mouse", "Bag"],
        ]
        # Build DataFrame
        rows = []
        for i, tx in enumerate(small_txs):
            for itm in tx:
                rows.append({
                    "Order_ID": f"ORD-{i+1}", "Product": itm,
                    "Supplier": "Sup1", "Warehouse": "WH1",
                    "Quantity": 1, "Order_Date": "2025-01-01"
                })
        small_df = pd.DataFrame(rows)
        small_basket = build_basket_matrix(small_df)
        
        freq_a, rules_a, _ = run_apriori_mining(small_basket, min_support=0.30, min_confidence=0.50, min_lift=1.0)
        freq_fp, rules_fp, _ = run_spark_fpgrowth(small_txs, min_support=0.30, min_confidence=0.50, min_lift=1.0)

        self.assertFalse(freq_a.empty)
        self.assertFalse(freq_fp.empty)
        self.assertFalse(rules_a.empty)
        self.assertFalse(rules_fp.empty)

    def test_05_mathematical_precision_verification(self):
        """
        Rigorously verify Support, Confidence, and Lift matching first-principles formulas:
        Support(A, B) = Count(A AND B) / N
        Confidence(A -> B) = Support(A, B) / Support(A)
        Lift(A -> B) = Confidence(A -> B) / Support(B)
        """
        freq, rules, _ = run_apriori_mining(
            self.basket_matrix,
            min_support=0.05,
            min_confidence=0.40,
            min_lift=1.20
        )
        self.assertFalse(rules.empty, "Rules should not be empty on canonical dataset")

        audit_table = generate_academic_verification_report(rules, self.basket_matrix, sample_size=10)
        self.assertFalse(audit_table.empty)

        for _, row in audit_table.iterrows():
            # Verify status
            self.assertEqual(row["Audit Status"], "EXACT MATCH (100% Correct)")
            # Verify numerical differences are zero within float epsilon
            self.assertAlmostEqual(row["Manual Supp"], row["Algo Supp"], places=4)
            self.assertAlmostEqual(row["Manual Conf"], row["Algo Conf"], places=4)
            self.assertAlmostEqual(row["Manual Lift"], row["Algo Lift"], places=2)

    def test_06_apriori_vs_fpgrowth_consistency(self):
        """
        Verify that Apriori and FP-Growth return consistent frequent itemsets on the same data.
        """
        min_supp = 0.08
        freq_a, _, _ = run_apriori_mining(self.basket_matrix, min_support=min_supp, max_len=3)
        freq_fp, _, _ = run_spark_fpgrowth(self.tx_list, min_support=min_supp)

        itemsets_a = set(freq_a["itemset_str"])
        itemsets_fp = set(freq_fp["itemset_str"])

        # Compare frequent itemsets found by both
        common = itemsets_a.intersection(itemsets_fp)
        self.assertGreater(len(common), 0)
        # Verify supports match for common itemsets
        supp_map_a = dict(zip(freq_a["itemset_str"], freq_a["support"]))
        supp_map_fp = dict(zip(freq_fp["itemset_str"], freq_fp["support"]))

        for itm in common:
            self.assertAlmostEqual(supp_map_a[itm], supp_map_fp[itm], places=3)

    def test_07_recommendations_generation(self):
        """Verify that recommendations properly separate statistical findings from business actions."""
        _, rules, _ = run_apriori_mining(self.basket_matrix, min_support=0.05, min_confidence=0.40, min_lift=1.20)
        recs = generate_supply_chain_recommendations(rules, self.df_clean, top_n=5)
        self.assertGreater(len(recs), 0)
        for r in recs:
            self.assertIn("Confidence", r["statistical_finding"])
            self.assertIn("business_recommendation", r)
            self.assertIn(r["priority"], ["High", "Medium", "Standard"])

    def test_08_accuracy_train_test_split(self):
        """Verify train/test split maintains transaction count, proportions, and determinism."""
        train_df, test_df = train_test_split_baskets(self.basket_matrix, test_size=0.20, random_state=42)
        total_n = len(self.basket_matrix)
        self.assertEqual(len(train_df) + len(test_df), total_n)
        self.assertAlmostEqual(len(test_df) / total_n, 0.20, places=1)
        self.assertEqual(list(train_df.columns), list(test_df.columns))

    def test_09_accuracy_generalization_manual_calculation(self):
        """
        Verify that out-of-sample test support, confidence, and lift match manual calculation
        on a known, controlled toy test split.
        """
        # Controlled toy test basket matrix (4 orders x 4 items)
        # Order 1: A, B
        # Order 2: A, B, C
        # Order 3: A
        # Order 4: C, D
        test_toy = pd.DataFrame([
            {"A": True,  "B": True,  "C": False, "D": False},
            {"A": True,  "B": True,  "C": True,  "D": False},
            {"A": True,  "B": False, "C": False, "D": False},
            {"A": False, "B": False, "C": True,  "D": True},
        ])
        
        # Simulated train rule: {A} -> {B} (train_conf=0.70, train_lift=1.40, train_supp=0.30)
        mock_train_rules = pd.DataFrame([{
            "rule_repr": "{A} -> {B}",
            "antecedents": ["A"],
            "consequents": ["B"],
            "support": 0.30,
            "confidence": 0.70,
            "lift": 1.40,
            "antecedents_str": "A",
            "consequents_str": "B"
        }])

        gen_df, gen_sum = evaluate_rule_generalization(mock_train_rules, test_toy)
        self.assertEqual(len(gen_df), 1)
        row = gen_df.iloc[0]

        # Manual derivations:
        # N_test = 4
        # Count(A) = 3 (Orders 1, 2, 3) -> Supp(A) = 3/4 = 0.75
        # Count(B) = 2 (Orders 1, 2)    -> Supp(B) = 2/4 = 0.50
        # Count(A & B) = 2 (Orders 1, 2) -> Supp(A & B) = 2/4 = 0.50
        # Test Confidence = 0.50 / 0.75 = 2/3 = 0.6667
        # Test Lift = (2/3) / 0.50 = 4/3 = 1.3333
        self.assertAlmostEqual(row["test_support"], 0.50, places=4)
        self.assertAlmostEqual(row["test_confidence"], 2.0 / 3.0, places=4)
        self.assertAlmostEqual(row["test_lift"], 4.0 / 3.0, places=4)
        self.assertAlmostEqual(row["confidence_drift"], (2.0 / 3.0) - 0.70, places=4)

    def test_10_accuracy_precision_recall_manual_calculation(self):
        """
        Verify that recommender Precision, Recall, and F1-Score match manual calculations
        on a known, controlled toy test split.
        """
        test_toy = pd.DataFrame([
            {"A": True,  "B": True,  "C": False, "D": False},  # Pred: B -> Target: B -> TP=1, FP=0 -> Prec=1.0, Rec=1.0
            {"A": True,  "B": True,  "C": True,  "D": False},  # Pred: B -> Target: B, C -> TP=1, FP=0, FN=1 -> Prec=1.0, Rec=0.5
            {"A": True,  "B": False, "C": False, "D": False},  # Pred: B -> Target: A -> TP=0, FP=1, FN=0 -> Prec=0.0, Rec=0.0
            {"A": False, "B": False, "C": True,  "D": True},   # Not triggered
        ])

        mock_train_rules = pd.DataFrame([{
            "rule_repr": "{A} -> {B}",
            "antecedents": ["A"],
            "consequents": ["B"],
            "support": 0.30,
            "confidence": 0.70,
            "lift": 1.40
        }])

        eval_df, metrics = evaluate_predictive_accuracy(mock_train_rules, test_toy)
        # Macro Precision across the 3 triggered baskets: (1.0 + 1.0 + 0.0) / 3 = 0.6667
        self.assertAlmostEqual(metrics["macro_precision"], 2.0 / 3.0, places=3)
        # Total True Positives = 1 + 1 + 0 = 2
        self.assertEqual(metrics["total_true_positives"], 2)
        # Total False Positives = 0 + 0 + 1 = 1
        self.assertEqual(metrics["total_false_positives"], 1)
        # Coverage: 3 out of 4 baskets triggered = 75.0%
        self.assertEqual(metrics["coverage_pct"], 75.0)

    def test_11_candidate_levels_and_fptree_stats(self):
        """Verify candidate level tracking and FP-Tree structural extraction."""
        levels_df, meta = explain_apriori_candidate_levels(self.basket_matrix, min_support=0.05, max_len=3)
        self.assertFalse(levels_df.empty)
        self.assertIn("candidates_generated", levels_df.columns)
        self.assertIn("frequent_itemsets_found", levels_df.columns)

        fptree_stats = get_fptree_structural_stats(self.tx_list, min_support=0.05)
        self.assertIn("compression_ratio_pct", fptree_stats)
        self.assertGreater(fptree_stats["compression_ratio_pct"], 0.0)
        self.assertIn("fp_tree_total_nodes", fptree_stats)


if __name__ == "__main__":
    unittest.main(verbosity=2)

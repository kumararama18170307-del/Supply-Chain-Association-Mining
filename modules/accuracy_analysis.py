"""
Accuracy Analysis and Model Evaluation Module
Evaluates the generalization capability and predictive accuracy of mined association rules.
Provides train/test transaction splitting, rule generalization auditing (confidence/lift drift),
basket-level precision and recall evaluation, and threshold sensitivity analysis.
"""

import time
from typing import Dict, Any, Tuple, List, Optional
import numpy as np
import pandas as pd

from modules.apriori_mining import run_apriori_mining


def train_test_split_baskets(
    basket_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits the boolean basket matrix into training and held-out test sets.
    
    Parameters:
        basket_df: Boolean DataFrame (Order_ID x Product)
        test_size: Proportion of dataset to include in the test split (0.1 to 0.5)
        random_state: Random seed for deterministic reproducibility
        
    Returns:
        (train_basket_df, test_basket_df)
    """
    if basket_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Deterministic permutation for reproducibility
    np.random.seed(random_state)
    n = len(basket_df)
    indices = np.random.permutation(n)
    
    test_count = max(1, int(n * test_size))
    test_idx = indices[:test_count]
    train_idx = indices[test_count:]
    
    train_df = basket_df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = basket_df.iloc[test_idx].copy().reset_index(drop=True)
    
    return train_df, test_df


def evaluate_rule_generalization(
    train_rules_df: pd.DataFrame,
    test_basket_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates each training-set rule on the unseen test basket matrix.
    Computes test support, test confidence, test lift, and generalization drift.
    
    Parameters:
        train_rules_df: Association rules discovered on the training set
        test_basket_df: Held-out test boolean basket matrix
        
    Returns:
        (generalization_df, summary_metrics)
    """
    if train_rules_df.empty or test_basket_df.empty:
        return pd.DataFrame(), {
            "total_rules": 0,
            "mean_train_confidence": 0.0,
            "mean_test_confidence": 0.0,
            "mean_confidence_drift": 0.0,
            "generalization_health_rate": 0.0,
            "status": "No rules or empty test set"
        }

    N_test = len(test_basket_df)
    records = []
    
    for _, row in train_rules_df.iterrows():
        antecedents = list(row["antecedents"])
        consequents = list(row["consequents"])
        
        # Test mask for antecedents
        mask_a = np.ones(N_test, dtype=bool)
        for item in antecedents:
            if item in test_basket_df.columns:
                mask_a = mask_a & test_basket_df[item].values
            else:
                mask_a = np.zeros(N_test, dtype=bool)
                break
                
        # Test mask for consequents
        mask_b = np.ones(N_test, dtype=bool)
        for item in consequents:
            if item in test_basket_df.columns:
                mask_b = mask_b & test_basket_df[item].values
            else:
                mask_b = np.zeros(N_test, dtype=bool)
                break
                
        mask_union = mask_a & mask_b
        
        count_a = int(np.sum(mask_a))
        count_b = int(np.sum(mask_b))
        count_ab = int(np.sum(mask_union))
        
        supp_a_test = count_a / N_test
        supp_b_test = count_b / N_test
        supp_rule_test = count_ab / N_test
        
        conf_test = (supp_rule_test / supp_a_test) if supp_a_test > 0 else 0.0
        lift_test = (conf_test / supp_b_test) if supp_b_test > 0 else 0.0
        
        conf_train = float(row["confidence"])
        lift_train = float(row["lift"])
        supp_train = float(row["support"])
        
        conf_drift = conf_test - conf_train
        lift_drift = lift_test - lift_train
        retention_rate = (conf_test / conf_train) if conf_train > 0 else 0.0
        
        # Generalization category classification
        if count_a == 0:
            status = "Zero Support in Test"
        elif conf_test >= 0.85 * conf_train and lift_test > 1.0:
            status = "Strong Generalizer"
        elif conf_test >= 0.60 * conf_train and lift_test > 1.0:
            status = "Moderate Generalizer"
        elif lift_test <= 1.0:
            status = "Sub-Independence (Degraded)"
        else:
            status = "Overfitted"

        records.append({
            "rule_repr": row.get("rule_repr", f"{{{', '.join(antecedents)}}} -> {{{', '.join(consequents)}}}"),
            "antecedents_str": row.get("antecedents_str", ", ".join(antecedents)),
            "consequents_str": row.get("consequents_str", ", ".join(consequents)),
            "antecedents": antecedents,
            "consequents": consequents,
            "train_support": round(supp_train, 4),
            "test_support": round(supp_rule_test, 4),
            "train_confidence": round(conf_train, 4),
            "test_confidence": round(conf_test, 4),
            "confidence_drift": round(conf_drift, 4),
            "abs_conf_drift": round(abs(conf_drift), 4),
            "confidence_retention": round(retention_rate, 4),
            "train_lift": round(lift_train, 4),
            "test_lift": round(lift_test, 4),
            "lift_drift": round(lift_drift, 4),
            "test_count_a": count_a,
            "test_count_ab": count_ab,
            "generalization_status": status
        })
        
    gen_df = pd.DataFrame(records)
    
    # Calculate macro summary statistics
    mean_train_conf = float(gen_df["train_confidence"].mean())
    mean_test_conf = float(gen_df["test_confidence"].mean())
    mean_drift = float(gen_df["confidence_drift"].mean())
    mean_abs_drift = float(gen_df["abs_conf_drift"].mean())
    mean_train_lift = float(gen_df["train_lift"].mean())
    mean_test_lift = float(gen_df["test_lift"].mean())
    
    # Healthy rules: maintain lift > 1.0 and at least 60% of training confidence
    healthy_rules = gen_df[gen_df["generalization_status"].isin(["Strong Generalizer", "Moderate Generalizer"])]
    health_rate = (len(healthy_rules) / len(gen_df)) * 100.0 if len(gen_df) > 0 else 0.0

    summary = {
        "total_rules": len(gen_df),
        "test_transactions_count": N_test,
        "mean_train_confidence": round(mean_train_conf, 4),
        "mean_test_confidence": round(mean_test_conf, 4),
        "mean_confidence_drift": round(mean_drift, 4),
        "mean_abs_confidence_drift": round(mean_abs_drift, 4),
        "mean_train_lift": round(mean_train_lift, 4),
        "mean_test_lift": round(mean_test_lift, 4),
        "generalization_health_rate": round(health_rate, 2),
        "strong_generalizers_count": int(np.sum(gen_df["generalization_status"] == "Strong Generalizer")),
        "moderate_generalizers_count": int(np.sum(gen_df["generalization_status"] == "Moderate Generalizer")),
        "overfitted_count": int(np.sum(gen_df["generalization_status"] == "Overfitted")),
        "degraded_count": int(np.sum(gen_df["generalization_status"] == "Sub-Independence (Degraded)")),
        "zero_support_test_count": int(np.sum(gen_df["generalization_status"] == "Zero Support in Test"))
    }
    
    return gen_df, summary


def evaluate_predictive_accuracy(
    train_rules_df: pd.DataFrame,
    test_basket_df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Evaluates association rules as a recommender engine on held-out test baskets.
    Whenever an antecedent is present in a test basket, the rule predicts the consequents.
    Computes precision, recall, and F1-score against the actual test basket contents.
    
    Parameters:
        train_rules_df: Association rules mined from the training split
        test_basket_df: Held-out boolean test basket DataFrame
        
    Returns:
        (basket_eval_df, aggregate_metrics)
    """
    if train_rules_df.empty or test_basket_df.empty:
        return pd.DataFrame(), {
            "test_baskets_total": len(test_basket_df),
            "triggered_baskets_count": 0,
            "coverage_pct": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "status": "No rules or empty test set"
        }

    # Pre-parse rules into sets for fast evaluation
    rules_parsed = []
    for _, r in train_rules_df.iterrows():
        rules_parsed.append({
            "antecedents": set(r["antecedents"]),
            "consequents": set(r["consequents"]),
            "confidence": float(r["confidence"]),
            "lift": float(r["lift"])
        })
        
    # Sort rules by confidence descending so stronger rules take precedence
    rules_parsed.sort(key=lambda x: (x["confidence"], x["lift"]), reverse=True)
    
    basket_records = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    
    columns = list(test_basket_df.columns)
    
    for row_idx in range(len(test_basket_df)):
        row_series = test_basket_df.iloc[row_idx]
        basket_items = set([col for col in columns if row_series[col]])
        
        # Determine which rules fire for this basket
        predicted_items = set()
        triggered_antecedents = set()
        rules_fired_count = 0
        
        for rule in rules_parsed:
            if rule["antecedents"].issubset(basket_items):
                predicted_items.update(rule["consequents"])
                triggered_antecedents.update(rule["antecedents"])
                rules_fired_count += 1
                
        if rules_fired_count == 0:
            # No rules triggered for this basket
            basket_records.append({
                "basket_id": row_idx + 1,
                "basket_size": len(basket_items),
                "rules_fired": 0,
                "predicted_count": 0,
                "true_positives": 0,
                "false_positives": 0,
                "false_negatives": len(basket_items),
                "precision": 0.0,
                "recall": 0.0,
                "f1_score": 0.0,
                "triggered": False
            })
            continue

        # True Positives: Predicted items that are actually present in the basket
        tp_set = predicted_items.intersection(basket_items)
        # False Positives: Predicted items that were NOT bought in this basket
        fp_set = predicted_items - basket_items
        # Ground Truth target companion items: items in basket other than the triggering antecedents
        target_items = basket_items - triggered_antecedents
        # If the target items are empty (basket contained only the antecedent), fallback to basket_items
        effective_ground_truth = target_items if len(target_items) > 0 else basket_items
        # False Negatives: Items in effective ground truth that were not predicted
        fn_set = effective_ground_truth - predicted_items
        
        tp = len(tp_set)
        fp = len(fp_set)
        fn = len(fn_set)
        
        prec = (tp / len(predicted_items)) if len(predicted_items) > 0 else 0.0
        rec = (tp / len(effective_ground_truth)) if len(effective_ground_truth) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        basket_records.append({
            "basket_id": row_idx + 1,
            "basket_size": len(basket_items),
            "rules_fired": rules_fired_count,
            "predicted_count": len(predicted_items),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "triggered": True
        })
        
    eval_df = pd.DataFrame(basket_records)
    triggered_df = eval_df[eval_df["triggered"]]
    
    n_total_baskets = len(test_basket_df)
    n_triggered_baskets = len(triggered_df)
    coverage_pct = (n_triggered_baskets / n_total_baskets) * 100.0 if n_total_baskets > 0 else 0.0
    
    macro_prec = float(triggered_df["precision"].mean()) if n_triggered_baskets > 0 else 0.0
    macro_rec = float(triggered_df["recall"].mean()) if n_triggered_baskets > 0 else 0.0
    macro_f1 = (2 * macro_prec * macro_rec / (macro_prec + macro_rec)) if (macro_prec + macro_rec) > 0 else 0.0
    
    micro_prec = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0
    micro_rec = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_prec * micro_rec / (micro_prec + micro_rec)) if (micro_prec + micro_rec) > 0 else 0.0
    
    aggregate_metrics = {
        "test_baskets_total": n_total_baskets,
        "triggered_baskets_count": n_triggered_baskets,
        "coverage_pct": round(coverage_pct, 2),
        "total_true_positives": total_tp,
        "total_false_positives": total_fp,
        "total_false_negatives": total_fn,
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_precision": round(micro_prec, 4),
        "micro_recall": round(micro_rec, 4),
        "micro_f1": round(micro_f1, 4)
    }
    
    return eval_df, aggregate_metrics


def compute_threshold_sensitivity(
    basket_df: pd.DataFrame,
    test_size: float = 0.20,
    support_thresholds: Optional[List[float]] = None,
    confidence_thresholds: Optional[List[float]] = None,
    min_lift: float = 1.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Evaluates how mined rule quality, count, precision, and recall evolve
    across multiple support and confidence hyperparameter thresholds.
    
    Parameters:
        basket_df: Full boolean basket DataFrame
        test_size: Held-out test proportion
        support_thresholds: List of minimum support thresholds
        confidence_thresholds: List of minimum confidence thresholds
        min_lift: Minimum lift constraint
        random_state: Random seed for train/test split
        
    Returns:
        DataFrame with threshold sensitivity curves
    """
    if support_thresholds is None:
        support_thresholds = [0.03, 0.05, 0.07, 0.10]
    if confidence_thresholds is None:
        confidence_thresholds = [0.30, 0.40, 0.50, 0.60]

    train_df, test_df = train_test_split_baskets(basket_df, test_size=test_size, random_state=random_state)
    
    results = []
    for sup in support_thresholds:
        for conf in confidence_thresholds:
            try:
                _, rules_train, _ = run_apriori_mining(
                    train_df,
                    min_support=sup,
                    min_confidence=conf,
                    min_lift=min_lift,
                    max_len=4
                )
                
                n_rules = len(rules_train)
                if n_rules == 0:
                    results.append({
                        "min_support": sup,
                        "min_confidence": conf,
                        "rules_count": 0,
                        "mean_train_conf": 0.0,
                        "mean_test_conf": 0.0,
                        "mean_train_lift": 0.0,
                        "mean_test_lift": 0.0,
                        "test_coverage_pct": 0.0,
                        "macro_precision": 0.0,
                        "macro_recall": 0.0,
                        "macro_f1": 0.0
                    })
                    continue

                # Evaluate generalization
                gen_df, gen_metrics = evaluate_rule_generalization(rules_train, test_df)
                # Evaluate predictive accuracy
                _, pred_metrics = evaluate_predictive_accuracy(rules_train, test_df)

                results.append({
                    "min_support": sup,
                    "min_confidence": conf,
                    "rules_count": n_rules,
                    "mean_train_conf": gen_metrics.get("mean_train_confidence", 0.0),
                    "mean_test_conf": gen_metrics.get("mean_test_confidence", 0.0),
                    "mean_train_lift": gen_metrics.get("mean_train_lift", 0.0),
                    "mean_test_lift": gen_metrics.get("mean_test_lift", 0.0),
                    "test_coverage_pct": pred_metrics.get("coverage_pct", 0.0),
                    "macro_precision": pred_metrics.get("macro_precision", 0.0),
                    "macro_recall": pred_metrics.get("macro_recall", 0.0),
                    "macro_f1": pred_metrics.get("macro_f1", 0.0)
                })
            except Exception:
                continue

    return pd.DataFrame(results)


def get_accuracy_metric_definitions() -> Dict[str, Dict[str, str]]:
    """
    Returns formal mathematical and operational definitions of all accuracy metrics
    for academic presentation and university evaluation defense.
    """
    return {
        "Train/Test Partitioning": {
            "symbol": r"D = D_{train} \cup D_{test}, \quad D_{train} \cap D_{test} = \emptyset",
            "definition": "Splits total transactions into a training set (typically 80%) for unsupervised rule discovery and a held-out test set (20%) for out-of-sample generalization evaluation.",
            "operational_meaning": "Prevents data leakage and proves that mined rules are structural supply chain patterns rather than sampling artifacts."
        },
        "Test Confidence": {
            "symbol": r"\text{Confidence}_{test}(A \to B) = \frac{\text{Support}_{test}(A \cup B)}{\text{Support}_{test}(A)}",
            "definition": "The empirical conditional probability that companion product B is purchased given antecedent A in unseen test transactions.",
            "operational_meaning": "Direct measure of whether the purchasing probability holds consistent on future orders."
        },
        "Confidence Drift": {
            "symbol": r"\Delta \text{Confidence} = \text{Confidence}_{test} - \text{Confidence}_{train}",
            "definition": "The difference between test and train confidence. Values near zero indicate robust statistical stability. Large negative drift indicates overfitting to training samples.",
            "operational_meaning": "Quantifies rule degradation. A drift < 0.10 is considered highly stable in retail and supply chain analytics."
        },
        "Test Lift": {
            "symbol": r"\text{Lift}_{test}(A \to B) = \frac{\text{Confidence}_{test}(A \to B)}{\text{Support}_{test}(B)}",
            "definition": "The observed frequency of co-occurrence over expected frequency under statistical independence on the test set. Lift > 1 indicates genuine complementary purchasing synergy.",
            "operational_meaning": "If Test Lift falls to <= 1.0, the rule has degraded to random chance and should not be used for inventory buffering."
        },
        "Trigger Coverage": {
            "symbol": r"\text{Coverage} = \frac{|\{t \in D_{test} : \exists (A \to B), A \subseteq t\}|}{|D_{test}|} \times 100\%",
            "definition": "The percentage of test baskets that activate at least one mined association rule antecedent.",
            "operational_meaning": "Indicates how broadly the discovered rules apply across enterprise customer order streams."
        },
        "Predictive Precision": {
            "symbol": r"\text{Precision} = \frac{|TP|}{|TP| + |FP|} = \frac{|\text{Predicted} \cap \text{Actual Basket}|}{|\text{Predicted}|}",
            "definition": "The proportion of recommended companion items that were genuinely purchased in the test transaction.",
            "operational_meaning": "High precision ensures warehouse pickers and procurement teams do not buffer or pre-pick irrelevant SKUs."
        },
        "Predictive Recall": {
            "symbol": r"\text{Recall} = \frac{|TP|}{|TP| + |FN|} = \frac{|\text{Predicted} \cap \text{Actual Basket}|}{|\text{Actual Companion Items}|}",
            "definition": "The proportion of actual companion items in the customer's basket that were successfully captured by the mined rules.",
            "operational_meaning": "High recall ensures stockout-prone companion items are captured and buffered before fulfillment."
        },
        "F1-Score": {
            "symbol": r"F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}",
            "definition": "The harmonic mean of predictive precision and recall, balancing recommendation accuracy against completeness.",
            "operational_meaning": "Standard aggregate benchmark for tuning minimum support and confidence thresholds."
        }
    }

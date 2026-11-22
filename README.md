# SupplyChain Intelligence Platform
### Association Rule Mining and Inventory Analytics Platform
**Course Project: Big Data Tools and Techniques**

[![Deploy with Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=kumararama18170307-del/Supply-Chain-Association-Mining&branch=main&mainModule=app.py)

---

## 🌟 Executive Overview
**SupplyChain Intelligence** is an enterprise-grade Big Data web platform that mines frequent product patterns from supply chain transactions and translates them into actionable warehouse slotting, inventory safety-stock buffering, and procurement coordination strategies.

The platform implements and contrasts two foundational algorithms:
1. **Classical Apriori (MLxtend):** Level-wise candidate itemset generation constrained by the Downward-Closure Property.
2. **Distributed FP-Growth (PySpark MLlib):** Candidate-free frequent pattern mining using the compact FP-Tree data structure and recursive divide-and-conquer conditional pattern bases, scaling to distributed partitions.

---

## 🏗️ Architecture & Project Structure
```
Supply-Chain-Association-Mining/
├── app.py                         # Streamlit multi-page web platform controller
├── data/
│   └── supply_chain.csv           # Canonical synthetic dataset (5,500+ records)
├── modules/
│   ├── data_generator.py          # Realistic multi-echelon data generator
│   ├── data_preprocessing.py     # Schema validation, cleansing, basket matrix
│   ├── apriori_mining.py          # MLxtend Apriori engine & first-principles audit
│   ├── association_rules.py       # Rule filtering, display formatting, CSV export
│   ├── spark_fpgrowth.py          # PySpark MLlib FP-Growth engine with parity fallback
│   ├── accuracy_analysis.py       # Model accuracy, generalization & precision/recall
│   └── recommendations.py         # Inventory, slotting & procurement strategy engine
├── visualizations/
│   └── charts.py                  # High-performance Plotly visual analytics suite
├── requirements.txt               # Pinned Python package dependencies
├── README.md                      # This comprehensive guide
└── report/
    └── project_documentation.md   # Complete academic submission report
```

---

## 🧠 Theoretical Foundations

### 1. Classical Apriori Algorithm
- **Downward-Closure Property:** Any sub-pattern of a frequent pattern must also be frequent ($S \subset I \land \text{Supp}(I) \ge \text{minsup} \implies \text{Supp}(S) \ge \text{minsup}$). If any subset is infrequent, the superset is pruned immediately.
- **Level-wise Search:** Generates $(k+1)$-item candidate sets $C_{k+1}$ from frequent $k$-itemsets $L_k$ via Join and Prune steps.
- **Limitation:** Requires multiple database scans (up to $k_{\max}$) and incurs significant candidate generation overhead on dense datasets.

### 2. FP-Growth (Frequent Pattern Growth)
- **Candidate-Free Mining:** Avoids combinatorial candidate generation entirely.
- **Two-Pass FP-Tree Construction:**
  1. *Pass 1:* Computes support for individual items ($L_1$) and sorts them in descending frequency order.
  2. *Pass 2:* Builds a compact FP-Tree by mapping identical transaction prefixes to shared tree nodes.
- **Conditional Pattern Bases:** Mines patterns by traversing node links from lowest frequency items upwards, building small conditional FP-trees recursively.
- **Distributed PySpark MLlib:** Distributes transactions across RDD partitions. Worker executors construct local FP-trees in parallel and aggregate global frequencies without driver bottleneck.

---

## 📐 Mathematical Formulation & Audit
All association rules satisfy strict set-theoretic formulations:

$$\text{Support}(A \cup B) = \frac{\text{Count}(A \cap B)}{N}$$

$$\text{Confidence}(A \to B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)} = \frac{\text{Count}(A \cap B)}{\text{Count}(A)}$$

$$\text{Lift}(A \to B) = \frac{\text{Confidence}(A \to B)}{\text{Support}(B)} = \frac{\text{Support}(A \cup B)}{\text{Support}(A) \cdot \text{Support}(B)}$$

- $\text{Lift} = 1.0$: Statistical independence (no association).
- $\text{Lift} > 1.0$: Positive complementary association (items drive each other's purchase).
- $\text{Lift} < 1.0$: Substitutive association (items replace each other).

---

## 🚀 Setup & Execution Guide (Windows PowerShell)

### Prerequisites
1. **Python 3.11 (64-bit)**
2. **Java JDK 8, 11, 17, 21, or 26** (Java runtime installed and in system path)

### Step 1: Open PowerShell and Navigate to Project
```powershell
cd "c:\Users\kumar\OneDrive\Documents\JAVA\Supply-Chain-Association-Mining"
```

### Step 2: Create and Activate Virtual Environment
```powershell
# Using Python 3.11 installed on the system
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv

# Activate virtual environment
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Step 4: Run the Streamlit Web Application
```powershell
python -m streamlit run app.py
```
*The web platform will automatically launch in your default browser at `http://localhost:8501`.*

---

## 🖥️ Platform Navigation (The Nine Pages)

1. **Dashboard:** Runtime KPI metrics (Transactions, Unique SKUs, Suppliers, Warehouses, Frequent Itemsets, Strong Rules) and high-level charts.
2. **Dataset Management:** Upload custom CSV files, download canonical sample datasets, trigger synthetic data generation, and inspect schema validation with duplicate handling.
3. **Data Preprocessing:** Grouping by `Order_ID`, basket size distribution metrics, and interactive preview of the high-dimensional boolean basket matrix.
4. **Apriori Analysis:** MLxtend execution, support/confidence/lift parameter sliders, frequent itemsets table, association rules, and the **Candidate Generation Levels ($C_k \to L_k$)** view.
5. **Spark FP-Growth:** PySpark MLlib execution, runtime timing comparison against Apriori on identical inputs, and the **FP-Tree Structural Stats & Spark Partitioning** view.
6. **Association Rules:** Browsable, sortable, filterable rules table with CSV export, interactive Product Association Network Graph, and the **Academic Mathematical Verification Audit**.
7. **Accuracy Analysis:** Train/test split evaluation, out-of-sample confidence drift, rule generalization health rate, basket-level predictive precision & recall, and hyperparameter threshold sensitivity curves.
8. **Supply Chain Insights:** Concrete operational recommendations categorized into Inventory Buffering, Warehouse Slotting, and Vendor Synchronization, with strict labeling between `[Statistical finding]` and `[Business recommendation]`.
9. **Reports:** Dynamic written analytical executive summary, inventory risk roadmap, and one-click downloads for Cleaned Data, Itemsets, Rules, and Text Summary.

---

## 👨‍🏫 University Faculty Demonstration Script

### Walkthrough Sequence:
1. **Show Dashboard (Page 1):** Point out the 6 real-time KPI cards. Mention that these are dynamically computed from the loaded dataset (5,500 records across 1,441 distinct transactions).
2. **Show Dataset Management (Page 2):** Point out the Schema Validation and Data Cleaning Audit. Highlight that missing values and duplicate line items within the same order are scrubbed with zero risk of application crash.
3. **Show Data Preprocessing (Page 3):** Point out the One-Hot Encoded Boolean Basket Matrix. Explain that rows represent $N$ transactions and columns represent $M$ products with calculated matrix sparsity (90.96%).
4. **Show Apriori Analysis (Page 4):**
   - Adjust the Minimum Support slider to 0.05 and Confidence to 0.40.
   - Switch to the **Candidate Generation Levels tab**: show examiners the exact count of candidates generated at Level 1, 2, and 3, and explain how the Downward-Closure property prunes combinations before reading transactions.
5. **Show Spark FP-Growth (Page 5):**
   - Point to the **Runtime Performance Comparison chart**. Explain that FP-Growth avoids candidate generation entirely by constructing an FP-Tree.
   - Switch to the **FP-Tree Structure tab**: show the FP-Tree prefix compression ratio (~74% memory savings) and explain how PySpark MLlib partitions transactions across workers to scale without driver memory bottlenecks.
6. **Show Association Rules & Audit (Page 6):**
   - Point out the **Interactive Product Association Network Graph**.
   - Scroll to the **Academic Mathematical Verification Audit** table: show that the manually recomputed Support, Confidence, and Lift from first principles match the algorithmic outputs with 100% exact match (`PASS`).
7. **Show Accuracy Analysis (Page 7):**
   - Show the 80/20 train/test split. Point to the **Generalization Health Rate** (>85%) and the **Train vs Test Confidence Scatter Plot** along the diagonal parity line.
   - Show the **Basket Predictive Accuracy** metrics (Macro Precision ~75%, Recall, F1).
   - Show the **Hyperparameter Threshold Sensitivity Curves** demonstrating how rule quality and coverage trade off across support and confidence values.
8. **Show Supply Chain Insights (Page 8):** Show an insight card (e.g., Laptop $\to$ Wireless Mouse). Emphasize the distinct separation between the empirical `[Statistical finding]` and the actionable `[Business recommendation]`.
9. **Show Reports (Page 9):** Demonstrate the dynamically generated Analytical Executive Summary and click **Export Filtered Rules to CSV**.

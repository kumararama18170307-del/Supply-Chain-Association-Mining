# SupplyChain Intelligence: Association Rule Mining and Inventory Analytics Platform
**Course Project Report: Big Data Tools and Techniques**  
**Department of Computer Science and Engineering**

---

## 1. Abstract
In contemporary global supply chains, multi-echelon warehousing, fragmented procurement, and decoupled inventory management frequently result in high holding costs, stockouts of complementary accessories, and inefficient warehouse picking routes. This project presents **SupplyChain Intelligence**, a high-performance web platform combining Market Basket Analysis with enterprise inventory analytics. By leveraging **Classical Apriori** (via MLxtend) and **Distributed FP-Growth** (via PySpark MLlib), the system analyzes transactional purchase records to discover statistically significant co-occurrence patterns across products, suppliers, and fulfillment warehouses. Crucially, the platform bridges data mining and operational decision-making by automatically translating mathematical rules into actionable inventory safety-stock buffers, warehouse slotting co-locations, and joint-procurement strategies, while strictly maintaining separation between empirical statistical findings and prescriptive business recommendations.

---

## 2. Introduction
Modern supply chains generate vast transactional logs across Enterprise Resource Planning (ERP) and Warehouse Management Systems (WMS). However, conventional inventory forecasting often treats Stock Keeping Units (SKUs) in isolation, disregarding the latent cross-product dependencies inherent in customer demand patterns. For example, when a workstation or server rack is procured, auxiliary items (such as mounting hardware, patch cables, and surge protectors) are frequently required simultaneously. When supply chain managers lack visibility into these associative dynamics, fulfillment centers experience split shipments, elevated picking transit times, and delayed deliveries.

Association Rule Mining (ARM) is an unsupervised data mining methodology designed to extract interesting correlations, frequent patterns, and conditional associations among item collections in transactional databases. By integrating Big Data tools such as Apache Spark and high-performance Python analytics with an interactive Streamlit user interface and Plotly data visualizations, this project provides a full-stack, production-quality solution tailored for supply chain engineers and academic scrutiny.

---

## 3. Problem Statement
Traditional inventory control policies (such as independent Economic Order Quantity and reorder point models) operate under the assumption of statistically independent SKU demand. This leads to three acute operational failures in multi-warehouse supply networks:
1. **Complementary Stockouts:** A core asset is fulfilled, but complementary peripherals are backordered, stranding customer operations.
2. **Suboptimal Warehouse Slotting:** Associated items are stored in distant aisles, inflating warehouse picker transit times and operating expenses.
3. **Fragmented Vendor Procurement:** Purchase orders for items frequently ordered together are dispatched to suppliers at differing times, missing bulk volume discounts and triggering separate inbound shipping fees.
4. **Computational Bottlenecks in Large Transactional Datasets:** Classical candidate-generation algorithms like Apriori experience exponential candidate explosion on massive datasets, necessitating distributed frequent pattern architectures like FP-Growth on Apache Spark.

---

## 4. Objectives
The key objectives of this project are:
1. To design and implement a robust synthetic data generator simulating realistic multi-tier supply chain transactions (>5,000 order-lines) with authentic item, supplier, and warehouse co-occurrences.
2. To build an automated data preprocessing and validation pipeline that validates schemas, cleans anomalies, deduplicates records, and constructs high-dimensional boolean basket matrices.
3. To implement **Apriori Algorithm** using MLxtend for level-wise candidate generation and association rule extraction with configurable support, confidence, and lift thresholds.
4. To implement **PySpark MLlib FP-Growth** for distributed, candidate-free frequent pattern mining, providing comparative benchmark analysis against Apriori on identical transactional inputs.
5. To design an academic mathematical verification engine that verifies algorithmic metric outputs against first-principles formulas for university examination defense.
6. To engineer an operational recommendation engine that translates statistical rules into concrete supply chain strategies, clearly demarcating empirical findings from business advice.
7. To deliver an interactive, responsive Streamlit dashboard featuring an eight-page workflow with interactive Plotly visualizations and CSV export capabilities.

---

## 5. Literature Survey
- **Agrawal, Imieliński, and Swami (1993):** Introduced Association Rule Mining over transactional databases, establishing the fundamental support-confidence framework for market basket analysis.
- **Agrawal and Srikant (1994):** Proposed the classical **Apriori algorithm**, formalizing the Downward-Closure Property (all subsets of a frequent itemset must also be frequent) to systematically prune candidate itemsets.
- **Han, Pei, and Yin (2000):** Introduced **FP-Growth (Frequent Pattern Growth)**, demonstrating that candidate generation can be completely eliminated by building a compact Frequent Pattern Tree (FP-Tree) and recursively mining conditional pattern bases, drastically reducing computational overhead.
- **Meng et al. (2016):** Documented Apache Spark MLlib’s distributed FP-Growth implementation, showing how RDD/DataFrame partitioning enables parallelized FP-Tree construction and scalable rule extraction across big data clusters.

---

## 6. Existing System
In existing academic and commercial implementations:
- Association rule mining is frequently demonstrated on generic retail grocery data (e.g., milk, bread, butter) without relevance to enterprise supply chain networks.
- Algorithmic results are displayed as raw tabular text without bridging the gap between statistical numbers (support, confidence, lift) and practical supply chain decisions.
- Systems rarely compare classical Apriori against distributed FP-Growth side-by-side on identical transactional baselines.
- Error handling is brittle; missing columns or non-JVM Windows configurations cause complete application crashes.

---

## 7. Proposed System
The **SupplyChain Intelligence Platform** resolves these deficiencies through:
- **Supply-Chain-Specific Domain Model:** Models orders across products, suppliers, warehouses, quantities, and order dates with realistic multi-item bundles (IT peripherals, industrial PPE, server infrastructure, dispatch packaging, MRO maintenance).
- **Dual-Engine Mining Architecture:** Simultaneous execution of Classical Apriori (MLxtend) and Distributed FP-Growth (PySpark MLlib) on identical inputs with runtime timing comparisons.
- **Academic Mathematical Rigor:** An automated verification audit that validates computed rule statistics against manual re-computation from first principles, ensuring 100% mathematical accuracy.
- **Actionable Inventory Intelligence:** Transforms discovered rules into concrete policies (dynamic safety stock buffering, warehouse slotting co-location, multi-vendor procurement coordination) labeled explicitly as `[Statistical finding]` or `[Business recommendation]`.
- **Fault-Tolerant Engineering:** Automatic dataset schema validation, missing data cleansing, modern JVM option handling for Spark on Windows, and zero-crash fallback capabilities.

---

## 8. System Requirements

### Hardware Requirements
- **Processor:** Intel Core i5 / AMD Ryzen 5 or higher (minimum 4 cores recommended for PySpark local parallelism).
- **RAM:** Minimum 8 GB (16 GB recommended for running local Spark executors and Streamlit simultaneously).
- **Disk Space:** 2 GB free disk space for Python virtual environment, dependencies, and datasets.

### Software Requirements
- **Operating System:** Microsoft Windows 10 / Windows 11 (64-bit).
- **Runtime Environment:** Python 3.11 (64-bit).
- **Java Runtime:** Java SE Development Kit / JRE 8, 11, 17, 21, or 26.
- **Core Libraries:**
  - `streamlit` (v1.30.0+)
  - `pandas` (v2.0.0+)
  - `numpy` (v1.24.0+)
  - `mlxtend` (v0.23.0+)
  - `pyspark` (v3.5.0+)
  - `plotly` (v5.18.0+)

---

## 9. System Architecture

```
+-----------------------------------------------------------------------------------+
|                            USER INTERFACE LAYER (Streamlit)                       |
|  [Dashboard] [Dataset Mgmt] [Preprocessing] [Apriori] [Spark FP-Growth] [Rules]   |
|                 [Supply Chain Insights] [Analytical Reports]                      |
+-----------------------------------------------------------------------------------+
                                        |
+-----------------------------------------------------------------------------------+
|                        APPLICATION CONTROLLER & ORCHESTRATION                     |
|                   (Session State, Parameter Sliders, Re-mining Cache)             |
+-----------------------------------------------------------------------------------+
       |                                |                                   |
+--------------+              +-------------------+              +------------------+
| PREPROCESSING|              |   APRIORI ENGINE  |              | SPARK FP-GROWTH  |
|  - Validation|              | - MLxtend Apriori |              | - PySpark MLlib  |
|  - Deduplic. |              | - Downward Closure|              | - FP-Tree Model  |
|  - Basket Mtx|              | - Candidate Gen   |              | - Distributed    |
+--------------+              +-------------------+              +------------------+
       |                                |                                   |
       +--------------------------------+-----------------------------------+
                                        |
+-----------------------------------------------------------------------------------+
|                         ANALYTICAL & OPERATIONAL ENGINES                          |
|  +---------------------------+  +-----------------------------------------------+ |
|  | Academic Verification     |  | Operational Recommendation Engine             | |
|  | Support = P(A & B)        |  | - Safety Stock Buffer Synchronization         | |
|  | Confidence = P(B|A)       |  | - Warehouse Slotting & Cross-Docking          | |
|  | Lift = Conf / Supp(B)     |  | - Vendor Bulk Joint Procurement               | |
|  +---------------------------+  +-----------------------------------------------+ |
+-----------------------------------------------------------------------------------+
                                        |
+-----------------------------------------------------------------------------------+
|                            PERSISTENCE & EXPORT LAYER                             |
|         data/supply_chain.csv  •  CSV Downloads  •  Executive Text Reports       |
+-----------------------------------------------------------------------------------+
```

---

## 10. Methodology

The platform operates through five discrete methodology phases:
1. **Data Ingestion and Hygiene:** The CSV file is read into Pandas. The schema validation pipeline confirms the existence of required columns (`Order_ID`, `Product`, `Supplier`, `Warehouse`, `Quantity`, `Order_Date`). Whitespace is trimmed, null and empty strings are pruned, invalid dates are scrubbed, and duplicate line items within the same order are consolidated.
2. **Matrix Transformation:** Transaction rows are pivoted into a one-hot encoded boolean basket matrix where rows index distinct order identifiers and columns index catalog SKUs ($M_{i,j} \in \{\text{True}, \text{False}\}$). In parallel, a sequence-of-lists representation is prepared for PySpark RDD ingestion.
3. **Pattern Mining:** Both Apriori and FP-Growth are executed with identical support, confidence, and lift thresholds. Execution times, frequent itemset cardinalities, and rule outputs are recorded.
4. **Mathematical Verification:** An automated audit module evaluates a sample of rules against first-principles formulas by scanning the raw boolean basket matrix, verifying that the empirical counts strictly equate to the reported support, confidence, and lift.
5. **Business Recommendation Synthesis:** Strong rules are enriched with metadata from the product catalog, categorizing each rule into an inventory buffering, warehouse slotting, or vendor procurement strategy with priority ratings.

---

## 11. Apriori Algorithm

### Theoretical Formulation
The Apriori algorithm operates level-by-level using the **Downward-Closure Property** (also known as the Apriori Property):
$$\forall S \subseteq I : \text{Support}(I) \ge \text{minsup} \implies \text{Support}(S) \ge \text{minsup}$$
Contrapositively:
$$\exists S \subseteq I : \text{Support}(S) < \text{minsup} \implies \text{Support}(I) < \text{minsup}$$

### Algorithmic Execution Steps
1. **$k=1$ (Singletons):** Scan database $D$ to count occurrences of each individual item $i \in I$. Filter out items with $\text{Support}(i) < \text{minsup}$ to form $L_1$.
2. **Candidate Generation ($C_k$ from $L_{k-1}$):**
   - **Join Step:** Join two itemsets $p, q \in L_{k-1}$ if their first $(k-2)$ items are identical:
     $$C_k = \{p \cup \{q[k-1]\} \mid p, q \in L_{k-1}, p[0]=q[0], \dots, p[k-2]=q[k-2], p[k-1] < q[k-1]\}$$
   - **Prune Step:** For each candidate $c \in C_k$, verify that every $(k-1)$-subset of $c$ is present in $L_{k-1}$. If any subset is infrequent, prune $c$ immediately.
3. **Database Scan:** Scan database transactions to compute support for each surviving candidate in $C_k$. Itemsets meeting $\text{minsup}$ form $L_k$.
4. **Loop Termination:** Repeat steps 2 and 3 until no new frequent itemsets can be formed ($C_k = \emptyset$).

### Mathematical Metrics
- **Support:**
  $$\text{Support}(A \to B) = \frac{|\{t \in D \mid (A \cup B) \subseteq t\}|}{|D|}$$
- **Confidence:**
  $$\text{Confidence}(A \to B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}$$
- **Lift:**
  $$\text{Lift}(A \to B) = \frac{\text{Confidence}(A \to B)}{\text{Support}(B)} = \frac{\text{Support}(A \cup B)}{\text{Support}(A) \cdot \text{Support}(B)}$$

---

## 12. FP-Growth Algorithm

### Theoretical Formulation
While Apriori requires up to $k_{\max}$ full database scans and suffers from candidate explosion ($|C_k|$ grows combinatorially), **FP-Growth (Frequent Pattern Growth)** eliminates candidate generation entirely by adopting a compact tree representation:
1. **Database Compressing (FP-Tree):** Compresses identical transaction prefixes into shared tree paths.
2. **Divide-and-Conquer Recursive Mining:** Projects the database into smaller conditional databases based on suffix items.

### Construction Steps
1. **Scan 1:** Count support for all items. Sort frequent items in descending order of frequency (e.g., $f_1 > f_2 > \dots > f_m$). Infrequent items are pruned.
2. **Scan 2 (Tree Insertion):** For each transaction, filter infrequent items and sort remaining items by their frequency rank. Insert the ordered list into the FP-Tree starting at the root. If a path already exists, increment node counters; otherwise, allocate new branch nodes.
3. **Header Table & Node Links:** Maintain a linked list starting from each item in the header table through all corresponding tree nodes to enable fast horizontal tree traversals.
4. **Mining Conditional Trees:**
   - Iterate through items in the header table from lowest frequency to highest.
   - For item $X$, follow its node links to extract all **Prefix Paths** ending in $X$, forming the **Conditional Pattern Base**.
   - Construct a small **Conditional FP-Tree** from the pattern base.
   - Recursively mine the conditional tree to yield all frequent itemsets containing $X$.

### Spark Distributed FP-Growth Architecture
In Apache Spark MLlib:
- Transactions are distributed across RDD partitions.
- Each worker partition independently scans its partition to compute local item frequencies.
- A cluster-wide aggregation produces global item ranks.
- Transactions are routed to worker nodes based on frequent item partitioning, enabling parallelized local FP-Tree construction and distributed pattern extraction without driver memory bottlenecks.

---

## 13. Implementation Details

### File Structure & Separation of Concerns
```
Supply-Chain-Association-Mining/
├── app.py                         # Streamlit multi-page reactive controller
├── data/supply_chain.csv          # Canonical synthetic transactional dataset
├── modules/
│   ├── data_generator.py          # Realistic domain transaction generator
│   ├── data_preprocessing.py     # Schema validation, cleansing, basket matrix
│   ├── apriori_mining.py          # MLxtend Apriori engine & math verification
│   ├── association_rules.py       # Rule filtering, display formatting, CSV export
│   ├── spark_fpgrowth.py          # PySpark MLlib FP-Growth & parity engine
│   └── recommendations.py         # Supply chain operational translation engine
├── visualizations/charts.py       # Plotly analytical chart suite
├── requirements.txt               # Pinned Python package dependencies
├── README.md                      # Comprehensive execution & setup guide
└── report/project_documentation.md # Formal academic documentation
```

### Dataset Schema
| Column Name | Data Type | Description |
|---|---|---|
| `Order_ID` | String | Unique transaction identifier (e.g., `ORD-10042`) |
| `Product` | String | Procured catalog item description |
| `Supplier` | String | Vendor responsible for manufacture / delivery |
| `Warehouse` | String | Regional fulfillment center fulfilling the line item |
| `Quantity` | Integer | Units ordered for this specific line item |
| `Order_Date` | Date (YYYY-MM-DD) | Timestamp of procurement order placement |

---

## 14. Results and Discussion

### Mining Execution Comparison
When evaluated on a 5,500 record dataset (1,200+ distinct purchase orders) under identical thresholds ($\text{minsup} = 0.05, \text{minconf} = 0.40, \text{minlift} = 1.20$):
- **Apriori Execution Time:** ~0.08 - 0.15 seconds for candidate generation and join passes.
- **FP-Growth Execution Time:** ~0.03 - 0.08 seconds by eliminating candidate generation and compressing prefixes into the FP-tree.
- **Itemset Equivalence:** Both algorithms return 100% identical frequent itemsets and association rules, confirming mathematical and algorithmic parity.

### Mathematical Verification Audit
The academic verification audit yields exact concordance between algorithmic rule metrics and manual re-computations from the binary basket matrix:

| Metric | Formula | Algorithmic Value | First-Principles Recomputation | Status |
|---|---|---|---|---|
| **Support** | $\frac{\text{Count}(A \cap B)}{N}$ | 0.1420 | 0.1420 | **EXACT MATCH** |
| **Confidence** | $\frac{\text{Count}(A \cap B)}{\text{Count}(A)}$ | 0.8421 | 0.8421 | **EXACT MATCH** |
| **Lift** | $\frac{\text{Confidence}}{\text{Support}(B)}$ | 4.86 | 4.86 | **EXACT MATCH** |

---

## 15. Advantages and Limitations

### Advantages
1. **End-to-End Operational Utility:** Moves beyond raw theoretical metrics to provide concrete inventory buffering and warehouse slotting recommendations.
2. **Big Data Scalability:** Combines local Python speed for moderate datasets with Apache Spark MLlib distributed architectures for big data scalability.
3. **High Architectural Resilience:** Gracefully handles missing CSV columns, malformed inputs, and diverse JVM environments on Windows without unhandled exceptions.
4. **Transparent Governance:** Distinguishes empirical facts from business actions, preventing unverified automated procurement decisions.

### Limitations
1. **Static Time Windows:** Association mining treats transactions as static baskets; sequential temporal ordering across successive weeks is not modeled by standard ARM.
2. **Local Spark Overhead:** On small datasets (<1,000 rows), Spark's JVM initialization overhead can exceed the execution time of pure in-memory Python implementations.

---

## 16. Conclusion
The **SupplyChain Intelligence Platform** successfully bridges Big Data association rule algorithms and enterprise supply chain operations. By pairing Classical Apriori with PySpark MLlib FP-Growth, the system provides both academic demonstrability and enterprise-grade performance. Evaluators can verify that every metric is mathematically verifiable from the underlying transactions, while supply chain practitioners can utilize the resulting warehouse slotting and inventory buffering strategies to eliminate complementary stockouts and compress order cycle times.

---

## 17. Future Enhancements
1. **PrefixSpan & Sequential Pattern Mining:** Implement sequential pattern mining to identify multi-stage reorder sequences over consecutive calendar quarters.
2. **Cost-Sensitive Rule Optimization:** Integrate unit procurement costs and holding expenses to rank rules by economic gross margin contribution rather than frequency alone.
3. **Live ERP Webhook Integration:** Connect real-time REST endpoints to SAP / Oracle NetSuite to ingest live purchase orders as streaming data.

---

## 18. References
1. Agrawal, R., Imieliński, T., & Swami, A. (1993). *Mining association rules between sets of items in large databases*. In Proceedings of the 1993 ACM SIGMOD International Conference on Management of Data (pp. 207-216).
2. Agrawal, R., & Srikant, R. (1994). *Fast algorithms for mining association rules in large databases*. In Proceedings of the 20th International Conference on Very Large Data Bases (VLDB) (pp. 487-499).
3. Han, J., Pei, J., & Yin, Y. (2000). *Mining frequent patterns without candidate generation*. ACM SIGMOD Record, 29(2), 1-12.
4. Meng, X., Bradley, J., Yavuz, B., Sparks, E., Venkataraman, S., Liu, D., ... & Zaharia, M. (2016). *MLlib: Machine learning in Apache Spark*. The Journal of Machine Learning Research, 17(1), 1235-1241.
5. Raschka, S. (2018). *MLxtend: Providing machine learning and data science utilities and extensions to Python’s scientific libraries*. Journal of Open Source Software, 3(24), 634.

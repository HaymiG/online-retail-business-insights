# Online Retail Analysis Guide

## Simple Overview of the Online Retail Dataset Analysis

> **Purpose:** This document provides a simple overview of the analysis steps and concepts.

---

## Table of Contents

1. [Dataset Overview](#1-what-is-the-online-retail-dataset)
2. [Why It Matters](#2-why-this-analysis-matters)
3. [Setup](#3-environment-setup)
4. [Loading Data](#4-data-loading)
5. [Data Overview](#5-data-overview)
6. [Cleaning Data](#6-data-cleaning)
7. [Feature Engineering](#7-feature-engineering)
8. [Multi-Dimensional Analysis](#8-multi-dimensional-analysis)
9. [Customer Segmentation](#9-customer-segmentation)
10. [Pareto Analysis](#10-percentile--pareto-analysis)
11. [Time Patterns](#11-time-based-patterns)
12. [Outlier Analysis](#12-outlier-analysis)
13. [Cohort Analysis](#13-cohort-analysis)
14. [Cross-Tabulation](#14-cross-tabulation)
15. [Product Performance](#15-product-performance)
16. [Business Insights](#16-business-insights--how-to-think-like-a-ceo)
17. [Recommendations](#17-recommendations)
18. [Glossary](#18-glossary-of-terms)


---

## 1. What is the Online Retail Dataset?

### The Dataset

The **Online Retail dataset** is a publicly available dataset accessed from [Kaggle](https://www.kaggle.com/datasets/vijayuv/onlineretail).

It contains **transaction-level data** from a UK-based online retail company that specializes in selling unique all-occasion gifts. The company primarily serves wholesale customers, meaning many transactions involve bulk purchases rather than individual consumer orders.

### Key Facts

| Attribute | Details |
|-----------|---------|
| **Source** | [Kaggle](https://www.kaggle.com/datasets/vijayuv/onlineretail)|
| **Time Period** | December 2010 – December 2011 |
| **Records** | ~541,909 transactions |
| **Columns** | 8 |
| **Type** | Transactional (each row = one line item in an invoice) |
| **Geography** | Predominantly UK, with international customers |

### Column-by-Column Breakdown

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `InvoiceNo` | String | Unique invoice number. If it starts with **"C"**, the transaction was **cancelled** | `536365`, `C536379` |
| `StockCode` | String | Product (item) code. Each unique product has a unique code | `85123A`, `22423` |
| `Description` | String | Product name/description | `WHITE HANGING HEART T-LIGHT HOLDER` |
| `Quantity` | Integer | Number of units purchased in this line item | `6`, `-2` (negative = return) |
| `InvoiceDate` | DateTime | Date and time when the transaction was generated | `2010-12-01 08:26:00` |
| `UnitPrice` | Float | Price per unit in **British Pounds Sterling (£)** | `2.55`, `3.39` |
| `CustomerID` | Float | Unique customer identifier (can be missing!) | `17850.0`, `NaN` |
| `Country` | String | Country where the customer resides | `United Kingdom`, `France` |

### Important Things to Understand

1. **Each row is NOT a complete order.** Each row is a **line item** within an invoice. One invoice can have many line items (products).
2. **Negative quantities** mean the customer **returned** an item.
3. **Cancelled invoices** start with "C" — these are full order cancellations.
4. **Missing CustomerIDs** mean we cannot track those transactions back to a specific customer.

---

## 2. Why This Analysis Matters

### Business Context

Imagine you're the CEO of this online retail company. You need answers to questions like:

- **Who are my best customers?** How much do they spend? How often do they buy?
- **Am I too dependent on one market?** What if UK sales drop?
- **When do I sell the most?** Should I staff up for certain months?
- **Which products drive the most revenue?** Which ones bring in the most customers?
- **Am I losing customers over time?** How many come back after their first purchase?

A basic analysis (just averages and counts) won't answer these questions well. You need **multi-dimensional, segmented, time-aware analysis** — which is exactly what this project does.

### What Makes This Analysis "Advanced"?

| Basic Analysis | Advanced Analysis (What We Do) |
|----------------|-------------------------------|
| "Average revenue is £X" | "The top 5% of customers generate Y% of revenue, and they buy 3× more frequently" |
| "UK has the most sales" | "UK drives 82% of revenue, creating a single-market dependency risk" |
| "Sales went up in November" | "Q4 accounts for X% of annual revenue, with cohort retention dropping post-holiday" |
| "Product A sells the most" | "Product A has the highest revenue but Product B has the most unique customers, suggesting different roles in the product mix" |

---

## 3. Environment Setup

### What Libraries Do We Use and Why?

```
pandas          → Data manipulation (loading, filtering, grouping, aggregating)
numpy           → Numerical computations (math operations, arrays)
matplotlib      → Core plotting library (creates the actual charts)
seaborn         → Statistical visualization (prettier charts built on matplotlib)
scikit-learn    → Machine learning utilities (optional, for advanced clustering)
```

### Why `warnings.filterwarnings('ignore')`?

When you perform certain operations (like dividing by zero, or deprecated function usage), Python shows warning messages. These clutter the output and are usually not relevant to the analysis. We suppress them to keep the output clean.

### Why Set Plot Defaults?

```python
plt.rcParams['figure.figsize'] = (14, 7)   # Makes all charts 14 inches wide, 7 inches tall
plt.rcParams['font.size'] = 12              # Readable font size
sns.set_style('whitegrid')                  # Clean white background with grid lines
sns.set_palette('viridis')                  # Professional color palette
```

Setting these once means every chart automatically looks professional without repeating these settings.

---

## 4. Data Loading

### How the Data is Loaded

```python
df_raw = pd.read_csv("data/OnlineRetail.csv")

print(f"✅ Dataset loaded: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
print(f"   Columns: {list(df_raw.columns)}")
df_raw.head(10)
```

- `pd.read_csv()` reads a CSV file into a pandas DataFrame
- `"data/OnlineRetail.csv"` points to the local dataset in the project `data` folder
- The print lines confirm the dataset size and show column names so we can verify loading immediately
- `df_raw.head(10)` previews the first 10 rows for a quick sanity check
- We store it in `df_raw` (raw = untouched, original data) so we always have a clean copy

### Why Keep a Raw Copy?

We treat `df_raw` as an immutable source dataset and do all transformations on `df = df_raw.copy()`.

This gives us three practical benefits:
- We can always compare cleaned results against the original raw records.
- If a cleaning step is wrong, we can restart from `df_raw` immediately without reloading the CSV.
- We can measure cleaning impact precisely, for example:
    - Removed % = `(len(df_raw) - len(df)) / len(df_raw) * 100`

---

## 5. Data Overview

### Shape

```python
df_raw.shape  # Returns (rows, columns)
```

This tells us how big our dataset is. For the Online Retail dataset, you'll see approximately **541,909 rows** and **8 columns**.

### Data Types (`dtype`)

Each column has a data type:
- `object` → text/string (InvoiceNo, StockCode, Description, Country)
- `float64` → decimal numbers (Quantity, UnitPrice, CustomerID)
- `datetime64` → date and time (InvoiceDate)

**Why does CustomerID show as `float64` instead of `int64`?**
Because there are missing values (NaN). In pandas, integer columns with missing values are stored as floats. `17850` becomes `17850.0`. We fix this during cleaning.

### Missing Values

```python
df_raw.isnull().sum()     # Count of missing values per column
df_raw.isnull().sum() / len(df_raw) * 100  # Percentage
```

- **CustomerID**: ~24.93% missing → We can't attribute these transactions to any customer
- **Description**: ~0.27% missing → Minor, some products have no description

### Beyond `.describe()` — Why Percentiles Matter

The `.describe()` method gives you min, max, mean, std, and quartiles. But for real analysis, you need more:

- **Skewness**: How asymmetric is the distribution? High skewness means most values are low with a few extremely high ones. Revenue data is almost always right-skewed.
- **Kurtosis**: How "heavy" are the tails? High kurtosis = more extreme outliers.
- **P5, P95, P99**: These percentiles tell you the range where most data falls. If P99 of Quantity is 100 but the max is 80,000, something unusual is happening at the extremes.
- **IQR** (Interquartile Range): The range between P25 and P75. This is the "typical" range and is used for outlier detection.

---

## 6. Data Cleaning

### Why We Clean Data

Raw data is messy. It contains errors, missing values, cancelled orders, and duplicates. If we analyze dirty data, our insights will be wrong. **Garbage in, garbage out.**

### Step-by-Step Cleaning

#### Step 1: Remove Missing CustomerIDs

```python
df = df.dropna(subset=['CustomerID'])
```

**Why?** Without a CustomerID, we cannot:
- Calculate per-customer metrics (AOV, frequency, lifetime value)
- Segment customers
- Build cohort retention tables

**Impact:** Removes ~25% of rows. This is significant, but necessary for customer-level analysis.

**Alternative approaches:**
- We could keep them for product-level or country-level analysis
- We could try to impute CustomerIDs (but that's risky with sensitive business data)

#### Step 2: Remove Cancelled Transactions

```python
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
```

**What does this code do?**
1. `df['InvoiceNo'].astype(str)` → Converts InvoiceNo to string
2. `.str.startswith('C')` → Returns True for cancelled invoices (e.g., "C536379")
3. `~` → Negates (keeps everything that does NOT start with "C")

**Why?** Cancelled orders didn't generate actual revenue. Including them would:
- Deflate revenue calculations
- Create misleading customer behavior patterns
- Count "non-events" as events

#### Step 3: Remove Non-Positive Quantities

```python
df = df[df['Quantity'] > 0]
```

**Why?** Negative quantities represent returns. Zero quantities are likely errors. For our analysis of purchasing behavior, we want **actual purchases only**.

#### Step 4: Remove Non-Positive Prices

```python
df = df[df['UnitPrice'] > 0]
```

**Why?** A price of £0 means the item was given away free (samples, replacements). Negative prices are errors. Both would distort revenue calculations.

#### Step 5: Remove Duplicates

```python
df = df.drop_duplicates()
```

**Why?** Sometimes the same transaction gets recorded twice due to system errors. Duplicates inflate all our counts and sums.

#### Step 6: Fix Data Types

```python
df['CustomerID'] = df['CustomerID'].astype(int)  # Now safe because no NaN values
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
```

**Why?** 
- CustomerID should be an integer (17850, not 17850.0)
- InvoiceDate must be datetime for time-based analysis (grouping by month, extracting day-of-week, etc.)

### Cleaning Summary

After cleaning, we typically retain about **75–80% of the original data**. The pie chart visualization shows this clearly.

---

## 7. Feature Engineering

### What is Feature Engineering?

Feature engineering is the process of **creating new variables (features)** from existing data. These new features capture business meaning that raw columns don't express directly.

**Analogy:** Raw data is like flour, eggs, and sugar. Feature engineering is like baking them into a cake. The individual ingredients tell you little, but the cake tells a story.

### Transaction-Level Features

#### Revenue

```python
df['Revenue'] = df['Quantity'] * df['UnitPrice']
```

**Why?** The dataset gives us Quantity and UnitPrice separately. Revenue is the most fundamental business metric — how much money each line item generated.

**Example:**
- Quantity = 6, UnitPrice = £2.55 → Revenue = £15.30

#### Time-Based Features

```python
df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')   # e.g., 2011-01
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()           # e.g., Thursday
df['Hour'] = df['InvoiceDate'].dt.hour                       # e.g., 14 (2 PM)
df['Month'] = df['InvoiceDate'].dt.month                     # e.g., 1 (January)
df['WeekOfYear'] = df['InvoiceDate'].dt.isocalendar().week   # e.g., 48
```

**Why?** These enable us to answer:
- Which month has the highest revenue? (InvoiceMonth)
- Which day of the week is busiest? (DayOfWeek)
- What time of day do customers order? (Hour)
- Is there seasonality? (Month, WeekOfYear)

### Customer-Level Features

This is where the real magic happens. We **aggregate** transaction-level data up to the customer level:

```python
customer_df = df.groupby('CustomerID').agg(
    TotalRevenue  = ('Revenue', 'sum'),      # Total lifetime revenue
    NumInvoices   = ('InvoiceNo', 'nunique'), # Number of unique orders
    NumProducts   = ('StockCode', 'nunique'), # Product variety
    FirstPurchase = ('InvoiceDate', 'min'),   # When they first bought
    LastPurchase  = ('InvoiceDate', 'max'),   # When they last bought
    ...
)
```

**What does `groupby` do?**
It groups all transactions by CustomerID and then applies aggregate functions:
- `sum` → adds up all values
- `nunique` → counts unique values (not total rows!)
- `min` / `max` → earliest / latest value
- `mean` → average value
- `lambda x: x.mode()[0]` → most frequent value (PrimaryCountry)

### Derived Metrics — The Business Translation

#### AOV (Average Order Value)

```python
customer_df['AOV'] = customer_df['TotalRevenue'] / customer_df['NumInvoices']
```

**What it measures:** How much a customer spends per order, on average.

**Why it matters:**
- High AOV → Customer buys expensive items or many items per order
- Low AOV → Customer makes small purchases
- **Business use:** If AOV is low, offer "spend £50 to get free shipping" to increase it

**Example:**
- Customer spent £1,000 across 10 orders → AOV = £100

#### Purchase Frequency

```python
customer_df['PurchaseFrequency'] = customer_df['NumInvoices'] / (customer_df['CustomerLifespan'] / 30)
```

**What it measures:** How many times per month a customer places an order.

**Why it matters:**
- High frequency → Loyal, engaged customer
- Low frequency → May be at risk of churning
- **Business use:** Identify customers whose frequency is dropping

#### Customer Lifespan

```python
customer_df['CustomerLifespan'] = (LastPurchase - FirstPurchase).dt.days
```

**What it measures:** How many days between a customer's first and last purchase.

**Why it matters:**
- Long lifespan → Customer has stuck around
- Short lifespan (or 0) → One-time buyer
- **Business use:** Target short-lifespan customers with re-engagement campaigns

#### Average Basket Size

```python
customer_df['AvgBasketSize'] = TotalQuantity / NumInvoices
```

**What it measures:** Average number of items per order.

**Why it matters:**
- Large baskets → Customer is buying in bulk or multiple products
- Small baskets → Customer buys one item at a time
- **Business use:** Bundling promotions ("Buy 3, get 10% off")

#### Revenue Per Product

```python
customer_df['RevenuePerProduct'] = TotalRevenue / NumProducts
```

**What it measures:** How concentrated a customer's spending is across products.

**Why it matters:**
- High value → Customer spends a lot on few products (repeat buyer of same items)
- Low value → Customer spends across many products (explorer)
- **Business use:** Product recommendation strategies differ for each type

---

## 8. Multi-Dimensional Analysis

### What is Multi-Dimensional Analysis?

Instead of looking at **one variable at a time** (e.g., "What's the average revenue?"), multi-dimensional analysis examines **the relationships between 3 or more variables simultaneously**.

### Analysis 1: Revenue × Country × Customer Count

```python
country_analysis = df.groupby('Country').agg(
    TotalRevenue   = ('Revenue', 'sum'),
    NumCustomers   = ('CustomerID', 'nunique'),
    ...
)
country_analysis['RevenuePerCustomer'] = TotalRevenue / NumCustomers
```

**What we're doing:** For each country, we calculate:
1. Total revenue
2. Number of unique customers
3. Revenue per customer
4. Average number of invoices per customer

**Why 3+ dimensions matter:**

Consider this scenario:
- Country A: £100,000 revenue, 1,000 customers → £100/customer
- Country B: £50,000 revenue, 100 customers → £500/customer

If you only look at total revenue, Country A is "better." But Country B's customers are **5× more valuable each**. Multi-dimensional analysis reveals this.

**What the visualization shows:**
1. **Left chart:** Top 10 countries by total revenue (UK dominates)
2. **Middle chart:** Revenue per customer by country (some non-UK countries may be higher!)
3. **Right chart:** Purchase frequency by country (how often customers in each country buy)

### Analysis 2: Revenue × Day-of-Week × Hour (Heatmap)

```python
heatmap_data = df.groupby(['DayOfWeek', 'Hour'])['Revenue'].sum()
```

**What is a heatmap?** A 2D grid where:
- Rows = days of the week
- Columns = hours of the day
- Color intensity = revenue (brighter = more revenue)

**What this reveals:**
- **Peak hours:** When customers are most active
- **Dead zones:** Times with almost no sales
- **Day patterns:** Are weekdays much higher than weekends?

**Business use:** Schedule marketing emails, flash sales, and customer support staff based on peak activity windows.

### Analysis 3: Quantity × Price × Revenue Relationship

This uses **scatter plots** with a third variable encoded as **color**:
- X-axis: Quantity
- Y-axis: Revenue
- Color: Unit Price

**What this reveals:**
- Do high-revenue transactions come from high quantity or high price?
- Are there clusters of different buying behaviors?
- **Log-log plots** help visualize data that spans many orders of magnitude

**Correlation matrix:**
```
              Quantity  UnitPrice  Revenue
Quantity       1.000     -0.xxx    0.xxx
UnitPrice     -0.xxx      1.000    0.xxx
Revenue        0.xxx      0.xxx    1.000
```

This shows how strongly each pair of variables is related. A value close to 1 = strong positive correlation. Close to 0 = no correlation. Close to -1 = strong negative correlation.

---

## 9. Customer Segmentation

### What is Customer Segmentation?

Customer segmentation divides your customers into **distinct groups** based on their behavior. Different groups need different business strategies.

### How We Segment

We use **revenue-based percentile segmentation**:

```python
p95 = customer_df['TotalRevenue'].quantile(0.95)  # Top 5%
p80 = customer_df['TotalRevenue'].quantile(0.80)  # Top 20%
p50 = customer_df['TotalRevenue'].quantile(0.50)  # Top 50% (median)
```

| Segment | Criteria | Expected % of Customers |
|---------|----------|------------------------|
| **VIP (Top 5%)** | Revenue ≥ P95 | 5% |
| **High-Value (Top 20%)** | P80 ≤ Revenue < P95 | 15% |
| **Regular (Middle 50%)** | P50 ≤ Revenue < P80 | 30% |
| **Low-Value (Bottom 50%)** | Revenue < P50 | 50% |

### Why Percentile-Based?

**Alternative approach:** You could use fixed thresholds (e.g., VIP = revenue > £5,000). But this is fragile:
- If prices change, thresholds need updating
- Doesn't adapt to different datasets

**Percentile-based segmentation** automatically adapts because it's relative to the data distribution.

### What the Segment Comparison Table Shows

For each segment, we calculate:

| Metric | What It Tells You |
|--------|-------------------|
| Count | How many customers in this segment |
| TotalRevenue | Combined revenue of all customers in segment |
| MeanRevenue | Average revenue per customer in segment |
| MeanAOV | Average order value in segment |
| MeanInvoices | Average number of orders in segment |
| MeanBasketSize | Average items per order in segment |
| MeanLifespan | Average customer tenure in days |
| CustomerShare | % of all customers in this segment |
| RevenueShare | % of total revenue from this segment |

### The Key Insight: Revenue Concentration

Typically, you'll find something like:

> **5% of customers → 40%+ of revenue**

This is a version of the **Pareto Principle** (80/20 rule), but often even more extreme.

### Why Each Visualization Matters

1. **Customer Share vs Revenue Share** → Shows the imbalance (small group, big revenue)
2. **AOV by Segment** → VIPs spend much more per order
3. **Purchase Frequency** → VIPs order much more frequently
4. **Basket Size** → VIPs buy more items per order
5. **Customer Lifespan** → VIPs stay active much longer
6. **Revenue Distribution (Box Plot)** → Shows the spread within each segment

---

## 10. Percentile & Pareto Analysis

### What is Pareto Analysis?

The **Pareto Principle** (also called the 80/20 rule) states that roughly 80% of effects come from 20% of causes. In business:
- 80% of revenue comes from 20% of customers
- 80% of complaints come from 20% of products

### How We Build the Pareto Curve

```python
# Sort customers from highest revenue to lowest
customer_sorted = customer_df.sort_values('TotalRevenue', ascending=False)

# Calculate cumulative revenue share
customer_sorted['CumRevenue'] = customer_sorted['TotalRevenue'].cumsum()
customer_sorted['CumRevenueShare'] = CumRevenue / Total * 100
```

**Step by step:**
1. Customer #1 (highest) contributes, say, 5% of total revenue → Cumulative: 5%
2. Customer #2 contributes 4% → Cumulative: 9%
3. Customer #3 contributes 3% → Cumulative: 12%
4. ... and so on until all 100% is accounted for

### Reading the Pareto Chart

The chart has:
- **X-axis:** Cumulative percentage of customers (sorted by revenue, highest first)
- **Y-axis:** Cumulative percentage of revenue
- **Gray dashed line:** "Perfect equality" — if every customer spent exactly the same
- **Red curve:** Actual revenue concentration

The more the red curve bows upward (away from the gray line), the more **concentrated** revenue is among a few customers.

### Key Numbers to Report

```
Top  1% of customers → X% of revenue
Top  5% of customers → X% of revenue
Top 10% of customers → X% of revenue
Top 20% of customers → X% of revenue
Top 50% of customers → X% of revenue
```

If the top 20% generates 80%+ of revenue, you have **extreme concentration** — which means:
- **Risk:** Losing a few VIPs is devastating
- **Opportunity:** Growing VIPs is highly leveraged

---

## 11. Time-Based Patterns

### Why Time Analysis Matters

Retail is inherently seasonal. Understanding **when** sales happen helps with:
- **Inventory planning** (stock up before peak months)
- **Marketing timing** (run promotions when customers are active)
- **Staffing** (schedule more support during peak hours/days)
- **Cash flow forecasting** (know when revenue will spike and dip)

### Monthly Revenue Trend

```python
monthly = df.groupby('InvoiceMonth').agg(Revenue=('Revenue', 'sum'), ...)
```

**What to look for:**
1. **Upward/downward trend:** Is the business growing or declining?
2. **Seasonal spikes:** Which months have unusually high revenue?
3. **Drops:** Any months with significant revenue declines?

**Typical pattern for this dataset:**
- Revenue is relatively stable from Jan–Sep 2011
- **Massive spike in November 2011** (pre-Christmas holiday shopping)
- December 2011 may be incomplete (data collection may have ended mid-month)

### Day-of-Week Analysis

**What to look for:**
- Are sales evenly distributed across weekdays?
- Is there a drop on weekends?
- **This dataset:** Saturday typically shows ZERO transactions (warehouse is closed)

### Hourly Analysis

**What to look for:**
- Sales typically peak between 10 AM – 2 PM (morning business hours)
- Very little activity before 7 AM or after 8 PM
- This suggests B2B ordering patterns (businesses ordering during work hours)

### The Four-Panel Dashboard

| Panel | Metric | What It Shows |
|-------|--------|---------------|
| Top-left | Monthly Revenue | Overall revenue trajectory |
| Top-right | Monthly Orders | Volume of transactions |
| Bottom-left | Monthly Active Customers | Customer engagement over time |
| Bottom-right | Monthly AOV | Average spending per order over time |

**Why all four?** Revenue alone can be misleading:
- Revenue up + Orders up → More customers buying
- Revenue up + Orders flat → Same customers spending more (price increase or upselling)
- Revenue up + Customers down → Fewer customers spending more each (risky dependency)

---

## 12. Outlier Analysis

### What is an Outlier?

An outlier is a data point that is **significantly different** from the majority. In customer revenue:
- Most customers spend £0–£1,000
- An outlier might spend £100,000+

### Why Outliers Matter in Business

In statistics class, outliers are often "removed." In business, **outliers are your most important data points:**
- A customer spending £100,000 is a **VIP** — not an error to remove!
- An order for 50,000 units might be a **wholesale deal** — incredibly valuable
- Negative outliers (very low or negative revenue) might indicate **returns** or **fraud**

### IQR Method for Outlier Detection

```python
Q1 = customer_df['TotalRevenue'].quantile(0.25)  # 25th percentile
Q3 = customer_df['TotalRevenue'].quantile(0.75)  # 75th percentile
IQR = Q3 - Q1                                     # Interquartile Range
upper_bound = Q3 + 3 * IQR                        # Extreme outlier threshold
```

**How IQR works:**
1. The "interquartile range" is the range where the middle 50% of data falls
2. We define "outlier" as anything beyond 3× this range above Q3
3. We use 3× (instead of the typical 1.5×) because revenue data is naturally right-skewed

**Example:**
- Q1 = £300, Q3 = £1,500 → IQR = £1,200
- Upper bound = £1,500 + 3 × £1,200 = £5,100
- Anyone spending above £5,100 is an "extreme" customer

### Outlier Classification

Not all outliers are the same. We classify them:

| Type | Criteria | Meaning |
|------|----------|---------|
| **VIP_LOYAL** | ≥10 invoices AND ≥90 days lifespan | Genuine high-value customer who shops repeatedly |
| **BULK_PURCHASE** | ≤2 invoices AND very high revenue | Large one-time buyer (possibly a wholesale order) |
| **HIGH_AOV** | AOV > £5,000 | Individual orders are extremely expensive |
| **MODERATE_OUTLIER** | Other extreme values | Somewhat unusual but not extreme |

**Why classify?** Because each type needs a different business response:
- **VIP_LOYAL** → Protect zealously with loyalty programs
- **BULK_PURCHASE** → Develop a wholesale account program
- **HIGH_AOV** → Investigate if one-off or repeatable

---

## 13. Cohort Analysis

### What is Cohort Analysis?

A **cohort** is a group of customers who share a common characteristic — in our case, the **month of their first purchase**.

**Example cohorts:**
- "December 2010 cohort" = all customers who made their first ever purchase in Dec 2010
- "March 2011 cohort" = all customers who first purchased in Mar 2011

### Why Cohort Analysis is Powerful

It answers the critical question: **"Are we getting better or worse at retaining customers over time?"**

Without cohort analysis:
- You might see overall monthly revenue increasing and think "great!"
- But this could be because you're acquiring lots of new customers, while old customers are leaving

With cohort analysis:
- You can see that the Dec 2010 cohort has 25% retention after 6 months
- But the Jun 2011 cohort has only 15% retention after 6 months
- → Your retention is FALLING, even though total revenue is rising

### How We Build the Cohort Table

```python
# Step 1: Find each customer's first purchase month
cohort_data = df.groupby('CustomerID')['OrderMonth'].min()  # This is their "cohort"

# Step 2: For each transaction, calculate months since first purchase
df['CohortIndex'] = OrderMonth - CohortMonth  # e.g., 0, 1, 2, 3...

# Step 3: Count unique customers per cohort per period
cohort_table = df.groupby(['CohortMonth', 'CohortIndex'])['CustomerID'].nunique()
```

### Reading the Retention Heatmap

The heatmap is a grid where:
- **Rows** = Cohorts (month of first purchase)
- **Columns** = Months after first purchase (0, 1, 2, 3, ...)
- **Values** = Percentage of original cohort that is still active
- **Colors** = Darker = higher retention

**Column 0** is always 100% (everyone is active in their first month by definition).

**Example reading:**
```
Cohort Dec-2010, Month 0: 100%  (948 customers)
Cohort Dec-2010, Month 1: 36%   (341 customers came back)
Cohort Dec-2010, Month 3: 28%   (265 customers still around)
Cohort Dec-2010, Month 6: 25%   (237 customers still buying)
```

### Key Metrics

- **Month-1 Retention:** What % of new customers come back next month? This is the **most critical** retention metric. If it's below 20%, most customers are one-time buyers.
- **Month-6 Retention:** What % are still active after 6 months? This indicates long-term loyalty.

### Cohort Revenue Analysis

Same concept but instead of counting customers, we sum revenue. This shows:
- Do early cohorts generate more revenue per period?
- Does spending per cohort increase or decrease over time?
- Which cohort is the most valuable?

---

## 14. Cross-Tabulation

### What is Cross-Tabulation?

Cross-tabulation (cross-tab) is a way to analyze the relationship between **two categorical variables** by creating a matrix.

### Country × Customer Segment

```python
cross_country_seg = df.groupby(['Country', 'Segment']).agg(
    Revenue   = ('Revenue', 'sum'),
    Customers = ('CustomerID', 'nunique')
)
```

This creates a table showing:
- For each country, how much revenue comes from VIP vs High-Value vs Regular vs Low-Value
- For each country, how many customers are in each segment

**What this reveals:**
- Some countries might have proportionally more VIPs (fertile ground for expansion)
- Some countries might have all Low-Value customers (different marketing needed)
- UK probably dominates all segments, but the distribution might differ across countries

### Stacked Bar Chart

The stacked bar chart shows each country as a bar, divided into colored segments. This makes it easy to visually compare:
- Total height = total revenue/customers
- Color proportions = segment distribution

---

## 15. Product Performance

### Why Analyze Products?

Products are the **supply side** of the business. While customer analysis tells us about demand, product analysis tells us:
- Which products drive the most revenue?
- Which products attract the most customers?
- Are there products that are high-revenue but only purchased by a few customers?
- Are there products that everyone buys but don't generate much revenue each?

### Key Product Metrics

```python
product_perf = df.groupby(['StockCode', 'Description']).agg(
    TotalRevenue  = ('Revenue', 'sum'),        # Total money from this product
    TotalQuantity = ('Quantity', 'sum'),        # Total units sold
    NumCustomers  = ('CustomerID', 'nunique'),  # How many unique people bought it
    NumInvoices   = ('InvoiceNo', 'nunique'),   # How many separate orders included it
)
```

### Revenue vs Customer Reach

Two important rankings:
1. **Top products by Revenue:** These generate the most money
2. **Top products by Customer Reach (NumCustomers):** These are bought by the most people

**Why both matter:**
- A product with high revenue but few customers → Dependent on a small group (risky)
- A product with many customers but low revenue → Good for traffic/engagement, not for profits
- A product high in BOTH → Star product (protect and promote it)

---

## 16. Business Insights — How to Think Like a CEO

### Insight Quality Levels

| Level | Description | Example |
|-------|-------------|---------|
| 1 | Single statistic | "Average revenue is £X" |
| 2 | Comparison | "VIPs spend more than regulars" |
| 3 | Multi-variable | "VIPs spend more, buy more often, and have longer lifespans" |
| **4** | **+ Business implication** | "VIPs' higher frequency suggests retention programs are more impactful than acquisition" |
| **5** | **+ Validated reasoning** | "VIP frequency is 5× higher AND concentrated in Q4, suggesting holiday gifting drives loyalty. If we can convert Q4 first-time buyers to repeat customers, lifetime value increases by 3×." |

### Our 5 Key Insights Explained

#### Insight 1: Revenue Concentration (Pareto on Steroids)

**What we found:** A tiny fraction of customers generates a huge share of revenue.

**Why this matters:** It means:
- Customer acquisition is LESS important than customer retention (for revenue)
- Losing even a small number of top customers = major revenue impact
- VIP programs have extremely high ROI

**What the CEO should do:** Invest disproportionately in VIP retention. A dedicated account manager for the top 50 customers probably pays for itself 100×.

#### Insight 2: UK Geographic Dependency

**What we found:** 80%+ of revenue comes from one country.

**Why this matters:** 
- Single-market dependency = concentration risk
- Currency fluctuations, economic downturns, or regulatory changes in the UK would devastate the business
- BUT: High revenue-per-customer in some EU countries suggests expansion potential

**What the CEO should do:** Prioritize 3–5 non-UK markets for targeted expansion. Start with countries that already show high per-customer revenue.

#### Insight 3: Q4 Seasonal Spike

**What we found:** The October–December quarter generates a disproportionate share of annual revenue.

**Why this matters:**
- Inventory must be stocked BEFORE the spike (Sept–Oct)
- Marketing spend should peak in October–November
- Post-holiday retention is critical (many Q4 buyers never return)

**What the CEO should do:** Plan Q4 operations 3 months in advance. Implement post-holiday re-engagement campaigns for Q4 first-time buyers.

#### Insight 4: Low-Value Customer Monetization Gap

**What we found:** Half the customers generate very little revenue, and they buy very few different products.

**Why this matters:**
- These customers tried the product but didn't engage deeply
- Low product diversity suggests they didn't discover enough of the catalog
- Converting even 10% of them to regular customers is significant revenue

**What the CEO should do:** Personalized recommendation emails. "You bought X, you might also like Y and Z."

#### Insight 5: Weekend Revenue Gap

**What we found:** Weekends show dramatically lower sales.

**Why this matters:**
- If customers can't order on weekends, you're losing 2/7 of potential sales
- Online shoppers often browse on weekends
- Automated order processing is relatively low-cost

**What the CEO should do:** Enable automated weekend order processing. Test weekend-only flash sales.

---

## 17. Recommendations

### How Recommendations Connect to Insights

Every recommendation must:
1. **Link to a specific insight** (not come out of nowhere)
2. **Be actionable** (specific enough to implement)
3. **Have an expected impact** (quantified if possible)
4. **Have a timeline** (when to start, when to expect results)

### Priority Levels

| Priority | Meaning | Criteria |
|----------|---------|----------|
| 🔴 CRITICAL | Do this immediately | High revenue impact, risk mitigation |
| 🟡 HIGH | Do this within 1–3 months | Significant growth opportunity |
| 🟢 MEDIUM | Do this within 3–6 months | Efficiency improvement |
| 🔵 ONGOING | Continuous improvement | Monitoring and optimization |

### Recommendation Details

| # | Recommendation | Connected Insight | Why Now |
|---|---------------|-------------------|---------|
| 1 | VIP loyalty program | Revenue concentration | Protect 40%+ of revenue |
| 2 | Q4 inventory preparation | Seasonal spike | Must plan months ahead |
| 3 | Low-value re-engagement | Monetization gap | Low cost, measurable ROI |
| 4 | International expansion | UK dependency | Diversification is risk management |
| 5 | Weekend automation | Revenue gap | Quick win with technology |
| 6 | Product cross-selling | Customer behavior | Increases AOV with minimal cost |
| 7 | Cohort monitoring dashboard | Retention tracking | Early warning system |

---

## 18. Glossary of Terms

| Term | Definition |
|------|-----------|
| **AOV** | Average Order Value — total revenue divided by number of orders |
| **Basket Size** | Number of items in a single order |
| **Churn** | When a customer stops buying from you |
| **Cohort** | A group of customers who share a common characteristic (e.g., same first-purchase month) |
| **Cross-tabulation** | A table showing the relationship between two categorical variables |
| **Customer Lifespan** | Days between first and last purchase |
| **Feature Engineering** | Creating new variables from existing data |
| **IQR** | Interquartile Range — the range between the 25th and 75th percentile |
| **Kurtosis** | Measure of how "heavy" the tails of a distribution are |
| **Multi-dimensional** | Analyzing 3+ variables simultaneously |
| **Outlier** | A data point significantly different from the majority |
| **Pareto Principle** | 80% of effects come from 20% of causes |
| **Percentile** | The value below which a given percentage of data falls (P90 = 90th percentile means 90% of data is below this value) |
| **Purchase Frequency** | How often a customer makes purchases (per month, per year, etc.) |
| **Retention Rate** | Percentage of customers who continue to be active over time |
| **Revenue Concentration** | How unevenly revenue is distributed across customers |
| **Segmentation** | Dividing customers into distinct groups based on behavior |
| **Skewness** | Measure of asymmetry in a distribution (positive skew = long right tail) |
| **VIP** | Very Important Person — top-tier customer by revenue or engagement |


---

*This guide was created for the Advanced E-Commerce Data Science Analysis project.*  
*For questions or clarifications, refer to the code in [README.md](./README.md).*

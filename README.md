# Pandas: From Messy Data to Real Insight

> 11 topics · 20 Jupyter notebooks · 1 applied project · built from scratch, practiced independently

I'm working toward a career as a **Machine Learning Engineer and Data Scientist**. Every model I'll ever train depends on data that someone had to clean, reshape, and understand first, and in Python that work happens in Pandas. This repository is how I built that foundation.

It's a complete, hands-on Pandas curriculum: each topic has a concept notebook and, for most, a practice notebook where I solved problems on my own. The goal was never to memorize syntax. It was to understand *why* an operation exists, *when* to use it, and *what goes wrong* when it's used carelessly.

I then tested the foundation on a real dataset in my **Superstore Sales Analysis** project. The project repository link will be added here once it is published on GitHub.

**Author:** [Abdullah Waseem](https://github.com/TheAbdullahWaseem)

---

## Contents

- [Skills Demonstrated](#skills-demonstrated)
- [Learning Roadmap](#learning-roadmap)
- [Applied Project: Superstore Sales Analysis](#applied-project-superstore-sales-analysis)
- [How I Approached Learning Pandas](#how-i-approached-learning-pandas)
- [A Taste of the Code](#a-taste-of-the-code)
- [Getting Started](#getting-started)
- [Lessons Learned](#lessons-learned)
- [What's Next](#whats-next)
- [About Me](#about-me)

---

## Skills Demonstrated

| Area | What the notebooks cover |
|---|---|
| **Foundations** | Series, DataFrames, data types, dataset inspection |
| **Selection and Indexing** | `loc`, `iloc`, Boolean masks, conditional filtering, index handling |
| **Data Cleaning** | Missing values, duplicates, type conversion, inconsistent entries, outliers |
| **String Operations** | The `.str` accessor, pattern matching, text standardization |
| **Transformation** | `apply`, `map`, derived columns, binning, conditional logic |
| **Aggregation and Grouping** | `groupby`, `agg`, named aggregations, grouped summaries |
| **Combining Data** | `concat`, `merge`, join types, mismatched keys |
| **Reshaping** | `pivot_table`, `melt`, `stack`, `unstack` |
| **File Handling** | Reading and writing data files |
| **Exploratory Data Analysis** | Distributions, relationships, trends, and patterns |

---

## Learning Roadmap

Notebooks are numbered so they can be followed in order. Each concept notebook teaches the idea, and its practice twin checks whether I actually understood it.

| No. | Topic | Notebook | Practice |
|---|---|---|---|
| 01 | Pandas Foundations | [Open](01_Pandas_Foundations.ipynb) | [Open](01_Pandas_Foundations_Practice.ipynb) |
| 02 | Data Selection and Indexing | [Open](02_Pandas_Data_Selection_and_Indexing.ipynb) | [Open](02_Pandas_Data_Selection_and_Indexing_Practice.ipynb) |
| 03 | Data Cleaning, Part 1 | [Open](03_Pandas_Data_Cleaning_1.ipynb) | [Open](03_Pandas_Data_Cleaning_1_Practice.ipynb) |
| 04 | Data Cleaning, Part 2 | [Open](04_Pandas_Data_Cleaning_2.ipynb) | [Open](04_Pandas_Data_Cleaning_2_Practice.ipynb) |
| 05 | String Operations | [Open](05_Pandas_String_Operations.ipynb) | [Open](05_Pandas_String_Operations_Practice.ipynb) |
| 06 | Data Transformation | [Open](06_Pandas_Data_Transformation.ipynb) | [Open](06_Pandas_Data_Transformation_Practice.ipynb) |
| 07 | Aggregation and Grouping | [Open](07_Pandas_Data_Aggregation_and_Grouping.ipynb) | [Open](07_Pandas_Data_Aggregation_and_Grouping_Practice.ipynb) |
| 08 | Combining Data | [Open](08_Pandas_Data_Combining.ipynb) | [Open](08_Pandas_Data_Combining_Practice.ipynb) |
| 09 | Data Reshaping | [Open](09_Pandas_Data_Reshaping.ipynb) | [Open](09_Pandas_Data_Reshaping_Practice.ipynb) |
| 10 | Working with Files | [Open](10_Pandas_Working_with_Files.ipynb) | n/a |
| 11 | Exploratory Data Analysis | [Open](11_Pandas_Exploratory_Data_Analysis.ipynb) | n/a |

Topics 10 and 11 are applied by nature, so they lead straight into the project below instead of ending in drills.

---

## Applied Project: Superstore Sales Analysis

Notebooks teach the tools. A real dataset shows whether you can use them.

I analyzed the **Superstore Sales dataset**, a retail dataset covering orders, customers, products, regions, sales, and profit. The aim was to take a raw table and answer the questions a business would actually ask.

**Questions the analysis set out to answer**

- Which categories and sub-categories drive profit, and which quietly erode it?
- How do regions and customer segments compare?
- How do sales and profit move over time?
- Who are the most valuable customers?

**How Pandas made it possible**

| Stage | What I did |
|---|---|
| **Cleaning** | Checked data types, converted order and ship dates to `datetime`, handled duplicates and inconsistencies |
| **Feature engineering** | Derived year, month, and shipping duration to enable time-based analysis |
| **Aggregation** | Used `groupby` and `agg` to break down sales and profit by category, region, segment, and customer |
| **Reshaping** | Built pivot tables to compare performance across dimensions side by side |
| **Exploration** | Examined trends and differences to surface patterns relevant to business decisions |

The project turned separate Pandas operations into one workflow: raw data, then a prepared dataset, then analysis, then findings. That is exactly the pipeline that sits underneath every machine learning project.

**Project repository:** The Superstore Sales Analysis project is being prepared for publication. The direct GitHub link will be added here once the repository is uploaded.

---

## How I Approached Learning Pandas

1. **Understand before memorizing.** For each operation I asked what problem it solves and when it's appropriate. Syntax can be looked up; judgment can't.
2. **Practice without a safety net.** Each practice notebook was solved independently, not copied from the examples.
3. **Respect data quality.** Wrong data types, missing values, and duplicate rows quietly produce wrong conclusions. Cleaning is not a warm-up; it's the job.
4. **Build, then reflect.** Finishing the notebooks wasn't the finish line. Applying them to a real dataset was.

---

## A Taste of the Code

A typical workflow from the notebooks: load, clean, derive a feature, and summarize.

```python
import pandas as pd

df = pd.read_csv("sales.csv")

# Clean: fix types and remove duplicates
df["order_date"] = pd.to_datetime(df["order_date"])
df = df.drop_duplicates()

# Transform: derive a monthly feature
df["order_month"] = df["order_date"].dt.to_period("M")

# Aggregate: sales, profit, and margin by category
summary = (
    df.groupby("category")
      .agg(
          total_sales=("sales", "sum"),
          total_profit=("profit", "sum")
      )
      .assign(margin=lambda x: x["total_profit"] / x["total_sales"])
      .sort_values("total_profit", ascending=False)
)

print(summary)
```

*Column names are illustrative.*

---

## Getting Started

**Requirements:** Python 3.10 or newer, plus `pandas`, `numpy`, and `jupyter`.

### View on GitHub

Open any notebook from the [roadmap](#learning-roadmap). GitHub renders notebooks with their outputs, so no setup is needed.

### Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/) and choose **File → Open notebook → GitHub**.
2. Enter `https://github.com/TheAbdullahWaseem/Pandas`.
3. Pick a notebook and run its cells.

### Run Locally

```bash
git clone https://github.com/TheAbdullahWaseem/Pandas.git
cd Pandas
pip install pandas numpy jupyter
jupyter notebook
```

Run cells in order. Some notebooks may use extra libraries such as `matplotlib`; install them with `pip` if prompted.

---

## Lessons Learned

- **Most analysis time is spent on data preparation**, not on the analysis itself. Good cleaning makes everything after it simpler and more trustworthy.
- **Readable code is part of the result.** Chaining operations clearly makes an analysis easier to check, explain, and reuse.
- **Always sanity-check after a transformation.** A merge, filter, or type conversion can silently change the data, so verifying shape and values after each step catches problems early.
- **Understanding beats syntax.** Once the logic of grouping, joining, or reshaping is clear, the exact method names are easy to find.

---

## What's Next

Pandas is one layer of a bigger stack, and most of the foundation is in place:

- [x] Python and SQL
- [x] Pandas for cleaning, transformation, and analysis
- [x] NumPy for numerical computing
- [x] Matplotlib and Seaborn for visualization
- [x] Power BI for dashboards
- [x] Applied project: Superstore Sales Analysis
- [ ] Machine learning with Scikit-learn: preprocessing, modeling, evaluation
- [ ] End-to-end ML projects for my portfolio

Pandas taught me to understand and prepare data. The next step is teaching models to learn from it.

---

## About Me

I'm **Abdullah Waseem**, a Computer Science graduate and self-taught data analyst working toward a career as a Machine Learning Engineer and Data Scientist. My toolkit includes Python, SQL, Pandas, NumPy, Matplotlib, Seaborn, and Power BI, and I learn best by building real things.

If you're hiring, collaborating, or working through Pandas yourself, feel free to reach out.

**GitHub:** [TheAbdullahWaseem](https://github.com/TheAbdullahWaseem)

If this repository helped you, a star is always appreciated.
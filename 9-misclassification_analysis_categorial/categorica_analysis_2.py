"""
category_misclassification_analysis.py

Corrected version:
- Robustly converts boolean-like columns (strings 'True'/'False', '1'/'0', 'Yes'/'No', etc.) to 0/1.
- Avoids DataFrame fragmentation by constructing converted columns and concatenating.
- Keeps exploded cuisine mapping (business -> possibly many cuisines) and merges many-to-many.
- Produces the same visualizations as before.
"""

import os
from pathlib import Path
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ------------------- User constants -------------------
MODELING_DATA_FILE = '../4-prep_model_data/data_filtered.csv'
MISCLASSIFICATION_FILE = '../7-deep_learning_textCNN/misclassification_analysis_textcnn.csv'

OUT_DIR = Path("./viz_outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ------------------- Helper: cuisine grouping rules -------------------
def infer_cuisine_group_from_colname(col_name: str) -> str:
    s = col_name.lower()
    if re.search(r"mex|taco|burrito|tex-?mex", s):
        return "Mexican"
    if re.search(r"pizza|ital", s):
        return "Italian"
    if re.search(r"sushi|ramen|japan|tonkatsu|japanese", s):
        return "Japanese"
    if re.search(r"china|chinese|cantonese|szechuan|szechuan", s):
        return "Chinese"
    if re.search(r"indian|pakistani|himalayan|nepalese", s):
        return "Indian"
    if re.search(r"burger|american|breakfast_and_brunch|diner|dinner", s):
        return "American"
    if re.search(r"fast_food|drive|chicken_wings|hot_dogs|donair|donairs|kebab", s):
        return "Fast Food"
    if re.search(r"thai", s):
        return "Thai"
    if re.search(r"vietnam|pho|taiwanese", s):
        return "Vietnamese"
    if re.search(r"mediterr|greek|lebanese|turkish|middle_eastern|persian|iranian|arabic|halal|falafel|shawarma", s):
        return "Middle Eastern / Mediterranean"
    if re.search(r"seafood|fish_and_chips|seafood_markets", s):
        return "Seafood"
    if re.search(r"bakery|patisserie|cake|cupcake|donut|gelato|dessert|macaron|gelato|ice", s):
        return "Bakery / Dessert"
    if re.search(r"latin|peruvian|colombian|brazil|venezuelan|puerto|salvadoran|nicaraguan", s):
        return "Latin American"
    if re.search(r"carib|cuban|puerto", s):
        return "Caribbean"
    if re.search(r"african|ethiopian|senegalese|moroccan", s):
        return "African"
    if re.search(r"french|spanish|german|belgian|british|portuguese|polish|scandi|european", s):
        return "European"
    if re.search(r"fusion|pan_asian|asian_fusion|modern_european", s):
        return "Fusion"
    if re.search(r"poke|hawaiian", s):
        return "Hawaiian / Poke"
    if re.search(r"salad|health|vegetarian|vegan|gluten", s):
        return "Health / Vegetarian"
    if re.search(r"steak|steakhouses|bbq|barbeque|smokehouse|smokehouse", s):
        return "Steakhouses / BBQ"
    if re.search(r"noodle|ramen|udon|soup|pho|noodles", s):
        return "Noodles / Ramen"
    if re.search(r"food|specialty|market|meat_shops", s):
        return "Specialty Food / Market"
    return "Other"

# ------------------- utility: robust boolean/flag conversion -------------------
def to_binary_series(s: pd.Series) -> pd.Series:
    """
    Convert a Series of mixed values into 0/1 ints.
    Handles 'True'/'False', 'true'/'false', '1'/'0', 'Yes'/'No', numpy booleans, and numeric strings.
    """
    s_orig = s.copy()
    s_filled = s.fillna("")  # replace NaN with empty string for safe processing
    s_str = s_filled.astype(str).str.strip().str.lower()

    truthy = {"1", "true", "t", "yes", "y"}
    falsy  = {"0", "false", "f", "no", "n", ""}

    result = pd.Series(index=s.index, dtype="float64")

    mask_truth = s_str.isin(truthy)
    mask_false = s_str.isin(falsy)
    result[mask_truth] = 1.0
    result[mask_false] = 0.0

    # for rest, try numeric conversion and threshold >0
    mask_other = ~(mask_truth | mask_false)
    if mask_other.any():
        numeric = pd.to_numeric(s_orig[mask_other], errors="coerce").fillna(0)
        result.loc[mask_other] = (numeric > 0).astype(float)

    result = result.fillna(0).astype(int)
    return result

# ------------------- Load data -------------------
print("Reading modeling data from:", MODELING_DATA_FILE)
print("Reading misclassification data from:", MISCLASSIFICATION_FILE)

df_model = pd.read_csv(MODELING_DATA_FILE, dtype=str, low_memory=False)
df_mis = pd.read_csv(MISCLASSIFICATION_FILE, low_memory=False)

# convert numeric columns in mis file if present
numeric_cols = ["True_Star_Rating", "Predicted_Star_Rating", "stars_x"]
for c in numeric_cols:
    if c in df_mis.columns:
        df_mis[c] = pd.to_numeric(df_mis[c], errors='coerce')

# ------------------- Identify category columns -------------------
category_cols = [c for c in df_model.columns if c.startswith("Category_")]
print(f"Found {len(category_cols)} category columns (example): {category_cols[:10]}")

col_to_cuisine = {col: infer_cuisine_group_from_colname(col) for col in category_cols}
cuisine_to_cols = defaultdict(list)
for col, cuisine in col_to_cuisine.items():
    cuisine_to_cols[cuisine].append(col)

print("Cuisine groups discovered and number of mapped raw columns:")
for cuisine, cols in sorted(cuisine_to_cols.items(), key=lambda x: (-len(x[1]), x[0])):
    print(f"  {cuisine:30s} : {len(cols)}")

# ------------------- Service & Ambience columns (as you requested) -------------------
# Use these if present in df_model
service_cols = [c for c in [
    "RestaurantsDelivery", "RestaurantsTakeOut", "RestaurantsReservations",
    "WheelchairAccessible", "HasParking", "RestaurantsTableService", "RestaurantsGoodForGroups"
] if c in df_model.columns]

ambience_cols = [c for c in [
    "Ambience_romantic", "Ambience_intimate", "Ambience_classy",
    "Ambience_hipster", "Ambience_touristy", "Ambience_trendy",
    "Ambience_upscale", "Ambience_casual"
] if c in df_model.columns]

goodfor_cols = [c for c in ["GoodForKids"] if c in df_model.columns]

print("Service columns used:", service_cols)
print("Ambience columns used:", ambience_cols)
print("Good-for columns used:", goodfor_cols)

# ------------------- Prepare business-level DataFrame with robust conversions -------------------
cols_needed = ["business_id"] + category_cols + service_cols + ambience_cols + goodfor_cols
available_cols = [c for c in cols_needed if c in df_model.columns]
print(f"Using {len(available_cols)} columns from modeling data for business aggregation.")

# Build df_business with only needed columns (still strings)
df_business = df_model[available_cols].copy()

# Convert flags in one pass to avoid fragmentation: build a DataFrame of converted columns
converted = {"business_id": df_business["business_id"].copy()}

# Category columns -> binary
for col in category_cols:
    if col in df_business.columns:
        converted[col] = to_binary_series(df_business[col])

# service / ambience / goodfor -> binary
for col in service_cols + ambience_cols + goodfor_cols:
    if col in df_business.columns:
        converted[col] = to_binary_series(df_business[col])

# Concatenate converted columns into DataFrame (defragmented)
df_business_conv = pd.DataFrame(converted)

# Deduplicate businesses by taking max (so if any row has the flag, it's 1)
df_business_agg = df_business_conv.groupby("business_id", as_index=False).max()
print("Unique businesses after aggregation:", df_business_agg.shape[0])

# Quick debug prints to verify there are non-zero ambience/service flags
print("Ambience column sums (post-agg):")
for c in ambience_cols:
    s = df_business_agg[c].sum() if c in df_business_agg.columns else None
    print(f"  {c}: {s}")

print("Service column sums (post-agg):")
for c in service_cols:
    s = df_business_agg[c].sum() if c in df_business_agg.columns else None
    print(f"  {c}: {s}")

# ------------------- Create business -> cuisine membership exploded table -------------------
rows = []
for _, row in df_business_agg.iterrows():
    bid = row["business_id"]
    any_flag = False
    for cat_col in category_cols:
        if cat_col in df_business_agg.columns and int(row.get(cat_col, 0)) == 1:
            cuisine = col_to_cuisine.get(cat_col, "Other")
            rows.append({"business_id": bid, "raw_category_col": cat_col, "cuisine_group": cuisine})
            any_flag = True
    if not any_flag:
        rows.append({"business_id": bid, "raw_category_col": None, "cuisine_group": "Other"})

df_bus_cuisine = pd.DataFrame(rows)
print("Exploded business-cuisine rows:", len(df_bus_cuisine))

# ------------------- Merge misclassification rows to business categories -------------------
# 1) Merge business attributes (one row per business) into mis dataframe
df_merge = df_mis.merge(df_business_agg, on="business_id", how="left")

# 2) Merge exploded cuisine groups (many-to-many) so each review can appear for each cuisine
df_merge = df_merge.merge(df_bus_cuisine, on="business_id", how="left")
df_merge["cuisine_group"] = df_merge["cuisine_group"].fillna("Other")

# Convert Is_Misclassified to boolean if present or infer it
if "Is_Misclassified" in df_merge.columns:
    df_merge["Is_Misclassified"] = df_merge["Is_Misclassified"].astype(str).str.lower().map(
        lambda x: True if x in ("true", "1", "yes", "y") else False if x in ("false", "0", "no", "n") else np.nan)
else:
    # safe inference if numeric columns exist
    if "True_Star_Rating" in df_merge.columns and "Predicted_Star_Rating" in df_merge.columns:
        df_merge["Is_Misclassified"] = (df_merge["True_Star_Rating"] != df_merge["Predicted_Star_Rating"])
    else:
        # fallback: presume not misclassified (shouldn't happen)
        df_merge["Is_Misclassified"] = False

# ------------------- Per-cuisine summary stats -------------------
grouped = df_merge.groupby("cuisine_group")
cuisine_stats = grouped["Is_Misclassified"].agg(
    total_reviews="count",
    misclassified_count=lambda x: x.sum()
).reset_index()
cuisine_stats["misclassification_rate"] = cuisine_stats["misclassified_count"] / cuisine_stats["total_reviews"]
cuisine_stats = cuisine_stats.sort_values("misclassified_count", ascending=False)

print("\nTop cuisine groups by misclassified_count:")
print(cuisine_stats.head(15).to_string(index=False))

# ------------------- Visuals -------------------
TOP_N = 12
top_cuisines_by_mis = cuisine_stats.sort_values("misclassified_count", ascending=False).head(TOP_N)

# Bar: misclassification rate
plt.figure(figsize=(10,6))
x = top_cuisines_by_mis["cuisine_group"]
y = top_cuisines_by_mis["misclassification_rate"]
bars = plt.bar(x, y)
plt.title(f"Misclassification Rate by Cuisine (top {TOP_N} cuisines by misclassified count)")
plt.ylabel("Misclassification rate")
plt.ylim(0, max(0.5, y.max()*1.2))
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.xticks(rotation=45, ha="right")
for rect, val in zip(bars, y):
    plt.gca().text(rect.get_x() + rect.get_width()/2, val + 0.01, f"{val:.1%}", ha='center', va='bottom', fontsize=9)
plt.tight_layout()
plt.savefig(OUT_DIR / "misclassification_rate_by_cuisine_topN.png", dpi=150)
plt.close()
print("Saved:", OUT_DIR / "misclassification_rate_by_cuisine_topN.png")

# Pie: share of misclassified reviews
mis_by_cuisine = cuisine_stats.set_index("cuisine_group")["misclassified_count"].sort_values(ascending=False)
top = mis_by_cuisine.head(TOP_N)
others = mis_by_cuisine.iloc[TOP_N:].sum()
pie_vals = list(top.values) + ([others] if others > 0 else [])
pie_labels = list(top.index) + (["Other"] if others > 0 else [])
plt.figure(figsize=(8,8))
plt.pie(pie_vals, labels=pie_labels, autopct="%1.1f%%", startangle=140, pctdistance=0.8)
plt.title(f"Share of Misclassified Reviews by Cuisine (top {TOP_N})")
plt.tight_layout()
plt.savefig(OUT_DIR / "misclassified_share_pie_topN.png", dpi=150)
plt.close()
print("Saved:", OUT_DIR / "misclassified_share_pie_topN.png")

# Ambience heatmap: average ambience flags for correct vs misclassified
if ambience_cols:
    amb_df = df_merge[[*ambience_cols, "Is_Misclassified"]].copy()
    for c in ambience_cols:
        amb_df[c] = pd.to_numeric(amb_df[c], errors='coerce').fillna(0)
    amb_grouped = amb_df.groupby("Is_Misclassified").mean()
    # ensure both rows exist
    amb_grouped = amb_grouped.reindex([False, True]).fillna(0)
    plt.figure(figsize=(10,3))
    plt.imshow(amb_grouped, aspect='auto', cmap='YlGnBu')
    plt.colorbar(label='Average (0-1)')
    plt.yticks([0,1], ["Correct", "Misclassified"])
    plt.xticks(range(len(ambience_cols)), ambience_cols, rotation=45, ha='right')
    plt.title("Average Ambience Flags: Misclassified vs Correct (0/1)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "ambience_misclassified_vs_correct_heatmap.png", dpi=150)
    plt.close()
    print("Saved:", OUT_DIR / "ambience_misclassified_vs_correct_heatmap.png")
else:
    print("No ambience columns present; skipping ambience heatmap.")

# Services comparison: average availability for correct vs misclassified
svc_cols_used = [c for c in (service_cols + goodfor_cols) if c in df_merge.columns]
if svc_cols_used:
    svc_df = df_merge[[*svc_cols_used, "Is_Misclassified"]].copy()
    for c in svc_cols_used:
        svc_df[c] = pd.to_numeric(svc_df[c], errors='coerce').fillna(0)
    svc_grouped = svc_df.groupby("Is_Misclassified").mean().reindex([False, True]).fillna(0)
    labels = svc_grouped.columns.tolist()
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(x - width/2, svc_grouped.loc[False].values, width, label='Correct')
    ax.bar(x + width/2, svc_grouped.loc[True].values, width, label='Misclassified')
    ax.set_ylabel('Average Availability (0-1)')
    ax.set_title('Service / GoodFor Features: Correct vs Misclassified')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "services_correct_vs_mis.png", dpi=150)
    plt.close()
    print("Saved:", OUT_DIR / "services_correct_vs_mis.png")
else:
    print("No service/goodfor columns present; skipping services comparison plot.")

# ------------------- Top misclassified businesses (table) -------------------
biz_stats = df_merge.groupby(["business_id"]).agg(
    total_reviews=("Is_Misclassified", "count"),
    misclassified_count=("Is_Misclassified", "sum")
).reset_index()
biz_stats["misclassification_rate"] = biz_stats["misclassified_count"] / biz_stats["total_reviews"]

if "name" in df_model.columns:
    name_map = df_model.groupby("business_id", as_index=False)["name"].first().set_index("business_id")["name"]
    biz_stats["business_name"] = biz_stats["business_id"].map(name_map)

main_cuisine = df_bus_cuisine.groupby("business_id")["cuisine_group"].apply(lambda s: ";".join(sorted(set(s)))).reset_index()
biz_stats = biz_stats.merge(main_cuisine, on="business_id", how="left")

top_mis_biz = biz_stats.sort_values("misclassified_count", ascending=False).head(20)
print("\nTop 20 businesses by misclassified review count (sample):")
print(top_mis_biz[["business_id", "business_name", "cuisine_group", "total_reviews", "misclassified_count", "misclassification_rate"]].to_string(index=False))
top_mis_biz.to_csv(OUT_DIR / "top_misclassified_businesses.csv", index=False)
print("Saved:", OUT_DIR / "top_misclassified_businesses.csv")

# Bar for top businesses misclassification rate
plt.figure(figsize=(10,6))
labels = top_mis_biz["business_name"].fillna(top_mis_biz["business_id"]).apply(lambda x: str(x)[:30])
vals = top_mis_biz["misclassification_rate"]
plt.barh(range(len(labels))[::-1], vals[::-1])
plt.yticks(range(len(labels)), labels[::-1])
plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
plt.xlabel("Misclassification rate")
plt.title("Top 20 Businesses by Misclassification Count (misclassification rate shown)")
plt.tight_layout()
plt.savefig(OUT_DIR / "top20_businesses_misclassification_rate.png", dpi=150)
plt.close()
print("Saved:", OUT_DIR / "top20_businesses_misclassification_rate.png")

# Save cuisine summary
cuisine_stats.sort_values("misclassified_count", ascending=False).to_csv(OUT_DIR / "cuisine_misclassification_summary.csv", index=False)
print("Saved:", OUT_DIR / "cuisine_misclassification_summary.csv")

print("\nAll done. Visualizations & CSVs saved in:", OUT_DIR.resolve())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Define Constants ---
# NOTE: Verify these relative paths are correct for your system!
MODELING_DATA_FILE = '../4-prep_model_data/data_filtered.csv' 
MISCLASSIFICATION_FILE = '../7-deep_learning_textCNN/misclassification_analysis_textcnn.csv' 

CATEGORY_PREFIX = 'Category_'
AMBIENCE_PREFIX = 'Ambience_'
MIN_REVIEWS_THRESHOLD = 20 # Lowered to ensure Ambience/Attributes with fewer reviews are included

# --- Feature Selection and Mapping ---
CUISINE_MAPPINGS = {
    'Japanese': 'Category_Japanese', 'Italian': 'Category_Italian', 'Vietnamese': 'Category_Vietnamese', 
    'Mediterranean': 'Category_Mediterranean', 'Mexican': 'Category_Mexican', 'Chinese': 'Category_Chinese', 
    'Indian': 'Category_Indian', 'French': 'Category_French', 'American (Traditional)': 'Category_American_(Traditional)', 
    'American (New)': 'Category_American_(New)', 'Burgers': 'Category_Burgers', 'Pizza': 'Category_Pizza',
    'Steakhouses': 'Category_Steakhouses'
}
CUISINE_COLS = list(CUISINE_MAPPINGS.values())

AMBIENCE_COLS = ['Ambience_romantic', 'Ambience_intimate', 'Ambience_classy', 
                 'Ambience_hipster', 'Ambience_touristy', 'Ambience_trendy', 
                 'Ambience_upscale', 'Ambience_casual']
AMBIENCE_DISPLAY_MAP = {col: col.replace(AMBIENCE_PREFIX, '').replace('_', ' ').title() for col in AMBIENCE_COLS}

ATTRIBUTE_COLS = ['OutdoorSeating', 'RestaurantsDelivery', 'RestaurantsReservations', 
                  'GoodForKids', 'RestaurantsGoodForGroups', 'RestaurantsTakeOut', 
                  'WheelchairAccessible', 'HasParking', 'DriveThr', 'BusinessAcceptsCreditCards']
ATTRIBUTE_DISPLAY_MAP = {col: col for col in ATTRIBUTE_COLS}

ALL_REPORT_COLS = CUISINE_COLS + AMBIENCE_COLS + ATTRIBUTE_COLS

# --- NEW FUNCTION: Robust Data Cleaning for Binary Features (Fixes Ambience/Service) ---
def clean_binary_features(df, cols):
    """Fills NaNs with 0 and converts binary columns to integer type (robustly)."""
    for col in cols:
        if col in df.columns:
            # Coerce non-numeric/missing values to 0, then ensure integer type
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    return df

# --- FUNCTION: Calculate Rates with Explicit Columns ---
def calculate_feature_rates(df, feature_cols, display_map):
    """Calculates the weighted misclassification rate for binary features."""
    
    results = []
    present_cols = [col for col in feature_cols if col in df.columns]

    for col in present_cols:
        # Crucial check: filters only where the binary feature is present (value == 1)
        df_feature = df[df[col] == 1]
        
        if not df_feature.empty:
            total_misclassified = df_feature['Total_Misclassified'].sum()
            total_tested = df_feature['Total_Reviews_Tested'].sum()
            
            rate = total_misclassified / total_tested if total_tested > 0 else 0
            
            display_name = display_map.get(col, col.replace('_', ' ').title())

            results.append([display_name, total_tested, rate])

    df_results = pd.DataFrame(results, columns=['Feature', 'Total_Reviews', 'Misclassification_Rate'])

    # Filter using the defined global threshold
    df_results = df_results[df_results['Total_Reviews'] >= MIN_REVIEWS_THRESHOLD]
    return df_results.sort_values(by='Misclassification_Rate', ascending=False)


# --- 1. Load Data ---
try:
    df_category_source = pd.read_csv(MODELING_DATA_FILE)
    df_misclass = pd.read_csv(MISCLASSIFICATION_FILE)
except FileNotFoundError as e:
    print(f"Error: Could not find one or both files at the specified path: {e}")
    exit()

# --- 2. Filter, Clean, and Merge Data ---

# Robust Cleaning of all Binary Features
df_category_source = clean_binary_features(df_category_source, CUISINE_COLS + AMBIENCE_COLS + ATTRIBUTE_COLS)

present_cols = ['business_id'] + [col for col in ALL_REPORT_COLS if col in df_category_source.columns]
df_features = df_category_source.filter(items=present_cols).drop_duplicates(subset=['business_id'])

business_misclass_rate = df_misclass.groupby('business_id')['Is_Misclassified'].agg(['sum', 'count']).reset_index()
business_misclass_rate.columns = ['business_id', 'Total_Misclassified', 'Total_Reviews_Tested']
business_misclass_rate['Misclassification_Rate'] = business_misclass_rate['Total_Misclassified'] / business_misclass_rate['Total_Reviews_Tested']

df_analysis = pd.merge(df_features, business_misclass_rate, on='business_id', how='inner')
print(f"Analysis Data merged successfully: {len(df_analysis)} unique businesses analyzed.")


# --- 3. Run Analysis on Feature Groups ---
df_cuisine_analysis = calculate_feature_rates(df_analysis, CUISINE_COLS, CUISINE_MAPPINGS)
df_ambience_analysis = calculate_feature_rates(df_analysis, AMBIENCE_COLS, AMBIENCE_DISPLAY_MAP)
df_attribute_analysis = calculate_feature_rates(df_analysis, ATTRIBUTE_COLS, ATTRIBUTE_DISPLAY_MAP)


# --- 4. Tabular Summary and Insights ---
print("\n" + "="*80)
print("COMPREHENSIVE MISCLASSIFICATION ANALYSIS REPORT")
print("="*80)

# CUISINE ANALYSIS
print("\n>>> A. CUISINE ANALYSIS (Top 5 Most & Least Confusing)")
if not df_cuisine_analysis.empty:
    df_top_cuisine = pd.concat([df_cuisine_analysis.head(5), df_cuisine_analysis.tail(5)]).sort_values(by='Misclassification_Rate', ascending=False)
    print(df_top_cuisine.to_markdown(index=False, floatfmt=".2%", numalign="right"))

# AMBIENCE ANALYSIS
print("\n>>> B. AMBIENCE ANALYSIS (How Atmosphere Affects Review Clarity)")
if not df_ambience_analysis.empty:
    print(df_ambience_analysis.to_markdown(index=False, floatfmt=".2%", numalign="right"))

# ATTRIBUTE ANALYSIS
print("\n>>> C. SERVICE ATTRIBUTE ANALYSIS (Operational Features)")
if not df_attribute_analysis.empty:
    print(df_attribute_analysis.to_markdown(index=False, floatfmt=".2%", numalign="right"))


# --- 5. Enhanced Visualization (All requested graphs) ---
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')

# Visualization 1: Pie Chart - Overall Model Performance
total_reviews = df_misclass['Is_Misclassified'].count()
misclassified_sum = df_misclass['Is_Misclassified'].sum()
correctly_classified = total_reviews - misclassified_sum

data = [correctly_classified, misclassified_sum]
labels = ['Correctly Classified', 'Misclassified']
colors = ['#5cb85c', '#d9534f']

plt.figure(figsize=(7, 7))
plt.pie(data, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
plt.title('Overall Model Performance Breakdown', fontsize=14)
plt.show()

# Visualization 2: Cuisine Comparison (Bar Chart)
if not df_cuisine_analysis.empty:
    df_plot_cuisine = pd.concat([df_cuisine_analysis.head(5), df_cuisine_analysis.tail(5)]).sort_values(by='Misclassification_Rate', ascending=True)
    plt.figure(figsize=(14, 6))
    sns.barplot(x='Misclassification_Rate', y='Feature', data=df_plot_cuisine, palette='coolwarm')
    plt.title('Misclassification Rate by Cuisine (Top/Bottom 5)', fontsize=16)
    plt.xlabel('Misclassification Rate', fontsize=12)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.show()

# Visualization 3: Ambience Comparison (Bar Chart)
if not df_ambience_analysis.empty:
    df_ambience_analysis = df_ambience_analysis.sort_values(by='Misclassification_Rate', ascending=False)
    plt.figure(figsize=(12, 5))
    sns.barplot(x='Misclassification_Rate', y='Feature', data=df_ambience_analysis, palette='viridis')
    plt.title('Misclassification Rate by Ambience Feature', fontsize=16)
    plt.xlabel('Misclassification Rate', fontsize=12)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.show()

# Visualization 4: Service Attribute Comparison (Bar Chart)
if not df_attribute_analysis.empty:
    df_attribute_analysis = df_attribute_analysis.sort_values(by='Misclassification_Rate', ascending=False)
    plt.figure(figsize=(12, 5))
    sns.barplot(x='Misclassification_Rate', y='Feature', data=df_attribute_analysis, palette='cividis')
    plt.title('Misclassification Rate by Service Attribute', fontsize=16)
    plt.xlabel('Misclassification Rate', fontsize=12)
    plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    plt.show()


# Visualization 5: Cuisine Breakdown (Pie Chart by Volume)
if not df_cuisine_analysis.empty:
    df_pie_data = df_cuisine_analysis.sort_values(by='Total_Reviews', ascending=False).head(10)
    
    plt.figure(figsize=(8, 8))
    plt.pie(df_pie_data['Total_Reviews'], 
            labels=df_pie_data['Feature'], 
            autopct='%1.1f%%', 
            startangle=90, 
            wedgeprops={'edgecolor': 'black'})
    plt.title('Top 10 Cuisines by Review Volume (Tested)', fontsize=14)
    plt.show()


# --- 6. Top Misclassified Businesses Deep Dive ---

df_top_businesses = business_misclass_rate.sort_values(by='Misclassification_Rate', ascending=False).head(5)
df_top_businesses_report = pd.merge(df_top_businesses, df_features, on='business_id', how='left')

print("\n" + "="*80)
print("TOP 5 MISCLASSIFIED BUSINESSES: WHAT ARE THEIR CHARACTERISTICS?")
print("="*80)

report_rows = []
for index, row in df_top_businesses_report.iterrows():
    
    active_cuisines_raw = [col for col in CUISINE_COLS if col in row.index and row[col] == 1]
    active_ambience_raw = [col for col in AMBIENCE_COLS if col in row.index and row[col] == 1]
    active_attributes_raw = [col for col in ATTRIBUTE_COLS if col in row.index and row[col] == 1]
    
    active_cuisines_clean = [CUISINE_MAPPINGS.get(col, col).replace(CATEGORY_PREFIX, '').replace('_', ' ') for col in active_cuisines_raw]
    active_ambience_clean = [AMBIENCE_DISPLAY_MAP.get(col, col).replace(AMBIENCE_PREFIX, '').replace('_', ' ') for col in active_ambience_raw]
    
    report_rows.append({
        'Business ID': row['business_id'],
        'Rate': f"{row['Misclassification_Rate']:.2%}",
        'Total Reviews': row['Total_Reviews_Tested'],
        'Cuisines': ", ".join(active_cuisines_clean) or 'N/A',
        'Ambience': ", ".join(active_ambience_clean) or 'N/A',
        'Attributes': ", ".join(active_attributes_raw) or 'N/A'
    })

df_report = pd.DataFrame(report_rows)
print(df_report.to_markdown(index=False))
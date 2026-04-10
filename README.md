# iron-kaggle-g4


👉 No NaN values

dataset has following issues:

1. Fake column
Unnamed: 0 → useless
Done

2. Wrong type
date → string (needs conversion)
# Why it's a problem:
A raw string like "2013-04-18" is meaningless to a model — it can't understand that April is spring, or that 2015 is after 2013. You need to break it into numeric features the model can actually use.
How to fix it — full date extraction:
# Step 1: Convert string → datetime
df['date'] = pd.to_datetime(df['date'])

# Step 2: Extract useful features
df['year']        = df['date'].dt.year
df['month']       = df['date'].dt.month
df['day']         = df['date'].dt.day
df['week']        = df['date'].dt.isocalendar().week.astype(int)
df['is_weekend']  = df['date'].dt.dayofweek.isin([5, 6]).astype(int)

# Step 3: Drop the original date column
df = df.drop(columns=['date'])


3. Categorical data
state_holiday → text (needs encoding)

Why it's a problem for Linear Regression specifically:
Linear regression does math on features — multiply, add, etc. If you label encode:
'0' → 0
'a' → 1
'b' → 2
'c' → 3
The model assumes c is 3x bigger than a and b is between a and c — that's mathematically wrong. Holidays have no natural order.
Tree models (Random Forest, XGBoost) don't care — they just split on values. Linear Regression does care.
Fix: One-Hot Encoding (OHE)
# Creates a binary column for each category
df = pd.get_dummies(df, columns=['state_holiday'], prefix='holiday', drop_first=True)

# Result:
# holiday_a   holiday_b   holiday_c
#     0           0           0      ← '0' (no holiday) - reference
#     1           0           0      ← 'a' (public holiday)
#     0           1           0      ← 'b' (easter)
#     0           0           1      ← 'c' (christmas)
drop_first=True drops holiday_0 to avoid the dummy variable trap (multicollinearity) — another Linear Regression specific problem.
Apply same encoding to test data safely:
# Train
df = pd.get_dummies(df, columns=['state_holiday'], prefix='holiday', drop_first=True)

# Test - reindex to guarantee same columns as training
df_test = pd.get_dummies(df_test, columns=['state_holiday'], prefix='holiday', drop_first=True)
df_test = df_test.reindex(columns=df.columns, fill_value=0)

Summary — which encoding per model:
Model Use
Linear Regression One-Hot Encoding (OHE)
Random Forest Label Encoding or OHE (both fine)
XGBoost Label Encoding or OHE (both fine)
Since you're using both Linear Regression and tree models → use OHE for everyone, it works correctly across all models.



4. Logical edge cases
drop all close stores
original row = 640840
new row = 532016

Customers walk in, browse, but don't buy
Store is open for maintenance/restocking
Promotion brought people in but nothing appealed to them
Verdict — it IS an anomaly, and here's why:
Out of 531,986 open days with customers, only 1 single day has sales=0. That's 0.0002%.
Store 948 normally makes €6,898 average sales when open — on that specific day it made €0 with 5 customers.
The difference between "real case" vs "anomaly":
ScenarioﾠReal case?ﾠThis dataset?
Open + customers → no salesﾠYes, possibleﾠHappens 1 out of 531,986 times
Open + customers → some salesﾠNormalﾠHappens 531,985 times
It's not impossible — but it's statistically extreme (0.0002%). In data science that's called an outlier, not a pattern. The model should not learn from it.
Conclusion — keep it or drop it?
# KEEP IT - it's 1 row out of 640K, negligible impact either way
For the presentation say: "We identified 1 statistical outlier — a store open with customers but zero sales. Given it represents 0.0002% of data we kept it as it has negligible impact on model performance."
That shows data awareness without wasting time on 1 row.


👉 These are data quality problems, not missing values
We verified that the dataset contained no missing values. However, we identified structural issues such as an unnecessary index column, categorical variables, and date formatting, which required preprocessing.



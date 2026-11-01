import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression

print('--- ڈیٹا فائلز لوڈ کی جا رہی ہیں ---')

# 1. لوکل پاتھز سے فائلیں لوڈ کریں (جو GitHub پر موجود ہیں)
train_df = pd.read_csv('train-test.csv')
val_data = pd.read_csv('validation.csv')
template_df = pd.read_csv('validation-predictions-template.csv')

print('--- تمام فائلیں کامیابی سے لوڈ ہو چکی ہیں! ---')

# 2. ڈیٹا کی صفائی (Negative weight ہٹانا)
train_df = train_df[train_df['weight'] > 0].copy()

# 3. تاریخ کو datetime میں بدل کر فیچرز بنانا
for df in [train_df, val_data]:
  df['date'] = pd.to_datetime(df['date'])
  df['year'] = df['date'].dt.year
  df['month'] = df['date'].dt.month
  df['day'] = df['date'].dt.day
  df['dayofweek'] = df['date'].dt.dayofweek
  df['dayofyear'] = df['date'].dt.dayofyear

# مشترکہ نُمیریکل فیچرز
common_num_features = [
    'distance',
    'weight',
    'year',
    'month',
    'day',
    'dayofweek',
    'dayofyear',
]

# 4. مسنگ ویلیوز فل کرنا (صرف ٹریننگ سیٹ کے میڈین سے تاکہ ڈیٹا لیکج نہ ہو)
train_median = train_df[common_num_features].median(numeric_only=True)
train_df[common_num_features] = train_df[common_num_features].fillna(
    train_median
)
val_data[common_num_features] = val_data[common_num_features].fillna(
    train_median
)

# 5. TF-IDF فیچرز تیار کرنا (Pickup, Delivery, Equipment)
train_df['cat_text'] = (
    train_df['pickup'].astype(str)
    + ' '
    + train_df['delivery'].astype(str)
    + ' '
    + train_df['equipment'].astype(str)
)
val_data['cat_text'] = (
    val_data['pickup'].astype(str)
    + ' '
    + val_data['delivery'].astype(str)
    + ' '
    + val_data['equipment'].astype(str)
)

tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(train_df['cat_text'])
X_val_tfidf = tfidf.transform(val_data['cat_text'])

X_train_num = train_df[common_num_features].values
X_val_num = val_data[common_num_features].values

X_train_final = sp.hstack([X_train_num, X_train_tfidf])
X_val_final = sp.hstack([X_val_num, X_val_tfidf])

y_train = train_df['posted_rate'].values

# 6. ماڈل ٹرین کریں (Linear Regression)
print('--- ماڈل ٹرین ہو رہا ہے ---')
model = LinearRegression()
model.fit(X_train_final, y_train)

# 7. ویلیڈیشن ڈیٹا پر پیشگوئی (Predictions) حاصل کریں
val_predictions = model.predict(X_val_final)
val_data['predicted_rate'] = val_predictions

# 8. ٹیمپلیٹ کے مطابق load_id اور predicted_rate کی فائل تیار کریں
submission_df = template_df[['load_id']].merge(
    val_data[['load_id', 'predicted_rate']], on='load_id', how='left'
)

# اگر کوئی مسنگ پریڈکشن ہو تو ٹریننگ کے اوسط سے فل کریں
submission_df['predicted_rate'] = submission_df['predicted_rate'].fillna(
    y_train.mean()
)

# 9. فائنل سبمشن فائل سیو کریں
submission_df.to_csv('validation_predictions.csv', index=False)
print(
    '--- validation_predictions.csv کامیابی سے تیار اور سیو ہو چکی ہے! ---'
)

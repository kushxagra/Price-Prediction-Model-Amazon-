import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import re
from xgboost import XGBRegressor
import optuna

# Load data (unchanged)
test_df = pd.read_csv('dataset/sample_test.csv')
sample_out_df = pd.read_csv('sample_test_out.csv')

# Clean text: Preserve alphanumeric
def clean_text(text):
    text = str(text).lower()
    text = ''.join([c if c.isalnum() or c == ' ' else ' ' for c in text])
    text = ' '.join(text.split())
    return text

# Extract numbers (unchanged)
def extract_numbers(text):
    numbers = re.findall(r'\d+\.?\d*', str(text))
    return [float(n) for n in numbers] if numbers else [0]

# Improved quantity/unit extraction
def extract_quantity_unit(text):
    pattern = r'(\d+\.?\d*)\s*([a-zA-Z]+)'
    matches = re.findall(pattern, str(text).lower())
    if matches:
        values = [float(m[0]) for m in matches]
        units = [m[1] for m in matches]
        return np.mean(values), units[0] if units else 'unknown'
    return 0.0, 'unknown'

# Extract multiplier (e.g., "pack of 4")
def extract_multiplier(text):
    match = re.search(r'(pack|combo|set|box)\s*(of|with)?\s*(\d+)', str(text).lower())
    return int(match.group(3)) if match else 1

# Unit conversions for normalization
unit_conversions = {
    'kg': 1000, 'g': 1, 'gm': 1, 'gram': 1, 'lb': 453.592,
    'ltr': 1000, 'ml': 1, 'fl': 29.5735, 'oz': 29.5735,
    'piece': 1, 'count': 1, 'dozen': 12, 'pack': 1, 'set': 1
}

def normalize_quantity(value, unit):
    unit = unit.lower()
    if unit in unit_conversions:
        return value * unit_conversions[unit]
    return value

# Feature engineering
def add_text_features(df):
    df['clean_content'] = df['catalog_content'].apply(clean_text)
    df['num_words'] = df['clean_content'].apply(lambda x: len(x.split()))
    df['num_unique_words'] = df['clean_content'].apply(lambda x: len(set(x.split())))
    df['num_chars'] = df['clean_content'].apply(len)
    df['avg_word_length'] = df['clean_content'].apply(lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)
    df['lexical_diversity'] = df['num_unique_words'] / (df['num_words'] + 1)
    df['numbers'] = df['catalog_content'].apply(extract_numbers)
    df['num_count'] = df['numbers'].apply(len)
    df['max_number'] = df['numbers'].apply(max)
    df['min_number'] = df['numbers'].apply(min)
    df['sum_numbers'] = df['numbers'].apply(sum)
    df['avg_number'] = df['numbers'].apply(np.mean)
    
    df['value_num'], df['unit_token'] = zip(*df['catalog_content'].apply(extract_quantity_unit))
    df['multiplier'] = df['catalog_content'].apply(extract_multiplier)
    df['adjusted_value'] = df['value_num'] * df['multiplier']
    df['normalized_value'] = df.apply(lambda row: normalize_quantity(row['adjusted_value'], row['unit_token']), axis=1)
    df['value_per_unit'] = df['value_num'] / (df['normalized_value'] + 1e-6)
    
    for kw in ['pack', 'combo', 'premium', 'offer', 'rs', 'price', 'mrp', 'off', 'discount', 'sale',
               'kg', 'gm', 'ml', 'ltr', 'gram', 'piece', 'count', 'oz', 'organic', 'natural', 'fresh', 'pure', 'best', 'quality']:
        df[f'kw_{kw}'] = df['clean_content'].apply(lambda x: x.count(kw))
    
    df['num_digits'] = df['catalog_content'].apply(lambda x: sum(c.isdigit() for c in str(x)))
    df['uppercase_ratio'] = df['catalog_content'].apply(lambda x: sum(1 for c in str(x) if c.isupper()) / (len(str(x)) + 1))
    df['brand'] = df['catalog_content'].apply(lambda x: str(x).split(' ')[0].lower())
    top_brands = df['brand'].value_counts().head(10).index.tolist()
    for brand in top_brands:
        df[f'brand_{brand}'] = df['brand'].apply(lambda x: int(x == brand))
    
    units_list = ['oz', 'ml', 'kg', 'g', 'gm', 'ltr', 'piece', 'count', 'lb', 'fl', 'dozen', 'pack', 'set']
    for unit in units_list:
        df[f'unit_{unit}'] = df['unit_token'].apply(lambda x: int(x == unit))
    
    return df

# Load and process data
train_df = pd.read_csv('dataset/train.csv')
test_full_df = pd.read_csv('dataset/test.csv')

# Outlier handling
q_low, q_high = train_df['price'].quantile([0.01, 0.99])
train_df = train_df[(train_df['price'] >= q_low) & (train_df['price'] <= q_high)]

# Apply feature engineering
train_df = add_text_features(train_df)
test_full_df = add_text_features(test_full_df)

# TF-IDF: Reduced features for speed
tfidf_char = TfidfVectorizer(max_features=100, ngram_range=(1,2), analyzer='char_wb', min_df=2, max_df=0.95, sublinear_tf=True)
tfidf_word = TfidfVectorizer(max_features=100, ngram_range=(1,3), analyzer='word', min_df=2, max_df=0.95, sublinear_tf=True)
tfidf_char_features_train = tfidf_char.fit_transform(train_df['clean_content'])
tfidf_word_features_train = tfidf_word.fit_transform(train_df['clean_content'])
tfidf_char_features_test = tfidf_char.transform(test_full_df['clean_content'])
tfidf_word_features_test = tfidf_word.transform(test_full_df['clean_content'])

# Feature columns
feature_cols = [
    'num_words', 'num_unique_words', 'num_chars', 'avg_word_length', 'lexical_diversity',
    'num_count', 'max_number', 'min_number', 'sum_numbers', 'avg_number',
    'value_num', 'value_per_unit', 'num_digits', 'uppercase_ratio',
    'multiplier', 'adjusted_value', 'normalized_value',
    'kw_pack', 'kw_combo', 'kw_premium', 'kw_offer', 'kw_rs', 'kw_price', 'kw_mrp', 'kw_off', 'kw_discount', 'kw_sale',
    'kw_kg', 'kw_gm', 'kw_ml', 'kw_ltr', 'kw_gram', 'kw_piece', 'kw_count', 'kw_oz',
    'kw_organic', 'kw_natural', 'kw_fresh', 'kw_pure', 'kw_best', 'kw_quality'
]
top_brands = train_df['brand'].value_counts().head(10).index.tolist()
units_list = ['oz', 'ml', 'kg', 'g', 'gm', 'ltr', 'piece', 'count', 'lb', 'fl', 'dozen', 'pack', 'set']
feature_cols += [f'brand_{b}' for b in top_brands] + [f'unit_{u}' for u in units_list]

# Combine features
X_train = np.hstack([tfidf_char_features_train.toarray(), tfidf_word_features_train.toarray(), train_df[feature_cols].values])
X_test = np.hstack([tfidf_char_features_test.toarray(), tfidf_word_features_test.toarray(), test_full_df[feature_cols].values])
y_train = train_df['price'].values

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
y_train_log = np.log1p(y_train)

# SMAPE function
def smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / np.maximum(denom, 1e-8)
    return 100 * np.mean(diff)

# Train-validation split
X_tr, X_val, y_tr_log, y_val_log = train_test_split(
    X_train, y_train_log, test_size=0.18, random_state=42
)
train_indices = train_df.index.tolist()
_, val_idx = train_test_split(train_indices, test_size=0.18, random_state=42)
y_val_actual = train_df.loc[val_idx, 'price'].values

# Optuna tuning for XGBoost with logging
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'max_depth': trial.suggest_int('max_depth', 8, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.5, 2),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2)
    }
    model = XGBRegressor(**params, random_state=42, n_jobs=4)  # Limit cores for stability
    model.fit(X_tr, y_tr_log)
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)
    preds = np.clip(preds, 0.01, None)
    smape_score = smape(y_val_actual, preds)
    print(f"Trial {trial.number}: SMAPE = {smape_score:.2f}, Params = {params}")
    return smape_score

print("Starting XGBoost tuning...")
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=10)  # Reduced for speed
best_params_xgb = study_xgb.best_params
print("Best XGBoost params:", best_params_xgb)

# Train XGBoost with best params
xgb = XGBRegressor(**best_params_xgb, random_state=42, n_jobs=4)
print("Training XGBoost on validation split...")
xgb.fit(X_tr, y_tr_log)
y_val_pred_log_xgb = xgb.predict(X_val)
y_val_pred_xgb = np.expm1(y_val_pred_log_xgb)
y_val_pred_xgb = np.clip(y_val_pred_xgb, 0.01, None)
print("XGBoost SMAPE validation:", smape(y_val_actual, y_val_pred_xgb))

# Feature selection
selector = SelectFromModel(xgb, prefit=True, threshold='median')
X_train_selected = selector.transform(X_train)
X_test_selected = selector.transform(X_test)

# Train on full data with selected features
print("Training XGBoost on full data...")
xgb.fit(X_train_selected, y_train_log)
y_pred_log_xgb = xgb.predict(X_test_selected)
y_pred = np.expm1(y_pred_log_xgb)
y_pred = np.clip(y_pred, 0.01, None)

# Output predictions
output = pd.DataFrame({'sample_id': test_full_df['sample_id'], 'price': y_pred})
output.to_csv('test_out.csv', index=False)
print("Final submission file written: test_out.csv")

# Error analysis
def error_analysis(val_df, y_val_actual, y_val_pred):
    val_df['pred'] = y_val_pred
    val_df['abs_error'] = np.abs(val_df['price'] - val_df['pred'])
    print("SMAPE by unit:", val_df.groupby('unit_token').apply(lambda g: smape(g['price'], g['pred'])))
    print("High error samples:", val_df.sort_values('abs_error', ascending=False).head(10))

val_df = train_df.loc[val_idx].copy()
error_analysis(val_df, y_val_actual, y_val_pred_xgb)
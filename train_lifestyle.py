import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib 
import os

# --- 1. CONFIGURATION ---
CSV_PATH = 'data/csv/PCOS_data_without_infertility.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# CORRECTED FEATURE MAP (Matches your specific CSV now)
feature_map = {
    ' Age (yrs)': 'Age',           # Added space at start
    'BMI': 'BMI',
    'Cycle length(days)': 'Cycle_Length',
    'Cycle(R/I)': 'Cycle_Regularity',
    'Weight gain(Y/N)': 'Weight_Gain',
    'hair growth(Y/N)': 'Hirsutism', # Lowercase 'h'
    'Skin darkening (Y/N)': 'Skin_Darkening',
    'Pimples(Y/N)': 'Pimples',
    'Fast food (Y/N)': 'Fast_Food',
    'Reg.Exercise(Y/N)': 'Exercise'
}

target_col = 'PCOS (Y/N)'

# --- 2. DATA LOADING & CLEANING FUNCTION ---
def load_and_clean_data(path):
    print(f"Loading data from {path}...")
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Error: {e}")
        return None, None

    # Rename columns to standard names
    try:
        # Create a copy with only the columns we want
        df_selected = df[list(feature_map.keys()) + [target_col]].copy()
        df_selected.rename(columns=feature_map, inplace=True)
    except KeyError as e:
        print(f"❌ Column mismatch! Check your CSV. Missing: {e}")
        print("Available columns:", df.columns.tolist())
        return None, None

    # --- SPECIFIC CLEANING RULES ---
    
    # 1. Convert objects to numeric
    for col in df_selected.columns:
        df_selected[col] = pd.to_numeric(df_selected[col], errors='coerce')

    # 2. Clean 'Cycle_Regularity' (2=Reg, 4=Irreg -> 0=Reg, 1=Irreg)
    df_selected['Cycle_Regularity'] = df_selected['Cycle_Regularity'].apply(lambda x: 1 if x == 4 else 0)

    # 3. Handle Missing Values (Imputation)
    fill_values = df_selected.median()
    df_clean = df_selected.fillna(fill_values)

    print("✅ Data Cleaned.")
    return df_clean, fill_values

# --- 3. MAIN EXECUTION ---

# Load
df, imputer_values = load_and_clean_data(CSV_PATH)
if df is None:
    exit()

# Split X and y
X = df.drop(columns=[target_col])
y = df[target_col]

# Split Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training on {len(X_train)} samples, Testing on {len(X_test)} samples.")

# Train XGBoost
print("--- Training XGBoost ---")
model = xgb.XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    use_label_encoder=False,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n🏆 Model Accuracy: {acc:.2%}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# --- 4. SAVE ARTIFACTS ---
joblib.dump(model, os.path.join(MODEL_DIR, 'xgb_pcos_lifestyle.pkl'))
joblib.dump(imputer_values, os.path.join(MODEL_DIR, 'imputer_values.pkl'))
joblib.dump(X.columns.tolist(), os.path.join(MODEL_DIR, 'feature_names.pkl'))

print(f"\n✅ All files saved to '{MODEL_DIR}/'.")
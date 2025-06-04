import pandas as pd
import numpy as np
import ast  # For safely evaluating string representations of lists

# Preprocessing & Feature Engineering
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Imbalance Handling
from imblearn.over_sampling import SMOTE
# If you don't have imblearn: pip install imbalanced-learn
from imblearn.pipeline import Pipeline as ImbPipeline  # Use imblearn pipeline for SMOTE

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Define file path (assuming the script is in the same directory as the CSV)
csv_file_path = 'netflix_imdb_clean.csv'

# --- 1. Load Data ---
try:
    df = pd.read_csv(csv_file_path)
    print(f"Successfully loaded data from {csv_file_path}. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: File not found at {csv_file_path}")
    print("Please ensure 'netflix_imdb_clean.csv' is in the same directory as the script.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the CSV: {e}")
    exit()

# --- 2. Prepare NLP Features (Keywords) ---
# The 'keywords' column might be stored as strings representing lists. Convert them back to lists.
# Then join list elements into a single string per row for TfidfVectorizer.
def process_keywords_for_tfidf(keywords_str):
    if pd.isna(keywords_str):
        return ""
    try:
        # Safely evaluate the string representation of the list
        keywords_list = ast.literal_eval(keywords_str)
        if isinstance(keywords_list, list):
            # Join the list elements into a space-separated string
            return " ".join(keywords_list)
        else:
            # Handle cases where the content might not be a list string
            return ""
    except (ValueError, SyntaxError):
        # Handle potential errors if the string is not a valid list representation
        # Check if it's already a string of words (less likely based on notebook)
        if isinstance(keywords_str, str):
             return keywords_str # Assume it's already processed? Needs inspection if error occurs.
        return "" # Fallback

df['keywords_text'] = df['keywords'].apply(process_keywords_for_tfidf)

# --- 3. Define Features (X) and Target (y) ---

# Target variable (defined in Clean_nlp_Popularity.ipynb)
target = 'popularity'
y = df[target]

# Identify feature columns
# Note: 'rating' column from the notebook was label encoded, treat as numerical/ordinal or one-hot encode later if preferred.
# Assuming 'type', 'rating', 'release_season', 'genre_trend' are already label encoded as per the notebook.
# We will scale numerical features and apply TF-IDF to keywords_text.

numerical_features = [
    'budget', 'production_company_track_record', 'director_success',
    'cast_popularity', 'marketing_budget',
    'release_year', 'duration_minutes', 'duration_seasons',
    'type', 'rating', 'release_season' # Encoded categorical treated as numerical for scaling here
    # Add other relevant numerical columns if needed
]
# The column 'genre_trend' was likely label encoded too but needs confirmation.
# If 'genre_trend' exists and is categorical/encoded, add it to numerical_features or handle separately.
if 'genre_trend' in df.columns and pd.api.types.is_numeric_dtype(df['genre_trend']):
     numerical_features.append('genre_trend')

text_feature = 'keywords_text' # The preprocessed text column for TF-IDF

# Combine numerical features and the text feature name for defining X
feature_columns = numerical_features + [text_feature]
X = df[feature_columns]

# --- 4. Train-Test Split ---
# Split data *before* applying SMOTE
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
print(f"Popularity distribution in training set:\n{y_train.value_counts(normalize=True)}")
print(f"Popularity distribution in test set:\n{y_test.value_counts(normalize=True)}")


# --- 5. Create Preprocessing & Modeling Pipelines ---

# Create a transformer for TF-IDF on the text feature
# Note: Adjust max_features as needed. More features might capture more nuance but increase complexity.
tfidf_vectorizer = TfidfVectorizer(max_features=200, stop_words='english')

# Create a transformer for scaling numerical features
numeric_transformer = StandardScaler()

# Use ColumnTransformer to apply different transformations to different columns
# It's crucial that the column names/indices match the structure of X
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_features),
        ('tfidf', tfidf_vectorizer, text_feature)
    ],
    remainder='passthrough' # Keep other columns if any (shouldn't be the case here)
)

# --- Define Models ---
# Model 1: Logistic Regression
# Using ImbPipeline to include SMOTE step specifically for training data
lr_pipeline = ImbPipeline([
    ('preprocess', preprocessor),
    ('smote', SMOTE(random_state=42)), # Apply SMOTE to handle imbalance
    ('classifier', LogisticRegression(class_weight='balanced', # Also use class_weight for robustness
                                      max_iter=1000,
                                      random_state=42,
                                      solver='liblinear')) # Good solver for smaller datasets
])

# Model 2: Random Forest
rf_pipeline = ImbPipeline([
    ('preprocess', preprocessor),
    ('smote', SMOTE(random_state=42)), # Apply SMOTE
    ('classifier', RandomForestClassifier(class_weight='balanced', # Use class_weight
                                           random_state=42,
                                           n_estimators=150, # Example: Increase estimators
                                           max_depth=10,     # Example: Limit tree depth
                                           min_samples_leaf=5)) # Example: Ensure leaves have min samples
])


# --- 6. Train and Evaluate Models ---

models = {
    "Logistic Regression": lr_pipeline,
    "Random Forest": rf_pipeline
}

for name, pipeline in models.items():
    print(f"\n--- Training and Evaluating {name} ---")

    # Train the model
    pipeline.fit(X_train, y_train)

    # Make predictions on the TEST set
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1] # Probability of class 1 (popular)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    # Use target_names for better report readability
    print(classification_report(y_test, y_pred, target_names=['Not Popular (0)', 'Popular (1)']))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# --- 7. Example Prediction (Optional) ---
# Let's predict on a few samples from the original data
print("\n--- Example Predictions ---")
sample_data = df.sample(5, random_state=1)
sample_X = sample_data[feature_columns]

# Use the trained Random Forest pipeline for prediction (example)
# Note: The pipeline handles preprocessing automatically
rf_predictions = rf_pipeline.predict(sample_X)
rf_probabilities = rf_pipeline.predict_proba(sample_X)[:, 1]

predictions_df = pd.DataFrame({
    'title': sample_data['title'].values,
    'actual_popularity': sample_data[target].values,
    'predicted_popularity (RF)': rf_predictions,
    'predicted_probability (RF)': rf_probabilities
})

print(predictions_df)

# --- Feature Importance (Random Forest Example) ---
print("\n--- Random Forest Feature Importance ---")
try:
    # Get feature names after preprocessing
    # TF-IDF feature names
    tfidf_feature_names = list(rf_pipeline.named_steps['preprocess']
                               .named_transformers_['tfidf']
                               .get_feature_names_out())
    # Combine all feature names in the correct order
    all_feature_names = numerical_features + tfidf_feature_names

    # Get importance scores from the classifier step
    importances = rf_pipeline.named_steps['classifier'].feature_importances_

    # Create DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': all_feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print("Top 20 Important Features:")
    print(feature_importance_df.head(20))

except Exception as e:
    print(f"Could not extract feature importances: {e}")
    print("This might happen if the pipeline structure changed or the model doesn't support feature_importances_.") 
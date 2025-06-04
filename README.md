# Netflix Show Popularity Prediction

## Project Overview
This project aims to predict the popularity of Netflix movies and TV shows *before* their release using machine learning techniques. By analyzing pre-release metadata such as genre, director, cast, budget, and textual descriptions, the model classifies content into "popular" or "not popular" categories. This provides a valuable tool for content acquisition, production, and marketing strategy.


## Table of Contents
1.  [Problem Statement](#problem-statement)
2.  [Project Scope](#project-scope)
3.  [Data Sources](#data-sources)
4.  [Methodology](#methodology)
    * [Data Preprocessing & EDA](#data-preprocessing--eda)
    * [Feature Engineering](#feature-engineering)
    * [Model Development & Evaluation](#model-development--evaluation)
5.  [Key Features Engineered](#key-features-engineered)
6.  [Model Performance](#model-performance)
7.  [Feature Importance](#feature-importance)
8.  [How to Run](#how-to-run)
9.  [File Structure](#file-structure)
10. [Future Work](#future-work)
11. [Reports](#reports)

## 1. Problem Statement
In the competitive streaming market, predicting the potential success of content *before* incurring significant production and marketing costs is invaluable. This project develops a predictive model to classify Netflix shows based on pre-release metadata, moving beyond reliance on post-launch metrics like IMDb ratings. The goal is to aid decision-making in the early planning stages of content development.

## 2. Project Scope
* Utilize pre-release metadata for prediction.
* Perform extensive data cleaning, preprocessing, and exploratory data analysis (EDA).
* Engineer a comprehensive set of features, including NLP-derived text features and custom metrics (e.g., budget, cast/director influence).
* Develop and evaluate classification models (Logistic Regression, Random Forest).
* Identify key features driving show popularity.

## 3. Data Sources
* **Netflix Titles:** Initial dataset from Kaggle (`netflix_titles.csv`) containing basic show information.
* **IMDb Data:** Historical IMDb scores and votes (e.g., from `show_scores.csv` or similar sources) were used *only* for feature engineering (like `director_success`, `cast_popularity`) and *not* as a direct target for the pre-release prediction model.
* **Wikipedia:** Leveraged for web scraping budget information using `requests` and `BeautifulSoup`.
* **Cleaned & Engineered Dataset:** `netflix_imdb_clean.csv` is the primary dataset used for the final model, resulting from filtering (US content, post-2000), cleaning, merging, and feature engineering processes.

## 4. Methodology

### Data Preprocessing & EDA (`preproc&eda.ipynb`)
* **Filtering:** Focused on content released in the "United States" and after the year "2000".
* **Missing Value Imputation:** Handled missing data for text columns ("Unknown") and numerical columns (e.g., 0 for historical scores if applicable).
* **Normalization & Cleaning:** Converted text to lowercase, stripped extra spaces, and ensured data type consistency (`date_added` to datetime).
* **Feature Extraction:** Separated 'duration' into 'duration_minutes' (movies) and 'duration_seasons' (TV shows).
* **Duplicate Removal:** Ensured data integrity.
* **Exploratory Data Analysis:** Visualized content type trends, genre distributions, and relationships between variables (detailed in project reports).

### Feature Engineering
*(Implemented in `preproc&eda.ipynb` and `feature_engineering.py`)*
* **Text-Based Features:**
    * `title_length`, `word_count` (from title).
    * **NLP on `description`:** Tokenization, stop-word removal (NLTK's `stopwords`, `punkt`), stemming (Porter Stemmer).
    * **Keyword Extraction:** TF-IDF Vectorizer to identify top 5-10 keywords from descriptions.
* **Categorical Feature Engineering:**
    * `age_group`: Mapped content ratings (e.g., 'PG-13', 'TV-MA') to broader audience categories.
    * `main_cast`: Extracted the first 1-5 actors.
    * `genre_category`: Mapped specific genres into broader categories.
    * Label Encoding applied where appropriate.
* **External & Derived Pre-Release Features:**
    * **`budget_filled`:** Estimated using web scraping from Wikipedia with a fallback to industry averages based on show type and release year.
    * **`director_success_rating`:** Calculated as the average historical IMDb rating of a director's past titles.
    * **`cast_popularity_score`:** A sophisticated custom function (`calculate_cast_popularity` in `feature_engineering.py`) considering metrics like number of past shows, average historical IMDb rating, genre diversity, budget levels of past projects, versatility, industry connections, longevity, and international presence.
    * **`production_company_track_record`:** Average past ratings of a company's shows.
    * **`genre_trend_score`:** Relative popularity of genres over time.
    * **`release_season_score`:** Categorization based on release timing.

### Model Development & Evaluation (`Final_ML_Model.py`)
* **Target Variable Definition:**
    * A custom **`popularity_score`** was created as a weighted combination of key pre-release features (e.g., `budget_filled`, `director_success_rating`, `cast_popularity_score`).
    * This `popularity_score` was then used to define a binary target variable **`is_popular`**: shows were labeled "popular" (top 40th percentile of the custom score) or "not popular" (bottom 60th percentile).
* **Models Explored:** Logistic Regression and Random Forest Classifier.
* **Chosen Model:** Random Forest Classifier, selected for its robustness, ability to handle mixed data types, and feature importance interpretability.
* **Preprocessing Pipeline:**
    * `ColumnTransformer` to apply `TfidfVectorizer` to 'description_cleaned' and `StandardScaler` to numerical features.
* **Imbalance Handling:** `SMOTE` (Synthetic Minority Over-sampling Technique) from `imblearn` was used within the pipeline to address class imbalance in the `is_popular` target.
* **Training & Evaluation:**
    * 80/20 train-test split.
    * Metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix, ROC-AUC.

## 5. Key Features Engineered
* **`budget_filled`:** Dynamically estimated budget with fallback logic.
* **`cast_popularity_score`:** Comprehensive, multi-faceted metric for cast influence.
* **`director_success_rating`:** Historical performance indicator for directors.
* **NLP-derived Keywords:** TF-IDF based keywords from show descriptions.
* `genre_trend_score`, `release_season_score`, `age_group`.

## 6. Model Performance
*(Summarize your key model performance metrics here from your reports. Example below, replace with your actuals)*
The Random Forest model achieved:
* **Accuracy:** [e.g., 75%]
* **Precision (Popular):** [e.g., 0.60]
* **Recall (Popular):** [e.g., 0.37] (as noted in one report, indicating good identification of "not popular" and room to improve "popular" identification)
* **F1-Score (Popular):** [e.g., 0.45]
* **ROC-AUC:** [e.g., 0.70]
* The model demonstrated strong performance in identifying shows likely to be "not popular" (e.g., 92% accuracy for "not popular" class as per `NetflixPopularity_Group4-2.pdf`) and provided a baseline for predicting "popular" shows using only pre-release data.

## 7. Feature Importance
*(List top 5-10 features from your Random Forest model as shown in `Final_ML_Model.py` output or reports. Example below)*
1.  `cast_popularity_score`
2.  `budget_filled`
3.  `director_success_rating`
4.  `duration_minutes` / `duration_seasons`
5.  `genre_trend_score`
6.  Keywords from TF-IDF (e.g., `tfidf_keyword_crime`, `tfidf_keyword_drama`)
*(Add more specific top features based on your model output)*

## 8. How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/netflix-popularity-prediction.git](https://github.com/YourUsername/netflix-popularity-prediction.git)
    cd netflix-popularity-prediction
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3.  **Install necessary libraries:**
    ```bash
    pip install pandas numpy scikit-learn nltk beautifulsoup4 requests imbalanced-learn jupyterlab
    ```
4.  **Download NLTK resources:** Run Python and then:
    ```python
    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    ```
5.  **Data Files:**
    * Place `netflix_titles.csv` in the root project directory.
    * The `preproc&eda.ipynb` notebook is expected to generate intermediate files like `show_scores.csv` (if applicable to your exact EDA workflow) and ultimately `netflix_imdb_clean.csv` which is used by `Final_ML_Model.py`. Ensure the paths in the scripts match your file locations or update them.
    * *Note on large files:* If `netflix_titles.csv` or generated CSVs are too large for GitHub, this README should be updated with instructions on where to obtain them.
6.  **Execution Order:**
    * Run the Jupyter Notebook `preproc&eda.ipynb` first. This performs initial data cleaning, EDA, and some feature engineering, likely saving an intermediate dataset.
    * Then, run `feature_engineering.py`. This script might take the output from the notebook (or `netflix_titles.csv` directly if it handles all preprocessing) and creates/enhances features like budget and cast popularity, saving `netflix_imdb_clean.csv`.
    * Finally, run `Final_ML_Model.py` to train and evaluate the models using `netflix_imdb_clean.csv`.
    ```bash
    jupyter lab preproc&eda.ipynb  # Or jupyter notebook preproc&eda.ipynb
    python feature_engineering.py
    python Final_ML_Model.py
    ```

## 9. File Structure

.
├── netflix_titles.csv              # Initial raw dataset
├── show_scores.csv                 # (If generated/used by preproc&eda.ipynb for historical scores)
├── netflix_imdb_clean.csv          # Cleaned and engineered dataset for the ML model
├── preproc&eda.ipynb               # Jupyter Notebook for preprocessing and EDA
├── feature_engineering.py          # Python script for advanced feature engineering
├── Final_ML_Model.py               # Python script for ML model training and evaluation
├── NetflixPopularity_Group4-2.pdf  # Project Presentation Slides
├── Technical Report Data Visualization.pdf # Detailed Project Report
├── .gitignore                      # Specifies intentionally untracked files
└── README.md                       # This file


## 10. Future Work

* **Advanced NLP:** Explore show description embeddings (e.g., Word2Vec, BERT), genre keyword deeper analysis, and sentiment analysis from related news/social media (if pre-release signals can be found).
* **Model Improvements:** Experiment with other ensemble methods (e.g., XGBoost, LightGBM), conduct extensive hyperparameter tuning with cross-validation.
* **Advanced Feature Engineering:** Incorporate temporal trends more explicitly (e.g., seasonality effects beyond just 'release_season'), analyze content similarity clustering.
* **Granular Modeling:** Develop separate models for movies versus TV series, as their success patterns and influential features might differ.
* **Interactive Tool:** Create an interactive prediction tool or dashboard (e.g., using Streamlit or Flask) for stakeholders to input show features and get popularity predictions.

## 11. Reports
* **Project Presentation:** [NetflixPopularity_Group4-2.pdf](NetflixPopularity_Group4-2.pdf)
* **Technical Report:** [Technical Report Data Visualization.pdf](Technical Report Data Visualization.pdf)



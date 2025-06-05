# Netflix Show Popularity Prediction

## Project Overview
This project, "Identifying Potential for High IMDb Ratings in Netflix Shows using Pre-release Features," develops a machine learning model to forecast the success of Netflix content *prior* to its release. By analyzing exclusively pre-release metadata—such as production budget, director's historical success, cast reputation, genre trends, and textual descriptions—the model classifies content into "popular" or "not popular" categories. This initiative aims to provide an empirical, data-driven tool for production companies and Netflix to make informed investment and content strategy decisions, moving beyond traditional post-launch evaluation metrics.

**Team (Original Academic Project):** Arya Mehta (018292885), Jane Heng (018321914), Prajwal Dambalkar (018318196), Vedika Sumbli (018305937) *(Note: This project is showcased here as an individual's portfolio piece, highlighting their contributions to the overall group effort in an academic setting.)*

## Table of Contents
1.  [Problem Statement](#problem-statement)
2.  [Project Scope](#project-scope)
3.  [Data Sources](#data-sources)
4.  [Methodology](#methodology)
    * [Data Preprocessing & EDA](#data-preprocessing--eda)
    * [Feature Engineering](#feature-engineering)
    * [Model Development & Evaluation](#model-development--evaluation)
5.  [Key Features Engineered](#key-features-engineered)
6.  [Power BI Dashboard Visualization](#power-bi-dashboard-visualization)
7.  [Model Performance](#model-performance)
8.  [Feature Importance](#feature-importance)
9.  [How to Run](#how-to-run)
10. [File Structure](#file-structure)
11. [Future Work](#future-work)

## 1. Problem Statement
In the highly competitive Over-The-Top (OTT) media landscape, accurately predicting the potential success of content *before* significant financial and resource commitments are made is crucial. This project addresses the challenge of forecasting Netflix show popularity using only pre-release metadata. The goal is to develop a classification model that aids production companies and Netflix in assessing investment viability and optimizing content strategy by identifying key pre-release indicators of potential audience engagement and IMDb rating success.

## 2. Project Scope
The project encompasses the end-to-end development of a popularity prediction model:
* **Data Collection and Preprocessing:** Utilizing the Kaggle Netflix dataset, enriching it with features like budget and historical performance metrics, and performing rigorous cleaning. This includes filtering for US-released content post-2000 to reduce data skewness and focus on a relevant market segment.
* **Feature Engineering:** Crafting a comprehensive set of predictive features, including advanced NLP techniques (TF-IDF on tokenized and stemmed descriptions using PorterStemmer) and custom-calculated metrics for budget, cast popularity, director success, and marketing budget (estimated at 20-25% of production cost). Numerical features are normalized using MinMaxScaler.
* **Model Development:** Implementing and comparing Logistic Regression and Random Forest classifiers. A key aspect is the creation of a custom `popularity_score` based on a weighted combination of pre-release features, which is then binarized (top 40% as "popular") to define the target variable, thus avoiding data leakage from post-release metrics.
* **Model Evaluation:** Assessing model performance using train-test splits (1,982 items; 1,585 training, 397 testing) and metrics such as accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrices. Five-fold cross-validation was also considered in the evaluation strategy.
* **Insight Generation:** Analyzing feature importance to understand the key drivers of predicted popularity.
* **Visualization:** Developing a Power BI dashboard to present data trends, EDA insights, and potentially model predictions in an accessible format.

## 3. Data Sources
* **Netflix Titles (Kaggle):** The primary dataset (`netflix_titles.csv`) providing show_id, type, title, director, cast, country, date_added, release_year, rating, duration, listed_in (genre), and description.
* **IMDb Data (for Feature Engineering Only):** Historical IMDb scores and vote counts were used *exclusively* to engineer pre-release features like `director_success_rating` and aspects of `cast_popularity_score`. These were *not* used as direct model targets.
* **Wikipedia (Web Scraping):** Budget information was scraped from Wikipedia using `requests` and `BeautifulSoup`, with fallbacks to industry averages.
* **Engineered Dataset (`netflix_imdb_clean.csv`):** The final dataset for modeling, derived after filtering (US, post-2000, resulting in ~4000 rows initially, refined to 1,982 items for final model training), cleaning, feature engineering (including new columns like `marketing_budget`, `director_popularity_score`, `cast_popularity_score`, `genre_trend`), and normalization.

## 4. Methodology

### Data Preprocessing & EDA (`preproc&eda.ipynb`)
The initial dataset was qualitative and highly varied. Preprocessing was pivotal:
* **Filtering & Cleaning:** Rows with null values were handled. To reduce skewness from wide year ranges (1970s-2020s) and diverse countries, data was filtered to content released in the **United States after 2000**. This reduced the dataset by about 50% to ~4000 rows, later refined to 1,982 instances for the final modeling stage.
* **Text Processing for Descriptions:** Descriptions were processed using **PorterStemmer** (via NLTK) for stemming after tokenization and stop-word removal (standard and custom stop-words) to identify core keywords.
* **Feature Transformation:** `date_added` converted to datetime; `duration` parsed into numerical minutes/seasons.
* **Normalization:** Numerical features were normalized using `MinMaxScaler` to a 0-1 scale.
* **EDA:** Visualizations (e.g., word clouds from descriptions, distributions of ratings, budget ranges, content types over 5-year periods, correlation heatmaps) were generated to understand data characteristics. KMeans clustering was also explored on IMDb ratings and log-transformed vote counts to identify natural popularity groupings. CDF plots were used to analyze rating and vote distributions.

### Feature Engineering
*(Implemented in `preproc&eda.ipynb` and `feature_engineering.py`)*
This stage was crucial for creating predictive signals from pre-release information.
* **NLP on Show Descriptions:**
    * A custom function was developed to convert string representations of keyword lists into space-separated strings, preparing them for TF-IDF.
    * **TF-IDF (Term Frequency-Inverse Document Frequency):** `TfidfVectorizer` was used to convert preprocessed text descriptions (tokenized, stemmed) into a numerical feature matrix, capturing the importance of specific words. A maximum of **200 text features** were extracted.
* **Categorical Encoding:** Features like `type`, `rating`, `release_season` were numerically encoded (e.g., OneHotEncoding or Label Encoding as appropriate).
* **Custom Calculated Features:**
    * **`budget_filled`:** Estimated production budget via Wikipedia scraping (`get_show_budget()` function) with fallbacks to industry averages based on type and release year.
    * **`marketing_budget`:** Estimated as 20-25% of the `budget_filled`.
    * **`director_success_rating`:** Based on the director's historical average IMDb ratings.
    * **`cast_popularity_score`:** A detailed custom scoring system (`calculate_cast_popularity` and `process_netflix_data` functions) considering actor experience, genre diversity, past ratings, project budget levels, versatility (movie vs. TV), industry connections, longevity, and international presence.
    * `production_company_track_record`, `genre_trend_score`, `release_season`.

### Model Development & Evaluation (`Final_ML_Model.py`)
* **Target Variable Definition (`is_popular`):**
    * A custom `popularity_score` was devised, calculated as a weighted combination of key pre-release features.
    * The top 40% of shows based on this custom score were labeled "popular" (1), and the rest "not popular" (0). Crucially, **no IMDb scores or other post-release metrics were used in this target creation**, ensuring the model's pre-release predictive integrity.
* **Dataset Split:** The final dataset of 1,982 items was split into training (n=1,585) and testing (n=397) sets, with stratified sampling to maintain class distribution (~80% non-popular, ~20% popular).
* **Scikit-learn Pipeline:**
    * `ColumnTransformer` was used for applying `TfidfVectorizer` (to processed keywords, max 200 features) and `StandardScaler` (to numerical features) simultaneously.
    * **SMOTE (Synthetic Minority Over-sampling Technique):** Integrated into an `imblearn.pipeline.Pipeline` to address the significant class imbalance by oversampling the minority "popular" class *only during training* to prevent data leakage.
* **Models Implemented:**
    1.  **Logistic Regression:** Served as a baseline. Configured with balanced class weights, 1000 maximum iterations, and the 'liblinear' solver (optimized for smaller datasets).
    2.  **Random Forest Classifier:** Chosen for its robustness with mixed data types and interpretability. Configured with 150 estimators (trees), a maximum depth of 10 (to prevent overfitting), a minimum of 5 samples per leaf node for generalization, and balanced class weights.
* **Evaluation Strategy:** Performance was assessed on the held-out test set using accuracy, precision, recall, F1-score (particularly for the "popular" class), confusion matrices, and ROC-AUC scores. Five-fold cross-validation was also noted as part of the broader evaluation methodology.

## 5. Key Features Engineered
The project's strength lies in its detailed and contextually relevant feature engineering:
* **`budget_filled` & `marketing_budget`:** Providing realistic financial framing for projects.
* **`cast_popularity_score`:** A nuanced, multi-dimensional measure of potential star influence.
* **`director_success_rating`:** Quantifying director's proven appeal.
* **TF-IDF Keywords from Descriptions:** Extracting thematic essence from textual narratives.
* Other context-rich features: `production_company_track_record`, `genre_trend_score`, `release_season`, normalized `duration_minutes`/`duration_seasons`, and encoded `rating`.

## 6. Power BI Dashboard Visualization
An interactive Power BI dashboard (`NetflixDashboard.pbix`) was developed to visualize data insights from the EDA phase and to present findings in an accessible manner.
*(Screenshot of Dashboard - Overview Page - Placeholder)*
* **Purpose:** To provide an intuitive visual exploration of the Netflix dataset, the factors influencing show ratings, and the interplay between various features.
* **Key Visualizations (derived from `DATA_230_Group4_NetflixDashboard.pdf` and Technical Report):**
    * **Content Overview:** KPIs for total movies/TV shows, average ratings, vote counts.
    * **Rating & Vote Distributions:** Histograms for IMDb ratings and log-transformed IMDb votes, CDF plots to identify percentile thresholds (e.g., 60th percentile for defining popularity).
    * **KMeans Clustering:** Visualization of titles clustered by IMDb rating and log-vote counts.
    * **Genre Analysis:** Pie charts showing IMDb rating variance by genre/genre\_trend; bar charts of average rating and title counts per genre.
    * **Financials vs. Ratings:** Scatter plots exploring budget/marketing budget against IMDb ratings, segmented by director success, content rating, and genre trends.
    * **Geospatial & Temporal Views:** World map of titles by country; stacked bar charts of titles/directors by year and country.
    * **Correlation Heatmap:** Visualizing relationships between key numerical features like `imdbrating`, `duration_minutes`, `budget`, `cast_popularity`, `director_success`, and the custom `popularity_score`.
    * **Content Distribution:** Bar charts for content by MPAA/TV rating and budget bins.
* **Interactivity:** The dashboard includes slicers for "Show Type" and other dimensions to allow dynamic exploration of the data.

*(Screenshot of Dashboard - Detailed Analysis Page - Placeholder)*

## 7. Model Performance
The models were evaluated on a 20% held-out test set (n=397).
* **Random Forest Classifier (Chosen Model):**
    * **Accuracy:** 0.8060 (approximately 80%)
    * **Precision (Popular Class):** 0.52
    * **Recall (Popular Class):** 0.37
    * **F1-score (Popular Class):** 0.43
    * **ROC-AUC:** 0.721
    * **Confusion Matrix (TP/FP/TN/FN from report page 11):**
        * True Negatives (Correctly ID'd Non-Popular): 291
        * False Positives (Incorrectly ID'd as Popular): 27
        * False Negatives (Missed Popular Content): 50
        * True Positives (Correctly ID'd Popular): 29
* **Logistic Regression (Baseline):**
    * **Accuracy:** 0.6801
    * **Precision (Popular Class):** 0.34
    * **Recall (Popular Class):** 0.65
    * **F1-score (Popular Class):** 0.45
    * **Confusion Matrix (TP/FP/TN/FN from report page 11):**
        * True Negatives: 219
        * False Positives: 99
        * False Negatives: 28
        * True Positives: 51

While Logistic Regression showed better recall for the popular class, the Random Forest model achieved significantly higher overall accuracy and precision for popular content, making it the preferred model. The Random Forest model demonstrated a strong capability to correctly identify non-popular content, crucial for risk mitigation.

## 8. Feature Importance
Analysis from the Random Forest model highlighted that intrinsic content characteristics held more predictive power than traditionally emphasized industry metrics:
1.  **Duration in minutes:** 0.109
2.  **Keyword: "finds" (from TF-IDF):** 0.084
3.  **Rating (Content Rating, e.g., PG, R, TV-MA):** 0.078
4.  **Keyword: "help" (from TF-IDF):** 0.074
5.  **Keyword: "young" (from TF-IDF):** 0.055
6.  **Release year:** 0.054

Notably, features often considered highly important, such as `production_company_track_record` (0.027), `director_success` (0.027), `budget` (0.026), `marketing_budget` (0.026), and `cast_popularity` (0.022), showed lower relative importance in this model. This suggests that fundamental content attributes and thematic elements captured by keywords are stronger pre-release predictors of popularity as defined in this project.

## 9. How to Run
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
    * The `preproc&eda.ipynb` notebook is expected to process this and intermediate files (like `show_scores.csv` if applicable for historical data) to generate `netflix_imdb_clean.csv`.
    * Ensure paths in scripts are correctly set if your file structure differs.
6.  **Execution Order:**
    * Run `preproc&eda.ipynb` first (for data cleaning, initial EDA, and potentially saving intermediate processed files).
    * Then, run `feature_engineering.py` (which likely takes data from the notebook's output or `netflix_titles.csv` and generates `netflix_imdb_clean.csv` with all custom features).
    * Finally, run `Final_ML_Model.py` (which loads `netflix_imdb_clean.csv` to train and evaluate the models).
    ```bash
    jupyter lab preproc&eda.ipynb
    python feature_engineering.py
    python Final_ML_Model.py
    ```
7.  **Power BI Dashboard:**
    * To view the dashboard, open `NetflixDashboard.pbix` using Microsoft Power BI Desktop.

## 10. File Structure

.
├── netflix_titles.csv              # Initial raw dataset from Kaggle
├── show_scores.csv                 # (If used/generated by preproc&eda.ipynb for historical IMDb data)
├── netflix_imdb_clean.csv          # Cleaned, engineered dataset for ML model input
├── preproc&eda.ipynb               # Jupyter Notebook: Data preprocessing, EDA, initial feature creation
├── feature_engineering.py          # Python script: Advanced and custom feature engineering
├── Final_ML_Model.py               # Python script: ML model training, evaluation, feature importance
├── NetflixDashboard.pbix           # Power BI dashboard file
├── .gitignore                      # Specifies intentionally untracked files by Git
└── README.md                       # This detailed project overview


## 11. Future Work
Based on the project outcomes and the technical report:
* **Advanced NLP:** Transition from TF-IDF to word embeddings (Word2Vec, GloVe, BERT) for richer semantic understanding of descriptions.
* **Model Optimization:** Explore other ensemble methods (XGBoost, LightGBM); conduct rigorous hyperparameter tuning using GridSearchCV or RandomizedSearchCV with cross-validation.
* **Expanded Feature Set:** Incorporate more granular temporal features (e.g., specific holiday release effects), analyze content similarity using clustering, and investigate ethical ways to include early social media buzz if available pre-release.
* **Granular Modeling:** Develop distinct models for "Movies" vs. "TV Shows" as their success factors may differ.
* **Deployment & Interactivity:** Create an interactive web application (e.g., using Streamlit or Flask) for stakeholders to input hypothetical show features and receive popularity predictions and insights.


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
6.  [Power BI Dashboard Visualization](#power-bi-dashboard-visualization)
7.  [Model Performance](#model-performance)
8.  [Feature Importance](#feature-importance)
9.  [How to Run](#how-to-run)
10. [File Structure](#file-structure)
11. [Future Work](#future-work)

## 1. Problem Statement
In the competitive streaming market, predicting the potential success of content *before* incurring significant production and marketing costs is invaluable. This project develops a predictive model to classify Netflix shows based on pre-release metadata, moving beyond reliance on post-launch metrics like IMDb ratings. The goal is to aid decision-making in the early planning stages of content development by identifying key pre-release indicators of potential audience engagement.

## 2. Project Scope
* Utilize pre-release metadata exclusively for prediction, simulating real-world pre-production decision scenarios.
* Perform extensive data cleaning, preprocessing, and exploratory data analysis (EDA) to ensure data quality and uncover initial insights.
* Engineer a comprehensive set of features, including NLP-derived text features from show descriptions and innovative custom metrics (e.g., estimated budget, multi-faceted cast/director influence scores).
* Develop, train, and evaluate classification models (Logistic Regression, Random Forest) to predict a binary popularity outcome.
* Identify and interpret key features driving show popularity, providing actionable insights.
* Visualize key data trends and model insights using a Power BI dashboard.

## 3. Data Sources
* **Netflix Titles:** Initial dataset from Kaggle (`netflix_titles.csv`) containing basic show information (title, director, cast, country, release year, rating, duration, listed_in, description).
* **IMDb Data (for Feature Engineering Only):** Historical IMDb scores and votes (e.g., from `show_scores.csv` or similar sources) were utilized *solely* for constructing pre-release features like `director_success_rating` and components of `cast_popularity_score`. These IMDb metrics were *not* used as direct target variables or as features reflecting post-release performance in the final model, maintaining the pre-release prediction integrity.
* **Wikipedia:** Leveraged for web scraping budget information for shows using Python libraries `requests` and `BeautifulSoup`.
* **Cleaned & Engineered Dataset:** `netflix_imdb_clean.csv` is the primary dataset fed into the machine learning model. It's the result of rigorous filtering (US content, post-2000), cleaning, merging of data sources, and the extensive feature engineering processes detailed below.

## 4. Methodology

### Data Preprocessing & EDA (`preproc&eda.ipynb`)
The initial phase focused on transforming raw data into a usable format and understanding its underlying structure.
* **Filtering:** Data was filtered to include content released in the "United States" and "after the year 2000" to focus on a specific market and contemporary content trends.
* **Missing Value Imputation:** Strategies were employed to handle missing data thoughtfully. Textual fields (e.g., 'director', 'cast') were imputed with "Unknown" to retain records, while numerical fields used for historical calculations were handled appropriately (e.g., with 0 or mean) to avoid data leakage.
* **Normalization & Cleaning:** Text data was standardized (lowercase, stripping extra spaces) for consistency. `date_added` was converted to datetime objects for potential time-based analysis. Duration strings were parsed into numerical `duration_minutes` (for movies) and `duration_seasons` (for TV shows).
* **Duplicate Removal:** Ensured data integrity by removing duplicate entries.
* **Exploratory Data Analysis (EDA):** Using Pandas, Matplotlib, and Seaborn, key aspects of the data were visualized to understand distributions, identify trends, and spot potential relationships. This included analyzing content types, release year trends, genre popularity, and IMDb rating distributions (as seen in the initial data exploration phase and dashboard).

### Feature Engineering
This was a critical stage focused on creating meaningful predictors from the available pre-release data. *(Implemented in `preproc&eda.ipynb` and `feature_engineering.py`)*
* **Text-Based Features from Show Descriptions (NLP):**
    * **Preprocessing:** Descriptions underwent tokenization, removal of common stop-words (using NLTK), and stemming (Porter Stemmer) to reduce words to their root form, standardizing the text.
    * **Keyword Extraction with TF-IDF:** `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) from Scikit-learn was employed. This technique identifies words that are significant to a specific document (show description) within the context of the entire corpus of descriptions. It assigns higher scores to terms that are frequent in a document but rare across all documents, capturing characteristic keywords. The top N TF-IDF scores/keywords were used as features.
* **Categorical Feature Transformation:**
    * `age_group`: Content ratings (e.g., 'PG-13', 'TV-MA') were mapped to broader audience segments.
    * `main_cast`: The first 1-5 actors were extracted, representing the lead cast.
    * `genre_category`: Specific genres were consolidated into broader, more manageable categories.
    * Label Encoding was applied to convert these categorical features into numerical representations suitable for the ML models.
* **Innovative External & Derived Pre-Release Features:**
    * **`budget_filled`:** Recognizing budget's importance but its frequent unavailability, a two-pronged approach was used:
        1.  Web scraping from Wikipedia using `requests` and `BeautifulSoup`.
        2.  If scraping failed, an intelligent fallback mechanism used industry average budgets based on show `type` (Movie/TV Show) and `release_year`, ensuring a robust budget estimate for all titles.
    * **`director_success_rating`:** Calculated based on the average historical IMDb performance of a director's previous works, serving as a proxy for their established track record.
    * **`cast_popularity_score`:** A custom, multi-faceted score (`calculate_cast_popularity` in `feature_engineering.py`) was developed. This went beyond simple metrics, considering an actor's filmography size, average past IMDb ratings, genre diversity, budget levels of past projects, movie vs. TV show versatility, co-star network, career longevity, and international presence. This aimed to create a nuanced measure of potential audience draw.
    * Additional features like `production_company_track_record`, `genre_trend_score`, and `release_season_score` were also engineered.

### Model Development & Evaluation (`Final_ML_Model.py`)
* **Target Variable Definition:**
    * A custom **`popularity_score`** was conceptualized and created as a weighted combination of key pre-release features (e.g., `budget_filled`, `director_success_rating`, `cast_popularity_score`). This score acted as an internal proxy for "potential for high popularity" based solely on pre-launch data.
    * This `popularity_score` was then binarized to create the target variable **`is_popular`**: shows ranking in the top 40th percentile of this custom score were labeled "popular" (1), and the rest "not popular" (0), transforming the problem into a binary classification task.
* **Models Explored:** Logistic Regression (as a baseline) and Random Forest Classifier.
* **Chosen Model: Random Forest Classifier:** Selected due to its strong performance with mixed data types, inherent ability to handle non-linear relationships, lower susceptibility to overfitting compared to single decision trees (due to ensemble nature), and its provision of feature importance scores for interpretability.
* **Structured Preprocessing with Scikit-learn Pipelines:**
    * A `ColumnTransformer` was used to apply different preprocessing steps to different types of columns simultaneously: `TfidfVectorizer` for the 'description_cleaned' text feature and `StandardScaler` for all numerical features (to normalize their scales and prevent features with larger magnitudes from dominating the model).
    * This `ColumnTransformer` was then integrated into a full `Pipeline`.
* **Handling Class Imbalance (SMOTE):** The dataset exhibited an imbalance between "popular" and "not popular" classes. To prevent the model from being biased towards the majority class, **SMOTE (Synthetic Minority Over-sampling Technique)** from the `imblearn` library was integrated into the pipeline (`imblearn.pipeline.Pipeline`). SMOTE generates synthetic samples for the minority class, leading to a more balanced training set and often improving the model's ability to predict the minority class.
* **Training & Evaluation:** The dataset was split into training (80%) and testing (20%) sets. The pipeline (including SMOTE and Random Forest) was trained on the training set. Performance was evaluated on the unseen test set using metrics like Accuracy, Precision, Recall, F1-Score, and ROC-AUC, along with the Confusion Matrix to understand the types of errors made.

## 5. Key Features Engineered
This project placed significant emphasis on creating high-impact features from pre-release data:
* **`budget_filled`:** A robust budget estimate derived from web scraping (Wikipedia) with an intelligent industry-average fallback system, addressing a common data gap.
* **`cast_popularity_score`:** A comprehensive, custom-calculated metric reflecting an actor's influence beyond simple film counts, incorporating factors like past project success, genre diversity, and project caliber.
* **`director_success_rating`:** A quantitative measure of a director's historical performance based on past IMDb ratings.
* **NLP-derived Keywords (TF-IDF):** Extracted significant thematic keywords from show descriptions, turning unstructured text into valuable model inputs.
* Other context-rich features like `genre_trend_score`, `release_season_score`, and refined `age_group` categories.

## 6. Power BI Dashboard Visualization
To effectively communicate the project's findings and data insights, an interactive Power BI dashboard (`NetflixDashboard.pbix`) was developed.

*(Screenshot of Dashboard - Overview Page - Placeholder)*
* **Purpose:** The dashboard aims to provide an intuitive visual exploration of the Netflix dataset, the factors influencing show ratings, and potentially, a way to interact with the model's predictions (depending on final dashboard features).
* **Key Visualizations & Insights (based on your `DATA_230_Group4_NetflixDashboard.pdf`):**
    * **Overall Content Landscape:** Cards showing total number of movies vs. TV shows, average ratings, total votes (Page 2 of PDF).
    * **Rating Distributions:** Bar chart showing the number of titles by IMDb rating groups (e.g., how many shows fall into 6-7, 7-8, 8-9 rating buckets) (Page 2 of PDF).
    * **Genre Analysis:**
        * Pie chart illustrating the variance of IMDb ratings across different genres and how `genre_trend` (trending, stable, declining) interacts with this (Page 2 of PDF).
        * Bar chart showing average rating and number of titles per genre (Page 5 of PDF).
    * **Budget vs. Rating Dynamics:** Scatter plot exploring the relationship between average budget, average IMDb rating, influenced by director success, content rating (PG, R, TV-MA), and genre trends (Page 2 of PDF).
    * **Geographical Distribution:** World map visualizing the count of titles by their main country of production (Page 3 of PDF).
    * **Temporal Trends:** Stacked bar chart showing the count of titles and directors by year and main country, highlighting production trends over time (Page 3 of PDF).
    * **Detailed Show Browser (Illustrative):** A table-like view that could allow users to see posters, ratings, genres, descriptions, and budget for individual shows, potentially with filtering capabilities (Page 4 of PDF).
* **Interactivity:** Describe any slicers, filters, or drill-through capabilities implemented (e.g., filtering by "Show Type" - Movie/TV Show, or "Popularity Score" - High/Low as seen on PDF).

*(Screenshot of Dashboard - Detailed Analysis Page - Placeholder)*

This dashboard serves as a powerful tool for stakeholders to quickly grasp key trends and the interplay of different factors without needing to delve into the raw data or code.

## 7. Model Performance
*(Summarize your key model performance metrics here from your original reports. Be specific and explain what they mean. Example below, replace with your actuals)*
The Random Forest model, after training with SMOTE for imbalance and evaluated on the unseen test set, achieved:
* **Accuracy:** [e.g., 75%] - Overall percentage of titles correctly classified.
* **Precision (Popular Class):** [e.g., 0.60] - Of all shows the model predicted as "popular," X% actually were.
* **Recall (Popular Class):** [e.g., 0.37] - The model correctly identified Y% of all actual "popular" shows.
* **F1-Score (Popular Class):** [e.g., 0.45] - A balanced measure of precision and recall for the "popular" class.
* **ROC-AUC:** [e.g., 0.70] - Indicates the model's ability to distinguish between popular and non-popular shows.
* **Specificity (Not Popular Class):** It's important to note if the model was particularly good at identifying "not popular" shows (e.g., "Correctly identified 92% of non-popular content"). This is valuable for de-risking investments by flagging potential underperformers.

The model demonstrated a strong ability to identify shows likely to be "not popular," providing a valuable tool for risk mitigation in content acquisition, and a foundational capability for predicting "popular" shows using solely pre-release data.

## 8. Feature Importance
*(List top 5-10 features from your Random Forest model as shown in `Final_ML_Model.py` output or original reports. Ensure these align with the "Key Features Engineered" section. Example below)*
The Random Forest model provided insights into which pre-release factors were most influential in its predictions:
1.  `cast_popularity_score`
2.  `budget_filled`
3.  `director_success_rating`
4.  `duration_minutes` / `duration_seasons`
5.  `genre_trend_score`
6.  Keywords from TF-IDF (e.g., `tfidf_keyword_crime`, `tfidf_keyword_drama`)
*(Add more specific top features based on your model output)*

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
    * The `preproc&eda.ipynb` notebook is expected to generate intermediate files like `show_scores.csv` (if applicable) and ultimately `netflix_imdb_clean.csv` which is used by `Final_ML_Model.py`. Ensure paths match.
    * *Note on large files:* If CSVs are too large for GitHub, provide download instructions.
6.  **Execution Order:**
    * Run `preproc&eda.ipynb` first for initial data processing.
    * Then, run `feature_engineering.py` to create/enhance features, saving `netflix_imdb_clean.csv`.
    * Finally, run `Final_ML_Model.py` to train and evaluate models using `netflix_imdb_clean.csv`.
    ```bash
    jupyter lab preproc&eda.ipynb
    python feature_engineering.py
    python Final_ML_Model.py
    ```
7.  **Power BI Dashboard:**
    * To view the dashboard, open `NetflixDashboard.pbix` using Microsoft Power BI Desktop.

## 10. File Structure

.
├── netflix_titles.csv              # Initial raw dataset
├── show_scores.csv                 # (If generated/used by preproc&eda.ipynb)
├── netflix_imdb_clean.csv          # Cleaned and engineered dataset for ML model
├── preproc&eda.ipynb               # Jupyter Notebook for preprocessing, EDA
├── feature_engineering.py          # Python script for advanced feature engineering
├── Final_ML_Model.py               # Python script for ML model training & evaluation
├── NetflixDashboard.pbix           # Power BI dashboard file
├── .gitignore                      # Specifies intentionally untracked files
└── README.md                       # This file


## 11. Future Work
* **Advanced NLP:** Explore show description embeddings (e.g., Word2Vec, BERT), deeper genre keyword analysis, and sentiment analysis.
* **Model Improvements:** Experiment with other ensemble methods (e.g., XGBoost, LightGBM), conduct extensive hyperparameter tuning.
* **Advanced Feature Engineering:** Incorporate more granular temporal trends, content similarity clustering.
* **Granular Modeling:** Develop separate models for movies versus TV series.
* **Interactive Prediction Tool:** Create a web-based tool (Streamlit/Flask) for stakeholders.


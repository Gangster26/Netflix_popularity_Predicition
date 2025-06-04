import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
from datetime import datetime

def get_show_budget(show_title, show_type, release_year):
    """
    Get budget information from multiple sources
    """
    budget = None
    
    # 1. Try to get from Wikipedia
    try:
        # Format title for Wikipedia URL
        wiki_title = show_title.replace(' ', '_')
        url = f"https://en.wikipedia.org/wiki/{wiki_title}"
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Look for budget information in infobox
        infobox = soup.find('table', {'class': 'infobox'})
        if infobox:
            for row in infobox.find_all('tr'):
                if 'Budget' in row.text:
                    budget_text = row.find('td').text
                    # Extract numeric value
                    budget = float(''.join(filter(str.isdigit, budget_text)))
                    break
    except:
        pass
    
    # 2. If Wikipedia fails, use industry averages based on type and year
    if budget is None:
        if show_type == 'TV Show':
            # TV Show budget estimates based on industry standards
            if release_year >= 2020:
                budget = 5_000_000  # Modern high-budget TV show
            elif release_year >= 2015:
                budget = 3_000_000  # Mid-budget TV show
            else:
                budget = 2_000_000  # Standard TV show
        else:
            # Movie budget estimates based on industry standards
            if release_year >= 2020:
                budget = 20_000_000  # Modern movie
            elif release_year >= 2015:
                budget = 15_000_000  # Mid-era movie
            else:
                budget = 10_000_000  # Older movie
    
    return budget

def calculate_budget_with_sources(df):
    """
    Calculate budget for each show using multiple sources
    """
    budgets = []
    
    for idx, row in df.iterrows():
        print(f"Processing {row['title']}...")
        
        # Get budget from multiple sources
        budget = get_show_budget(
            show_title=row['title'],
            show_type=row['type'],
            release_year=row['release_year']
        )
        
        # For TV shows, multiply by number of episodes
        if row['type'] == 'TV Show':
            try:
                episodes = int(row['duration'].split()[0])
                budget *= episodes
            except:
                # If can't parse episodes, use default of 10 episodes
                budget *= 10
        
        budgets.append(budget)
        
        # Be nice to the websites
        time.sleep(1)
    
    return budgets

def calculate_features(df):
    # 1. Budget
    def calculate_budget(row):
        # Budget is typically in millions
        # For TV shows, budget is calculated per episode
        if row['type'] == 'TV Show':
            # Estimate budget based on number of episodes and typical TV show budgets
            episodes = int(row['duration'].split()[0])  # Extract number of episodes
            return episodes * 2_000_000  # $2M per episode is a typical budget
        else:
            # For movies, use the actual budget
            return row['budget']
    
    df['budget'] = df.apply(calculate_budget, axis=1)
    
    # 2. Production Company Track Record
    def calculate_company_track_record(row):
        # Calculate based on previous successful shows from the same company
        company = row['production_company']
        company_shows = df[df['production_company'] == company]
        
        if len(company_shows) > 1:  # If company has other shows
            # Calculate success rate based on ratings and viewership
            success_rate = company_shows['rating'].mean() / 10  # Normalize to 0-1
            return success_rate * 100  # Scale to 0-100
        return 50  # Default value for new companies
    
    df['production_company_track_record'] = df.apply(calculate_company_track_record, axis=1)
    
    # 3. Director Success
    def calculate_director_success(row):
        director = row['director']
        if pd.isna(director):
            return 50  # Default value for unknown directors
            
        director_shows = df[df['director'] == director]
        if len(director_shows) > 1:
            # Calculate success based on previous works
            success_score = (
                director_shows['rating'].mean() * 0.6 +  # 60% weight to ratings
                director_shows['budget'].mean() * 0.4    # 40% weight to budget
            )
            return success_score
        return 50  # Default for new directors
    
    df['director_success'] = df.apply(calculate_director_success, axis=1)
    
    # 4. Cast Popularity
    def calculate_cast_popularity(row):
        if pd.isna(row['cast']):
            return 50  # Default for unknown cast
            
        cast_members = row['cast'].split(', ')
        popularity_scores = []
        
        for actor in cast_members:
            # Find all shows featuring this actor
            actor_shows = df[df['cast'].str.contains(actor, na=False)]
            if len(actor_shows) > 0:
                # Calculate actor's popularity based on their shows' success
                actor_score = (
                    actor_shows['rating'].mean() * 0.7 +  # 70% weight to ratings
                    actor_shows['budget'].mean() * 0.3    # 30% weight to budget
                )
                popularity_scores.append(actor_score)
        
        return np.mean(popularity_scores) if popularity_scores else 50
    
    df['cast_popularity'] = df.apply(calculate_cast_popularity, axis=1)
    
    # 5. Genre Trend
    def calculate_genre_trend(row):
        genres = row['listed_in'].split(', ')
        trend_scores = []
        
        for genre in genres:
            # Find recent shows in this genre
            genre_shows = df[df['listed_in'].str.contains(genre, na=False)]
            recent_shows = genre_shows[genre_shows['release_year'] >= 2020]
            
            if len(recent_shows) > 0:
                # Calculate trend based on recent popularity
                trend_score = (
                    recent_shows['rating'].mean() * 0.5 +  # 50% weight to ratings
                    len(recent_shows) * 0.5               # 50% weight to number of recent shows
                )
                trend_scores.append(trend_score)
        
        return np.mean(trend_scores) if trend_scores else 50
    
    df['genre_trend'] = df.apply(calculate_genre_trend, axis=1)
    
    # 6. Release Season
    def calculate_release_season(row):
        if pd.isna(row['date_added']):
            return 'Unknown'
            
        date = pd.to_datetime(row['date_added'])
        month = date.month
        
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['release_season'] = df.apply(calculate_release_season, axis=1)
    
    # 7. Marketing Budget
    def calculate_marketing_budget(row):
        # Marketing budget is typically 20-25% of production budget
        marketing_ratio = np.random.uniform(0.2, 0.25)  
        return row['budget'] * marketing_ratio
    
    df['marketing_budget'] = df.apply(calculate_marketing_budget, axis=1)
    
    # Normalize all numeric features to 0-1 scale
    scaler = MinMaxScaler()
    numeric_features = [
        'budget',
        'production_company_track_record',
        'director_success',
        'cast_popularity',
        'genre_trend',
        'marketing_budget'
    ]
    
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    
    return df

# Example usage:
# df = pd.read_csv('netflix_titles.csv')
# df_with_features = calculate_features(df)

(feature_path, conf_rf_path)

def calculate_cast_popularity(df):
    """
    Calculate cast popularity based on lead actor's (cast_name) influence and success metrics
    """
    # Create actor statistics dictionary
    global actor_stats
    actor_stats = {}
    
    # Calculate statistics for each lead actor
    for actor in df['cast_name'].unique():
        if pd.isna(actor):
            continue
            
        actor_shows = df[df['cast_name'] == actor]
        
        # Calculate actor's statistics
        actor_stats[actor] = {
            'total_shows': len(actor_shows),
            'recent_shows': len(actor_shows[actor_shows['release_year'] >= 2020]),
            'genre_diversity': len(actor_shows['listed_in'].unique()),
            'avg_budget': actor_shows['budget'].mean(),
            'show_types': len(actor_shows['type'].unique()),  # Movies vs TV Shows
            'production_companies': len(actor_shows['production_company'].unique()),
            'years_active': actor_shows['release_year'].max() - actor_shows['release_year'].min(),
            'total_budget': actor_shows['budget'].sum(),
            'international_shows': len(actor_shows[actor_shows['country'].str.contains(',', na=False)])
        }
    
    def calculate_actor_popularity(row):
        actor = row['cast_name']
        if pd.isna(actor) or actor not in actor_stats:
            return 50  # Default score
            
        stats = actor_stats[actor]
        
        # Calculate popularity score components
        experience_score = min(stats['total_shows'] / 10, 1) * 0.15  # 15% weight to total shows
        recent_activity = min(stats['recent_shows'] / 3, 1) * 0.15  # 15% weight to recent work
        genre_diversity = min(stats['genre_diversity'] / 5, 1) * 0.15  # 15% weight to genre diversity
        budget_level = min(stats['avg_budget'] / 10000000, 1) * 0.15  # 15% weight to budget
        versatility = min(stats['show_types'] / 2, 1) * 0.1  # 10% weight to show type diversity
        industry_connections = min(stats['production_companies'] / 5, 1) * 0.1  # 10% weight to industry connections
        longevity = min(stats['years_active'] / 10, 1) * 0.1  # 10% weight to years active
        international_presence = min(stats['international_shows'] / 5, 1) * 0.1  # 10% weight to international presence
        
        # Calculate final score
        popularity_score = (
            experience_score +
            recent_activity +
            genre_diversity +
            budget_level +
            versatility +
            industry_connections +
            longevity +
            international_presence
        ) * 100  # Scale to 0-100
        
        return popularity_score
    
    # Apply the calculation
    df['cast_popularity'] = df.apply(calculate_actor_popularity, axis=1)
    
    return df

def process_netflix_data(df):
    # Clean and prepare data
    df = df.copy()
    
    # Calculate cast popularity
    df = calculate_cast_popularity(df)
    
    # Add additional metrics
    df['actor_experience'] = df['cast_name'].map(
        lambda x: actor_stats[x]['total_shows'] if x in actor_stats else 0
    )
    
    df['actor_genre_diversity'] = df['cast_name'].map(
        lambda x: actor_stats[x]['genre_diversity'] if x in actor_stats else 0
    )
    
    df['actor_versatility'] = df['cast_name'].map(
        lambda x: actor_stats[x]['show_types'] if x in actor_stats else 0
    )
    
    return df

# Main execution
if __name__ == "__main__":
    # Read the input CSV
    df = pd.read_csv('Netflix_Titles__US__After_2000__Enriched_cast.csv')
    
    # Process the data
    processed_df = process_netflix_data(df)
    
    # Save to new CSV
    processed_df.to_csv('Netflix_Titles__US__After_2000__Enriched_cast_updated.csv', index=False)
    
    print("Processing complete. New CSV file created with updated cast popularity scores.")

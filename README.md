# Movie Recommendation System

A machine learning project that provides personalized movie recommendations by analyzing movie features and similarities. This system uses content-based filtering to suggest movies that are similar to a user's past preferences.

## Project Overview
This recommendation system leverages content-based filtering using movie metadata such as genres, plot, and keywords. By computing the similarity between movies, the system provides a list of similar movies based on user input.

![hi](https://www.heartoflongmont.org/wp-content/uploads/2019/02/Movie-Recommendation.jpg)

## Features
- **Content-Based Filtering**: Recommends movies similar to those the user likes by analyzing the movie metadata.
- **Similarity Computation**: Uses TF-IDF vectorization and cosine similarity to find movies with the most similar content.
- **Efficient Search**: Quickly retrieves recommendations based on user-selected movies.

## Technologies and Libraries
- **Python**: Primary programming language.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical computations.
- **Scikit-Learn**: Provides tools for TF-IDF vectorization and cosine similarity calculation.
- **Difflib**: For fuzzy matching, helping to match user input with movie titles.

## How the Recommendation System Works
1. **Data Collection**: The dataset includes information on movie titles, genres, plots, and other metadata.
2. **Data Preprocessing**: The TF-IDF Vectorizer converts movie metadata into numerical vectors.
3. **Similarity Calculation**: Calculates cosine similarity scores between movies to find those with the most similar metadata.
4. **Recommendation Generation**: Based on a user-selected movie, recommends other movies with high cosine similarity scores.

## Libraries Used
```python
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import re
import os
from utils import clean_movie_name # Import from utils


def aggregate_textual_bias_to_film_and_decade(analysis_data, songs_frequency_data):
    """
    Aggregates textual bias scores from row level to film and decade levels.

    Args:
        analysis_data (pd.DataFrame): DataFrame containing row-level textual bias scores
                                      ('Movie Name', 'row_character_bias_score').
        songs_frequency_data (pd.DataFrame): DataFrame containing movie names and years
                                             ('MOVIE NAME', ' YEAR').

    Returns:
        tuple: A tuple containing the film-level and decade-level textual bias DataFrames:
               (film_bias_scores, decade_bias_scores). Returns empty DataFrames if aggregation fails.
    """
    film_bias_scores = pd.DataFrame()
    decade_bias_scores = pd.DataFrame()

    if analysis_data is None or analysis_data.empty or songs_frequency_data is None or songs_frequency_data.empty:
        print("Analysis data or songs frequency data is empty. Skipping textual bias aggregation.")
        return film_bias_scores, decade_bias_scores

    print("\nCalculating film-level textual bias scores...")
    if 'Movie Name' in analysis_data.columns and 'row_character_bias_score' in analysis_data.columns:
        # Calculate film-level scores first
        film_bias_scores = analysis_data.groupby('Movie Name')['row_character_bias_score'].sum().reset_index()
        film_bias_scores.rename(columns={'row_character_bias_score': 'film_bias_score'}, inplace=True)

        # Add the cleaned movie name column to film_bias_scores
        film_bias_scores['Movie Name_cleaned'] = film_bias_scores['Movie Name'].apply(clean_movie_name)

        print("Film-level textual bias scores calculated.")
    else:
         print("Required columns ('Movie Name', 'row_character_bias_score') not found in analysis_data. Skipping film-level textual bias calculation.")
         return film_bias_scores, decade_bias_scores # Return empty if essential data is missing

    print("\nCalculating decade-level textual bias scores...")
    # Ensure songs_frequency_data has a cleaned movie name column and a year column
    if 'MOVIE NAME' in songs_frequency_data.columns and ' YEAR' in songs_frequency_data.columns:
         songs_frequency_data['MOVIE NAME_cleaned'] = songs_frequency_data['MOVIE NAME'].apply(clean_movie_name)

         # Merge film_bias_scores (which now has Movie Name_cleaned) with songs_frequency_data to get the year
         merged_data = pd.merge(film_bias_scores[['Movie Name_cleaned', 'film_bias_score']], # Use film_bias_scores with cleaned name
                               songs_frequency_data[['MOVIE NAME_cleaned', ' YEAR']].drop_duplicates(),
                               left_on='Movie Name_cleaned', right_on='MOVIE NAME_cleaned', how='inner')

         merged_data.rename(columns={' YEAR': 'YEAR'}, inplace=True)
         merged_data.dropna(subset=['YEAR'], inplace=True)

         if not merged_data.empty:
            merged_data['YEAR'] = merged_data['YEAR'].astype(int)
            merged_data['Decade'] = (merged_data['YEAR'] // 10) * 10

            # Calculate decade-level bias score (sum of film bias scores per decade)
            decade_bias_scores = merged_data.groupby('Decade')['film_bias_score'].sum().reset_index()
            decade_bias_scores.rename(columns={'film_bias_score': 'decade_bias_score'}, inplace=True)
            print("Decade-level textual bias scores calculated.")
         else:
             print("No matching movies with year information found after merging for decade bias calculation.")
    else:
        print("Required columns ('MOVIE NAME', ' YEAR') not found in songs_frequency_data, skipping decade bias calculation.")


    return film_bias_scores, decade_bias_scores

def aggregate_visual_bias_to_film_and_decade(quantified_visual_bias_df, songs_frequency_data):
    """
    Aggregates visual bias scores from image level to film and decade levels.

    Args:
        quantified_visual_bias_df (pd.DataFrame): DataFrame containing image-level visual bias scores
                                                  ('image_filename', 'total_image_bias_score').
        songs_frequency_data (pd.DataFrame): DataFrame containing movie names and years
                                             ('MOVIE NAME', ' YEAR').

    Returns:
        tuple: A tuple containing the film-level and decade-level visual bias DataFrames:
               (film_visual_bias_scores, decade_visual_bias_scores). Returns empty DataFrames if aggregation fails.
    """
    film_visual_bias_scores = pd.DataFrame()
    decade_visual_bias_scores = pd.DataFrame()


    if quantified_visual_bias_df is None or quantified_visual_bias_df.empty or songs_frequency_data is None or songs_frequency_data.empty:
        print("Quantified visual bias data or songs frequency data is empty. Skipping visual bias aggregation.")
        return film_visual_bias_scores, decade_visual_bias_scores

    print("\nAggregating visual bias scores to film and decade levels...")

    # Extract movie title from image filename
    if 'image_filename' in quantified_visual_bias_df.columns:
        quantified_visual_bias_df['Movie Name_from_image'] = quantified_visual_bias_df['image_filename'].apply(
            lambda x: '_img_' in x and os.path.splitext(x)[0].split('_img_')[0] or None
        )

        quantified_visual_bias_df.dropna(subset=['Movie Name_from_image'], inplace=True)

        if not quantified_visual_bias_df.empty:
            # Clean movie names for merging
            quantified_visual_bias_df['Movie Name_cleaned'] = quantified_visual_bias_df['Movie Name_from_image'].apply(clean_movie_name)

            # Ensure songs_frequency_data has a cleaned movie name column and a year column
            if 'MOVIE NAME' in songs_frequency_data.columns and ' YEAR' in songs_frequency_data.columns:
                 songs_frequency_data['MOVIE NAME_cleaned'] = songs_frequency_data['MOVIE NAME'].apply(clean_movie_name)

                 # Merge with songs_frequency_data to get the year
                 merged_visual_data = pd.merge(quantified_visual_bias_df,
                                               songs_frequency_data[['MOVIE NAME_cleaned', ' YEAR']].drop_duplicates(),
                                               left_on='Movie Name_cleaned', right_on='MOVIE NAME_cleaned', how='inner')

                 merged_visual_data.rename(columns={' YEAR': 'YEAR'}, inplace=True)
                 merged_visual_data.dropna(subset=['YEAR'], inplace=True)

                 if not merged_visual_data.empty:
                     merged_visual_data['YEAR'] = merged_visual_data['YEAR'].astype(int)
                     merged_visual_data['Decade'] = (merged_visual_data['YEAR'] // 10) * 10

                     # Calculate film-level visual bias score
                     if 'total_image_bias_score' in merged_visual_data.columns:
                         film_visual_bias_scores = merged_visual_data.groupby('Movie Name_from_image')['total_image_bias_score'].sum().reset_index()
                         film_visual_bias_scores.rename(columns={'total_image_bias_score': 'film_visual_bias_score'}, inplace=True)
                         print("\nFilm-level visual bias scores calculated.")
                     else:
                         print("'total_image_bias_score' column not found in merged visual data. Skipping film-level visual bias calculation.")
                         return film_visual_bias_scores, decade_visual_bias_scores # Return empty if essential data is missing


                     # Calculate decade-level visual bias score
                     decade_visual_bias_scores = merged_visual_data.groupby('Decade')['total_image_bias_score'].sum().reset_index()
                     decade_visual_bias_scores.rename(columns={'total_image_bias_score': 'decade_visual_bias_score'}, inplace=True)
                     print("\nDecade-level visual bias scores calculated.")

                 else:
                     print("No matching movies with year information found after merging for visual bias aggregation.")

            else:
                print("Required columns ('MOVIE NAME', ' YEAR') not found in songs_frequency_data. Cannot aggregate visual bias scores.")

        else:
            print("No movie names could be extracted from image filenames in quantified_visual_bias_df.")
    else:
        print("'image_filename' column not found in quantified_visual_bias_df. Skipping visual bias aggregation.")


    return film_visual_bias_scores, decade_visual_bias_scores

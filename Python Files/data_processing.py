import os
import pandas as pd
import re

def load_all_data(base_path='Bollywood-Data'):
    """
    Loads various datasets from specified CSV files within the base path.

    Args:
        base_path (str): The base path to the cloned repository.

    Returns:
        tuple: A tuple containing the loaded DataFrames:
               (plot_synopses_data, songs_data, songs_frequency_data,
                image_plot_mentions_data, trailer_transcripts_data, trailers_list_data)
               Returns None for any data that could not be loaded.
    """
    wikipedia_data_path = os.path.join(base_path, 'wikipedia-data')
    plot_synopses_csv_path = os.path.join(wikipedia_data_path, 'coref_plot.csv')
    songs_data_csv_path = os.path.join(wikipedia_data_path, 'songsDB.csv')
    songs_frequency_csv_path = os.path.join(wikipedia_data_path, 'songsFrequency.csv')
    image_plot_mentions_csv_path = os.path.join(wikipedia_data_path, 'image_and_plot_mentions_fequency.csv')

    trailer_data_path = os.path.join(base_path, 'trailer-data')
    trailer_transcripts_csv_path = os.path.join(trailer_data_path, 'complete-data.csv')
    trailers_list_csv_path = os.path.join(trailer_data_path, 'trailers_list.csv')

    plot_synopses_data = None
    songs_data = None
    songs_frequency_data = None
    image_plot_mentions_data = None
    trailer_transcripts_data = None
    trailers_list_data = None

    if os.path.exists(plot_synopses_csv_path):
        try:
            plot_synopses_data = pd.read_csv(plot_synopses_csv_path)
            print("Plot Synopses data loaded successfully.")
        except Exception as e:
            print(f"Error reading plot synopses CSV: {e}")
    else:
        print(f"Plot synopses CSV not found at: {plot_synopses_csv_path}")

    if os.path.exists(songs_data_csv_path):
        try:
            songs_data = pd.read_csv(songs_data_csv_path, on_bad_lines='skip')
            print("Songs data loaded successfully.")
        except Exception as e:
            print(f"Error reading songs data CSV: {e}")
    else:
        print(f"Songs data CSV not found at: {songs_data_csv_path}")

    if os.path.exists(songs_frequency_csv_path):
        try:
            songs_frequency_data = pd.read_csv(songs_frequency_csv_path, on_bad_lines='skip')
            print("Songs frequency data loaded successfully.")
        except Exception as e:
            print(f"Error reading songs frequency data CSV: {e}")
    else:
        print(f"Songs frequency data CSV not found at: {songs_frequency_csv_path}")

    if os.path.exists(image_plot_mentions_csv_path):
        try:
            image_plot_mentions_data = pd.read_csv(image_plot_mentions_csv_path, on_bad_lines='skip')
            print("Image and Plot Mentions frequency data loaded successfully")
        except Exception as e:
            print(f"Error reading image and plot mentions CSV: {e}")
    else:
        print(f"Image and Plot Mentions frequency data CSV not found at: {image_plot_mentions_csv_path}")

    if os.path.exists(trailer_transcripts_csv_path):
        try:
            trailer_transcripts_data = pd.read_csv(trailer_transcripts_csv_path)
            print("Trailer Transcripts data loaded successfully.")
        except Exception as e:
            print(f"Error reading trailer transcripts CSV: {e}")
    else:
        print(f"Trailer transcripts CSV not found at: {trailer_transcripts_csv_path}")

    if os.path.exists(trailers_list_csv_path):
        try:
            trailers_list_data = pd.read_csv(trailers_list_csv_path, on_bad_lines='skip')
            print("Trailers list data loaded successfully.")
        except Exception as e:
            print(f"Error reading trailers list CSV: {e}")
    else:
        print(f"Trailers list CSV not found at: {trailers_list_csv_path}")

    return (plot_synopses_data, songs_data, songs_frequency_data,
            image_plot_mentions_data, trailer_transcripts_data, trailers_list_data)

def clean_plot_synopses(df):
    """
    Cleans and preprocesses the plot synopses DataFrame.

    Args:
        df (pd.DataFrame): The raw plot synopses DataFrame.

    Returns:
        pd.DataFrame: The cleaned plot synopses DataFrame, or the original
                      DataFrame if input is None.
    """
    if df is None:
        print("Plot Synopses DataFrame is None, skipping cleaning.")
        return None

    cleaned_df = df.copy()

    # Drop the 'Unnamed: 0' column
    if 'Unnamed: 0' in cleaned_df.columns:
        cleaned_df = cleaned_df.drop('Unnamed: 0', axis=1)

    # Handle missing values in 'Coref Plot' column
    cleaned_df['Coref Plot'] = cleaned_df['Coref Plot'].fillna('')

    # Basic text cleaning for 'Coref Plot'
    cleaned_df['Cleaned Plot'] = cleaned_df['Coref Plot'].apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', x.lower()))
    cleaned_df['Cleaned Plot'] = cleaned_df['Cleaned Plot'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    print("Plot Synopses data cleaned.")
    return cleaned_df

def clean_trailer_transcripts(df):
    """
    Cleans and preprocesses the trailer transcripts DataFrame.

    Args:
        df (pd.DataFrame): The raw trailer transcripts DataFrame.

    Returns:
        pd.DataFrame: The cleaned trailer transcripts DataFrame, or the original
                      DataFrame if input is None.
    """
    if df is None:
        print("Trailer Transcripts DataFrame is None, skipping cleaning.")
        return None

    cleaned_df = df.copy()

    # Handle missing values - drop rows with missing 'emotion' or 'gender'
    cleaned_df.dropna(subset=['emotion', 'gender'], inplace=True)

    # Convert 'emotion' and 'gender' to lowercase for consistency
    cleaned_df['emotion'] = cleaned_df['emotion'].str.lower()
    cleaned_df['gender'] = cleaned_df['gender'].str.lower()

    print("Trailer Transcripts data cleaned.")
    return cleaned_df

# run_analysis.py
import os
import pandas as pd
import spacy # Need to import spacy for the textual analysis functions
from PIL import Image # Need to import PIL for the visual analysis functions

# Import functions from other modules
import data_processing
import text_bias_analysis
import visual_bias_analysis
import bias_aggregation
import remediation
import utils # Import the utilities module

# Define title_patterns here or import them if they are in utils
title_patterns = [
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "mr"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "mrs"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "ms"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "miss"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "dr"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "prof"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "inspector"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "detective"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "officer"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "constable"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "sir"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "madam"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "king"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "queen"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "prince"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "princess"}]},
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "father"}]}, # Sometimes used as a title/address
    {"label": "PERSON_TITLE", "pattern": [{"LOWER": "mother"}]}, # Sometimes used as a title/address
]


# Load spaCy model here (as it's used in text_bias_analysis functions applied in the loop)
try:
    nlp = spacy.load("en_core_web_sm")
    # Add entity ruler if not already present
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(title_patterns) # Add patterns here
    else:
         ruler = nlp.get_pipe("entity_ruler")
         ruler.add_patterns(title_patterns) # Add patterns here as well if it exists
except OSError:
    print("Downloading en_core_web_sm model for run_analysis...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        ruler.add_patterns(title_patterns) # Add patterns here
    else:
         ruler = nlp.get_pipe("entity_ruler")
         ruler.add_patterns(title_patterns) # Add patterns here as well if it exists


if __name__ == "__main__":
    print("Starting film bias analysis...")

    # Define the base path to the cloned repository
    base_path = 'Bollywood-Data'
    images_path = os.path.join(base_path, 'images-data')

    # 1. Load Data
    print("\nStep 1: Loading data...")
    (plot_synopses_data, songs_data, songs_frequency_data,
     image_plot_mentions_data, trailer_transcripts_data, trailers_list_data) = data_processing.load_all_data(base_path)
    print("Data loading complete.")

    # 2. Clean Data
    print("\nStep 2: Cleaning data...")
    cleaned_plot_synopses = data_processing.clean_plot_synopses(plot_synopses_data)
    # cleaned_trailer_transcripts = data_processing.clean_trailer_transcripts(trailer_transcripts_data) # Optional for this workflow
    print("Data cleaning complete.")

    # Ensure cleaned_plot_synopses is not None or empty before proceeding with textual analysis
    if cleaned_plot_synopses is None or cleaned_plot_synopses.empty:
        print("\nCleaned plot synopses data is not available or is empty. Skipping textual analysis.")
        textual_analysis_results = pd.DataFrame() # Initialize as empty DataFrame
    else:
        # 3. Initialize Gemini Models
        print("\nStep 3: Initializing Gemini models...")
        # Initialize text model (used for remediation suggestions)
        gemini_text_model = utils.initialize_gemini_model('gemini-1.5-flash-latest')
        # Initialize vision model (used for visual analysis)
        # gemini_vision_model = utils.initialize_gemini_vision_model('gemini-1.5-flash-latest') # Initialized within visual_bias_analysis


        # 4. Perform Textual Analysis
        print("\nStep 4: Performing textual analysis...")
        textual_analysis_results = cleaned_plot_synopses.copy() # Work on a copy
        # Apply the textual analysis functions row by row
        # Need to pass the nlp model to the extraction function
        textual_analysis_results['character_info'] = textual_analysis_results['Cleaned Plot'].apply(
            lambda x: text_bias_analysis.extract_character_info(x, nlp)
        )
        textual_analysis_results['stereotypes'] = textual_analysis_results['character_info'].apply(
            text_bias_analysis.categorize_stereotypes
        )
        textual_analysis_results['row_character_bias_score'] = textual_analysis_results['stereotypes'].apply(
            text_bias_analysis.calculate_textual_bias_score
        )
        print("Textual analysis complete.")


    # 5. Perform Visual Analysis (uses Gemini Vision internally)
    print("\nStep 5: Performing visual analysis...")
    # Perform visual analysis on images directory (process a subset for testing if needed)
    subset_size = 50 # Set to None to process all images
    visual_analysis_df = visual_bias_analysis.perform_visual_analysis(images_path, subset_size=subset_size)
    print("Visual analysis complete.")

    # 6. Quantify Visual Bias
    print("\nStep 6: Quantifying visual bias...")
    quantified_visual_bias_df = visual_bias_analysis.quantify_visual_bias(visual_analysis_df)
    print("Visual bias quantification complete.")

    # 7. Aggregate Bias Scores
    print("\nStep 7: Aggregating bias scores...")

    # Aggregate Textual Bias
    if not textual_analysis_results.empty and songs_frequency_data is not None:
         film_text_bias_scores, decade_text_bias_scores = bias_aggregation.aggregate_textual_bias_to_film_and_decade(
             textual_analysis_results, songs_frequency_data
         )
         print("Textual bias aggregation complete.")
    else:
         print("Textual analysis results or songs frequency data not available. Skipping textual bias aggregation.")
         film_text_bias_scores = pd.DataFrame()
         decade_text_bias_scores = pd.DataFrame()

    # Aggregate Visual Bias
    if not quantified_visual_bias_df.empty and songs_frequency_data is not None:
         film_visual_bias_scores, decade_visual_bias_scores = bias_aggregation.aggregate_visual_bias_to_film_and_decade(
             quantified_visual_bias_df, songs_frequency_data
         )
         print("Visual bias aggregation complete.")
    else:
        print("Quantified visual bias data or songs frequency data not available. Skipping visual bias aggregation.")
        film_visual_bias_scores = pd.DataFrame()
        decade_visual_bias_scores = pd.DataFrame()

    # 8. Generate Sample Bias Report (Optional)
    print("\nStep 8: Generating sample bias report (Optional)...")

    # Find a sample movie title with some bias for the report
    sample_movie_title = None

    # Try to find a movie with textual bias first
    if not film_text_bias_scores.empty and 'film_bias_score' in film_text_bias_scores.columns:
        biased_text_movies = film_text_bias_scores[film_text_bias_scores['film_bias_score'] > 0]
        if not biased_text_movies.empty:
            sample_movie_title = biased_text_movies.iloc[0]['Movie Name']

    # If no textual bias, try to find a movie with visual bias
    if sample_movie_title is None and not film_visual_bias_scores.empty and 'film_visual_bias_score' in film_visual_bias_scores.columns:
         biased_visual_movies = film_visual_bias_scores[film_visual_bias_scores['film_visual_bias_score'] > 0]
         if not biased_visual_movies.empty:
             # Use the original movie name from the visual aggregation result if available
             if 'Movie Name_from_image' in biased_visual_movies.columns:
                 sample_movie_title = biased_visual_movies.iloc[0]['Movie Name_from_image']
             else:
                 # Fallback to the cleaned name or just skip if original name is hard to get
                 print("Could not determine original movie name for visual bias, skipping report.")


    if sample_movie_title is not None and gemini_text_model is not None:
        print(f"Generating bias report for movie: {sample_movie_title}")
        try:
            # Ensure necessary dataframes are available for the report function
            if 'textual_analysis_results' in locals() and not textual_analysis_results.empty and \
               'film_text_bias_scores' in locals() and not film_text_bias_scores.empty and \
               'quantified_visual_bias_df' in locals() and not quantified_visual_bias_df.empty and \
               'film_visual_bias_scores' in locals() and not film_visual_bias_scores.empty:

                 remediation.generate_bias_report(
                     sample_movie_title,
                     textual_analysis_results,
                     film_text_bias_scores,
                     quantified_visual_bias_df,
                     film_visual_bias_scores,
                     gemini_text_model, # Pass the text model for remediation suggestions
                     output_filename=f"{utils.clean_movie_name(sample_movie_title)}_bias_report.pdf"
                 )
            else:
                 print("Required dataframes for report generation are not available or are empty. Skipping report.")

        except Exception as e:
            print(f"An error occurred during report generation: {e}")

    elif sample_movie_title is None:
        print("No movie with detected bias found. Skipping sample report generation.")
    elif gemini_text_model is None:
        print("Gemini text model not initialized. Cannot generate remediation suggestions in the report. Skipping report.")


    print("\nFilm bias analysis complete.")

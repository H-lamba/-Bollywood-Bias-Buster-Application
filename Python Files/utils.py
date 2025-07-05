
import google.generativeai as genai
from google.colab import userdata
import re
import pandas as pd # Although clean_movie_name doesn't directly use pandas, it's common in data processing utils.

def clean_movie_name(name):
    """
    Cleans movie names by removing years in parentheses or at the end,
    non-alphanumeric characters, and extra whitespace.

    Args:
        name (str): The raw movie name string.

    Returns:
        str or None: The cleaned movie name string, or None if input is not a string.
    """
    if isinstance(name, str):
        # Remove year in parentheses or at the end (e.g., " (2017)" or "_2017")
        name = re.sub(r'\s*\(?\d{4}\)?$', '', name)
        name = re.sub(r'_\d{4}$', '', name)
        # Remove non-alphanumeric characters and convert to lowercase
        name = re.sub(r'[^a-z0-9\s]', '', name.lower())
        # Replace multiple spaces with a single space and strip whitespace
        name = re.sub(r'\s+', ' ', name).strip()
        return name
    return None

def configure_gemini_api():
    """
    Configures the Gemini API using the GOOGLE_API_KEY from Colab secrets.

    Returns:
        bool: True if API is configured successfully, False otherwise.
    """
    try:
        GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Gemini API configured successfully.")
        return True
    except userdata.SecretNotFoundError:
        print("Warning: GOOGLE_API_KEY not found in Colab secrets.")
        print("Please add it to the secrets manager under the '🔑' in the left panel.")
        return False
    except Exception as e:
        print(f"An error occurred while configuring the Gemini API: {e}")
        return False

def initialize_gemini_model(model_name='gemini-1.5-flash-latest'):
    """
    Initializes a Gemini generative model after configuring the API.

    Args:
        model_name (str): The name of the Gemini model to initialize.

    Returns:
        genai.GenerativeModel or None: The initialized Gemini model if successful,
                                         None otherwise.
    """
    if configure_gemini_api():
        try:
            gemini_model = genai.GenerativeModel(model_name)
            print(f"Gemini model '{model_name}' initialized.")
            return gemini_model
        except Exception as e:
            print(f"Error initializing Gemini model '{model_name}': {e}")
            return None
    else:
        print("Gemini API not configured, cannot initialize model.")
        return None

def initialize_gemini_vision_model(model_name='gemini-1.5-flash-latest'):
    """
    Initializes a Gemini generative model suitable for vision tasks after configuring the API.

    Args:
        model_name (str): The name of the Gemini vision model to initialize.

    Returns:
        genai.GenerativeModel or None: The initialized Gemini vision model if successful,
                                         None otherwise.
    """
    return initialize_gemini_model(model_name) # Vision models are also generative models

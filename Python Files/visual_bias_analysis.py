
import os
import json
import pandas as pd
from PIL import Image
import glob
import google.generativeai as genai
from google.colab import userdata # Used to securely store your API key

def initialize_gemini_vision_model():
    """
    Configures the Gemini API and initializes the Gemini Vision model.

    Returns:
        genai.GenerativeModel or None: The initialized Gemini Vision model
                                         if successful, None otherwise.
    """
    try:
        GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
        genai.configure(api_key=GOOGLE_API_KEY)
        print("Gemini API configured successfully.")
    except userdata.SecretNotFoundError:
        print("Warning: GOOGLE_API_KEY not found in Colab secrets.")
        print("Please add it to the secrets manager under the '🔑' in the left panel.")
        return None
    except Exception as e:
        print(f"An error occurred while configuring the Gemini API: {e}")
        return None

    gemini_vision_model = None
    try:
        # Use a model that supports multimodal input (text and images)
        gemini_vision_model = genai.GenerativeModel('gemini-1.5-flash-latest') # Or 'gemini-1.5-pro-latest'
        print("Gemini Vision model initialized.")
    except Exception as e:
        print(f"Error initializing Gemini Vision model: {e}")
        return None

    return gemini_vision_model

def find_image_files(images_path):
    """
    Finds all image files (jpg and png) in the specified directory and its subdirectories.

    Args:
        images_path (str): The path to the directory containing the images.

    Returns:
        list: A list of paths to the found image files.
    """
    image_files = []
    if os.path.exists(images_path):
        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(images_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
    else:
        print(f"Images directory not found at: {images_path}")
    return image_files


def analyze_image_with_gemini(image_path, gemini_model):
    """
    Analyzes a single image using the Gemini Vision model to identify potential stereotypes.

    Args:
        image_path (str): The path to the image file.
        gemini_model: The initialized Gemini generative model capable of vision tasks.

    Returns:
        dict: A dictionary containing the analysis data parsed from the Gemini response,
              or a dictionary with an 'error' key if analysis fails.
    """
    analysis_data = {'image_filename': os.path.basename(image_path), 'characters': []} # Initialize with default structure

    if gemini_model is None:
        analysis_data['error'] = 'Gemini model not initialized'
        return analysis_data

    try:
        img = Image.open(image_path)

        # Craft a more structured prompt to extract specific information
        prompt = """Analyze this movie poster for potential gender stereotypes.
        Identify all people in the poster. For each person, provide the following information in a structured format:
        - Apparent gender (e.g., male, female, ambiguous, unknown)
        - Brief description of clothing
        - Brief description of pose or action
        - List of prominent objects or symbols associated with the person
        - Any visual elements that suggest traditional gender roles or stereotypes for that person.

        Format the output as a JSON object with a list of "characters", where each character is an object with keys like "id", "gender", "clothing", "pose", "associated_objects", and "stereotypes_detected". If no people are detected, the "characters" list should be empty.
        """

        # Send the prompt and image to the Gemini model
        response = gemini_model.generate_content([prompt, img])

        # Attempt to parse the JSON response
        raw_response_text = response.text.strip()

        try:
            # Attempt to load JSON, handling potential markdown code blocks
            if raw_response_text.startswith('```json'):
                raw_response_text = raw_response_text[7:-3].strip() # Remove ```json and ```

            parsed_json = json.loads(raw_response_text)

            if 'characters' in parsed_json and isinstance(parsed_json['characters'], list):
                 analysis_data['characters'] = parsed_json['characters']
            else:
                 print(f"Warning: 'characters' key not found or not a list in JSON for {os.path.basename(image_path)}. Raw response:")
                 print(response.text)
                 analysis_data['error'] = "'characters' key missing or not a list"
                 analysis_data['raw_response'] = response.text

        except json.JSONDecodeError:
            print(f"Warning: Could not parse JSON response for {os.path.basename(image_path)}. Raw response:")
            print(response.text)
            analysis_data['error'] = 'JSON parsing failed'
            analysis_data['raw_response'] = response.text
        except Exception as e:
             print(f"Warning: Error processing response for {os.path.basename(image_path)}: {e}")
             analysis_data['error'] = f'Response processing failed: {e}'
             analysis_data['raw_response'] = response.text

        img.close() # Close the image

    except Exception as e:
        print(f"Error analyzing image {os.path.basename(image_path)}: {e}")
        analysis_data['error'] = f'Image analysis failed: {e}'

    return analysis_data

# Define a scoring mechanism for different types of visual stereotypes identified by Gemini
# These scores can be adjusted based on domain expertise or further analysis
visual_stereotype_scores = {
    'Passive pose, emphasizing beauty and adornment': 2,
    'Traditional attire suggests a focus on femininity and domesticity': 1,
    'Serious expression may suggest a character burdened by responsibility or a brooding nature, often associated with masculine stereotypes': 1,
    'Bright, possibly flamboyant clothing, suggesting a more outgoing or potentially rebellious character, although this alone is not a definitive stereotype. Could be contrasted with character 2, representing different masculine archetypes.': 0.5, # Lower score as it's less definitive
    'The man is positioned centrally and prominently, suggesting a leading role. The \'Follow Me\' shirt reinforces a more assertive, possibly dominant, stereotype of masculinity.': 3,
    'The woman\'s pose and closeness to the man suggest a traditional, romantic stereotype of a woman being supportive and dependent. Her clothing is somewhat traditional, which could also be seen as reinforcing a gender stereotype.': 2.5,
    'The woman\'s headscarf might be interpreted as a traditional representation of modesty and possibly passivity, depending on cultural context and viewer interpretation. Her downcast eyes might also suggest a submissive or melancholic persona.': 2,
    'Passive pose suggesting a damsel-in-distress or romantic interest archetype.': 2,
    'Subdued appearance might imply traditional feminine ideals of demureness.': 1.5,
    'Formal attire suggesting a professional or established figure. This could be interpreted as a stereotypically masculine image of authority or success.': 1,
    'Casual attire contrasts with the more formal clothing of the other male character.': 0, # Not necessarily a stereotype on its own
    'Pose is relatively relaxed compared to other figures, perhaps suggesting a more approachable or less serious persona.': 0, # Not necessarily a stereotype on its own
    'Aggressive, serious demeanor often associated with male action heroes.': 2,
    'Innocent, possibly a victim or falsely accused, although the context suggests potential rebellion.': 0.5, # Less of a strong gender stereotype
    'Soft features and a somewhat passive pose, suggesting stereotypical feminine vulnerability; this is somewhat subdued compared to the male figures.': 2.5,
    'Typical representation of a rebellious or antagonistic character, often associated with male roles in action films.': 1.5,
    'Passive expression suggesting a potential romantic interest': 1.5,
    'Serious, brooding expression, potentially associated with a masculine archetype of a strong character': 1.5,
    'Dominant pose, suggesting a controlling or protective stance over a female character': 3,
    'Submissive, looking slightly away, appears almost held captive': 3.5,
    'Bangles could be interpreted as traditional adornment for women': 1,
    'Emphasis on appearance/beauty, consistent with the objectification of women in some film genres.': 3,
    'The pose and the text on the T-shirt suggest an assertive and possibly dominant role, which aligns with some traditional male stereotypes.': 2.5,
    'The demure pose and traditional clothing could be interpreted as reinforcing stereotypical notions of femininity and passivity.': 2,
    'The head covering could be interpreted in various ways, some of which might align with religious or cultural expectations, but it also appears in a somewhat melancholic context which may be played up as a traditional female trope.': 1.5,
    'The woman\'s pose and attire suggest a traditional, perhaps passive, feminine role. The presence of flowers may allude to a romantic or passive role within the story. The lack of active involvement in the poster\'s main image further reinforces potential passive stereotypes.': 2.5,

    # Adding stereotype strings exactly as identified by Gemini in the last output
    'Men depicted as aggressive, holding a weapon.': 2,
    'Passive, demure feminine beauty, traditional attire reinforces expectations of femininity and submissiveness.': 2,
    'More flamboyant style of dress, breaking from traditional norms for males, potentially suggesting a more rebellious or charismatic persona. The contrast with character 2 is notable.': 0.5,
    'Serious, brooding male character, possibly indicating a strong, perhaps troubled, masculine figure.': 1.5,
    'The woman is portrayed in a more passive position, her gaze and posture suggesting submissiveness or dependence on the man.': 2.5,
    'Her clothing style aligns with traditional representations of South Asian women.': 1,
    'The woman\'s head covering might suggest a portrayal of traditional modesty or religious observance, which can be seen as stereotypical in some contexts.': 2,
    'The man is presented as dominant, indicated by his position and the text on his shirt suggesting a pursuit of the woman.': 3,
    'Her downcast pose suggests some form of emotional reserve or sadness.': 1.5,
    'Passive pose, suggesting a more demure or romantic role': 2,
    'Formal attire suggesting professionalism or a more serious demeanor, a common male stereotype.': 1,
    'Subdued expression aligns with potential stereotypes of feminine emotional restraint': 1.5,
    'More casual attire compared to the other male, suggesting a potential difference in social standing or personality within the story, a potential trope.': 0,
    'Potentially presented as innocent or a victim based on clothing and pose.': 0.5,
    'Women depicted in a passive role, with a worried expression.': 2.5,
    'The woman\'s smile and pose could be interpreted as passively receptive, a common stereotype of women in some media.': 2,
    'The man\'s pose, holding the woman, might suggest a controlling or possessive aspect, a stereotype of some male characters in movies.': 3,
    'The man\'s central placement and serious expression might reinforce a stereotypical idea of the male protagonist as strong and decisive.': 1.5,
    'The woman\'s pose and the fact that she\'s being held by a male character could reinforce traditional gender stereotypes related to passivity and dependence on men.': 3.5,
    'Traditional portrayal of a woman\'s role. The soft expression and pose might suggest a passive or romantic character archetype.': 2,
    'The serious expression might align with traditional stereotypes of men as being stoic and strong. The head-on gaze could imply assertiveness.': 1.5,
    'The phrase \'FOLLOW ME\' on his shirt might suggest a dominant, assertive male role in the relationship.': 2.5,
    'The traditional clothing might reinforce a traditional, subservient female role, especially in the context of the man\'s assertive pose. The placement of the woman being behind the man also suggests a more passive role.': 2,
    'The headscarf could be interpreted as reinforcing stereotypical religious or cultural norms associated with women. Her subdued expression might suggest a passive role or emotional restraint. The partial visibility also contributes to a feeling of her being marginalized in the image.': 1.5,
    'The bindi and traditional clothing could be interpreted as reinforcing stereotypical representations of South Asian women as demure or traditional. The pose also contributes to this feeling.': 1.5,
    'The reflection in the mirror and serious expression could reinforce stereotypes of the male as a mysterious or perhaps somewhat brooding figure. The inclusion of roses suggests a potential romantic interest which is a common trope in media.': 1,
    'Passive pose, suggesting a more traditionally feminine role. The elaborate jewelry and attire may reinforce expectations of beauty and adornment for women.': 2,
    'Serious expression might be interpreted as depicting traditional masculine traits like strength or brooding intensity.': 1.5,
    'The bright clothing and confident expression could be seen as reflecting traditional masculine traits like boldness or charisma. The clothing is more flamboyant than that of character 2, suggesting a possible difference in character types.': 0.5,
    'Passive, demure expression often associated with feminine ideals in some cultures.': 1.5,
    'Wearing a headscarf, possibly suggesting religious piety or modesty, which can be a stereotype linked to certain gender roles.': 1.5,
    'Central position might suggest a protagonist role.': 0,

    # Stereotypes identified in the last run that were not in the initial dictionary
    'Minor character, part of a background scene of potential chaos or action.': 0.1, # Assign a low score
    'Men as strong, serious figures. Association with violence through guns.': 2.5, # Assign a score related to aggression
    'Passive, submissive role in contrast to the other male characters. The white clothing could subtly suggest innocence or purity. The Gandhi cap adds a layer of possible political symbolism.': 2.8, # Assign a score related to submissiveness
    'Relatively passive role; shown as worried, suggesting a damsel-in-distress archetype. Less prominent than the male characters on the poster.': 2.2, # Assign a score related to passivity/victimhood
}


def calculate_visual_bias_scores(analysis_results_list, stereotype_scores_dict):
    """
    Calculates character-level and image-level visual bias scores from the analysis results.

    Args:
        analysis_results_list (list): A list of dictionaries from the visual analysis results.
        stereotype_scores_dict (dict): A dictionary mapping stereotype strings to numerical scores.

    Returns:
        list: A list of dictionaries, each representing an image with aggregated bias scores.
    """
    quantified_bias_results = []

    for image_data in analysis_results_list:
        image_filename = image_data.get('image_filename', 'Unknown')
        characters = image_data.get('characters', [])
        image_bias_score = 0
        character_bias_details = []

        if isinstance(characters, list):
            for character in characters:
                character_id = character.get('id', 'Unknown')
                stereotypes_detected = character.get('stereotypes_detected', [])
                character_score = 0
                stereotype_breakdown = {}

                if isinstance(stereotypes_detected, list):
                    for stereotype in stereotypes_detected:
                        # Ensure stereotype is a string and in our scoring dictionary
                        if isinstance(stereotype, str):
                            score = stereotype_scores_dict.get(stereotype, 0) # Use .get with default 0 for unknown stereotypes
                            character_score += score
                            stereotype_breakdown[stereotype] = stereotype_breakdown.get(stereotype, 0) + score # Track score per stereotype

                image_bias_score += character_score # Add character score to image total

                character_bias_details.append({
                    'character_id': character_id,
                    'character_bias_score': character_score,
                    'stereotypes_breakdown': stereotype_breakdown
                })

        quantified_bias_results.append({
            'image_filename': image_filename,
            'total_image_bias_score': image_bias_score,
            'character_bias_details': character_bias_details
        })

    return quantified_bias_results

def perform_visual_analysis(images_path, subset_size=None):
    """
    Performs systematic visual analysis of movie posters using the Gemini Vision model.

    Args:
        images_path (str): The path to the directory containing the images.
        subset_size (int, optional): The number of images to process. If None, process all.

    Returns:
        pd.DataFrame: A DataFrame containing the analysis results for each image.
    """
    gemini_vision_model = initialize_gemini_vision_model()
    if gemini_vision_model is None:
        print("Gemini Vision model could not be initialized. Skipping visual analysis.")
        return pd.DataFrame()

    image_files = find_image_files(images_path)
    if not image_files:
        print("No image files found for visual analysis.")
        return pd.DataFrame()

    if subset_size is not None:
        images_to_process = image_files[:subset_size]
        print(f"\nStarting systematic analysis of a subset of {len(images_to_process)} images...")
    else:
        images_to_process = image_files
        print(f"\nStarting systematic analysis of {len(images_to_process)} images...")


    image_analysis_results = []
    for i, image_file in enumerate(images_to_process):
        # print(f"Analyzing image {i+1}/{len(images_to_process)}: {os.path.basename(image_file)}")
        analysis_data = analyze_image_with_gemini(image_file, gemini_vision_model)
        image_analysis_results.append(analysis_data)

    print("\nSystematic image analysis completed.")

    # Convert the results to a pandas DataFrame
    visual_bias_data = pd.DataFrame(image_analysis_results)

    return visual_bias_data

def quantify_visual_bias(visual_analysis_df):
    """
    Quantifies visual bias scores based on the analysis results DataFrame.

    Args:
        visual_analysis_df (pd.DataFrame): DataFrame containing visual analysis results
                                           with a 'characters' column.

    Returns:
        pd.DataFrame: A DataFrame containing the quantified visual bias scores per image.
    """
    if visual_analysis_df is None or visual_analysis_df.empty:
        print("Visual analysis DataFrame not available or is empty. Skipping visual bias quantification.")
        return pd.DataFrame()

    print("\nCalculating visual bias scores...")

    # Handle potential errors or non-list format in 'characters' column
    # Convert the DataFrame rows to a list of dictionaries for easier processing
    visual_analysis_list = visual_analysis_df.to_dict('records')

    # Use the global visual_stereotype_scores dictionary
    quantified_visual_bias_results = calculate_visual_bias_scores(visual_analysis_list, visual_stereotype_scores)

    # Convert the results back to a DataFrame for easier manipulation
    quantified_visual_bias_df = pd.DataFrame(quantified_visual_bias_results)

    print("Visual bias scores calculated.")

    return quantified_visual_bias_df

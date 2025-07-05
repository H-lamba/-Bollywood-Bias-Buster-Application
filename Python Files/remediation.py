import google.generativeai as genai
import pandas as pd # Import pandas for DataFrame manipulation
import os # Import os for path joining
from fpdf import FPDF # Import FPDF for PDF generation
# from google.colab import userdata # Assume API key is handled elsewhere
# from PIL import Image # Assuming image object is handled before calling remediation

def suggest_poster_remediation(image_filename, visual_bias_details, gemini_model):
    """
    Generates textual suggestions for poster remediation based on identified visual biases
    using the Gemini LLM.

    Args:
        image_filename (str): The filename of the image (for context in the prompt).
        visual_bias_details (list): A list of dictionaries containing character bias details
                                     (from quantified_visual_bias_df['character_bias_details']).
        gemini_model: The initialized Gemini generative model (text-capable).

    Returns:
        str: A string containing the suggested remediation alternatives, or an error message.
    """
    if not visual_bias_details:
        return "No specific character bias details available for this image to suggest remediation."

    bias_description = "Identified visual biases in the poster:\n"
    for character_detail in visual_bias_details:
        character_id = character_detail.get('character_id', 'Unknown Character')
        bias_score = character_detail.get('character_bias_score', 0)
        stereotypes = character_detail.get('stereotypes_breakdown', {})

        if bias_score > 0 and stereotypes:
            bias_description += f"- For {character_id} (Bias Score: {bias_score}):\n"
            for stereotype, score in stereotypes.items():
                bias_description += f"  - Stereotype: '{stereotype}' (Score: {score})\n"

    if gemini_model is not None:
        prompt = f"""
        Based on the following identified visual biases in a movie poster ({image_filename}):

        {bias_description}

        Please suggest ways to alter the poster's visual elements (e.g., character poses, clothing, composition, associated objects) to reduce or eliminate these gender stereotypes, while maintaining the overall theme or genre of a Bollywood movie poster.

        Provide concise and actionable suggestions.

        Remediation Suggestions:
        """

        try:
            # Make the API call to Gemini
            response = gemini_model.generate_content(prompt)
            # Extract the generated text
            return response.text.strip()

        except Exception as e:
            return f"Error generating remediation suggestions with Gemini: {e}"

    else:
        return "Gemini model not initialized. Cannot generate remediation suggestions."


def remediate_bias(text, stereotypes, gemini_model):
    """
    Generates bias-free rewrites or suggestions for text based on identified stereotypes
    using the Gemini LLM if available, otherwise uses a placeholder.

    Args:
        text (str): The original text snippet flagged for bias.
        stereotypes (list): A list of identified stereotype categories associated with the text.
        gemini_model: The initialized Gemini generative model (text-capable).

    Returns:
        dict: A dictionary containing the original text, identified stereotypes,
              and suggested bias-free alternatives.
    """
    suggested_alternatives = []

    if not stereotypes:
        return {
            "original_text": text,
            "identified_stereotypes": [],
            "suggested_alternatives": ["No stereotypes identified, no alternatives suggested."]
        }

    if gemini_model is not None:
        # Create a prompt for the Gemini model
        prompt = f"""
        The following text excerpt from a movie plot synopsis has been identified with potential gender stereotypes.
        Original text: "{text}"
        Identified stereotypes: {', '.join(stereotypes)}

        Please provide a few alternative phrasings or suggestions to rewrite this text to reduce or eliminate these stereotypes, while preserving the original narrative intent as much as possible.

        Suggestions:
        """

        try:
            # Make the API call to Gemini
            response = gemini_model.generate_content(prompt)
            # Extract the generated text
            gemini_suggestions = response.text.strip().split('\n')
            suggested_alternatives = [s for s in gemini_suggestions if s] # Remove empty strings

            if not suggested_alternatives:
                 suggested_alternatives.append("Gemini did not return any specific suggestions.")

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            suggested_alternatives = [f"Error generating suggestions with Gemini: {e}"]

    else:
        # Placeholder logic if Gemini model is not available
        suggested_alternatives.append("Gemini model not initialized. Cannot generate alternatives using LLM.")
        # Add some basic rule-based suggestions here as a fallback if needed
        # These rules are illustrative and would need to be more sophisticated
        # if 'Stereotypical Female Profession' in stereotypes:
        #     suggested_alternatives.append(f"Consider revising description of profession to be gender-neutral or highlight diverse roles.")
        # if 'Potential Passive Agency' in stereotypes:
        #     suggested_alternatives.append(f"Rewrite sentences to give the character more active verbs and agency.")
        # if 'Potential Authority/Provider Role' in stereotypes:
        #      suggested_alternatives.append(f"Ensure other characters, regardless of gender, also demonstrate authority or act as providers.")
        # if 'Potential Appearance Stereotype (Female)' in stereotypes:
        #      suggested_alternatives.append(f"Focus descriptions on skills, personality, or actions rather than solely on physical appearance.")


    return {
        "original_text": text,
        "identified_stereotypes": stereotypes,
        "suggested_alternatives": suggested_alternatives
    }

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Film Bias Analysis Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(2)

    def chapter_body(self, body):
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, body)
        self.ln()

def generate_bias_report(movie_title, textual_analysis_df, film_text_bias_scores, quantified_visual_bias_df, film_visual_bias_scores, gemini_text_model, output_filename="bias_report.pdf"):
    """
    Generates a PDF bias report for a specific movie, including textual and visual
    bias details and remediation suggestions.

    Args:
        movie_title (str): The title of the movie for the report.
        textual_analysis_df (pd.DataFrame): DataFrame containing row-level textual analysis results.
        film_text_bias_scores (pd.DataFrame): DataFrame containing film-level textual bias scores.
        quantified_visual_bias_df (pd.DataFrame): DataFrame containing image-level visual bias scores.
        film_visual_bias_scores (pd.DataFrame): DataFrame containing film-level visual bias scores.
        gemini_text_model: The initialized Gemini generative model (text-capable) for remediation.
        output_filename (str): The name for the output PDF file.
    """
    pdf = PDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.chapter_title(f"Bias Analysis Report for: {movie_title}")

    # Textual Bias Summary
    pdf.chapter_title("Textual Bias Summary")
    text_bias_summary = film_text_bias_scores[film_text_bias_scores['Movie Name'] == movie_title]
    if not text_bias_summary.empty:
        pdf.chapter_body(f"Total Textual Bias Score: {text_bias_summary.iloc[0]['film_bias_score']:.2f}")

        # Detailed Textual Bias (from row-level data for this movie)
        movie_text_analysis = textual_analysis_df[textual_analysis_df['Movie Name'] == movie_title]
        if not movie_text_analysis.empty:
            pdf.chapter_body("\nDetailed Textual Bias Findings:")
            for index, row in movie_text_analysis.iterrows():
                if row['row_character_bias_score'] > 0:
                    pdf.chapter_body(f"  - Snippet (Row {index}): '{row['Cleaned Plot'][:150]}...'")
                    pdf.chapter_body(f"    Identified Stereotypes: {row['stereotypes']}")
                    pdf.chapter_body(f"    Bias Score for Snippet: {row['row_character_bias_score']:.2f}")

                    # Textual Remediation Suggestion
                    if row['stereotypes'] and gemini_text_model:
                         try:
                             remediation_suggestion_dict = remediate_bias(row['Cleaned Plot'], list(row['stereotypes'].values())[0], gemini_text_model)
                             pdf.chapter_body("    Suggested Remediation:")
                             for suggestion in remediation_suggestion_dict['suggested_alternatives']:
                                 pdf.chapter_body(f"      - {suggestion}")
                         except Exception as e:
                             pdf.chapter_body(f"    Error generating textual remediation: {e}")
                    elif not gemini_text_model:
                         pdf.chapter_body("    Gemini text model not initialized, skipping textual remediation suggestion.")

    else:
        pdf.chapter_body("No textual bias data found for this movie.")

    # Visual Bias Summary
    pdf.chapter_title("Visual Bias Summary")
    visual_bias_summary = film_visual_bias_scores[film_visual_bias_scores['Movie Name_from_image'] == movie_title]
    if not visual_bias_summary.empty:
        pdf.chapter_body(f"Total Visual Bias Score: {visual_bias_summary.iloc[0]['film_visual_bias_score']:.2f}")

        # Detailed Visual Bias (from image-level data for this movie)
        movie_visual_analysis = quantified_visual_bias_df[quantified_visual_bias_df['Movie Name_from_image'] == movie_title]
        if not movie_visual_analysis.empty:
            pdf.chapter_body("\nDetailed Visual Bias Findings:")
            for index, row in movie_visual_analysis.iterrows():
                if row['total_image_bias_score'] > 0:
                    pdf.chapter_body(f"  - Image: {row['image_filename']}")
                    pdf.chapter_body(f"    Total Image Bias Score: {row['total_image_bias_score']:.2f}")
                    pdf.chapter_body("    Character Bias Details:")
                    for char_detail in row['character_bias_details']:
                        pdf.chapter_body(f"      - Character ID: {char_detail['character_id']}")
                        pdf.chapter_body(f"        Character Bias Score: {char_detail['character_bias_score']:.2f}")
                        pdf.chapter_body("        Stereotypes Breakdown:")
                        for stereotype, score in char_detail['stereotypes_breakdown'].items():
                            pdf.chapter_body(f"          - '{stereotype}': {score:.2f}")

                    # Visual Remediation Suggestion
                    if row['character_bias_details'] and gemini_text_model:
                         try:
                             remediation_suggestion = suggest_poster_remediation(row['image_filename'], row['character_bias_details'], gemini_text_model)
                             pdf.chapter_body("    Suggested Poster Remediation:")
                             pdf.chapter_body(remediation_suggestion)
                         except Exception as e:
                              pdf.chapter_body(f"    Error generating visual remediation: {e}")
                    elif not gemini_text_model:
                         pdf.chapter_body("    Gemini text model not initialized, skipping visual remediation suggestion.")


    else:
        pdf.chapter_body("No visual bias data found for this movie.")


    pdf.output(output_filename)
    print(f"\nBias report saved as {output_filename}")

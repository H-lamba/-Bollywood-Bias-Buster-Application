import spacy
import re
import pandas as pd

# Load spaCy model and add entity ruler (ensure this runs only once)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading en_core_web_sm model...")
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Add a simple pattern for common titles that might indicate gender/role
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

# Add entity_ruler only if it's not already in the pipeline
if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(title_patterns)
else:
    # If it exists, clear existing patterns and add new ones
    ruler = nlp.get_pipe("entity_ruler")
    ruler.clear()
    ruler.add_patterns(title_patterns)


def extract_character_info(plot_text, nlp_model):
    """
    Extracts character information and attributes from plot text using spaCy.

    Args:
        plot_text (str): The cleaned text of the movie plot synopsis.
        nlp_model: The initialized spaCy language model with an entity ruler.

    Returns:
        dict: A dictionary where keys are character lemmas and values are
              dictionaries containing 'mentions' (list of positions) and
              'attributes' (dictionary of lists for profession, agency,
              relationship, appearance, and gender_hints).
    """
    doc = nlp_model(plot_text)
    character_info = {}
    potential_characters_lemmas = set()

    # Extract capitalized words within this function
    capitalized_words = re.findall(r'\b[A-Z][a-zA-Z]*\b', plot_text)

    # 1. Identify potential characters using NER (including custom titles) and subjects of verbs
    for ent in doc.ents:
        if ent.label_ == "PERSON" or ent.label_ == "PERSON_TITLE":
            potential_characters_lemmas.add(ent.text.lower())

    for token in doc:
        if token.pos_ == "VERB":
            for child in token.children:
                if child.dep_ == "nsubj" or child.dep_ == "nsubjpass":
                    potential_characters_lemmas.add(child.lemma_.lower())

    # Clean up potential character list - remove common non-character terms
    common_non_characters = {'it', 'this', 'that', 'which', 'who', 'he', 'she', 'they', 'we', 'you', 'i', 'there', 'what', 'move', 'thing', 'man', 'woman', 'boy', 'girl', 'people', 'someone', 'somebody', 'nobody', 'everyone', 'everybody'}
    potential_characters_lemmas = {char for char in potential_characters_lemmas if char not in common_non_characters and len(char) > 1}

    # 2. Process each potential character lemma to find mentions and attributes
    for char_lemma in potential_characters_lemmas:
        character_info[char_lemma] = {
            'mentions': [],
            'attributes': {
                'profession': [],
                'agency': [],
                'relationship': [],
                'appearance': [],
                'gender_hints': []
            }
        }

        # Find mentions: look for the lemma or capitalized versions of the lemma
        mention_pattern = r'\b' + re.escape(char_lemma) + r'\b'
        capitalized_variation_pattern = r'\b' + char_lemma.capitalize() + r'[a-zA-Z]*\b'

        for match in re.finditer(mention_pattern, plot_text.lower()):
             character_info[char_lemma]['mentions'].append(match.start())

        for match in re.finditer(capitalized_variation_pattern, plot_text):
             if match.start() not in character_info[char_lemma]['mentions']:
                character_info[char_lemma]['mentions'].append(match.start())

        character_info[char_lemma]['mentions'].sort()

        # 3. Extract attributes and gender hints from sentences containing mentions or likely related pronouns/names
        for sent in doc.sents:
             sent_text_lower = sent.text.lower()
             is_relevant_sentence = False

             if re.search(r'\b' + re.escape(char_lemma) + r'\b', sent_text_lower):
                 is_relevant_sentence = True
             elif re.search(r'\b' + char_lemma.capitalize() + r'[a-zA-Z]*\b', sent.text):
                  is_relevant_sentence = True

             if char_lemma in character_info:
                 current_gender_hints = character_info[char_lemma]['attributes']['gender_hints']
                 if (' he ' in sent_text_lower or ' him ' in sent_text_lower or ' his ' in sent_text_lower) and any(hint.startswith('male') for hint in current_gender_hints):
                     is_relevant_sentence = True
                 elif (' she ' in sent_text_lower or ' her ' in sent_text_lower or ' hers ' in sent_text_lower) and any(hint.startswith('female') for hint in current_gender_hints):
                     is_relevant_sentence = True

             if is_relevant_sentence:
                context = sent_text_lower

                professions = ['doctor', 'engineer', 'teacher', 'police', 'businessman', 'lawyer', 'student', 'worker',
                               'manager', 'artist', 'nurse', 'secretary', 'chef', 'driver', 'servant', 'politician',
                               'criminal', 'detective', 'inspector', 'actor', 'actress', 'director', 'producer',
                               'musician', 'singer', 'dancer', 'writer', 'journalist', 'scientist', 'engineer', 'architect',
                               'shopkeeper', 'farmer', 'labourer', 'clergy', 'soldier', 'captain', 'major', 'general']
                for prof in professions:
                    if prof in context:
                        character_info[char_lemma]['attributes']['profession'].append(prof)

                sentence_doc = nlp_model(sent.text)
                for token in sentence_doc:
                    if token.pos_ == "VERB":
                         for child in token.children:
                            if (child.dep_ == "nsubj" or child.dep_ == "nsubjpass"):
                                if child.lemma_.lower() == char_lemma:
                                     verb_text = token.text
                                     character_info[char_lemma]['attributes']['agency'].append(verb_text)
                                elif child.text in capitalized_words and child.text.lower().startswith(char_lemma):
                                     verb_text = token.text
                                     character_info[char_lemma]['attributes']['agency'].append(verb_text)

                relationships = ['father', 'mother', 'son', 'daughter', 'brother', 'sister', 'husband', 'wife', 'friend',
                                 'lover', 'boss', 'colleague', 'uncle', 'aunt', 'cousin', 'grandma', 'grandpa', 'neighbor',
                                 'partner', 'enemy', 'stranger', 'fiance', 'guardian', 'relative', 'child', 'parent',
                                 'family', 'couple', 'widow', 'widower', 'orphan']
                for rel in relationships:
                    if rel in context:
                         character_info[char_lemma]['attributes']['relationship'].append(rel)

                appearances = ['beautiful', 'handsome', 'old', 'young', 'tall', 'short', 'strong', 'weak', 'dressed',
                               'wearing', 'pretty', 'ugly', 'fat', 'thin', 'rich', 'poor', 'blind', 'deaf', 'disabled',
                               'fair', 'dark', 'attractive', 'unattractive', 'healthy', 'sick', 'injured', 'charming', 'kind', 'cruel', 'evil', 'good', 'brave', 'cowardly']
                for app in appearances:
                     if app in context:
                         character_info[char_lemma]['attributes']['appearance'].append(app)

                if ' he ' in context or ' him ' in context or ' his ' in context:
                     character_info[char_lemma]['attributes']['gender_hints'].append('male_pronoun')
                if ' she ' in context or ' her ' in context or ' hers ' in context:
                     character_info[char_lemma]['attributes']['gender_hints'].append('female_pronoun')
                if ' mr ' in context:
                     character_info[char_lemma]['attributes']['gender_hints'].append('mr_title')
                if ' mrs ' in context or ' ms ' in context or ' miss ' in context:
                     character_info[char_lemma]['attributes']['gender_hints'].append('female_title')
                gendered_relationships = {'father': 'male', 'mother': 'female', 'son': 'male', 'daughter': 'female',
                                          'brother': 'male', 'sister': 'female', 'husband': 'male', 'wife': 'female',
                                          'uncle': 'male', 'aunt': 'female', 'grandpa': 'male', 'grandma': 'female',
                                          'widower': 'male', 'widow': 'female'}
                for rel, gender in gendered_relationships.items():
                    if rel in context:
                        character_info[char_lemma]['attributes']['gender_hints'].append(gender + '_relationship')

        for char, info in character_info.items():
             for attr_type in info['attributes']:
                info['attributes'][attr_type] = list(set(info['attributes'][attr_type]))

        if char_lemma in character_info and not character_info[char_lemma]['mentions'] and not any(info for attr, info in character_info[char_lemma]['attributes'].items() if attr != 'gender_hints'):
             del character_info[char_lemma]

    return character_info


def categorize_stereotypes(character_info_dict):
    """
    Categorizes potential stereotypes based on extracted character attributes and gender hints.

    Args:
        character_info_dict (dict): A dictionary containing extracted character information.

    Returns:
        dict: A dictionary where keys are character lemmas and values are
              lists of identified stereotype categories.
    """
    stereotypes = {}

    male_hints = ['male_pronoun', 'mr_title', 'male_relationship']
    female_hints = ['female_pronoun', 'female_title', 'female_relationship']

    for char_lemma, info in character_info_dict.items():
        potential_genders = set()
        for hint in info['attributes']['gender_hints']:
            if hint in male_hints:
                potential_genders.add('male')
            elif hint in female_hints:
                potential_genders.add('female')

        if 'male' in potential_genders and 'female' in potential_genders:
            gender = 'ambiguous'
        elif 'male' in potential_genders:
            gender = 'male'
        elif 'female' in potential_genders:
            gender = 'female'
        else:
            gender = 'unknown'

        if gender == 'female':
            stereotypical_female_professions = ['nurse', 'secretary', 'teacher', 'housewife', 'mother']
            if any(prof in info['attributes']['profession'] for prof in stereotypical_female_professions):
                stereotypes[char_lemma] = stereotypes.get(char_lemma, []) + ['Stereotypical Female Profession']

            passive_verbs = ['wait', 'receive', 'listen', 'support', 'obey', 'cry', 'suffer', 'wish']
            if any(verb.lower() in info['attributes']['agency'] for verb in passive_verbs):
                 stereotypes[char_lemma] = stereotypes.get(char_lemma, []) + ['Potential Passive Agency']

            caregiver_relationships = ['mother', 'wife', 'sister', 'daughter', 'nurse']
            if any(rel in info['attributes']['relationship'] for rel in caregiver_relationships) and not any(info['attributes']['agency']):
                 stereotypes[char_lemma] = stereotypes.get(char_lemma, []) + ['Potential Primary Caregiver Role']

        elif gender == 'male':
            stereotypical_male_professions = ['engineer', 'police', 'businessman', 'manager', 'detective', 'criminal', 'soldier', 'driver', 'boss']
            if any(prof in info['attributes']['profession'] for prof in stereotypical_male_professions):
                stereotypes[char_lemma] = stereotypes.get(char_lemma, []) + ['Stereotypical Male Profession']

            active_verbs = ['fight', 'lead', 'save', 'kill', 'plan', 'decide', 'investigate', 'pursue', 'build', 'destroy']
            if any(verb.lower() in info['attributes']['agency'] for verb in active_verbs):
                 stereotypes[char_lemma] = stereotypes.get(char_lemma, []) + ['Potential Active/Heroic Agency']

            authority_relationships = ['father', 'husband', 'boss', 'leader', 'inspector', 'manager']
            if any(rel in info['attributes']['relationship'] for rel in authority_relationships) and any(info['attributes']['agency']):
                 stereotypes[char_lemma] = stereotypes.get(char_lemma, []) + ['Potential Authority/Provider Role']

        if 'beautiful' in info['attributes']['appearance'] and gender == 'female':
             stereotypes[char_lemma] = stereotypes.get(char_lemma, []) + ['Potential Appearance Stereotype (Female)']
        if 'handsome' in info['attributes']['appearance'] and gender == 'male':
             stereotypes[char_lemma] = stereotypes.get(char_lemma, []) + ['Potential Appearance Stereotype (Male)']
        if 'strong' in info['attributes']['appearance'] and gender == 'male':
             stereotypes[char_lemma] = stereotypes.get(char_lemma, []) + ['Potential Appearance Stereotype (Strong Male)']
        if 'weak' in info['attributes']['appearance'] and gender == 'female':
             stereotypes[char_lemma] = stereotypes.get(char_lemma, []) + ['Potential Appearance Stereotype (Weak Female)']

        if char_lemma in stereotypes:
            stereotypes[char_lemma] = list(set(stereotypes[char_lemma]))
            if not stereotypes[char_lemma]:
                 del stereotypes[char_lemma]

    return stereotypes

# Define a scoring mechanism for different types of stereotypes
stereotype_scores = {
    'Stereotypical Female Profession': 2,
    'Potential Passive Agency': 3,
    'Potential Primary Caregiver Role': 2,
    'Stereotypical Male Profession': 1,
    'Potential Active/Heroic Agency': 1,
    'Potential Authority/Provider Role': 1,
    'Potential Appearance Stereotype (Female)': 1,
    'Potential Appearance Stereotype (Male)': 0.5,
    'Potential Appearance Stereotype (Strong Male)': 0.5,
    'Potential Appearance Stereotype (Weak Female)': 1,
}

def calculate_textual_bias_score(stereotypes_dict):
    """
    Calculates a character-level textual bias score by summing stereotype scores.

    Args:
        stereotypes_dict (dict): A dictionary where keys are character lemmas and
                                 values are lists of identified stereotype categories.

    Returns:
        float: The total bias score for the text excerpt.
    """
    row_character_bias = 0
    if isinstance(stereotypes_dict, dict):
         for char, stereotype_list in stereotypes_dict.items():
            char_score = 0
            if isinstance(stereotype_list, list):
                for stereotype in stereotype_list:
                    if isinstance(stereotype, str):
                         char_score += stereotype_scores.get(stereotype, 0)
            row_character_bias += char_score
    return row_character_bias

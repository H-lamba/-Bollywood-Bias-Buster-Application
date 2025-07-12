# Bollywood Bias Buster

_Bollywood Bias Buster_ is an AI-powered application that analyzes Bollywood movie posters and plot synopses to detect, quantify, and remediate gender bias using a combination of multimodal AI models and NLP techniques.

---

## 🚀 AI Strategy

The pipeline integrates both image and text analysis:

- **Image Analysis:**  
  - Uses Google Gemini Vision to analyze movie posters.
  - Extracts structured information about each character (gender, clothing, pose, associated objects) and identifies visual stereotypes.
  - Results are parsed and scored using a custom stereotype dictionary.

- **Text Analysis:**  
  - Uses spaCy for entity recognition and dependency parsing on plot synopses.
  - Extracts character mentions, professions, agency (verbs), relationships, and appearance descriptors.
  - Categorizes these attributes into stereotype classes using rule-based logic.
  - Prompts Gemini LLM for remediation suggestions on biased text excerpts.

- **Aggregation & Visualization:**  
  - Bias scores are aggregated at the character, film, and decade levels.
  - Results are visualized using bar plots for both films and decades.
  - Automated PDF reports are generated for individual movies, summarizing findings and remediation suggestions.

---

## 🏷️ Bias Taxonomy

Bias is detected and scored using a domain-specific taxonomy:

- **Textual Stereotypes:**
  - *Professions:* E.g., women as nurses/teachers, men as police/businessmen.
  - *Agency:* Passive verbs for women (e.g., "waits", "cries"), active/heroic verbs for men (e.g., "fights", "leads").
  - *Relationships:* Caregiver roles for women, authority/provider roles for men.
  - *Appearance:* Emphasis on beauty for women, strength for men.

- **Visual Stereotypes:**
  - *Poses:* Passive/demure for women, dominant/assertive for men.
  - *Clothing:* Traditional attire reinforcing gender roles.
  - *Composition:* Central/dominant placement of male characters.
  - *Objects/Symbols:* Gendered props or visual cues.

Each stereotype is assigned a bias score reflecting its severity or prevalence.

---

## ✅ Validation Approach

To ensure reliability and interpretability:

- **Manual Review:**  
  - Random samples of Gemini outputs (both image and text) are manually checked for accuracy and relevance.
  - Stereotype detection is cross-checked against the taxonomy for coverage and correct mapping.

- **Consistency Checks:**  
  - Aggregated scores are compared across films and decades to ensure trends are plausible and not artifacts of the pipeline.

- **Remediation Validation:**  
  - LLM-generated remediation suggestions are reviewed for clarity and narrative preservation.

- **Reporting:**  
  - Automated PDF reports are generated for selected films, summarizing both findings and actionable suggestions.

---

## 📂 Project Structure

```

├── Bollywood-bias-analyze/
│   ├── Notebook Practical 2/
│   │   └── text(1).ipynb
|   │   └── poster and trailers.ipynb.ipynb
|   │   └── wiki.ipynb
├── README.md
```

---

## 📖 More Information

For full implementation details, see the [Implementation Notebook](Notebook%20Practical/Implementation.ipynb).

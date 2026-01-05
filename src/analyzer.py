import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import time

# Load environment variables
load_dotenv()

# def analyze_article(text):
#     """
#     MOCK VERSION: Returns static data to avoid API usage during testing.
#     """
#     time.sleep(1.5) # Simulate API latency
#     return {
#         "ideological_dimensions": {
#             "National Positioning": "Strongly Pro-Moroccan",
#             "Diplomatic Framing": "Pro–EU Strategic Alignment",
#             "Conflict Framing": "Delegitimizing Opposing Claims"
#         },
#         "narrative_alignment": [
#             "Aligns with official Moroccan government positions",
#             "Reinforces EU–Morocco strategic partnership framing"
#         ],
#         "objectivity_level": {
#             "assessment": "Low",
#             "range": "21 – 40",
#             "confidence": "High",
#             "definitions": "Frequent subjective framing; facts are present but subordinated"
#         },
#         "score": 30.0,
#         "score_explanation": "Wf (188) / (Wf (188) + Weighted Ws (170.0)) * 100",
#         "subjective_claims": {
#             "Pre-emptive Delegitimization": [
#                 {"severity": "Severe", "quote": "بلا أفق (Dead end)", "analysis": "Dismisses legal action before adjudication"},
#                 {"severity": "Moderate", "quote": "مجرد تحركات رمزية... (Just symbolic moves)", "analysis": "Minimizes legal significance"}
#             ],
#             "Adversarial Framing": [
#                 {"severity": "Moderate", "quote": "سياسة التشويش (Policy of obfuscation)", "analysis": "Frames actions as disruption"},
#                 {"severity": "Moderate", "quote": "إفشال الضغوط... (Thwarting pressures)", "analysis": "Casts diplomacy as hostile pressure"}
#             ],
#             "Identity Labeling": [
#                 {"severity": "Moderate", "quote": "جبهة البوليساريو الانفصالية (The separatist Polisario Front)", "analysis": "Consistent pejorative labeling"}
#             ],
#             "Emotive Intensification": [
#                 {"severity": "Mild", "quote": "ضربة قاسية (Severe blow)", "analysis": "Emotional amplification"}
#             ],
#             "Positive Self-Framing": [
#                 {"severity": "Mild", "quote": "شريكا موثوقا واستثنائيا (A reliable and exceptional partner)", "analysis": "Adulatory language"}
#             ]
#         },
#         "claims": [
#             "Peter Medawar won the Nobel Prize in Medicine in 1960 for his research on acquired immunological tolerance.",
#             "Anwar Sadat was the first Arab leader to sign a peace treaty with Israel after the 1973 war.",
#             "Naguib Mahfouz was the first Arab writer to receive the Nobel Prize in Literature in 1988.",
#             "Elias James Corey won the Nobel Prize in Chemistry in 1990 for developing methodologies for organic synthesis.",
#             "Yasser Arafat won the Nobel Peace Prize in 1994 shared with Yitzhak Rabin and Shimon Peres.",
#             "Ahmed Zewail won the Nobel Prize in Chemistry in 1999 for his invention of femtochemistry.",
#             "Mohamed ElBaradei won the Nobel Peace Prize in 2005 for his work with the IAEA.",
#             "Tawakkol Karman was the first Arab woman to win the Nobel Peace Prize in 2011.",
#             "The Tunisian National Dialogue Quartet won the Nobel Peace Prize in 2015 for facilitating a peaceful democratic transition.",
#             "Abdulrazak Gurnah won the Nobel Prize in Literature in 2021.",
#             "Moungi Bawendi won the Nobel Prize in Chemistry in 2023 for the development of quantum dots.",
#             "Omar Yaghi is cited in the text as a 2025 Nobel Prize winner in Chemistry."
#         ],
#         "notable_omissions": [
#             "Legal basis of the ECJ rulings involving distinct territory status",
#             "Perspectives from international legal observers or UN resolutions",
#             "Direct citations from the opposing party's official statements regarding the specific appeal"
#         ]
#     }

def analyze_article(text):
    """
    Analyzes the article text using Gemini API for political orientation, 
    group alignment, and objectivity score.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
        
    client = genai.Client(api_key=api_key)
    
    prompt = f"""
    You are a professional Media Bias Analyst / Public Editor / Fact-Checker.

    Goal: Analyze the following article text and provide a structured assessment.
    
    1. Ideological Dimensions: Provide a structured object with the following keys:
       - "National Positioning": (e.g. Strongly Pro-Government, Critical, Neutral)
       - "Diplomatic Framing": (e.g. Pro-Western, Globalist, Sovereignist)
       - "Conflict Framing": (e.g. Delegitimizing Opponent, Peace-Seeking, Aggressive)
    2. Narrative Alignment: List specific narratives, official positions, or strategic frameworks the article reinforces or aligns with. Limit to the top 2-3 main narratives.
    3. Subjective Claims: Analyze the text for subjective or biased language. Group your findings by "Rhetorical Technique" (e.g., Pre-emptive Delegitimization, Adversarial Framing, Identity Labeling, Emotive Intensification, etc.). For each technique, provide a list of claims containing:
       - "severity": "Mild", "Moderate", or "Severe"
       - "quote": The specific text
       - "analysis": Brief explanation of why it is biased
    4. Counterfactual Context & Notable Omissions: List 2-3 specific elements, perspectives, or legal/historical facts that are commonly present in neutral or multi-perspective coverage of this topic but are missing, underrepresented, or dismissed in this article.
    5. Factual Claims: Identify key factual claims made in the article.

    Text content:
    {text[:30000]}  # Truncate to avoid context limit issues if extremely long
    
    Provide the output in valid JSON format with keys: "ideological_dimensions", "narrative_alignment", "subjective_claims", "notable_omissions", "claims".
    "subjective_claims" should be a dictionary where keys are technique names and values are lists of objects.
    "claims" should be a list of strings.
    "narrative_alignment" should be a list of strings.
    "notable_omissions" should be a list of strings.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-3-flash-preview', 
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type='application/json'
            )
        )
        
        # Access text directly
        result_text = response.text
        
        # --- Log Raw Response ---
        try:
            log_dir = os.path.join(os.getcwd(), "raw_responses")
            os.makedirs(log_dir, exist_ok=True)
            timestamp = int(time.time())
            log_file = os.path.join(log_dir, f"response_{timestamp}.json")
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(result_text)
            print(f"Raw response saved to: {log_file}")
        except Exception as e:
            print(f"Warning: Failed to log raw response: {e}")

        data = json.loads(result_text.strip())
        
        # --- Python-based Score Calculation ---
        factual_claims = data.get('claims', [])
        # subjective_claims is now a dict {Technique: [List of claim objects]}
        subjective_claims_data = data.get('subjective_claims', {})
        
        # Calculate Wf (Word count of Factual Claims)
        wf = sum(len(claim.split()) for claim in factual_claims)
        
        # Calculate Ws * I (Weighted Subjective Word Count)
        # Intensity: Mild=1.0, Moderate=1.5, Severe=2.0
        ws_weighted_sum = 0
        
        # Flatten the dictionary to iterate over all claims for scoring
        all_subjective_items = []
        if isinstance(subjective_claims_data, dict):
            for technique, items in subjective_claims_data.items():
                if isinstance(items, list):
                    all_subjective_items.extend(items)
        elif isinstance(subjective_claims_data, list):
             # Fallback if model returns list
            all_subjective_items = subjective_claims_data

        for claim_obj in all_subjective_items:
            # Handle both object (new) and string (fallback) formats
            if isinstance(claim_obj, dict):
                content = claim_obj.get('quote', '')
                severity = claim_obj.get('severity', 'Mild').lower()
                
                if 'severe' in severity:
                    intensity = 2.0
                elif 'moderate' in severity:
                    intensity = 1.5
                else:
                    intensity = 1.0
            else:
                # Fallback for string format
                content = str(claim_obj)
                intensity = 1.0 # Default
                lower_claim = content.lower()
                if '[severe]' in lower_claim:
                    intensity = 2.0
                    content = content.replace('[Severe]', '').replace('[severe]', '')
                elif '[moderate]' in lower_claim:
                    intensity = 1.5
                    content = content.replace('[Moderate]', '').replace('[moderate]', '')
                elif '[mild]' in lower_claim:
                    intensity = 1.0
                    content = content.replace('[Mild]', '').replace('[mild]', '')
                
            word_count = len(content.split())
            ws_weighted_sum += (word_count * intensity)
            
        # Formula: (Wf / (Wf + (Ws * I))) * 100
        # Denominator check
        denominator = wf + ws_weighted_sum
        
        if denominator == 0:
            final_score = 0.0 # Edge case: empty text
        else:
            final_score = (wf / denominator) * 100
            
        data['score'] = round(final_score, 1)
        data['score_explanation'] = f"Wf ({wf}) / (Wf ({wf}) + Weighted Ws ({ws_weighted_sum})) * 100"
        
        # --- Bucket Logic for Levels ---
        s = final_score
        if s <= 20:
            assessment = "Very Low"
            r_range = "0 – 20"
            definition = "Dominated by rhetoric, emotive framing, and evaluative language"
        elif s <= 40:
            assessment = "Low"
            r_range = "21 – 40"
            definition = "Frequent subjective framing; facts are present but subordinated"
        elif s <= 60:
            assessment = "Moderate"
            r_range = "41 – 60"
            definition = "Mix of factual reporting and interpretative language"
        elif s <= 80:
            assessment = "High"
            r_range = "61 – 80"
            definition = "Largely factual with limited rhetorical framing"
        else:
            assessment = "Very High"
            r_range = "81 – 100"
            definition = "Primarily descriptive; minimal evaluative or emotive language"
            
        # --- Confidence Heuristic ---
        # Based on length of analyzed text (approximation) and presence of extracted signals
        text_len = len(text)
        claims_count = len(factual_claims) + len(all_subjective_items)
        
        if text_len > 1500 and claims_count > 5:
            confidence = "High"
        elif text_len < 500 or claims_count < 3:
            confidence = "Low"
        else:
            confidence = "Medium"
            
        data['objectivity_level'] = {
            "assessment": assessment,
            "range": r_range,
            "confidence": confidence,
            "definitions": definition
        }
        
        return data
        
    except Exception as e:
        raise Exception(f"Analysis failed: {str(e)}")

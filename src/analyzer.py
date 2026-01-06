import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import time

# Load environment variables
load_dotenv()

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
        
        # --- Score Calculation ---
        factual_claims = data.get('claims', [])
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
            all_subjective_items = subjective_claims_data

        for claim_obj in all_subjective_items:
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

import os
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class MultiAgentAnalyzer:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-3-flash-preview' # Using Flash for speed in multi-step

    def _call_model(self, prompt, response_schema=None):
        """Helper to call Gemini with JSON enforcement."""
        config = types.GenerateContentConfig(
            response_mime_type='application/json'
        )
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=config
            )
            return json.loads(response.text.strip())
        except Exception as e:
            print(f"Model call failed: {e}")
            return {}

    def step_1_analyze_content(self, text):
        """
        Role: The Reader (Objective Extraction)
        Goal: Extract what is physically in the text without judging it.
        """
        print("   [Step 1/4] Analyzing Content...")
        prompt = f"""
        Role: Objective Content Extractor.
        Task: Read the text and extract verifiable data points. Do NOT evaluate bias.

        Text: {text[:30000]}

        Output JSON with keys:
        - "main_topic": String (The core subject).
        - "key_entities": List of strings (People, Org, Countries involved).
        - "factual_claims": List of strings (Specific assertions made).
        - "narrative_arc": String (The story being told).
        - "tone_keywords": List of strings (Adjectives/Verbs used most frequently).
        """
        return self._call_model(prompt)

    def step_2_get_context(self, analysis_data):
        """
        Role: The Historian (Blind Context Retrieval)
        Goal: Retrieve standard perspectives on the topic WITHOUT seeing the article.
        """
        print("   [Step 2/4] Retrieving Context...")
        topic = analysis_data.get("main_topic", "General News")
        entities = analysis_data.get("key_entities", [])
        
        prompt = f"""
        Role: Political Historian / Context Specialist.
        Task: List the standard, multi-perspective viewpoints associated with this topic.
        
        Topic: {topic}
        Key Entities: {', '.join(entities[:5])}

        Output JSON with keys:
        - "standard_viewpoints": List of strings (The 3-4 major stances on this issue, e.g., Pro-Gov, Opposition, International Law).
        - "key_historical_facts": List of strings (Indisputable facts usually cited in comprehensive coverage).
        - "controversies": List of strings (The usual points of contention).
        """
        return self._call_model(prompt)

    def step_3_compare(self, analysis_data, context_data):
        """
        Role: The Judge (Gap Analysis)
        Goal: Compare what was specific in the article vs what exists in the world.
        """
        print("   [Step 3/4] Comparing & Detecting Bias...")
        prompt = f"""
        Role: Bias Comparator.
        Task: Compare the Article's content against the Standard Context to find Omissions and Framing.

        Article Narrative: {analysis_data.get('narrative_arc')}
        Article Claims: {json.dumps(analysis_data.get('factual_claims', []))}

        Standard Context: {json.dumps(context_data.get('standard_viewpoints', []))}
        Standard Facts: {json.dumps(context_data.get('key_historical_facts', []))}

        Output JSON with keys:
        - "omissions": List of strings (Important context or facts from Standard Context NOT present in Article).
        - "framing_bias": List of strings (How the article's narrative deviates from a neutral stance).
        - "ideological_stance": Object {{ "National": "...", "Diplomatic": "...", "Conflict": "..." }}
        """
        return self._call_model(prompt)

    def step_4_synthesize(self, analysis_data, context_data, comparison_data, original_text):
        """
        Role: The Narrator (Final Formatting)
        Goal: Format the data into the structure required by the frontend.
        """
        print("   [Step 4/4] Synthesizing Final Report...")
        prompt = f"""
        Role: Final Report Editor.
        Task: Synthesize the findings into a structured report for the UI.

        Analysis: {json.dumps(analysis_data)}
        Comparison: {json.dumps(comparison_data)}
        Context: {json.dumps(context_data)}
        Original Text Snippet: {original_text[:1000]}

        Requirements:
        1. "subjective_claims": Group by Rhetorical Technique (e.g., "Adversarial Framing", "Emotive Intensification"). keys=Technique, value=List of {{severity, quote, analysis}}.
        2. "objectivity_level": Assess based on the ratio of Omissions/Framing.

        Output JSON with keys:
        - "ideological_dimensions": (From Comparison)
        - "narrative_alignment": List of strings (The specific narrative the article pushes).
        - "subjective_claims": Dictionary (Technique -> List of objects).
        - "notable_omissions": List of strings (From Comparison 'omissions').
        - "claims": List of strings (From Analysis 'factual_claims').
        - "score": Float (0-100, where 100 is neutral/complete).
        - "score_explanation": String (Why this score?).
        - "objectivity_level": {{ "assessment": "...", "range": "...", "confidence": "...", "definitions": "..." }}
        """
        return self._call_model(prompt)

    def run(self, text):
        # 1. Analyze
        s1 = self.step_1_analyze_content(text)
        print("   [Rate Limit] Pausing 60s to respect free tier quota...")
        time.sleep(60)
        
        # 2. Context
        s2 = self.step_2_get_context(s1)
        print("   [Rate Limit] Pausing 60s to respect free tier quota...")
        time.sleep(60)
        
        # 3. Compare
        s3 = self.step_3_compare(s1, s2)
        print("   [Rate Limit] Pausing 60s to respect free tier quota...")
        time.sleep(60)
        
        # 4. Synthesize
        final_output = self.step_4_synthesize(s1, s2, s3, text)
        
        # Save raw traces for debugging
        self._log_trace(s1, s2, s3, final_output)
        
        return final_output

    def _log_trace(self, s1, s2, s3, final):
        try:
            log_dir = os.path.join(os.getcwd(), "raw_responses")
            os.makedirs(log_dir, exist_ok=True)
            timestamp = int(time.time())
            trace = {
                "1_analysis": s1,
                "2_context": s2,
                "3_comparison": s3,
                "4_final": final
            }
            with open(os.path.join(log_dir, f"trace_{timestamp}.json"), "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to log trace: {e}")

# --- Main Entry Point ---

def analyze_article(text):
    """
    Orchestrator function that replaces the old monolithic one.
    """
    # MOCK MODE TOGGLE (Set to True to skip API during dev repetitions if needed)
    # USE_MOCK = True 
    
    # if USE_MOCK:
    #     # Returns the previous static mock data
    #     time.sleep(1.0)
    #     return get_mock_data()
        
    try:
        agent = MultiAgentAnalyzer()
        return agent.run(text)
    except Exception as e:
        print(f"Analysis Error: {e}")
        # Fallback to mock if API fails? Or re-raise?
        # For now, let's re-raise to see the error.
        raise e


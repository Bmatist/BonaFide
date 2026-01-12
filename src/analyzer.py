import os
import json
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tavily import TavilyClient

# Load environment variables
load_dotenv()

class MultiAgentAnalyzer:
    def __init__(self):
        gemini_key = os.getenv("GEMINI_API_KEY")
        tavily_key = os.getenv("TAVILY_API_KEY")
        if not gemini_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables.")
        if not tavily_key:
             print("Warning: TAVILY_API_KEY not found. RAG will be disabled.")
        
        self.client = genai.Client(api_key=gemini_key)
        self.tavily = TavilyClient(api_key=tavily_key) if tavily_key else None
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

    def _search_tavily(self, query):
        """Perform search to get real-world context snippets."""
        print(f"   [Step 1.5] Searching for context: {query}...")
        try:
            # Get search results with snippets
            results = self.tavily.search(query=query, search_depth="advanced", max_results=5)
            snippets = []
            for res in results.get('results', []):
                snippets.append(f"Source: {res['url']}\nContent: {res['content']}\n")
            return {
                "snippets": "\n".join(snippets),
                "raw": results
            }
        except Exception as e:
            print(f"Tavily search failed: {e}")
            return {"snippets": "Search failed.", "raw": None}

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

    def step_2_get_context(self, analysis_data, search_results):
        """
        Role: The Historian (Knowledge Retrieval)
        Goal: Retrieve standard perspectives grounded in real-world data.
        """
        print("   [Step 2/4] Retrieving Context (RAG)...")
        topic = analysis_data.get("main_topic", "General News")
        entities = analysis_data.get("key_entities", [])
        
        prompt = f"""
        Role: Political Historian / Context Specialist.
        Task: Based on the provided search results and your knowledge, list the standard viewpoints on this topic.
        
        Topic: {topic}
        Key Entities: {', '.join(entities[:5])}

        Retrieved Context Snippets:
        {search_results}

        Output JSON with keys:
        - "standard_viewpoints": List of strings (The major stances, grounded in the search snippets).
        - "key_historical_facts": List of strings (Facts confirmed by the snippets or standard knowledge).
        - "controversies": List of strings (The points of friction identified).
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

    def _get_objectivity_level(self, score):
        """Maps a numeric score to its textual bucket."""
        s = float(score)
        if s <= 20:
            return {
                "assessment": "Very Low",
                "range": "0 – 20",
                "definitions": "Dominated by rhetoric, emotive framing, and evaluative language"
            }
        elif s <= 40:
            return {
                "assessment": "Low",
                "range": "21 – 40",
                "definitions": "Frequent subjective framing; facts are present but subordinated"
            }
        elif s <= 60:
            return {
                "assessment": "Moderate",
                "range": "41 – 60",
                "definitions": "Mix of factual reporting and interpretative language"
            }
        elif s <= 80:
            return {
                "assessment": "High",
                "range": "61 – 80",
                "definitions": "Largely factual with limited rhetorical framing"
            }
        else:
            return {
                "assessment": "Very High",
                "range": "81 – 100",
                "definitions": "Primarily descriptive; minimal evaluative or emotive language"
            }

    def run(self, text):
        # 1. Analyze
        s1 = self.step_1_analyze_content(text)
        time.sleep(1)
        
        # 1.5 Search (RAG)
        # Search for the main topic and entities
        search_query = f"{s1.get('main_topic', '')} perspective controversy"
        search_data = self._search_tavily(search_query)
        s1_5_snippets = search_data["snippets"]
        s1_5_raw = search_data["raw"]

        # 2. Context
        s2 = self.step_2_get_context(s1, s1_5_snippets)
        time.sleep(1)
        
        # 3. Compare
        s3 = self.step_3_compare(s1, s2)
        time.sleep(1)
        
        # 4. Synthesize
        final_output = self.step_4_synthesize(s1, s2, s3, text)
        
        # Ensure consistency regardless of model's internal logic
        score = final_output.get('score', 50.0)
        level_data = self._get_objectivity_level(score)
        
        # Preserve confidence from model, but override labels
        model_level = final_output.get('objectivity_level', {})
        level_data['confidence'] = model_level.get('confidence', 'Medium')
        final_output['objectivity_level'] = level_data

        # Save raw traces for debugging
        self._log_trace(s1, s1_5_raw, s2, s3, final_output)
        
        return final_output

    def _log_trace(self, s1, s1_5, s2, s3, final):
        try:
            log_dir = os.path.join(os.getcwd(), "raw_responses")
            os.makedirs(log_dir, exist_ok=True)
            timestamp = int(time.time())
            trace = {
                "1_analysis": s1,
                "1_5_search": s1_5,
                "2_context": s2,
                "3_comparison": s3,
                "4_final": final
            }
            with open(os.path.join(log_dir, f"trace_{timestamp}.json"), "w", encoding="utf-8") as f:
                json.dump(trace, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Failed to log trace: {e}")


def analyze_article(text):
    """
    Orchestrator function that replaces the old monolithic one.
    """
    # USE_MOCK = True
    
    # if USE_MOCK:
    #     time.sleep(1.0)
    #     return get_mock_data()

    try:
        agent = MultiAgentAnalyzer()
        return agent.run(text)
    except Exception as e:
        print(f"Analysis Error: {e}")
        raise e


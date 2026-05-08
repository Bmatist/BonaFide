import asyncio
import os
import json
import time
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv
from tavily import TavilyClient
from mock_data import get_mock_data

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
        self.model = 'gemini-flash-latest' 
        self.pro_model = 'gemini-pro-latest' 

    async def _call_model_async(self, prompt, response_schema=None, use_pro=False, max_retries=3):
        """Helper to call Gemini with JSON enforcement and retry logic."""
        config = types.GenerateContentConfig(
            response_mime_type='application/json'
        )
        target_model = self.pro_model if use_pro else self.model
        
        for attempt in range(max_retries):
            try:
                response = await self.client.aio.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=config
                )
                return json.loads(response.text.strip())
            except Exception as e:
                error_msg = str(e)
                print(f"Model call failed (Attempt {attempt+1}/{max_retries}): {error_msg}")
                if attempt < max_retries - 1:
                    # Exponential backoff only if error
                    sleep_time = 5 * (attempt + 1)
                    if "retry in" in error_msg:
                        match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_msg)
                        if match:
                            sleep_time = float(match.group(1)) + 1
                    print(f"Retrying in {sleep_time:.1f} seconds...")
                    await asyncio.sleep(sleep_time)
                else:
                    # Final attempt fallback: if Pro fails, try Flash as a last resort
                    if use_pro:
                        print("   [Fallback] Pro failed, attempting with Flash...")
                        return await self._call_model_async(prompt, response_schema, use_pro=False, max_retries=1)
                    return {}

    async def _search_tavily_async(self, query):
        """Perform search to get real-world context snippets."""
        print(f"   [Step 1.5] Searching for context: {query}...")
        try:
            # Wrap synchronous Tavily call in a thread
            results = await asyncio.to_thread(
                self.tavily.search, query=query, search_depth="advanced", max_results=5
            )
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

    async def step_1_analyze_content(self, text):
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
    - "article_metadata": {{
        "genre": String (News Report, Opinion, Editorial, Interview, Academic Analysis, or Feature),
        "expected_neutrality": String (High, Medium, or Low - e.g. High for News, Low for Op-Eds)
      }},
    - "key_entities": List of strings (People, Org, Countries involved).
    - "factual_claims": List of strings (Specific assertions made).
    - "narrative_arc": String (The story being told).
    - "tone_keywords": List of strings (Adjectives/Verbs used most frequently).
    """
        return await self._call_model_async(prompt)

    async def step_2_get_context(self, text, search_results):
        """
        Role: The Researcher (RAG Context)
        """
        print("   [Step 2/4] Retrieving Global Context...")
        prompt = f"""
        Role: Neutral Context Researcher.
        Task: Provide missing context for the article based on external search results.
        
        Article Summary/Topic: {text[:2000]}
        
        Search Results (Context):
        {json.dumps(search_results, indent=2)}

        Requirement: Identify critical facts, events, or perspectives NOT in the article.
        For each point, identify the 'source_url' from the search results provided.

        Output JSON with keys:
        - "broader_context": String (Historical/Geopolitical background).
        - "competing_narratives": List of strings (Alternative ways this story is told).
        - "external_facts": List of {{'fact': string, 'source_url': string}} (Specific data points found).
        """
        return await self._call_model_async(prompt)

    async def step_3_compare(self, analysis, context):
        """
        Role: The Fact-Checker (Bias by Omission)
        """
        print("   [Step 3/4] Comparing Content vs Context...")
        prompt = f"""
        Role: Comparative Analyst.
        Task: Identify 'Bias by Omission' by comparing what was reported vs what exists in context.

        Internal Reporting: {json.dumps(analysis, indent=2)}
        External Context: {json.dumps(context, indent=2)}

        Requirement: 
        1. Be specific about what was LEFT OUT and provide the source_url for each omission.
        2. Verify the 'factual_claims' from Internal Reporting against the External Context. Assign a status: "Verified", "Disputed", "Single Source", or "Unverified".

        3. Perform Narrative Fingerprinting:
           - First, identify the geopolitical Region/Context (e.g. MENA, Latin America, US Domestic).
           - Identify the 'Editorial Ecosystem' this text most closely resembles within that region. 
           - **CRITICAL**: In 'closest_match', map the text to EXACTLY ONE of these archetypes and include specific media examples:
             * **Western-Liberal / Centrist** (e.g. NYT, BBC, The Guardian, Le Monde)
             * **State-Aligned / Official** (e.g. Asharq Al-Awsat, Al Arabiya, CCTV, RT, Xinhua)
             * **Pan-Arabist / Regional Network** (e.g. Al Jazeera, Al-Araby Al-Jadeed)
             * **Business-Institutional / Pragmatic** (e.g. Al-Monitor, Reuters, Bloomberg, WSJ)
             * **Populist-Nativist / Partisan** (e.g. Fox News, Daily Mail, Breitbart)
             * **Academic-Institutional / Policy-Heavy** (e.g. Brookings, Chatham House, specialized think-tanks)
           - Identify specific 'shared traits' (vocabulary, framing, omission patterns).

        Output JSON with keys:
    - "editorial_proximity": {{'region': string, 'closest_match': string (Must include specific media names as examples), 'shared_traits': List[string]}}.
    - "omissions": List of {{
        "omission": string, 
        "details": string, 
        "source_url": string,
        "relevance": string (Critical, Important, or Contextual),
        "intentionality": string (Likely, Unclear, or Unlikely),
        "justification": string (One sentence explaining why this omission matters given the article's genre)
      }}.
    - "verified_claims": List of {{'claim': string, 'status': string, 'support': string}}.
    - "framing_bias": List of strings (How the article slants what it DOES include).
    - "ideological_stance": Dictionary (How do they view the conflict/topic?).
    """
        return await self._call_model_async(prompt, use_pro=True)

    async def step_4_synthesize(self, analysis, context, comparison, original_text, target_language="English"):
        """
        Role: The Narrator (Final Report)
        """
        print("   [Step 4/4] Synthesizing Final Report...")
        prompt = f"""
        Role: Senior Analytical Narrator.
        Task: Create a final, polished report of bias.
        
        Input Data for Synthesis:
        1. Initial Analysis: {json.dumps(analysis, indent=2)}
        2. External Context: {json.dumps(context, indent=2)}
        3. Gap Comparison: {json.dumps(comparison, indent=2)}
        
        Reference Text: {original_text[:2000]}

        Requirements:
        1. "subjective_claims": Use the Reference Text to find quotes that support the Framing Bias identified in Comparison. Group by Rhetorical Technique. 
           Each object MUST include:
           - severity: "Mild", "Moderate", or "Severe"
           - quote_original: The verbatim quote in the article's language.
           - quote_translated: English translation.
           - analysis: A brief explanation of why this quote is biased.
        2. "notable_omissions": Merge information from 'External Context' and 'Gap Comparison'. Provide {{'text': string, 'url': string}}.
        3. "claims": Use the 'verified_claims' from Gap Comparison. transform to list of objects including status and support.
        4. "editorial_proximity": Pass through from Comparison step.

        Output JSON with keys:
    - "article_metadata": (From Step 1)
    - "ideological_dimensions": (From Comparison)
    - "narrative_alignment": List of strings (The specific narrative the article pushes).
    - "subjective_claims": Dictionary (Technique -> List of objects with severity, quote_original, quote_translated, and analysis).
    - "notable_omissions": List of objects (text, url, relevance, intentionality, justification).
    - "claims": List of objects (text, confidence, support).
    - "editorial_proximity": {{'region': string, 'closest_match': string (Ensure specific media names are included), 'shared_traits': List[string]}}.
    - "score": Float (Raw 0-100 score).
    - "adjusted_score": Float (0-100 score CALIBRATED for genre).
    - "score_breakdown": {{
        "completeness": int, 
        "neutrality": int, 
        "factuality": int
      }}
    - "score_explanation": String (Brief reasoning for the score).
    - "reader_risk": String (Interpretative consequences).
    - "objectivity_level": {{ "assessment": "...", "range": "...", "confidence": "...", "definitions": "..." }}
    
    IMPORTANT: You must generate the narrative content (explanation, reader_risk, analysis, technique names, and all text descriptions) in {target_language}. 
    HOWEVER, for the specific categorical tags (severity, confidence, relevance, intentionality, and claim status), you MUST use the EXACT English keywords listed in the requirements (e.g., "Severe", "Moderate", "Mild", "Verified", "Disputed", "Unverified", "Critical", "Important", "Contextual", "Likely", "Unlikely"). This is necessary for the UI styling to function.
    """
        return await self._call_model_async(prompt, use_pro=True)

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

    async def run(self, text, url=None, target_language="English"):
        # 1. Analyze
        s1 = await self.step_1_analyze_content(text)
        
        if isinstance(s1, list):
            s1 = s1[0] if s1 else {}
        
        # 1.5 Search (RAG)
        topic = s1.get('main_topic', 'political news')
        search_query = f"{topic} perspective controversy"
        search_data = await self._search_tavily_async(search_query)
        s1_5_snippets = search_data["snippets"]
        s1_5_raw = search_data["raw"]

        # 2. Context
        await asyncio.sleep(1) # Tiny breather
        s2 = await self.step_2_get_context(text, s1_5_snippets)
        
        # 3. Compare
        await asyncio.sleep(1) # Tiny breather
        s3 = await self.step_3_compare(s1, s2)
        
        # 4. Synthesize
        await asyncio.sleep(1) # Tiny breather
        final_output = await self.step_4_synthesize(s1, s2, s3, text, target_language)
        
        if isinstance(final_output, list):
            if len(final_output) > 0:
                final_output = final_output[0]
            else:
                final_output = {}
        
        # Ensure consistency regardless of model's internal logic
        score = final_output.get('score', 50.0)
        level_data = self._get_objectivity_level(score)
        
        # Preserve confidence from model, but override labels
        model_level = final_output.get('objectivity_level', {})
        level_data['confidence'] = model_level.get('confidence', 'Medium')
        final_output['objectivity_level'] = level_data

        # Save raw traces for debugging
        self._log_trace(s1, s1_5_raw, s2, s3, final_output, url)
        
        return final_output

    def _log_trace(self, s1, s1_5, s2, s3, final, url=None):
        try:
            log_dir = os.path.join(os.getcwd(), "raw_responses")
            os.makedirs(log_dir, exist_ok=True)
            timestamp = int(time.time())
            trace = {
                "url": url,
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


async def analyze_article(text, url=None, language="English"):
    """
    Orchestrator function that replaces the old monolithic one.
    """
    USE_MOCK = True
    
    if USE_MOCK:
        await asyncio.sleep(1.0)
        return get_mock_data()

    try:
        agent = MultiAgentAnalyzer()
        return await agent.run(text, url, language)
    except Exception as e:
        print(f"Analysis Error: {e}")
        raise e

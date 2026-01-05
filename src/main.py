import sys
import argparse
from scraper import scrape_article
from analyzer import analyze_article

def main():
    parser = argparse.ArgumentParser(description="Analyze the political bias of an article.")
    parser.add_argument("url", help="The URL of the article to analyze.")
    args = parser.parse_args()
    
    print(f"Fetching article from: {args.url}")
    try:
        text = scrape_article(args.url)
        print("Article fetched successfully. Analyzing...")
        
        analysis = analyze_article(text)
        
        print("\n" + "="*40)
        print(" ANALYSIS RESULTS")
        print("="*40)
        dims = analysis.get('ideological_dimensions')
        if dims and isinstance(dims, dict):
            print("Ideological Dimensions:")
            for k, v in dims.items():
                print(f"  - {k}: {v}")
        else:
            print(f"Political Orientation: {analysis.get('orientation', 'N/A')}")
        narratives = analysis.get('narrative_alignment')
        if narratives and isinstance(narratives, list):
            print("Narrative Alignment:")
            for item in narratives:
                print(f"  - {item}")
        else:
            print(f"Group Alignment:       {analysis.get('alignment', 'N/A')}")
        
        print("\nObjectivity Assessment")
        obj = analysis.get('objectivity_level', {})
        if obj:
            print(f"Assessment:      {obj.get('assessment', 'N/A')}")
            print(f"Estimated Range: {obj.get('range', 'N/A')}")
            print(f"Confidence:      {obj.get('confidence', 'N/A')}")
        else:
            print(f"Objectivity Score:     {analysis.get('score', 'N/A')}/100")
            print(f"Score Calculation:     {analysis.get('score_explanation', 'N/A')}")
        
        if 'notable_omissions' in analysis:
            print("\nCounterfactual Context & Notable Omissions:")
            for omission in analysis['notable_omissions']:
                print(f"- {omission}")
        
        print("\nSubjective Claims (Evidence of Bias):")
        subj_claims = analysis.get('subjective_claims', [])
        if isinstance(subj_claims, dict):
            for technique, items in subj_claims.items():
                print(f"  [{technique}]")
                for item in items:
                    if isinstance(item, dict):
                        severity = item.get('severity', 'Mild')
                        quote = item.get('quote', '')
                        analysis_text = item.get('analysis', '')
                        print(f"    - [{severity}] \"{quote}\" -> {analysis_text}")
                    else:
                        print(f"    - \"{item}\"")
        else:
            for claim in subj_claims:
                print(f"- \"{claim}\"")
            
        print("\nFactual Claims (Extracted):")
        for claim in analysis.get('claims', []):
            print(f"- {claim}")
            
        print("="*40 + "\n")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

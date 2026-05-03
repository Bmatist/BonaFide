from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scraper import scrape_article
from analyzer import analyze_article






app = FastAPI(title="Political Bias Detector")

# Mount static files
app.mount("/static", StaticFiles(directory="src/static"), name="static")

from urllib.parse import urlparse

# Templates
templates = Jinja2Templates(directory="src/templates")

def get_domain(url: str):
    try:
        domain = urlparse(url).netloc
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return "source"

templates.env.filters["domain"] = get_domain

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

from pydantic import AnyHttpUrl, ValidationError

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, url: str = Form(...)):
    try:
        # Validate URL
        try:
            valid_url = AnyHttpUrl(url)
            if valid_url.scheme not in ['http', 'https']:
                raise ValueError("Only http and https schemes are allowed.")
        except (ValidationError, ValueError) as ve:
            return templates.TemplateResponse("partials/error.html", {
                "request": request,
                "error": f"Invalid URL: {str(ve)}"
            })

        # Scrape
        text = scrape_article(url)
        
        # Analyze
        analysis = analyze_article(text, url)
        
        return templates.TemplateResponse("partials/result.html", {
            "request": request, 
            "analysis": analysis,
            "url": url
        })
    except Exception as e:
        import traceback
        error_msg = str(e)
        
        # Map known errors
        if "timeout" in error_msg.lower():
            user_friendly_msg = "We couldn't reach the website in time. It might be down or blocking our request."
            error_title = "Connection Timeout"
        elif "forbidden" in error_msg.lower() or "403" in error_msg:
            user_friendly_msg = "The news source is actively blocking our AI from reading the article (likely a strict paywall)."
            error_title = "Access Blocked"
        elif "token" in error_msg.lower():
            user_friendly_msg = "The article is too long for our AI to process in a single pass."
            error_title = "Context Limit Reached"
        else:
            user_friendly_msg = "An unexpected error occurred while analyzing the text."
            error_title = "Analysis Failed"

        return templates.TemplateResponse("partials/error.html", {
            "request": request,
            "error_title": error_title,
            "error": user_friendly_msg,
            "technical_details": error_msg
        })

@app.post("/compare", response_class=HTMLResponse)
async def compare_narratives(request: Request, url_a: str = Form(...), url_b: str = Form(...)):
    try:
        # Validate URLs
        for u in [url_a, url_b]:
            valid_url = AnyHttpUrl(u)
            if valid_url.scheme not in ['http', 'https']:
                raise ValueError(f"Only http and https schemes are allowed for {u}.")
                
        # In this prototype, we'll skip actual scraping and go straight to the analyzer
        # since the mock analyzer is synchronous and ignores the text.
        text_a = "mock text A"
        text_b = "mock text B"
        
        from src.analyzer import analyze_comparative
        analysis = analyze_comparative(text_a, url_a, text_b, url_b)
        
        return templates.TemplateResponse("partials/compare_result.html", {
            "request": request, 
            "analysis": analysis,
            "url_a": url_a,
            "url_b": url_b
        })
    except Exception as e:
        error_msg = str(e)
        return templates.TemplateResponse("partials/error.html", {
            "request": request,
            "error_title": "Comparison Failed",
            "error": "We couldn't complete the comparative analysis.",
            "technical_details": error_msg
        })

@app.post("/render_pdf_view", response_class=HTMLResponse)
async def render_pdf_view(request: Request):
    try:
        # We expect a JSON payload containing the analysis data and url
        data = await request.json()
        analysis = data.get("analysis", {})
        url = data.get("url", "Unknown Source")
        
        return templates.TemplateResponse("report_pdf.html", {
            "request": request,
            "analysis": analysis,
            "url": url
        })
    except Exception as e:
        return f"Error rendering PDF layout: {str(e)}"

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

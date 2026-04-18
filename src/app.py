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
        error_msg = str(e)
        user_friendly_msg = "An unexpected error occurred while analyzing the article."
        error_title = "Analysis Failed"
        
        if "Failed to scrape" in error_msg or "403" in error_msg or "404" in error_msg:
            error_title = "Couldn't Read Article"
            user_friendly_msg = "We couldn't access this article. The website might be blocking scrapers, or the article might be protected by a strict paywall."
        elif "rate limit" in error_msg.lower() or "429" in error_msg or "quota" in error_msg.lower():
            error_title = "System Overloaded"
            user_friendly_msg = "The AI service is currently experiencing high traffic or rate limits. Please wait a moment and try again."
        elif "too large" in error_msg.lower() or "context length" in error_msg.lower() or "too long" in error_msg.lower():
            error_title = "Article Too Long"
            user_friendly_msg = "This article exceeds the maximum length our AI can process at one time."
            
        return templates.TemplateResponse("partials/error.html", {
            "request": request,
            "error_title": error_title,
            "error": user_friendly_msg,
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

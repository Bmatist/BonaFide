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

# Templates
templates = Jinja2Templates(directory="src/templates")

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze", response_class=HTMLResponse)
async def analyze(request: Request, url: str = Form(...)):
    try:
        # Scrape
        text = scrape_article(url)
        
        # Analyze
        analysis = analyze_article(text)
        
        return templates.TemplateResponse("partials/result.html", {
            "request": request, 
            "analysis": analysis,
            "url": url
        })
    except Exception as e:
        return templates.TemplateResponse("partials/error.html", {
            "request": request,
            "error": str(e)
        })

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

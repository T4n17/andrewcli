from src.tools.google_scraper.scraper import google_search
from src.tools.google_scraper.url_fetcher import fetch_url
from src.tools.google_scraper.ai_chat import AIChat
from src.tools.rss import FetchRSS
from src.core.tool import Tool
import json

class GoogleSearch(Tool):
    name: str = "google_search"
    description: str = "Search Google and return the top organic results plus the AI Overview if present. Args: query: The search query string. num_results: How many results to return (default 10)."
    
    def execute(self, query: str, num_results: int = 10) -> str:
        data = google_search(query, num_results, debug=True, headless=True)
        if not data["results"] and not data["ai_overview"]:
            return json.dumps({
                "error": "No results found",
                "hint": "Check debug_page.html in the google-scraper directory for the raw page Google served.",
            }, indent=2)
        return json.dumps(data, indent=2, ensure_ascii=False)

class FetchPage(Tool):
    name: str = "fetch_page"
    description: str = "Fetch a web page and return its readable text content (dynamic JS content included). Args: url: The full URL to fetch (including https://). timeout: Request timeout in seconds (default 30)."
    
    def execute(self, url: str, timeout: int = 30) -> str:
        data = fetch_url(url, timeout=timeout, headless=True)
        return json.dumps(data, indent=2, ensure_ascii=False)

class AskGoogleAI(Tool):
    name: str = "ask_google_ai"
    description: str = "Send a prompt to Google AI Mode and return its response. Use this tool for hard tasks that require deep reasoning: coding problems, debugging, architecture decisions, mathematical proofs, complex comparisons, ethical dilemmas, research synthesis, or any question where a thorough, well-reasoned answer matters more than a quick lookup. Args: prompt: The question or task to send to Google AI."
    
    def execute(self, prompt: str) -> str:
        with AIChat(headless=True) as session:
            response = session.chat(prompt)

        if not response:
            return json.dumps({
                "error": "No response received from Google AI",
                "prompt": prompt,
                "hint": "Run with --debug to inspect the page HTML.",
            }, indent=2)

        return json.dumps({"prompt": prompt, "response": response}, indent=2, ensure_ascii=False)

class GetLatestNews(Tool):
    name: str = "get_latest_news"
    description: str = "Get the latest news from Google News"

    def execute(self):
        json_news = FetchRSS().execute("https://news.google.com/rss?hl=it&gl=IT&ceid=IT:it", limit=10)
        news = json.loads(json_news)
        for item in news:
            item.pop("link", None)
            item.pop("summary", None)
            item.pop("published", None)
        return json.dumps(news, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    print(GetLatestNews().execute())
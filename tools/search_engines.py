import re
from typing import List
from urllib.parse import parse_qs, unquote, urlparse

import requests
import yt_dlp
from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    ArxivLoader,
    WebBaseLoader,
    WikipediaLoader,
)
from langchain_core.tools import tool
from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright
from youtube_transcript_api import YouTubeTranscriptApi


@tool
def arxiv_search(query: str) -> str:
    """Search Arxiv for a query and return maximum 3 result.

    Args:
        query: The search query."""
    search_docs = ArxivLoader(query=query, load_max_docs=3).load()
    formatted_search_docs = "\n\n---\n\n".join(
        [
            f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content[:1000]}\n</Document>'
            for doc in search_docs
        ]
    )
    return {"arvix_results": formatted_search_docs}


@tool
def wiki_search(query: str) -> str:
    """Search Wikipedia (max 2 results)."""
    docs = WikipediaLoader(query=query, load_max_docs=2).load()
    return "\n\n".join(
        [f"Source: {doc.metadata['source']}\n{doc.page_content}" for doc in docs]
    )


@tool
def web_search(query: str) -> str:
    """Search the web using DuckDuckGo. Return search content and domain.

    Args:
        query: The search query."""
    try:
        domain = None
        web_search = WebSearchTool()

        if "site:" in query:
            query, domain = query.split("site:", 1)

        search_result = web_search.search(query.strip())
        if domain:
            search_result += f"\nDomain filter: {domain.strip()}"
        return search_result
    except Exception as e:
        return f"Web search failed: {str(e)}"


@tool
def web_page_text_extractor(url: str) -> str:
    """
    Extracts and returns plain text content from a given web page URL
    using WebBaseLoader.

    Steps:
      1. Strips whitespace from the URL.
      2. Validates that the URL has a valid HTTP/HTTPS scheme and network location.
      3. Loads the page content via WebBaseLoader.
      4. Returns the concatenated text or a clear error message.

    Args:
        url (str): The URL of the web page to extract content from.

    Returns:
        str: The extracted plain text content, prefixed with the tool name,
             or an error message if validation or loading fails.
    """
    # 1. Clean up the URL
    url_clean = url.strip().lower()

    # 2. Validate URL format
    parsed = urlparse(url_clean)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return f"[Tool: WebBaseLoader ERROR] Invalid URL: {url_clean}"

    try:
        # 3. Load page content
        loader = WebBaseLoader(url_clean)
        documents = loader.load()
        if not documents:
            return f"[Tool: WebBaseLoader] No content found at {url_clean}"

        # 4. Concatenate and return text
        content = "\n\n".join(doc.page_content for doc in documents)
        return f"[Tool: WebBaseLoader]\n{content}"

    except Exception as e:
        return f"[Tool: WebBaseLoader ERROR] Failed to extract content from {url_clean}: {e}"


@tool
def web_form_filler(spec: dict) -> str:
    """
    Fills and submits a web form, then returns the resulting page text.
    This tool is useful for automating form submissions and scraping
    the results.

    Args:
        spec (dict): {
            "url": str,
            "fields": { "<css_selector>": "<value>", ... },
            "submit_selector": str,
            "wait_for_selector"?: str  # optional element to wait for after submit
        }

    Returns:
        str: Plain text of the post-submission page, or an error message.
    """
    url = spec.get("url", "").strip()
    if not re.match(r"^https?://", url):
        return f"[Tool: web_form_filler ERROR] Invalid URL: {url}"

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=15000)
            # Fill each field
            for selector, value in spec.get("fields", {}).items():
                page.fill(selector, value)
            # Submit the form
            page.click(spec["submit_selector"])
            # Optionally wait for a post-submit element
            wait_sel = spec.get("wait_for_selector")
            if wait_sel:
                page.wait_for_selector(wait_sel, timeout=10000)
            # Extract text
            content = page.content()
            # Strip HTML tags for plain text
            text = re.sub(r"<[^>]+>", "", content)
            browser.close()
            return f"[Tool: web_form_filler]\n{text}"
    except PWTimeout as e:
        return f"[Tool: web_form_filler ERROR] Timeout navigating or waiting: {e}"
    except Exception as e:
        return f"[Tool: web_form_filler ERROR] {e}"


@tool
def youtube_video_extractor(url: str) -> str:
    """
    Extracts title, description, and full transcript from a YouTube video URL.

    1. Validates the URL format.
    2. Parses the video ID using parse_qs or a regex fallback.
    3. Uses yt_dlp to fetch title and description.
    4. Uses youtube-transcript-api to fetch the transcript.

    Args:
        url (str): A YouTube video URL (http:// or https://).

    Returns:
        str: "Title: <title>\nDescription: <description>\nTranscript: <transcript>"
             or an error message on failure.
    """
    # 1. URL validation
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return "[Tool: youtube_video_extractor ERROR] Invalid URL format."

    # 2. Extract video ID
    video_id = ""
    # Standard query parameter
    qs = parse_qs(parsed.query)
    if "v" in qs and qs["v"]:
        video_id = qs["v"][0]
    else:
        # Regex fallback for short URLs or embed links
        m = re.search(r"(?:v=|youtu\.be/)([0-9A-Za-z_-]{11})", url)
        if m:
            video_id = m.group(1)

    # 3. Fetch metadata with yt_dlp
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        "no_playlist": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_id, download=False)
            title = info.get("title", "").strip()
            description = info.get("description", "").strip().replace("\n", " ")
    except Exception as e:
        return f"[Tool: youtube_video_extractor ERROR] Metadata fetch failed: {e}"

    # 4. Fetch transcript with youtube-transcript-api
    if video_id:
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            # pick English or first available
            transcript = transcript_list.find_transcript(["en"])
            fetched = transcript.fetch()
            transcript_text = " ".join(item["text"] for item in fetched)
        except Exception as e:
            transcript_text = f"Transcript unavailable: {e}"
    else:
        transcript_text = "Transcript unavailable: Video ID not found."

    return (
        f"Title: {title}\n"
        f"Description: {description}\n"
        f"Transcript: {transcript_text}"
    )


class WebSearchTool:
    """DuckDuckGo web search tool enhanced for AI agents workflows"""

    def __init__(self, max_results: int = 5):
        self.max_results = max_results
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0.0.0"}
        )

    def search(self, query: str) -> str:
        """Main search method that actually works"""
        try:
            html = self._fetch_search_html(query)
            return self._parse_and_format_results(html)
        except Exception as e:
            return f"Search error: {str(e)}"

    def _fetch_search_html(self, query: str) -> str:
        """Get raw search results HTML"""
        response = self.session.get(
            "https://html.duckduckgo.com/html",
            params={"q": query, "kl": "wt-wt"},
            timeout=10,
        )
        response.raise_for_status()
        return response.text

    def _parse_and_format_results(self, html: str) -> str:
        """Direct parsing without unnecessary filtering"""
        soup = BeautifulSoup(html, "html.parser")
        results = []

        for result in soup.find_all("div", class_="result"):
            try:
                link = result.find("a", class_="result__a")
                snippet = result.find("a", class_="result__snippet")

                if not (link and snippet):
                    continue

                raw_url = link.get("href", "")
                url = self._unwrap_ddg_redirect(raw_url)
                domain = urlparse(url).netloc

                results.append(
                    {
                        "title": link.get_text(strip=True),
                        "url": url,
                        "snippet": snippet.get_text(strip=True),
                        "domain": domain,
                    }
                )

                if len(results) >= self.max_results:
                    break

            except Exception:
                continue

        return self._format_output(results)

    def _unwrap_ddg_redirect(self, ddg_url: str) -> str:
        """Handle DuckDuckGo's URL redirection"""
        if ddg_url.startswith("https://duckduckgo.com/l/"):
            try:
                query = parse_qs(urlparse(ddg_url).query)
                return unquote(query["uddg"][0])
            except:
                return ddg_url
        return ddg_url

    def _format_output(self, results: List[dict]) -> str:
        """Clean markdown formatting"""
        if not results:
            return "No results found."

        output = ["## Search Results"]
        for idx, result in enumerate(results, 1):
            output.append(
                f"{idx}. **[{result['title']}]({result['url']})**\n"
                f"*{result['domain']}*\n"
                f"{result['snippet']}\n"
            )

        return "\n".join(output)

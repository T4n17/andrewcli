import json
import re
import xml.etree.ElementTree as ET
from urllib.request import Request, urlopen

from src.core.tool import Tool

# Common namespaces used by Atom and RSS extensions.
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def _strip_html(text: str) -> str:
    if not text:
        return ""
    # Remove tags and collapse whitespace for a compact summary.
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _text(elem, path, ns=None) -> str:
    if elem is None:
        return ""
    found = elem.find(path, ns) if ns else elem.find(path)
    if found is None:
        return ""
    return (found.text or "").strip()


def _atom_link(entry) -> str:
    # Prefer rel="alternate" or the first link without a rel attribute.
    links = entry.findall("atom:link", _NS)
    for link in links:
        rel = link.get("rel")
        if rel in (None, "alternate"):
            href = link.get("href")
            if href:
                return href.strip()
    if links and links[0].get("href"):
        return links[0].get("href").strip()
    return ""


def _parse_rss(root) -> list[dict]:
    items = []
    for item in root.findall(".//item"):
        items.append({
            "title": _text(item, "title"),
            "link": _text(item, "link"),
            "published": _text(item, "pubDate") or _text(item, "dc:date", _NS),
            "summary": _strip_html(
                _text(item, "description") or _text(item, "content:encoded", _NS)
            ),
        })
    return items


def _parse_atom(root) -> list[dict]:
    items = []
    for entry in root.findall("atom:entry", _NS):
        items.append({
            "title": _text(entry, "atom:title", _NS),
            "link": _atom_link(entry),
            "published": _text(entry, "atom:published", _NS)
            or _text(entry, "atom:updated", _NS),
            "summary": _strip_html(
                _text(entry, "atom:summary", _NS) or _text(entry, "atom:content", _NS)
            ),
        })
    return items


class FetchRSS(Tool):
    name: str = "fetch_rss"
    description: str = "Fetch and parse an RSS or Atom feed from a URL. Returns a JSON list of items with title, link, published date, and summary."

    def execute(self, url: str, limit: int = 20, timeout: int = 10) -> str:
        req = Request(url, headers={"User-Agent": "awcli/1.0"})
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()

        try:
            root = ET.fromstring(data)
        except ET.ParseError as e:
            return f"[fetch_rss] Failed to parse feed: {e}"

        tag = root.tag.lower()
        if tag.endswith("rss") or tag.endswith("rdf"):
            items = _parse_rss(root)
        elif tag.endswith("feed"):
            items = _parse_atom(root)
        else:
            # Fallback: try both, RSS first.
            items = _parse_rss(root) or _parse_atom(root)

        if limit and limit > 0:
            items = items[:limit]

        return json.dumps(items, ensure_ascii=False, indent=2)

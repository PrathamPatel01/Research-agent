import argparse
import json
import re
import sys
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from lxml import html


class WebsiteScraperAgent:
    """Simple agent that fetches and extracts structured data from a web pageee."""

    def __init__(
        self,
        timeout: int = 15,
        max_links: int = 50,
        respect_robots_txt: bool = True,
        user_agent: str = "WebsiteScraperAgent/1.0 (+https://google.com)",
    ) -> None:
        self.timeout = timeout
        self.max_links = max_links
        self.respect_robots_txt = respect_robots_txt
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    def scrape(self, url: str) -> dict[str, Any]:
        normalized_url = self._normalize_url(url)
        self._validate_robots_txt(normalized_url)

        response = self.session.get(normalized_url, timeout=self.timeout)
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type:
            raise ValueError(
                f"URL does not appear to be HTML. Content-Type received: {content_type}"
            )

        return self._extract_page_data(normalized_url, response.text)

    def _normalize_url(self, raw_url: str) -> str:
        raw_url = raw_url.strip()
        if not raw_url:
            raise ValueError("Empty URL provided.")

        if not raw_url.startswith(("http://", "https://")):
            raw_url = "https://" + raw_url

        parsed = urlparse(raw_url)
        if not parsed.netloc:
            raise ValueError(f"Invalid URL: {raw_url}")

        return raw_url

    def _validate_robots_txt(self, url: str) -> None:
        if not self.respect_robots_txt:
            return

        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

        robots_parser = RobotFileParser()
        robots_parser.set_url(robots_url)
        robots_parser.read()

        can_fetch = robots_parser.can_fetch(
            self.session.headers.get("User-Agent", "*"), url
        )
        if not can_fetch:
            raise PermissionError(
                f"robots.txt disallows scraping this page: {url}. "
                "Use --ignore-robots if you have permission."
            )

    def _extract_page_data(self, url: str, page_html: str) -> dict[str, Any]:
        tree = html.fromstring(page_html)

        title = self._clean_text(tree.xpath("string(//title)"))
        meta_description = self._clean_text(
            " ".join(tree.xpath("//meta[translate(@name, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')='description']/@content"))
        )

        headings = {
            "h1": self._extract_text_list(tree, "//h1"),
            "h2": self._extract_text_list(tree, "//h2"),
            "h3": self._extract_text_list(tree, "//h3"),
        }

        paragraphs = self._extract_text_list(tree, "//p")
        links = self._extract_links(tree, url)

        body_text = self._clean_text(" ".join(tree.xpath("//body//text()")))
        body_text = re.sub(r"\s+", " ", body_text).strip()

        return {
            "url": url,
            "title": title,
            "meta_description": meta_description,
            "headings": headings,
            "paragraphs": paragraphs,
            "links": links,
            "body_text_preview": body_text[:3000],
            "stats": {
                "paragraph_count": len(paragraphs),
                "link_count": len(links),
                "body_text_chars": len(body_text),
            },
        }

    def _extract_text_list(self, tree: html.HtmlElement, xpath_expr: str) -> list[str]:
        values = []
        for node in tree.xpath(xpath_expr):
            text = self._clean_text(" ".join(node.xpath(".//text()")))
            if text:
                values.append(text)
        return values

    def _extract_links(self, tree: html.HtmlElement, base_url: str) -> list[str]:
        seen = set()
        links = []
        for href in tree.xpath("//a[@href]/@href"):
            absolute = urljoin(base_url, href.strip())
            parsed = urlparse(absolute)
            if parsed.scheme not in {"http", "https"}:
                continue
            absolute = parsed._replace(fragment="").geturl()
            if absolute in seen:
                continue
            seen.add(absolute)
            links.append(absolute)
            if len(links) >= self.max_links:
                break
        return links

    @staticmethod
    def _clean_text(text: str) -> str:
        return re.sub(r"\s+", " ", text or "").strip()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Scrape structured content from a website URL."
    )
    parser.add_argument(
        "url",
        nargs="?",
        help="Website URL to scrape. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="scraped_data.json",
        help="Path to save JSON output (default: scraped_data.json).",
    )
    parser.add_argument(
        "--max-links",
        type=int,
        default=50,
        help="Maximum number of links to extract (default: 50).",
    )
    parser.add_argument(
        "--ignore-robots",
        action="store_true",
        help="Ignore robots.txt checks (only use if you have permission).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    url = args.url or input("Enter website URL: ").strip()
    if not url:
        print("No URL provided.")
        return 1

    agent = WebsiteScraperAgent(
        max_links=args.max_links,
        respect_robots_txt=not args.ignore_robots,
    )

    try:
        data = agent.scrape(url)
    except Exception as exc:
        print(f"Scraping failed: {exc}")
        return 1

    with open(args.output, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)

    print(f"Scraped data saved to: {args.output}")
    print(f"Title: {data.get('title', 'N/A')}")
    print(
        "Stats:"
        f" paragraphs={data['stats']['paragraph_count']},"
        f" links={data['stats']['link_count']},"
        f" body_chars={data['stats']['body_text_chars']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

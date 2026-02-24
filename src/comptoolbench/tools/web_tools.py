"""Web and network tools: HTTP requests, HTML parsing, DNS, URL utilities.

Live mode notes:
- Pure functions (parse_html, extract_links, encode_url, extract_domain,
  generate_url) behave identically in live and simulated modes.
- Network-dependent tools (http_request, check_url_status, dns_lookup,
  rss_feed_parse, ip_geolocation) use deterministic hashing in simulated
  mode and are NOT implemented in live mode (would require real network
  calls outside the benchmark's scope).
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any
from urllib.parse import (
    parse_qs,
    quote,
    unquote,
    urlencode,
    urlparse,
)

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)

# ---------------------------------------------------------------------------
# Simulated data pools
# ---------------------------------------------------------------------------

_HTTP_STATUS_CODES = [200, 200, 200, 200, 301, 302, 403, 404, 500]

_COUNTRIES = [
    "United States", "United Kingdom", "Germany", "Japan", "Canada",
    "France", "Australia", "Brazil", "India", "Netherlands",
    "South Korea", "Singapore", "Sweden", "Switzerland", "Ireland",
]

_CITIES = [
    "San Francisco", "London", "Berlin", "Tokyo", "Toronto",
    "Paris", "Sydney", "Sao Paulo", "Mumbai", "Amsterdam",
    "Seoul", "Singapore", "Stockholm", "Zurich", "Dublin",
]

_RSS_TITLES = [
    "Breaking: Major Update Released",
    "New Study Reveals Surprising Findings",
    "Tech Industry Report Q1 2026",
    "Opinion: The Future of AI Development",
    "Market Analysis: Trends to Watch",
    "Interview with Industry Leader",
    "Product Review: Latest Innovations",
    "Research Paper: Novel Approach Demonstrated",
    "Conference Recap: Key Takeaways",
    "Guide: Best Practices for 2026",
]

_RESPONSE_BODIES = [
    '{"message":"OK","data":{"id":1,"status":"active"}}',
    '{"result":"success","items":[{"name":"item1"},{"name":"item2"}]}',
    '{"status":"healthy","version":"2.1.0","uptime_hours":720}',
    '{"users":150,"active":98,"timestamp":"2026-01-15T10:30:00Z"}',
    '{"products":[{"id":101,"name":"Widget","price":29.99}]}',
]


def _sim_hash(seed: str, *args: Any) -> int:
    """Deterministic hash -> integer for simulated data."""
    raw = json.dumps([seed, *args], sort_keys=True, default=str)
    return int(hashlib.sha256(raw.encode()).hexdigest(), 16)


def _sim_hex(seed: str, *args: Any) -> str:
    """Deterministic hash -> hex string (12 chars)."""
    raw = json.dumps([seed, *args], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 1. http_request
# ---------------------------------------------------------------------------


@register_tool
class HttpRequest(BaseTool):
    """Make an HTTP request (simulated)."""

    name = "http_request"
    schema = ToolSchema(
        name="http_request",
        description=(
            "Make an HTTP request to a URL with a specified method. "
            "Returns the response status code, body, and headers."
        ),
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="url",
                type="string",
                description="The full URL to request (e.g. 'https://api.example.com/data')",
            ),
            ToolParameter(
                name="method",
                type="string",
                description="HTTP method to use",
                enum=["GET", "POST", "PUT", "DELETE"],
            ),
        ],
        returns="HTTP response with status_code, body, and headers",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        # Not implemented in live mode — would require real network calls
        return self.execute_simulated(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        url = kwargs["url"]
        method = kwargs["method"].upper()
        h = _sim_hash("http_request", url, method)

        status_code = _HTTP_STATUS_CODES[h % len(_HTTP_STATUS_CODES)]
        body = _RESPONSE_BODIES[h % len(_RESPONSE_BODIES)]

        return {
            "status_code": status_code,
            "body": body,
            "headers": {
                "content-type": "application/json",
                "x-request-id": _sim_hex("http_req_id", url, method),
                "server": "nginx/1.25.3",
                "content-length": str(len(body)),
            },
        }


# ---------------------------------------------------------------------------
# 2. parse_html
# ---------------------------------------------------------------------------


@register_tool
class ParseHtml(BaseTool):
    """Extract text content from HTML."""

    name = "parse_html"
    schema = ToolSchema(
        name="parse_html",
        description=(
            "Extract readable text content from an HTML string. "
            "Returns the plain text, title (if present), and number of links found."
        ),
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="html",
                type="string",
                description="HTML content to parse",
            ),
        ],
        returns="Extracted text, page title, and link count",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._parse(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._parse(**kwargs)

    def _parse(self, **kwargs: Any) -> dict[str, Any]:
        """Parse HTML using regex (no external dependency needed)."""
        html = kwargs["html"]

        # Extract title
        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        title = title_match.group(1).strip() if title_match else ""

        # Count links
        links = re.findall(r'<a\s[^>]*href\s*=\s*["\']([^"\']+)["\']', html, re.IGNORECASE)

        # Strip HTML tags to get text
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.IGNORECASE | re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        return {
            "text": text,
            "title": title,
            "links_count": len(links),
        }


# ---------------------------------------------------------------------------
# 3. extract_links
# ---------------------------------------------------------------------------


@register_tool
class ExtractLinks(BaseTool):
    """Extract all URLs from text or HTML content."""

    name = "extract_links"
    schema = ToolSchema(
        name="extract_links",
        description=(
            "Extract all URLs from a given text or HTML string. "
            "Returns a list of unique URLs and their count."
        ),
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="content",
                type="string",
                description="Text or HTML content to extract URLs from",
            ),
        ],
        returns="List of extracted URLs and their count",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._extract(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._extract(**kwargs)

    def _extract(self, **kwargs: Any) -> dict[str, Any]:
        """Extract URLs via regex."""
        content = kwargs["content"]

        # Match URLs in href attributes and plain text
        url_pattern = r'https?://[^\s<>"\')\]}>]+'
        href_pattern = r'href\s*=\s*["\']([^"\']+)["\']'

        plain_urls = re.findall(url_pattern, content)
        href_urls = re.findall(href_pattern, content, re.IGNORECASE)

        # Combine and deduplicate, preserving order
        seen: set[str] = set()
        all_urls: list[str] = []
        for url in plain_urls + href_urls:
            cleaned = url.rstrip(".,;:!?")
            if cleaned not in seen and cleaned.startswith(("http://", "https://")):
                seen.add(cleaned)
                all_urls.append(cleaned)

        return {
            "links": all_urls,
            "count": len(all_urls),
        }


# ---------------------------------------------------------------------------
# 4. check_url_status
# ---------------------------------------------------------------------------


@register_tool
class CheckUrlStatus(BaseTool):
    """Check if a URL is reachable (simulated)."""

    name = "check_url_status"
    schema = ToolSchema(
        name="check_url_status",
        description=(
            "Check whether a URL is reachable and return its HTTP status code "
            "and estimated response time."
        ),
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="url",
                type="string",
                description="The URL to check (e.g. 'https://example.com')",
            ),
        ],
        returns="URL status, HTTP status code, and response time in milliseconds",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        # Not implemented in live mode — would require real network calls
        return self.execute_simulated(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        url = kwargs["url"]
        h = _sim_hash("check_url", url)

        status_code = _HTTP_STATUS_CODES[h % len(_HTTP_STATUS_CODES)]
        response_time_ms = 20 + (h % 480)  # 20–499 ms

        if status_code < 400:
            status = "reachable"
        elif status_code < 500:
            status = "client_error"
        else:
            status = "server_error"

        return {
            "url": url,
            "status": status,
            "status_code": status_code,
            "response_time_ms": response_time_ms,
        }


# ---------------------------------------------------------------------------
# 5. dns_lookup
# ---------------------------------------------------------------------------


@register_tool
class DnsLookup(BaseTool):
    """Perform a DNS lookup (simulated)."""

    name = "dns_lookup"
    schema = ToolSchema(
        name="dns_lookup",
        description=(
            "Perform a DNS lookup for a domain and return the resolved "
            "IP address and record type."
        ),
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="domain",
                type="string",
                description="Domain name to look up (e.g. 'example.com')",
            ),
        ],
        returns="Resolved IP address and DNS record type",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        # Not implemented in live mode — would require real DNS resolution
        return self.execute_simulated(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        domain = kwargs["domain"].lower().strip()
        h = _sim_hash("dns", domain)

        # Generate a deterministic IPv4 address from the hash
        octets = [
            (h >> 24) & 0xFF or 1,  # Avoid 0 in first octet
            (h >> 16) & 0xFF,
            (h >> 8) & 0xFF,
            h & 0xFF,
        ]
        ip_address = f"{octets[0]}.{octets[1]}.{octets[2]}.{octets[3]}"

        return {
            "domain": domain,
            "ip_address": ip_address,
            "record_type": "A",
            "ttl_seconds": 300 + (h % 3300),  # 300–3599
        }


# ---------------------------------------------------------------------------
# 6. ip_geolocation
# ---------------------------------------------------------------------------


@register_tool
class IpGeolocation(BaseTool):
    """Geolocate an IP address (simulated)."""

    name = "ip_geolocation"
    schema = ToolSchema(
        name="ip_geolocation",
        description=(
            "Get the approximate geographic location of an IP address, "
            "including country, city, and coordinates."
        ),
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="ip_address",
                type="string",
                description="IPv4 address to geolocate (e.g. '8.8.8.8')",
            ),
        ],
        returns="Country, city, latitude, and longitude for the IP",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        # Not implemented in live mode — would require a geolocation API
        return self.execute_simulated(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        ip = kwargs["ip_address"]
        h = _sim_hash("geolocation", ip)

        country = _COUNTRIES[h % len(_COUNTRIES)]
        city = _CITIES[h % len(_CITIES)]
        latitude = round((h % 18000 - 9000) / 100, 4)   # -90.00 to 90.00
        longitude = round((h % 36000 - 18000) / 100, 4)  # -180.00 to 180.00

        return {
            "ip_address": ip,
            "country": country,
            "city": city,
            "latitude": latitude,
            "longitude": longitude,
        }


# ---------------------------------------------------------------------------
# 7. rss_feed_parse
# ---------------------------------------------------------------------------


@register_tool
class RssFeedParse(BaseTool):
    """Parse an RSS feed (simulated)."""

    name = "rss_feed_parse"
    schema = ToolSchema(
        name="rss_feed_parse",
        description=(
            "Parse an RSS feed URL and return the latest items with their "
            "titles, links, and publication dates."
        ),
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="url",
                type="string",
                description="RSS feed URL (e.g. 'https://example.com/feed.xml')",
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="Maximum number of items to return (default 5)",
                required=False,
                default=5,
            ),
        ],
        returns="List of RSS feed items with titles, links, and dates",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        # Not implemented in live mode — would require fetching remote XML
        return self.execute_simulated(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        url = kwargs["url"]
        limit = int(kwargs.get("limit", 5))
        limit = max(1, min(limit, 20))

        # Extract a domain-like name for realistic-looking items
        parsed = urlparse(url)
        source = parsed.hostname or "example.com"

        items: list[dict[str, str]] = []
        for i in range(limit):
            h = _sim_hash("rss_item", url, i)
            title = _RSS_TITLES[h % len(_RSS_TITLES)]
            day = 1 + (h % 28)
            month = 1 + (h % 12)
            items.append({
                "title": title,
                "link": f"https://{source}/article/{_sim_hex('rss_link', url, i)}",
                "published": f"2026-{month:02d}-{day:02d}T{(h % 24):02d}:00:00Z",
                "summary": f"Summary of: {title}",
            })

        return {
            "feed_url": url,
            "source": source,
            "items": items,
            "count": len(items),
        }


# ---------------------------------------------------------------------------
# 8. encode_url
# ---------------------------------------------------------------------------


@register_tool
class EncodeUrl(BaseTool):
    """URL encode or decode a string."""

    name = "encode_url"
    schema = ToolSchema(
        name="encode_url",
        description=(
            "URL encode or decode a text string. Encodes special characters "
            "for safe use in URLs, or decodes percent-encoded strings back "
            "to readable text."
        ),
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to encode or decode",
            ),
            ToolParameter(
                name="action",
                type="string",
                description="Whether to encode or decode",
                enum=["encode", "decode"],
            ),
        ],
        returns="The encoded or decoded result string",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._process(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._process(**kwargs)

    def _process(self, **kwargs: Any) -> dict[str, str]:
        """Perform URL encoding/decoding (pure function)."""
        text = kwargs["text"]
        action = kwargs["action"]

        if action == "encode":
            result = quote(text, safe="")
        else:
            result = unquote(text)

        return {"result": result}


# ---------------------------------------------------------------------------
# 9. extract_domain
# ---------------------------------------------------------------------------


@register_tool
class ExtractDomain(BaseTool):
    """Extract domain components from a URL."""

    name = "extract_domain"
    schema = ToolSchema(
        name="extract_domain",
        description=(
            "Extract the domain, subdomain, and top-level domain (TLD) "
            "from a given URL."
        ),
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="url",
                type="string",
                description="Full URL to extract the domain from (e.g. 'https://www.example.com/page')",
            ),
        ],
        returns="Domain, subdomain, and TLD components",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._extract(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._extract(**kwargs)

    def _extract(self, **kwargs: Any) -> dict[str, str]:
        """Parse URL and extract domain components (pure function)."""
        url = kwargs["url"]
        parsed = urlparse(url)
        hostname = parsed.hostname or ""

        parts = hostname.split(".")
        if len(parts) >= 3:
            # e.g. ["www", "example", "com"] or ["api", "v2", "example", "co", "uk"]
            tld = parts[-1]
            domain = parts[-2]
            subdomain = ".".join(parts[:-2])
        elif len(parts) == 2:
            tld = parts[-1]
            domain = parts[0]
            subdomain = ""
        else:
            tld = ""
            domain = hostname
            subdomain = ""

        return {
            "domain": domain,
            "subdomain": subdomain,
            "tld": tld,
        }


# ---------------------------------------------------------------------------
# 10. generate_url
# ---------------------------------------------------------------------------


@register_tool
class GenerateUrl(BaseTool):
    """Build a URL from components."""

    name = "generate_url"
    schema = ToolSchema(
        name="generate_url",
        description=(
            "Build a complete URL from a base URL, path, and query parameters. "
            "Parameters should be provided as a JSON string of key-value pairs."
        ),
        category=ToolCategory.INFORMATION_RETRIEVAL,
        parameters=[
            ToolParameter(
                name="base",
                type="string",
                description="Base URL (e.g. 'https://api.example.com')",
            ),
            ToolParameter(
                name="path",
                type="string",
                description="URL path (e.g. '/v1/search')",
            ),
            ToolParameter(
                name="params",
                type="string",
                description='Query parameters as JSON string (e.g. \'{"q": "test", "page": "1"}\')',
            ),
        ],
        returns="The fully constructed URL",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self._build(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        return self._build(**kwargs)

    def _build(self, **kwargs: Any) -> dict[str, str]:
        """Construct a URL from its components (pure function)."""
        base = kwargs["base"].rstrip("/")
        path = kwargs["path"]
        params_str = kwargs["params"]

        # Ensure path starts with /
        if path and not path.startswith("/"):
            path = "/" + path

        # Parse JSON params
        try:
            params_dict = json.loads(params_str)
        except (json.JSONDecodeError, TypeError):
            params_dict = {}

        query_string = urlencode(params_dict)
        url = f"{base}{path}"
        if query_string:
            url = f"{url}?{query_string}"

        return {"url": url}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "HttpRequest",
    "ParseHtml",
    "ExtractLinks",
    "CheckUrlStatus",
    "DnsLookup",
    "IpGeolocation",
    "RssFeedParse",
    "EncodeUrl",
    "ExtractDomain",
    "GenerateUrl",
]

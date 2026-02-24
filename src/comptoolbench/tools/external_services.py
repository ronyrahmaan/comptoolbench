"""External service tools: weather, currency, location, stock prices.

Live mode uses FREE APIs (no keys needed for most).
Simulated mode returns deterministic outputs.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import requests

from comptoolbench.tools.base import (
    BaseTool,
    ToolCategory,
    ToolParameter,
    ToolSchema,
    register_tool,
)

# ---------------------------------------------------------------------------
# Simulated data pools (used for deterministic fallback)
# ---------------------------------------------------------------------------

_WEATHER_CONDITIONS = ["sunny", "cloudy", "rainy", "partly cloudy", "snowy", "windy", "foggy"]

_CITY_COORDS: dict[str, tuple[float, float]] = {
    "tokyo": (35.6762, 139.6503),
    "paris": (48.8566, 2.3522),
    "london": (51.5074, -0.1278),
    "new york": (40.7128, -74.0060),
    "sydney": (-33.8688, 151.2093),
    "berlin": (52.5200, 13.4050),
    "mumbai": (19.0760, 72.8777),
    "dubai": (25.2048, 55.2708),
    "toronto": (43.6532, -79.3832),
    "singapore": (1.3521, 103.8198),
    "seoul": (37.5665, 126.9780),
    "beijing": (39.9042, 116.4074),
    "cairo": (30.0444, 31.2357),
    "moscow": (55.7558, 37.6173),
    "rome": (41.9028, 12.4964),
    "san francisco": (37.7749, -122.4194),
    "los angeles": (34.0522, -118.2437),
    "chicago": (41.8781, -87.6298),
}

_EXCHANGE_RATES: dict[str, float] = {
    "USD": 1.0, "EUR": 0.92, "GBP": 0.79, "JPY": 149.50, "AUD": 1.53,
    "CAD": 1.36, "CHF": 0.88, "CNY": 7.24, "INR": 83.12, "KRW": 1325.50,
    "BRL": 4.97, "MXN": 17.15, "SGD": 1.34, "HKD": 7.82, "SEK": 10.42,
}


def _sim_hash(seed: str, *args: Any) -> int:
    """Deterministic hash → integer for simulated data."""
    raw = json.dumps([seed, *args], sort_keys=True, default=str)
    return int(hashlib.sha256(raw.encode()).hexdigest(), 16)


@register_tool
class GetWeather(BaseTool):
    """Get current weather for a city."""

    name = "get_weather"
    schema = ToolSchema(
        name="get_weather",
        description="Get the current weather for a given city, including temperature, humidity, wind speed, and conditions.",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="city",
                type="string",
                description="The city name, e.g. 'Tokyo' or 'New York'",
            ),
        ],
        returns="Current weather data including temperature (Celsius), humidity, wind speed, and condition",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        city = kwargs["city"]
        # Use wttr.in — free, no API key, returns JSON
        resp = requests.get(
            f"https://wttr.in/{city}?format=j1",
            timeout=10,
            headers={"User-Agent": "CompToolBench/0.1"},
        )
        resp.raise_for_status()
        data = resp.json()
        current = data["current_condition"][0]
        return {
            "city": city,
            "temperature_celsius": int(current["temp_C"]),
            "humidity_percent": int(current["humidity"]),
            "wind_speed_kmh": int(current["windspeedKmph"]),
            "condition": current["weatherDesc"][0]["value"].lower(),
            "feels_like_celsius": int(current["FeelsLikeC"]),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        city = kwargs["city"].lower().strip()
        h = _sim_hash("weather", city)
        return {
            "city": kwargs["city"],
            "temperature_celsius": (h % 45) - 5,  # -5 to 39
            "humidity_percent": 30 + (h % 60),     # 30-89
            "wind_speed_kmh": h % 50,              # 0-49
            "condition": _WEATHER_CONDITIONS[h % len(_WEATHER_CONDITIONS)],
            "feels_like_celsius": (h % 45) - 7,
        }


@register_tool
class GetExchangeRate(BaseTool):
    """Get currency exchange rate."""

    name = "get_exchange_rate"
    schema = ToolSchema(
        name="get_exchange_rate",
        description="Get the current exchange rate between two currencies. Returns the rate and converted amount if a value is provided.",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="from_currency",
                type="string",
                description="Source currency code (e.g. 'USD', 'EUR', 'GBP')",
            ),
            ToolParameter(
                name="to_currency",
                type="string",
                description="Target currency code (e.g. 'JPY', 'EUR')",
            ),
            ToolParameter(
                name="amount",
                type="number",
                description="Amount to convert (default 1.0)",
                required=False,
                default=1.0,
            ),
        ],
        returns="Exchange rate and converted amount",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        from_curr = kwargs["from_currency"].upper()
        to_curr = kwargs["to_currency"].upper()
        amount = float(kwargs.get("amount", 1.0))

        # Use exchangerate.host — free, no key
        resp = requests.get(
            "https://api.exchangerate.host/latest",
            params={"base": from_curr, "symbols": to_curr, "amount": amount},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("success", True) or to_curr not in data.get("rates", {}):
            # Fallback to simulated
            return self.execute_simulated(**kwargs)

        rate = data["rates"][to_curr] / amount if amount != 0 else 0
        return {
            "from_currency": from_curr,
            "to_currency": to_curr,
            "rate": round(rate, 6),
            "amount": amount,
            "converted_amount": round(data["rates"][to_curr], 4),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        from_curr = kwargs["from_currency"].upper()
        to_curr = kwargs["to_currency"].upper()
        amount = float(kwargs.get("amount", 1.0))

        from_rate = _EXCHANGE_RATES.get(from_curr, 1.0)
        to_rate = _EXCHANGE_RATES.get(to_curr, 1.0)
        rate = to_rate / from_rate
        return {
            "from_currency": from_curr,
            "to_currency": to_curr,
            "rate": round(rate, 6),
            "amount": amount,
            "converted_amount": round(amount * rate, 4),
        }


@register_tool
class GetStockPrice(BaseTool):
    """Get current stock price."""

    name = "get_stock_price"
    schema = ToolSchema(
        name="get_stock_price",
        description="Get the current stock price and basic info for a given ticker symbol.",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="symbol",
                type="string",
                description="Stock ticker symbol, e.g. 'AAPL', 'GOOGL', 'MSFT'",
            ),
        ],
        returns="Current stock price, change, and basic info",
        returns_type="object",
    )

    _STOCK_DATA: dict[str, dict[str, Any]] = {
        "AAPL": {"name": "Apple Inc.", "price": 189.84, "change": 1.23, "market_cap": "2.95T"},
        "GOOGL": {"name": "Alphabet Inc.", "price": 141.80, "change": -0.54, "market_cap": "1.76T"},
        "MSFT": {"name": "Microsoft Corp.", "price": 415.20, "change": 2.10, "market_cap": "3.08T"},
        "AMZN": {"name": "Amazon.com Inc.", "price": 178.25, "change": 0.87, "market_cap": "1.86T"},
        "NVDA": {"name": "NVIDIA Corp.", "price": 875.35, "change": 12.40, "market_cap": "2.16T"},
        "TSLA": {"name": "Tesla Inc.", "price": 238.45, "change": -3.20, "market_cap": "758B"},
        "META": {"name": "Meta Platforms", "price": 505.75, "change": 4.50, "market_cap": "1.29T"},
    }

    def execute_live(self, **kwargs: Any) -> Any:
        # Live would use yfinance, but for benchmark we use fixed data
        # to ensure reproducibility across runs
        return self.execute_simulated(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        symbol = kwargs["symbol"].upper()
        if symbol in self._STOCK_DATA:
            data = self._STOCK_DATA[symbol]
            return {
                "symbol": symbol,
                "name": data["name"],
                "price_usd": data["price"],
                "change_usd": data["change"],
                "change_percent": round(data["change"] / data["price"] * 100, 2),
                "market_cap": data["market_cap"],
            }
        # Unknown symbol — generate deterministic data
        h = _sim_hash("stock", symbol)
        price = 50 + (h % 500)
        change = round((h % 200 - 100) / 10, 2)
        return {
            "symbol": symbol,
            "name": f"{symbol} Corp.",
            "price_usd": price,
            "change_usd": change,
            "change_percent": round(change / price * 100, 2),
            "market_cap": f"{h % 500}B",
        }


@register_tool
class GetLocationInfo(BaseTool):
    """Get geographic info about a location."""

    name = "get_location_info"
    schema = ToolSchema(
        name="get_location_info",
        description="Get geographic information about a city or location, including coordinates, country, timezone, and population.",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="location",
                type="string",
                description="City or location name, e.g. 'Tokyo' or 'Paris, France'",
            ),
        ],
        returns="Geographic information including coordinates, country, and timezone",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        location = kwargs["location"]
        # Use Nominatim (OpenStreetMap) — free, no key
        resp = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": location, "format": "json", "limit": 1, "addressdetails": 1},
            timeout=10,
            headers={"User-Agent": "CompToolBench/0.1"},
        )
        resp.raise_for_status()
        results = resp.json()

        if not results:
            return self.execute_simulated(**kwargs)

        r = results[0]
        addr = r.get("address", {})
        return {
            "location": location,
            "display_name": r.get("display_name", location),
            "latitude": round(float(r["lat"]), 4),
            "longitude": round(float(r["lon"]), 4),
            "country": addr.get("country", "Unknown"),
            "country_code": addr.get("country_code", "").upper(),
            "type": r.get("type", "city"),
        }

    def execute_simulated(self, **kwargs: Any) -> Any:
        location = kwargs["location"].lower().strip()
        coords = _CITY_COORDS.get(location, (0.0, 0.0))
        h = _sim_hash("location", location)

        return {
            "location": kwargs["location"],
            "display_name": f"{kwargs['location']}, Country",
            "latitude": coords[0] if coords != (0.0, 0.0) else round((h % 180) - 90, 4),
            "longitude": coords[1] if coords != (0.0, 0.0) else round((h % 360) - 180, 4),
            "country": "Japan" if "tokyo" in location else "France" if "paris" in location else "Unknown",
            "country_code": "JP" if "tokyo" in location else "FR" if "paris" in location else "XX",
            "type": "city",
        }


@register_tool
class TranslateText(BaseTool):
    """Translate text between languages."""

    name = "translate_text"
    schema = ToolSchema(
        name="translate_text",
        description="Translate text from one language to another.",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="text",
                type="string",
                description="The text to translate",
            ),
            ToolParameter(
                name="from_language",
                type="string",
                description="Source language code (e.g. 'en', 'fr', 'ja', 'de')",
            ),
            ToolParameter(
                name="to_language",
                type="string",
                description="Target language code (e.g. 'en', 'fr', 'ja', 'de')",
            ),
        ],
        returns="The translated text",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        text = kwargs["text"]
        from_lang = kwargs["from_language"]
        to_lang = kwargs["to_language"]

        # Use LibreTranslate (free, open source)
        try:
            resp = requests.post(
                "https://libretranslate.com/translate",
                json={"q": text, "source": from_lang, "target": to_lang},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "original_text": text,
                "translated_text": data["translatedText"],
                "from_language": from_lang,
                "to_language": to_lang,
            }
        except Exception:
            return self.execute_simulated(**kwargs)

    # Common phrase translations for realistic simulated output
    _TRANSLATIONS: dict[str, dict[str, str]] = {
        "hello": {"fr": "bonjour", "es": "hola", "de": "hallo", "ja": "こんにちは", "it": "ciao", "pt": "olá", "zh": "你好", "ko": "안녕하세요", "ru": "привет"},
        "hello world": {"fr": "bonjour le monde", "es": "hola mundo", "de": "hallo welt", "ja": "ハローワールド", "it": "ciao mondo"},
        "good morning": {"fr": "bonjour", "es": "buenos días", "de": "guten morgen", "ja": "おはようございます", "it": "buongiorno"},
        "thank you": {"fr": "merci", "es": "gracias", "de": "danke", "ja": "ありがとう", "it": "grazie", "pt": "obrigado"},
        "how are you": {"fr": "comment allez-vous", "es": "¿cómo estás?", "de": "wie geht es Ihnen", "ja": "お元気ですか"},
        "goodbye": {"fr": "au revoir", "es": "adiós", "de": "auf wiedersehen", "ja": "さようなら", "it": "arrivederci"},
        "the weather is nice today": {"fr": "le temps est beau aujourd'hui", "es": "el clima está agradable hoy", "de": "das Wetter ist heute schön"},
    }

    _LANG_NAMES: dict[str, str] = {
        "en": "English", "fr": "French", "es": "Spanish", "de": "German",
        "ja": "Japanese", "it": "Italian", "pt": "Portuguese", "zh": "Chinese",
        "ko": "Korean", "ar": "Arabic", "ru": "Russian",
    }

    def execute_simulated(self, **kwargs: Any) -> Any:
        text = kwargs["text"]
        from_lang = kwargs["from_language"]
        to_lang = kwargs["to_language"]

        # Try lookup table first for common phrases
        key = text.lower().strip().rstrip(".")
        if key in self._TRANSLATIONS and to_lang in self._TRANSLATIONS[key]:
            translated = self._TRANSLATIONS[key][to_lang]
        else:
            # Deterministic fallback with language name
            lang_name = self._LANG_NAMES.get(to_lang, to_lang.upper())
            translated = f"{text} ({lang_name} translation)"

        return {
            "original_text": text,
            "translated_text": translated,
            "from_language": from_lang,
            "to_language": to_lang,
        }


@register_tool
class GetDirections(BaseTool):
    """Get directions between two locations."""

    name = "get_directions"
    schema = ToolSchema(
        name="get_directions",
        description="Get directions from one location to another, including distance, estimated travel time, and step-by-step directions.",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="origin",
                type="string",
                description="Starting location (e.g. 'New York' or '350 5th Ave, New York')",
            ),
            ToolParameter(
                name="destination",
                type="string",
                description="Destination location (e.g. 'Boston' or 'Logan Airport')",
            ),
            ToolParameter(
                name="mode",
                type="string",
                description="Travel mode",
                enum=["driving", "walking", "transit", "cycling"],
                default="driving",
                required=False,
            ),
        ],
        returns="Directions with distance, duration, and route steps",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        return self.execute_simulated(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        origin = kwargs["origin"]
        destination = kwargs["destination"]
        mode = kwargs.get("mode", "driving")
        h = _sim_hash("directions", origin, destination, mode)

        speed = {"driving": 60, "walking": 5, "transit": 40, "cycling": 15}[mode]
        distance_km = 10 + (h % 500)
        duration_min = round(distance_km / speed * 60)

        steps = [
            f"Head north from {origin}",
            f"Continue on main highway for {distance_km * 0.6:.0f} km",
            f"Take exit toward {destination}",
            f"Arrive at {destination}",
        ]

        return {
            "origin": origin,
            "destination": destination,
            "mode": mode,
            "distance_km": round(distance_km, 1),
            "distance_miles": round(distance_km * 0.621371, 1),
            "duration_minutes": duration_min,
            "duration_text": f"{duration_min // 60}h {duration_min % 60}min" if duration_min >= 60 else f"{duration_min} min",
            "steps": steps,
        }


@register_tool
class SearchProducts(BaseTool):
    """Search for products with prices."""

    name = "search_products"
    schema = ToolSchema(
        name="search_products",
        description="Search for products by name or category and get prices, ratings, and availability.",
        category=ToolCategory.EXTERNAL_SERVICES,
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="Product search query, e.g. 'laptop under $1000' or 'wireless headphones'",
            ),
            ToolParameter(
                name="max_results",
                type="integer",
                description="Maximum number of results to return (default 5)",
                required=False,
                default=5,
            ),
        ],
        returns="List of matching products with prices and ratings",
        returns_type="object",
    )

    def execute_live(self, **kwargs: Any) -> Any:
        # No free product API — always simulated
        return self.execute_simulated(**kwargs)

    def execute_simulated(self, **kwargs: Any) -> Any:
        query = kwargs["query"]
        max_results = int(kwargs.get("max_results", 5))
        h = _sim_hash("products", query)

        products = []
        for i in range(min(max_results, 5)):
            h2 = _sim_hash("product_item", query, i)
            price = round(20 + (h2 % 980), 2)
            rating = round(3.0 + (h2 % 20) / 10, 1)
            products.append({
                "name": f"{query.title()} - Option {i + 1}",
                "price_usd": price,
                "rating": min(rating, 5.0),
                "reviews": 50 + (h2 % 9950),
                "in_stock": h2 % 5 != 0,
            })

        return {
            "query": query,
            "results": products,
            "total_found": len(products),
        }

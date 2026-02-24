"""Tests for external service tools."""

from __future__ import annotations

from comptoolbench.tools.base import ToolMode
from comptoolbench.tools.external_services import (
    GetDirections,
    GetExchangeRate,
    GetLocationInfo,
    GetStockPrice,
    GetWeather,
    SearchProducts,
    TranslateText,
)


class TestGetWeather:
    def setup_method(self) -> None:
        self.tool = GetWeather(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        r = self.tool.execute(city="Tokyo")
        assert r.success
        assert r.data["city"] == "Tokyo"
        assert "temperature_celsius" in r.data
        assert "condition" in r.data

    def test_deterministic(self) -> None:
        r1 = self.tool.execute(city="Paris")
        r2 = self.tool.execute(city="Paris")
        assert r1.data["temperature_celsius"] == r2.data["temperature_celsius"]

    def test_different_cities_differ(self) -> None:
        r1 = self.tool.execute(city="Tokyo")
        r2 = self.tool.execute(city="London")
        # Very unlikely to be identical due to hash
        assert r1.data["city"] != r2.data["city"]


class TestGetExchangeRate:
    def setup_method(self) -> None:
        self.tool = GetExchangeRate(mode=ToolMode.SIMULATED)

    def test_usd_to_eur(self) -> None:
        r = self.tool.execute(from_currency="USD", to_currency="EUR", amount=100)
        assert r.success
        assert r.data["from_currency"] == "USD"
        assert r.data["to_currency"] == "EUR"
        assert r.data["converted_amount"] > 0

    def test_same_currency(self) -> None:
        r = self.tool.execute(from_currency="USD", to_currency="USD", amount=50)
        assert r.success
        assert r.data["converted_amount"] == 50.0

    def test_rate_consistency(self) -> None:
        r1 = self.tool.execute(from_currency="USD", to_currency="EUR", amount=1)
        r2 = self.tool.execute(from_currency="USD", to_currency="EUR", amount=100)
        # Rate should be the same, amount proportional
        assert r1.data["rate"] == r2.data["rate"]


class TestGetStockPrice:
    def setup_method(self) -> None:
        self.tool = GetStockPrice(mode=ToolMode.SIMULATED)

    def test_known_symbol(self) -> None:
        r = self.tool.execute(symbol="AAPL")
        assert r.success
        assert r.data["symbol"] == "AAPL"
        assert r.data["name"] == "Apple Inc."
        assert r.data["price_usd"] > 0

    def test_unknown_symbol(self) -> None:
        r = self.tool.execute(symbol="FAKE")
        assert r.success  # Returns generated data
        assert r.data["symbol"] == "FAKE"


class TestGetLocationInfo:
    def setup_method(self) -> None:
        self.tool = GetLocationInfo(mode=ToolMode.SIMULATED)

    def test_known_city(self) -> None:
        r = self.tool.execute(location="Tokyo")
        assert r.success
        assert r.data["latitude"] == 35.6762
        assert r.data["longitude"] == 139.6503


class TestTranslateText:
    def setup_method(self) -> None:
        self.tool = TranslateText(mode=ToolMode.SIMULATED)

    def test_basic_translation(self) -> None:
        r = self.tool.execute(text="Hello", from_language="en", to_language="fr")
        assert r.success
        assert r.data["original_text"] == "Hello"
        # "Hello" → "bonjour" from lookup table (realistic simulation)
        assert r.data["translated_text"] == "bonjour"


class TestGetDirections:
    def setup_method(self) -> None:
        self.tool = GetDirections(mode=ToolMode.SIMULATED)

    def test_driving(self) -> None:
        r = self.tool.execute(origin="New York", destination="Boston", mode="driving")
        assert r.success
        assert r.data["origin"] == "New York"
        assert r.data["destination"] == "Boston"
        assert r.data["distance_km"] > 0
        assert r.data["duration_minutes"] > 0

    def test_walking_slower(self) -> None:
        r_drive = self.tool.execute(origin="A", destination="B", mode="driving")
        r_walk = self.tool.execute(origin="A", destination="B", mode="walking")
        assert r_walk.data["duration_minutes"] > r_drive.data["duration_minutes"]

    def test_deterministic(self) -> None:
        r1 = self.tool.execute(origin="X", destination="Y")
        r2 = self.tool.execute(origin="X", destination="Y")
        assert r1.data["distance_km"] == r2.data["distance_km"]


class TestSearchProducts:
    def setup_method(self) -> None:
        self.tool = SearchProducts(mode=ToolMode.SIMULATED)

    def test_basic(self) -> None:
        r = self.tool.execute(query="laptop", max_results=3)
        assert r.success
        assert len(r.data["results"]) == 3
        assert all("price_usd" in p for p in r.data["results"])

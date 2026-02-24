"""New L2 (parallel fork-join) task generators for CompToolBench.

This module defines 10 additional L2 generators that produce parallel
composition tasks. Each generator is a ``@staticmethod`` that accepts a
:class:`CompositionEngine` instance and returns a fully-populated
:class:`Task`.

Generators
----------
1.  multi_hash           - Hash same text with 3 algorithms in parallel
2.  multi_exchange       - Convert same amount to 3 currencies in parallel
3.  multi_format_date    - Format same date in 3 formats in parallel
4.  multi_keyword_extract - Extract keywords from 3 texts in parallel
5.  multi_detect_language - Detect language of 3 texts in parallel
6.  parallel_weather_stock - Weather + stock price simultaneously
7.  multi_word_count     - Count words in 3 texts in parallel
8.  multi_extract_numbers - Extract numbers from 3 entity texts in parallel
9.  parallel_weather_time - Weather + current time for same city/timezone
10. multi_business_days  - Business days between 3 date pairs in parallel
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from comptoolbench.tasks.models import (
    CompositionLevel,
    ExpectedTrace,
    Task,
    ToolCall,
)

if TYPE_CHECKING:
    from comptoolbench.generators.composition_engine import CompositionEngine

# Re-import parameter pools used by the generators so that the main
# composition engine module does not need restructuring.
from comptoolbench.generators.composition_engine import (
    CITIES_EXPANDED,
    CURRENCIES_EXPANDED,
    DATE_PAIRS_EXPANDED,
    ENTITY_TEXTS_EXPANDED,
    STOCKS_EXPANDED,
    TEXTS_EXPANDED,
)

# ---------------------------------------------------------------------------
# City-to-timezone mapping (used by parallel_weather_time)
# ---------------------------------------------------------------------------

CITY_TIMEZONE_MAP: dict[str, str] = {
    "Tokyo": "Asia/Tokyo",
    "Paris": "Europe/Paris",
    "London": "Europe/London",
    "New York": "US/Eastern",
    "Sydney": "Australia/Sydney",
    "Berlin": "Europe/Berlin",
    "Mumbai": "Asia/Kolkata",
    "Dubai": "Asia/Dubai",
    "Toronto": "America/Toronto",
    "Singapore": "Asia/Singapore",
    "Seoul": "Asia/Seoul",
    "Beijing": "Asia/Shanghai",
    "Cairo": "Africa/Cairo",
    "Moscow": "Europe/Moscow",
    "Rome": "Europe/Rome",
    "San Francisco": "US/Pacific",
    "Los Angeles": "US/Pacific",
    "Chicago": "US/Central",
    "Bangkok": "Asia/Bangkok",
    "Istanbul": "Europe/Istanbul",
    "Lagos": "Africa/Lagos",
    "Jakarta": "Asia/Jakarta",
    "Mexico City": "America/Mexico_City",
    "Buenos Aires": "America/Argentina/Buenos_Aires",
    "Lima": "America/Lima",
    "Nairobi": "Africa/Nairobi",
    "Johannesburg": "Africa/Johannesburg",
    "Stockholm": "Europe/Stockholm",
    "Oslo": "Europe/Oslo",
    "Copenhagen": "Europe/Copenhagen",
    "Helsinki": "Europe/Helsinki",
    "Warsaw": "Europe/Warsaw",
    "Prague": "Europe/Prague",
    "Vienna": "Europe/Vienna",
    "Zurich": "Europe/Zurich",
    "Amsterdam": "Europe/Amsterdam",
    "Brussels": "Europe/Brussels",
    "Lisbon": "Europe/Lisbon",
    "Madrid": "Europe/Madrid",
    "Barcelona": "Europe/Madrid",
    "Athens": "Europe/Athens",
    "Dublin": "Europe/Dublin",
    "Edinburgh": "Europe/London",
    "Montreal": "America/Toronto",
    "Vancouver": "America/Vancouver",
    "Osaka": "Asia/Tokyo",
    "Shanghai": "Asia/Shanghai",
    "Delhi": "Asia/Kolkata",
    "Kuala Lumpur": "Asia/Kuala_Lumpur",
    "Manila": "Asia/Manila",
}

# ---------------------------------------------------------------------------
# Multilingual texts pool (for multi_detect_language)
# ---------------------------------------------------------------------------

MULTILINGUAL_TEXTS: list[tuple[str, str]] = [
    ("The weather is beautiful today and I feel great.", "english"),
    (
        "Le temps est magnifique aujourd'hui et les fleurs sont en pleine floraison.",
        "french",
    ),
    ("Das Wetter ist heute wunderschön und die Blumen blühen.", "german"),
    ("El clima está hermoso hoy y las flores están floreciendo.", "spanish"),
    ("Il tempo è bellissimo oggi e i fiori stanno sbocciando.", "italian"),
    ("O tempo está lindo hoje e as flores estão desabrochando.", "portuguese"),
    ("Het weer is prachtig vandaag en de bloemen staan in bloei.", "dutch"),
    ("Artificial intelligence has revolutionized the way we work and live.", "english"),
    (
        "L'intelligence artificielle a révolutionné notre façon de travailler et de vivre.",
        "french",
    ),
    (
        "Künstliche Intelligenz hat unsere Art zu arbeiten und zu leben revolutioniert.",
        "german",
    ),
    (
        "La inteligencia artificial ha revolucionado nuestra forma de trabajar y vivir.",
        "spanish",
    ),
    (
        "L'intelligenza artificiale ha rivoluzionato il nostro modo di lavorare e vivere.",
        "italian",
    ),
    (
        "A inteligência artificial revolucionou a forma como trabalhamos e vivemos.",
        "portuguese",
    ),
    (
        "Kunstmatige intelligentie heeft de manier waarop we werken en leven gerevolutioneerd.",
        "dutch",
    ),
    ("The city streets are full of people enjoying the weekend sunshine.", "english"),
    (
        "Les rues de la ville sont pleines de gens qui profitent du soleil du week-end.",
        "french",
    ),
    (
        "Die Straßen der Stadt sind voll mit Menschen die das Wochenende genießen.",
        "german",
    ),
    (
        "Las calles de la ciudad están llenas de gente disfrutando del sol del fin de semana.",
        "spanish",
    ),
    (
        "Le strade della città sono piene di persone che si godono il sole del fine settimana.",
        "italian",
    ),
    (
        "As ruas da cidade estão cheias de pessoas aproveitando o sol do fim de semana.",
        "portuguese",
    ),
    ("Education is the foundation of a prosperous and equitable society.", "english"),
    ("L'éducation est le fondement d'une société prospère et équitable.", "french"),
    (
        "Bildung ist die Grundlage einer wohlhabenden und gerechten Gesellschaft.",
        "german",
    ),
    ("La educación es la base de una sociedad próspera y equitativa.", "spanish"),
    ("L'istruzione è il fondamento di una società prospera e equa.", "italian"),
    ("A educação é a base de uma sociedade próspera e equitativa.", "portuguese"),
    ("De kennis van vandaag is de basis van de welvaart van morgen.", "dutch"),
    (
        "Science and technology continue to push the boundaries of human knowledge.",
        "english",
    ),
    (
        "La science et la technologie continuent de repousser les limites de la connaissance.",
        "french",
    ),
    (
        "Wissenschaft und Technologie erweitern weiterhin die Grenzen des menschlichen Wissens.",
        "german",
    ),
]


# ---------------------------------------------------------------------------
# L2 Prompt Variants
# ---------------------------------------------------------------------------

NEW_L2_PROMPT_VARIANTS: dict[str, list[str]] = {
    "multi_hash": [
        'Compute the SHA-256, MD5, and SHA-512 hashes of: "{text}"',
        'Hash the text "{text}" using sha256, md5, and sha512 simultaneously.',
        'Generate all three hashes (SHA-256, MD5, SHA-512) for: "{text}"',
        'Calculate the md5, sha256, and sha512 digests of: "{text}"',
    ],
    "multi_exchange": [
        "Convert {amount} {from_c} to {currencies} simultaneously.",
        "How much is {amount} {from_c} in {currencies}?",
        "Exchange {amount} {from_c} into {currencies} at the same time.",
        "Get the value of {amount} {from_c} in each of {currencies}.",
    ],
    "multi_format_date": [
        "Format the date {date} in short, long, and ISO formats simultaneously.",
        "Show {date} in three formats: short, long, and iso.",
        "Convert {date} into short, long, and ISO date representations.",
        "Display the date {date} in short format, long format, and ISO format at once.",
    ],
    "multi_keyword_extract": [
        'Extract keywords from these three texts: 1) "{t1}" 2) "{t2}" 3) "{t3}"',
        'Find the key terms in each of these texts simultaneously: "{t1}", "{t2}", "{t3}"',
        'Identify keywords from three passages at once: "{t1}", "{t2}", "{t3}"',
    ],
    "multi_detect_language": [
        'Detect the language of each text: 1) "{t1}" 2) "{t2}" 3) "{t3}"',
        'What languages are these texts written in? "{t1}", "{t2}", "{t3}"',
        'Identify the language for each: "{t1}", "{t2}", "{t3}"',
    ],
    "parallel_weather_stock": [
        "Get the weather in {city} and the stock price of {symbol} at the same time.",
        "Simultaneously check the weather for {city} and look up {symbol}'s stock price.",
        "In parallel: weather in {city} and stock price for {symbol}.",
        "Fetch the current weather in {city} alongside {symbol}'s latest stock price.",
    ],
    "multi_word_count": [
        'Count the words in each of these texts: 1) "{t1}" 2) "{t2}" 3) "{t3}"',
        'How many words are in each text? "{t1}", "{t2}", "{t3}"',
        'Get the word count for three texts simultaneously: "{t1}", "{t2}", "{t3}"',
    ],
    "multi_extract_numbers": [
        'Extract numbers from each of these texts: 1) "{t1}" 2) "{t2}" 3) "{t3}"',
        'Find all numeric values in these three passages: "{t1}", "{t2}", "{t3}"',
        'Pull out the numbers from each text simultaneously: "{t1}", "{t2}", "{t3}"',
    ],
    "parallel_weather_time": [
        "Get the weather in {city} and the current time in {timezone} simultaneously.",
        "Check both the weather for {city} and the current time in {timezone}.",
        "In parallel: weather in {city} and current time in {timezone}.",
        "Fetch the weather for {city} and look up what time it is in {timezone}.",
    ],
    "multi_business_days": [
        "Calculate the business days between these date pairs: {pair1}, {pair2}, {pair3}.",
        "How many business days are there in each period? {pair1}, {pair2}, {pair3}.",
        "Count business days for three date ranges simultaneously: {pair1}, {pair2}, {pair3}.",
    ],
}


# ---------------------------------------------------------------------------
# Generator static methods
# ---------------------------------------------------------------------------


class _NewL2Generators:
    """Container for new L2 generator static methods.

    Each method follows the same interface as existing generators on
    :class:`CompositionEngine`: it receives the engine instance and
    returns a fully populated :class:`Task`.
    """

    # 1. multi_hash ---------------------------------------------------------

    @staticmethod
    def _l2_multi_hash(engine: CompositionEngine) -> Task:
        """Hash same text with sha256, md5, sha512 in parallel."""
        text = engine._pick(TEXTS_EXPANDED)
        algos = ["sha256", "md5", "sha512"]
        hashes: dict[str, Any] = {}
        steps: list[ToolCall] = []
        for i, algo in enumerate(algos):
            h = engine._execute_tool("hash_text", text=text, algorithm=algo)
            hashes[algo] = h
            steps.append(
                ToolCall(
                    step_id=f"step_{i + 1}",
                    tool_name="hash_text",
                    arguments={"text": text, "algorithm": algo},
                    output_key=f"hash_{algo}",
                    depends_on=[],
                )
            )
        text_preview = text[:60] + "..." if len(text) > 60 else text
        prompt = engine._format_prompt(
            "multi_hash",
            NEW_L2_PROMPT_VARIANTS,
            text=text_preview,
        )
        distractors = engine._pick_distractors(["hash_text"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["hash_text"] + distractors,
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=hashes,
            metadata={
                "category": "parallel",
                "tools": ["hash_text"],
                "pattern": "fan-out-hash",
            },
        )

    # 2. multi_exchange -----------------------------------------------------

    @staticmethod
    def _l2_multi_exchange(engine: CompositionEngine) -> Task:
        """Convert same amount from one currency to 3 others in parallel."""
        from_c = engine._pick(CURRENCIES_EXPANDED)
        # Pick 3 target currencies different from the source
        candidates = [c for c in CURRENCIES_EXPANDED if c != from_c]
        to_currencies = engine._pick_n(candidates, 3)
        amount = round(engine.rng.uniform(10, 5000), 2)

        results: dict[str, Any] = {}
        steps: list[ToolCall] = []
        for i, to_c in enumerate(to_currencies):
            r = engine._execute_tool(
                "get_exchange_rate",
                from_currency=from_c,
                to_currency=to_c,
                amount=amount,
            )
            results[to_c] = r
            steps.append(
                ToolCall(
                    step_id=f"step_{i + 1}",
                    tool_name="get_exchange_rate",
                    arguments={
                        "from_currency": from_c,
                        "to_currency": to_c,
                        "amount": amount,
                    },
                    output_key=f"exchange_{to_c.lower()}",
                    depends_on=[],
                )
            )
        prompt = engine._format_prompt(
            "multi_exchange",
            NEW_L2_PROMPT_VARIANTS,
            amount=amount,
            from_c=from_c,
            currencies=", ".join(to_currencies),
        )
        distractors = engine._pick_distractors(["get_exchange_rate"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["get_exchange_rate"] + distractors,
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=results,
            metadata={
                "category": "parallel",
                "tools": ["get_exchange_rate"],
                "pattern": "fan-out-exchange",
            },
        )

    # 3. multi_format_date --------------------------------------------------

    @staticmethod
    def _l2_multi_format_date(engine: CompositionEngine) -> Task:
        """Format the same date in short, long, and ISO formats in parallel."""
        date_pair = engine._pick(DATE_PAIRS_EXPANDED)
        date_str = date_pair[0]
        formats = ["short", "long", "iso"]

        formatted: dict[str, Any] = {}
        steps: list[ToolCall] = []
        for i, fmt in enumerate(formats):
            f = engine._execute_tool("format_date", date=date_str, format=fmt)
            formatted[fmt] = f
            steps.append(
                ToolCall(
                    step_id=f"step_{i + 1}",
                    tool_name="format_date",
                    arguments={"date": date_str, "format": fmt},
                    output_key=f"date_{fmt}",
                    depends_on=[],
                )
            )
        prompt = engine._format_prompt(
            "multi_format_date",
            NEW_L2_PROMPT_VARIANTS,
            date=date_str,
        )
        distractors = engine._pick_distractors(["format_date"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["format_date"] + distractors,
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=formatted,
            metadata={
                "category": "parallel",
                "tools": ["format_date"],
                "pattern": "fan-out-format",
            },
        )

    # 4. multi_keyword_extract ----------------------------------------------

    @staticmethod
    def _l2_multi_keyword_extract(engine: CompositionEngine) -> Task:
        """Extract keywords from 3 different texts in parallel."""
        texts = engine._pick_n(TEXTS_EXPANDED, 3)

        keywords: dict[str, Any] = {}
        steps: list[ToolCall] = []
        for i, text in enumerate(texts):
            k = engine._execute_tool("keyword_extract", text=text)
            keywords[text[:40]] = k
            steps.append(
                ToolCall(
                    step_id=f"step_{i + 1}",
                    tool_name="keyword_extract",
                    arguments={"text": text},
                    output_key=f"keywords_{i + 1}",
                    depends_on=[],
                )
            )
        previews = [t[:50] + "..." if len(t) > 50 else t for t in texts]
        prompt = engine._format_prompt(
            "multi_keyword_extract",
            NEW_L2_PROMPT_VARIANTS,
            t1=previews[0],
            t2=previews[1],
            t3=previews[2],
        )
        distractors = engine._pick_distractors(["keyword_extract"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["keyword_extract"] + distractors,
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=list(keywords.values()),
            metadata={
                "category": "parallel",
                "tools": ["keyword_extract"],
                "pattern": "fan-out-keywords",
            },
        )

    # 5. multi_detect_language ----------------------------------------------

    @staticmethod
    def _l2_multi_detect_language(engine: CompositionEngine) -> Task:
        """Detect the language of 3 texts in parallel."""
        items = engine._pick_n(MULTILINGUAL_TEXTS, 3)

        detections: dict[str, Any] = {}
        steps: list[ToolCall] = []
        for i, (text, _expected_lang) in enumerate(items):
            d = engine._execute_tool("detect_language", text=text)
            detections[text[:40]] = d
            steps.append(
                ToolCall(
                    step_id=f"step_{i + 1}",
                    tool_name="detect_language",
                    arguments={"text": text},
                    output_key=f"lang_{i + 1}",
                    depends_on=[],
                )
            )
        previews = [t[:50] + "..." if len(t) > 50 else t for t, _ in items]
        prompt = engine._format_prompt(
            "multi_detect_language",
            NEW_L2_PROMPT_VARIANTS,
            t1=previews[0],
            t2=previews[1],
            t3=previews[2],
        )
        distractors = engine._pick_distractors(["detect_language"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["detect_language"] + distractors,
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=list(detections.values()),
            metadata={
                "category": "parallel",
                "tools": ["detect_language"],
                "pattern": "fan-out-detect",
            },
        )

    # 6. parallel_weather_stock ---------------------------------------------

    @staticmethod
    def _l2_parallel_weather_stock(engine: CompositionEngine) -> Task:
        """Get weather for a city and stock price simultaneously."""
        city = engine._pick(CITIES_EXPANDED)
        symbol = engine._pick(STOCKS_EXPANDED)

        weather = engine._execute_tool("get_weather", city=city)
        stock = engine._execute_tool("get_stock_price", symbol=symbol)

        prompt = engine._format_prompt(
            "parallel_weather_stock",
            NEW_L2_PROMPT_VARIANTS,
            city=city,
            symbol=symbol,
        )
        distractors = engine._pick_distractors(["get_weather", "get_stock_price"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["get_weather", "get_stock_price"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(
                        step_id="step_1",
                        tool_name="get_weather",
                        arguments={"city": city},
                        output_key="weather",
                        depends_on=[],
                    ),
                    ToolCall(
                        step_id="step_2",
                        tool_name="get_stock_price",
                        arguments={"symbol": symbol},
                        output_key="stock",
                        depends_on=[],
                    ),
                ],
                final_answer_source="step_1",
            ),
            expected_final_answer={"weather": weather, "stock": stock},
            metadata={
                "category": "parallel",
                "tools": ["get_weather", "get_stock_price"],
                "pattern": "independent-merge",
            },
        )

    # 7. multi_word_count ---------------------------------------------------

    @staticmethod
    def _l2_multi_word_count(engine: CompositionEngine) -> Task:
        """Count words in 3 different texts in parallel."""
        texts = engine._pick_n(TEXTS_EXPANDED, 3)

        counts: dict[str, Any] = {}
        steps: list[ToolCall] = []
        for i, text in enumerate(texts):
            c = engine._execute_tool("word_count", text=text)
            counts[text[:40]] = c
            steps.append(
                ToolCall(
                    step_id=f"step_{i + 1}",
                    tool_name="word_count",
                    arguments={"text": text},
                    output_key=f"wc_{i + 1}",
                    depends_on=[],
                )
            )
        previews = [t[:50] + "..." if len(t) > 50 else t for t in texts]
        prompt = engine._format_prompt(
            "multi_word_count",
            NEW_L2_PROMPT_VARIANTS,
            t1=previews[0],
            t2=previews[1],
            t3=previews[2],
        )
        distractors = engine._pick_distractors(["word_count"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["word_count"] + distractors,
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=list(counts.values()),
            metadata={
                "category": "parallel",
                "tools": ["word_count"],
                "pattern": "fan-out-count",
            },
        )

    # 8. multi_extract_numbers ----------------------------------------------

    @staticmethod
    def _l2_multi_extract_numbers(engine: CompositionEngine) -> Task:
        """Extract numbers from 3 entity texts in parallel."""
        texts = engine._pick_n(ENTITY_TEXTS_EXPANDED, 3)

        extractions: dict[str, Any] = {}
        steps: list[ToolCall] = []
        for i, text in enumerate(texts):
            e = engine._execute_tool("extract_numbers", text=text)
            extractions[text[:40]] = e
            steps.append(
                ToolCall(
                    step_id=f"step_{i + 1}",
                    tool_name="extract_numbers",
                    arguments={"text": text},
                    output_key=f"numbers_{i + 1}",
                    depends_on=[],
                )
            )
        previews = [t[:50] + "..." if len(t) > 50 else t for t in texts]
        prompt = engine._format_prompt(
            "multi_extract_numbers",
            NEW_L2_PROMPT_VARIANTS,
            t1=previews[0],
            t2=previews[1],
            t3=previews[2],
        )
        distractors = engine._pick_distractors(["extract_numbers"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["extract_numbers"] + distractors,
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=list(extractions.values()),
            metadata={
                "category": "parallel",
                "tools": ["extract_numbers"],
                "pattern": "fan-out-extract",
            },
        )

    # 9. parallel_weather_time ----------------------------------------------

    @staticmethod
    def _l2_parallel_weather_time(engine: CompositionEngine) -> Task:
        """Get weather and current time for the same city/timezone in parallel."""
        # Pick a city that has a known timezone mapping
        mapped_cities = [c for c in CITIES_EXPANDED if c in CITY_TIMEZONE_MAP]
        city = engine._pick(mapped_cities)
        timezone = CITY_TIMEZONE_MAP[city]

        weather = engine._execute_tool("get_weather", city=city)
        time_result = engine._execute_tool("get_current_time", timezone=timezone)

        prompt = engine._format_prompt(
            "parallel_weather_time",
            NEW_L2_PROMPT_VARIANTS,
            city=city,
            timezone=timezone,
        )
        distractors = engine._pick_distractors(["get_weather", "get_current_time"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["get_weather", "get_current_time"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(
                        step_id="step_1",
                        tool_name="get_weather",
                        arguments={"city": city},
                        output_key="weather",
                        depends_on=[],
                    ),
                    ToolCall(
                        step_id="step_2",
                        tool_name="get_current_time",
                        arguments={"timezone": timezone},
                        output_key="time",
                        depends_on=[],
                    ),
                ],
                final_answer_source="step_1",
            ),
            expected_final_answer={"weather": weather, "current_time": time_result},
            metadata={
                "category": "parallel",
                "tools": ["get_weather", "get_current_time"],
                "pattern": "independent-merge",
            },
        )

    # 10. multi_business_days -----------------------------------------------

    @staticmethod
    def _l2_multi_business_days(engine: CompositionEngine) -> Task:
        """Calculate business days between 3 date pairs in parallel."""
        pairs = engine._pick_n(DATE_PAIRS_EXPANDED, 3)

        results: dict[str, Any] = {}
        steps: list[ToolCall] = []
        for i, (start, end) in enumerate(pairs):
            r = engine._execute_tool(
                "business_days_between",
                start_date=start,
                end_date=end,
            )
            results[f"{start}_to_{end}"] = r
            steps.append(
                ToolCall(
                    step_id=f"step_{i + 1}",
                    tool_name="business_days_between",
                    arguments={"start_date": start, "end_date": end},
                    output_key=f"bdays_{i + 1}",
                    depends_on=[],
                )
            )
        pair_strs = [f"{s} to {e}" for s, e in pairs]
        prompt = engine._format_prompt(
            "multi_business_days",
            NEW_L2_PROMPT_VARIANTS,
            pair1=pair_strs[0],
            pair2=pair_strs[1],
            pair3=pair_strs[2],
        )
        distractors = engine._pick_distractors(["business_days_between"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["business_days_between"] + distractors,
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=results,
            metadata={
                "category": "parallel",
                "tools": ["business_days_between"],
                "pattern": "fan-out-business-days",
            },
        )


# ---------------------------------------------------------------------------
# Registry: maps human-readable name -> callable
# ---------------------------------------------------------------------------

NEW_L2_REGISTRY: dict[str, Any] = {
    "multi_hash": _NewL2Generators._l2_multi_hash,
    "multi_exchange": _NewL2Generators._l2_multi_exchange,
    "multi_format_date": _NewL2Generators._l2_multi_format_date,
    "multi_keyword_extract": _NewL2Generators._l2_multi_keyword_extract,
    "multi_detect_language": _NewL2Generators._l2_multi_detect_language,
    "parallel_weather_stock": _NewL2Generators._l2_parallel_weather_stock,
    "multi_word_count": _NewL2Generators._l2_multi_word_count,
    "multi_extract_numbers": _NewL2Generators._l2_multi_extract_numbers,
    "parallel_weather_time": _NewL2Generators._l2_parallel_weather_time,
    "multi_business_days": _NewL2Generators._l2_multi_business_days,
}

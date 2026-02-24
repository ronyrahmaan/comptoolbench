"""New L1 (2-tool chain) generators for CompToolBench composition engine.

This module defines 18 additional L1 generators covering new tool combinations:
utility, NLP, math, datetime, web, and productivity tools. Each generator
is a ``@staticmethod`` on :class:`CompositionEngine` that returns a :class:`Task`.

The module exports:
- ``NEW_L1_PROMPT_VARIANTS`` — prompt template dict to merge into ``L1_PROMPT_VARIANTS``
- ``NEW_L1_GENERATORS`` — registry dict mapping names to static method callables
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from comptoolbench.tasks.models import (
    CompositionLevel,
    ExpectedTrace,
    Task,
    ToolCall,
)

if TYPE_CHECKING:
    from comptoolbench.generators.composition_engine import CompositionEngine

# Re-import parameter pools used by generators
from comptoolbench.generators.composition_engine import (
    CITIES_EXPANDED,
    CURRENCIES_EXPANDED,
    EMAIL_RECIPIENTS_EXPANDED,
    ENTITY_TEXTS_EXPANDED,
    FILE_PATHS_EXPANDED,
    MEETING_TITLES_EXPANDED,
    SEARCH_QUERIES_EXPANDED,
    STOCKS_EXPANDED,
    TEXTS_EXPANDED,
)


# ---------------------------------------------------------------------------
# Prompt template variants — 3-4 per generator
# ---------------------------------------------------------------------------

NEW_L1_PROMPT_VARIANTS: dict[str, list[str]] = {
    # 1. word_count → format_number
    "word_count_format": [
        'Count the words in "{text}" and format the count with comma separators.',
        'How many words are in "{text}"? Give me the count formatted nicely.',
        'Get the word count of "{text}" and display it in a formatted way.',
        'Count words in "{text}", then format the number with commas.',
    ],
    # 2. keyword_extract → web_search
    "keyword_search": [
        'Extract keywords from "{text}" and search the web for the top keyword.',
        'Find the main keywords in "{text}" and look up the most important one online.',
        'What are the keywords in "{text}"? Search for the top one on the web.',
        'Pull keywords from "{text}", then do a web search for the primary keyword.',
    ],
    # 3. detect_language → translate_text
    "detect_translate": [
        'Detect the language of "{text}" and then translate it to English.',
        'What language is "{text}" in? Translate it to English.',
        'Identify the language of "{text}" and convert it to English.',
        'Figure out what language "{text}" is written in, then translate to English.',
    ],
    # 4. percentage_change → format_number
    "pctchange_format": [
        "What is the percentage change from {old} to {new}? Format the result as a percentage.",
        "Calculate the percent change between {old} and {new}, then format the result nicely.",
        "Compute the percentage difference from {old} to {new} and display it formatted.",
        "Find the % change from {old} to {new} and present it in a formatted style.",
    ],
    # 5. extract_numbers → statistical_analysis
    "extract_nums_stats": [
        'Extract all numbers from "{text}" and compute statistics on them.',
        'Find the numbers in "{text}" and run a statistical analysis.',
        'Pull out every number from "{text}", then calculate summary statistics.',
        'Get the numeric values in "{text}" and analyze them statistically.',
    ],
    # 6. parse_date → get_weekday
    "parse_weekday": [
        'Parse the date "{date_str}" and tell me what day of the week it falls on.',
        'What day of the week is "{date_str}"? Parse the date first.',
        'Convert "{date_str}" to a standard date, then find out the weekday.',
        'Figure out the day of the week for "{date_str}".',
    ],
    # 7. read_file → keyword_extract
    "read_keywords": [
        "Read the file at {path} and extract the keywords from its contents.",
        "Open {path} and pull out the most important keywords.",
        "Read {path}, then identify the key topics and keywords.",
        "Load the file {path} and extract keywords from the text.",
    ],
    # 8. read_file → word_count
    "read_wordcount": [
        "Read the file at {path} and count how many words it contains.",
        "Open {path} and tell me the word count.",
        "Read {path}, then count the total number of words.",
        "How many words are in the file at {path}?",
    ],
    # 9. get_weather → log_event
    "weather_log": [
        "Get the weather in {city} and log the weather report as an event.",
        "Check the weather for {city} and create a log entry with the conditions.",
        "Look up the weather in {city}, then log this as an info event.",
        "Fetch {city}'s weather and record it in the event log.",
    ],
    # 10. calculator → round_number
    "calc_round": [
        "Calculate {expr} and round the result to {decimals} decimal places.",
        "Compute {expr}, then round the answer to {decimals} decimals.",
        "What is {expr}? Round the result to {decimals} decimal places.",
        "Evaluate {expr} and give me the answer rounded to {decimals} places.",
    ],
    # 11. extract_numbers → min_max
    "extract_nums_minmax": [
        'Extract all numbers from "{text}" and find the minimum and maximum.',
        'Find all the numbers in "{text}" and tell me the min and max.',
        'Pull out the numbers from "{text}", then identify the smallest and largest.',
        'Get the numeric values in "{text}" and compute their range.',
    ],
    # 12. create_invoice → send_email
    "invoice_email": [
        "Create an invoice for {client} and email it to {email}.",
        "Generate an invoice for {client} and send it via email to {email}.",
        "Make an invoice for {client}, then email the details to {email}.",
        "Prepare an invoice for {client} and dispatch it to {email} by email.",
    ],
    # 13. generate_report → write_file
    "report_save": [
        'Generate a report titled "{title}" and save it to {path}.',
        'Create a "{title}" report and write it to the file at {path}.',
        'Build a report called "{title}", then save the output to {path}.',
        'Produce a "{title}" report and store it in {path}.',
    ],
    # 14. encrypt_text → write_file
    "encrypt_save": [
        'Encrypt the text "{text}" using {method} and save the result to {path}.',
        'Encrypt "{text}" with {method} encryption, then write it to {path}.',
        'Use {method} to encrypt "{text}" and store the encrypted version at {path}.',
        'Apply {method} encryption to "{text}" and save the output to {path}.',
    ],
    # 15. slugify → generate_url
    "slugify_url": [
        'Create a URL-safe slug from "{title}" and build a URL with it on {base}.',
        'Slugify "{title}" and generate a full URL using base {base}.',
        'Convert "{title}" to a slug and construct a URL with base {base}.',
        'Turn "{title}" into a slug, then build the complete URL from {base}.',
    ],
    # 16. extract_entities → mask_pii
    "entity_mask": [
        'Extract entities from "{text}" and then mask any PII found in it.',
        'Find named entities in "{text}", then redact all personally identifiable information.',
        'Identify entities in "{text}" and mask emails, phone numbers, and other PII.',
        'Analyze "{text}" for entities, then remove any personal information.',
    ],
    # 17. get_stock_price → percentage_change
    "stock_pctchange": [
        "Get the current price of {symbol} and calculate the percentage change from {base_price}.",
        "Look up {symbol}'s stock price and compute the % change from a baseline of {base_price}.",
        "What is {symbol}'s stock price? Calculate the percentage change relative to {base_price}.",
        "Fetch {symbol}'s price and determine the percent difference from {base_price}.",
    ],
    # 18. get_weather → create_calendar_event
    "weather_calendar": [
        "Check the weather in {city} and create an outdoor event called \"{event_title}\" if conditions are good.",
        "Get the weather for {city}. If it's nice, schedule an outdoor calendar event titled \"{event_title}\".",
        "Look up the weather in {city} and create a calendar event \"{event_title}\" including the forecast.",
        "Fetch {city}'s weather, then set up a calendar event called \"{event_title}\" with the weather details.",
    ],
}


# ---------------------------------------------------------------------------
# Additional parameter pools for new generators
# ---------------------------------------------------------------------------

# Files that are guaranteed to exist in the virtual file system
_READABLE_PATHS = [
    "/data/report.txt", "/data/config.json", "/data/employees.csv",
]

_DATE_STRINGS_EXPANDED = [
    "March 15, 2026", "02/14/2026", "2026-06-01", "Jan 1, 2027",
    "12/25/2026", "July 4, 2026", "2026-09-15", "Nov 11, 2026",
    "04/01/2026", "August 20, 2026", "2026-10-31", "Feb 22, 2026",
    "05/05/2026", "December 31, 2026", "2026-03-20", "Sep 1, 2026",
    "06/15/2026", "October 10, 2026", "2026-11-28", "Apr 15, 2026",
    "01/20/2026", "March 8, 2026", "2026-07-20", "May 30, 2026",
    "08/08/2026", "January 15, 2026", "2026-04-22", "Jun 21, 2026",
    "11/15/2026", "February 1, 2026",
]

_MULTILINGUAL_TEXTS = [
    "Bonjour, comment allez-vous aujourd'hui?",
    "Guten Morgen, wie geht es Ihnen?",
    "Hola, me llamo Carlos y soy de Madrid.",
    "Buongiorno, come sta oggi?",
    "Bom dia, tudo bem com voce?",
    "Vandaag is het mooi weer in Nederland.",
    "Bonjour, je cherche la gare la plus proche.",
    "Ich habe gestern ein gutes Buch gelesen.",
    "Me gusta mucho la comida mexicana.",
    "Il ristorante e aperto fino alle dieci di sera.",
    "Eu gostaria de reservar uma mesa para dois.",
    "Wij gaan morgen naar het museum.",
    "Le temps est magnifique ce matin.",
    "Koennen Sie mir bitte helfen?",
    "Donde esta la biblioteca mas cercana?",
    "La vita e bella quando il sole splende.",
    "O Brasil e um pais muito bonito.",
    "De trein vertrekt om acht uur.",
    "Je voudrais un cafe au lait, s'il vous plait.",
    "Wir fahren naechste Woche in den Urlaub.",
]

_NUMBER_TEXTS_EXPANDED = [
    "The building has 42 floors and was built in 1998. It cost $350 million to construct.",
    "She ran the 26.2 mile marathon in 3 hours and 45 minutes, finishing in 127th place.",
    "The recipe calls for 2.5 cups of flour, 3 eggs, and 175 grams of sugar.",
    "Population grew from 8.3 million in 2010 to 9.1 million in 2020, an increase of 800,000.",
    "The telescope detected light from a star 4.2 light years away with a magnitude of -1.46.",
    "Revenue was $12.5 million in Q1, $14.3 million in Q2, and $11.8 million in Q3.",
    "The car accelerates from 0 to 60 mph in 3.2 seconds and has a top speed of 155 mph.",
    "Average temperature was 72 degrees with a high of 89 and a low of 54.",
    "The study involved 500 participants aged 18 to 65, with a mean age of 34.7 years.",
    "Stock rose 15.3 percent from 145.50 to 167.76 over the past 30 days.",
    "The bridge spans 1,280 feet and rises 220 feet above the water surface.",
    "Inflation hit 6.2 percent in 2022, up from 1.4 percent the previous year.",
    "Battery capacity is 4,500 mAh with a charging time of 1.5 hours at 65 watts.",
    "The garden covers 2.5 acres with over 300 species of plants and 15 fountains.",
    "Quarterly sales were 1,250 units at $49.99 each for total revenue of $62,487.50.",
    "The satellite orbits at 35,786 km altitude completing 1 revolution every 24 hours.",
    "Company has 1,200 employees across 8 offices with annual revenue of $45 million.",
    "Test scores ranged from 65 to 98, with a median of 82 and mean of 79.5.",
    "The lake is 12.4 km long, 3.8 km wide, and has a maximum depth of 125 meters.",
    "Budget allocated $5.2 million for research, $3.1 million for development, and $1.7 million for marketing.",
]

_INVOICE_CLIENTS = [
    "Acme Corporation", "TechStart Inc.", "Global Dynamics", "Spark Solutions",
    "NextGen Labs", "Blue Horizon", "Pinnacle Systems", "Vertex Analytics",
    "CloudBridge", "DataForge", "Stellar Innovations", "Quantum Leap",
    "Nova Industries", "Atlas Digital", "Meridian Group",
]

_INVOICE_ITEMS_POOL = [
    [{"description": "Web Development", "quantity": 40, "unit_price": 150.0}],
    [{"description": "Consulting Hours", "quantity": 10, "unit_price": 200.0},
     {"description": "Travel Expenses", "quantity": 1, "unit_price": 450.0}],
    [{"description": "Software License", "quantity": 5, "unit_price": 99.99}],
    [{"description": "Design Work", "quantity": 20, "unit_price": 125.0},
     {"description": "Logo Design", "quantity": 1, "unit_price": 500.0}],
    [{"description": "API Integration", "quantity": 1, "unit_price": 3500.0}],
    [{"description": "Monthly Hosting", "quantity": 12, "unit_price": 49.99}],
    [{"description": "Data Analysis", "quantity": 8, "unit_price": 175.0}],
    [{"description": "Training Session", "quantity": 3, "unit_price": 800.0}],
    [{"description": "Security Audit", "quantity": 1, "unit_price": 5000.0}],
    [{"description": "Content Writing", "quantity": 15, "unit_price": 75.0},
     {"description": "SEO Optimization", "quantity": 1, "unit_price": 1200.0}],
]

_REPORT_DATA_POOL = [
    '{"sales": 15000, "returns": 320, "customers": 1250}',
    '{"cpu_usage": 72.5, "memory_usage": 68.3, "disk_usage": 45.1}',
    '{"revenue": 125000, "expenses": 98000, "profit": 27000}',
    '{"users_active": 8500, "new_signups": 340, "churn_rate": 2.1}',
    '{"requests_per_sec": 1250, "avg_latency_ms": 45, "error_rate": 0.3}',
    '{"tasks_completed": 87, "tasks_pending": 13, "velocity": 42}',
    '{"tickets_open": 23, "tickets_closed": 156, "avg_resolution_hours": 4.5}',
    '{"page_views": 250000, "bounce_rate": 34.5, "avg_session_min": 3.2}',
]

_REPORT_TITLES = [
    "Monthly Sales Report", "System Health Dashboard", "Q1 Financial Summary",
    "User Activity Report", "Performance Metrics", "Sprint Review Summary",
    "Customer Support Report", "Website Analytics Report",
]

_BASE_URLS = [
    "https://www.example.com", "https://blog.company.io",
    "https://docs.platform.dev", "https://news.site.org",
    "https://api.service.com", "https://shop.store.net",
    "https://learn.academy.edu", "https://wiki.project.dev",
]

_ARTICLE_TITLES_FOR_SLUG = [
    "How to Build a REST API", "Top 10 Python Libraries for Data Science",
    "The Future of Artificial Intelligence", "A Guide to Machine Learning",
    "Understanding Quantum Computing", "Best Practices for Code Review",
    "Introduction to Kubernetes", "Why Remote Work Is Here to Stay",
    "Building Scalable Microservices", "The Art of Technical Writing",
    "Modern Web Development Tools", "Getting Started with Docker",
    "Deep Learning for Beginners", "Cloud Infrastructure Best Practices",
    "The Rise of Edge Computing", "Mastering Git Workflows",
    "Data Engineering Fundamentals", "Designing Accessible Web Apps",
    "Cybersecurity Essentials 2026", "Serverless Architecture Patterns",
]

_OUTDOOR_EVENT_TITLES = [
    "Park Picnic", "Outdoor Yoga Session", "Team BBQ", "Garden Party",
    "Morning Jog Meetup", "Open Air Concert", "Beach Volleyball",
    "Outdoor Photography Walk", "Farmers Market Visit", "Rooftop Brunch",
    "Bird Watching Excursion", "Outdoor Book Club", "Cycling Tour",
    "Sunset Hike", "Stargazing Night", "Outdoor Painting Class",
]

_ENCRYPTION_METHODS = ["aes256", "rsa"]


# ---------------------------------------------------------------------------
# Generator static methods
# ---------------------------------------------------------------------------


def _l1_word_count_format(engine: CompositionEngine) -> Task:
    """word_count -> format_number: count words then format the count nicely."""
    text = engine._pick(TEXTS_EXPANDED)
    wc = engine._execute_tool("word_count", text=text)
    count = wc.get("words", wc.get("word_count", 0))
    formatted = engine._execute_tool("format_number", value=count, format="comma")

    prompt = engine._format_prompt(
        "word_count_format", NEW_L1_PROMPT_VARIANTS,
        text=text[:60] + "...",
    )
    distractors = engine._pick_distractors(["word_count", "format_number"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["word_count", "format_number"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="word_count",
                    arguments={"text": text},
                    output_key="wc", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="format_number",
                    arguments={"value": count, "format": "comma"},
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=formatted,
        metadata={
            "category": "chain",
            "tools": ["word_count", "format_number"],
            "pattern": "count-format",
        },
    )


def _l1_keyword_search(engine: CompositionEngine) -> Task:
    """keyword_extract -> web_search: extract keywords, search for the top one."""
    text = engine._pick(TEXTS_EXPANDED)
    kw_result = engine._execute_tool("keyword_extract", text=text)
    keywords = kw_result.get("keywords", [])
    top_keyword = keywords[0] if keywords else "default"

    search_result = engine._execute_tool("web_search", query=top_keyword)

    prompt = engine._format_prompt(
        "keyword_search", NEW_L1_PROMPT_VARIANTS,
        text=text[:60] + "...",
    )
    distractors = engine._pick_distractors(["keyword_extract", "web_search"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["keyword_extract", "web_search"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="keyword_extract",
                    arguments={"text": text},
                    output_key="keywords", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="web_search",
                    arguments={"query": top_keyword},
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=search_result,
        metadata={
            "category": "chain",
            "tools": ["keyword_extract", "web_search"],
            "pattern": "extract-search",
        },
    )


def _l1_detect_translate(engine: CompositionEngine) -> Task:
    """detect_language -> translate_text: detect language then translate to English."""
    text = engine._pick(_MULTILINGUAL_TEXTS)
    detected = engine._execute_tool("detect_language", text=text)
    lang_code = detected.get("language_code", "und")

    translated = engine._execute_tool(
        "translate_text", text=text, from_language=lang_code, to_language="en",
    )

    prompt = engine._format_prompt(
        "detect_translate", NEW_L1_PROMPT_VARIANTS,
        text=text[:60] + "...",
    )
    distractors = engine._pick_distractors(["detect_language", "translate_text"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["detect_language", "translate_text"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="detect_language",
                    arguments={"text": text},
                    output_key="detected", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="translate_text",
                    arguments={
                        "text": text,
                        "from_language": lang_code,
                        "to_language": "en",
                    },
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=translated,
        metadata={
            "category": "chain",
            "tools": ["detect_language", "translate_text"],
            "pattern": "detect-translate",
            "source_language_code": lang_code,
        },
    )


def _l1_pctchange_format(engine: CompositionEngine) -> Task:
    """percentage_change -> format_number: compute % change then format result."""
    old_value = round(engine.rng.uniform(10, 500), 2)
    new_value = round(engine.rng.uniform(10, 500), 2)

    pct = engine._execute_tool(
        "percentage_change", old_value=old_value, new_value=new_value,
    )
    change = pct.get("change_percent", 0)
    formatted = engine._execute_tool(
        "format_number", value=change, format="percent",
    )

    prompt = engine._format_prompt(
        "pctchange_format", NEW_L1_PROMPT_VARIANTS,
        old=old_value, new=new_value,
    )
    distractors = engine._pick_distractors(["percentage_change", "format_number"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["percentage_change", "format_number"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="percentage_change",
                    arguments={"old_value": old_value, "new_value": new_value},
                    output_key="pct", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="format_number",
                    arguments={"value": change, "format": "percent"},
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=formatted,
        metadata={
            "category": "chain",
            "tools": ["percentage_change", "format_number"],
            "pattern": "compute-format",
        },
    )


def _l1_extract_nums_stats(engine: CompositionEngine) -> Task:
    """extract_numbers -> statistical_analysis: extract numbers, compute stats."""
    text = engine._pick(_NUMBER_TEXTS_EXPANDED)
    extracted = engine._execute_tool("extract_numbers", text=text)
    numbers = extracted.get("numbers", [])

    # statistical_analysis requires at least one number
    if not numbers:
        numbers = [0]
    stats = engine._execute_tool(
        "statistical_analysis", numbers=numbers, operation="summary",
    )

    prompt = engine._format_prompt(
        "extract_nums_stats", NEW_L1_PROMPT_VARIANTS,
        text=text[:60] + "...",
    )
    distractors = engine._pick_distractors(["extract_numbers", "statistical_analysis"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["extract_numbers", "statistical_analysis"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="extract_numbers",
                    arguments={"text": text},
                    output_key="extracted", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="statistical_analysis",
                    arguments={"numbers": numbers, "operation": "summary"},
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=stats,
        metadata={
            "category": "chain",
            "tools": ["extract_numbers", "statistical_analysis"],
            "pattern": "extract-analyze",
        },
    )


def _l1_parse_weekday(engine: CompositionEngine) -> Task:
    """parse_date -> get_weekday: parse a date string, then get the weekday."""
    date_str = engine._pick(_DATE_STRINGS_EXPANDED)
    parsed = engine._execute_tool("parse_date", date_string=date_str)
    iso_date = parsed.get("iso_date", "2026-01-01")

    weekday_result = engine._execute_tool("get_weekday", date=iso_date)

    prompt = engine._format_prompt(
        "parse_weekday", NEW_L1_PROMPT_VARIANTS,
        date_str=date_str,
    )
    distractors = engine._pick_distractors(["parse_date", "get_weekday"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["parse_date", "get_weekday"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="parse_date",
                    arguments={"date_string": date_str},
                    output_key="parsed", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="get_weekday",
                    arguments={"date": iso_date},
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=weekday_result,
        metadata={
            "category": "chain",
            "tools": ["parse_date", "get_weekday"],
            "pattern": "parse-lookup",
        },
    )


def _l1_read_keywords(engine: CompositionEngine) -> Task:
    """read_file -> keyword_extract: read file contents, extract keywords."""
    # Only pick from files known to exist in the virtual FS
    path = engine._pick(_READABLE_PATHS)
    content = engine._execute_tool("read_file", path=path)
    raw = content.get("content")
    content_str = (raw if raw else json.dumps(content, default=str))[:500]

    keywords = engine._execute_tool("keyword_extract", text=content_str)

    prompt = engine._format_prompt(
        "read_keywords", NEW_L1_PROMPT_VARIANTS,
        path=path,
    )
    distractors = engine._pick_distractors(["read_file", "keyword_extract"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["read_file", "keyword_extract"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="read_file",
                    arguments={"path": path},
                    output_key="content", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="keyword_extract",
                    arguments={"text": content_str},
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=keywords,
        metadata={
            "category": "chain",
            "tools": ["read_file", "keyword_extract"],
            "pattern": "read-extract",
        },
    )


def _l1_read_wordcount(engine: CompositionEngine) -> Task:
    """read_file -> word_count: read file, count words."""
    # Only pick from files known to exist in the virtual FS
    path = engine._pick(_READABLE_PATHS)
    content = engine._execute_tool("read_file", path=path)
    raw = content.get("content")
    content_str = (raw if raw else json.dumps(content, default=str))[:500]

    wc = engine._execute_tool("word_count", text=content_str)

    prompt = engine._format_prompt(
        "read_wordcount", NEW_L1_PROMPT_VARIANTS,
        path=path,
    )
    distractors = engine._pick_distractors(["read_file", "word_count"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["read_file", "word_count"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="read_file",
                    arguments={"path": path},
                    output_key="content", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="word_count",
                    arguments={"text": content_str},
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=wc,
        metadata={
            "category": "chain",
            "tools": ["read_file", "word_count"],
            "pattern": "read-count",
        },
    )


def _l1_weather_log(engine: CompositionEngine) -> Task:
    """get_weather -> log_event: get weather, log it as an event."""
    city = engine._pick(CITIES_EXPANDED)
    weather = engine._execute_tool("get_weather", city=city)
    weather_msg = (
        f"Weather in {city}: {weather.get('condition', 'N/A')}, "
        f"{weather.get('temperature_celsius', 'N/A')}°C"
    )

    logged = engine._execute_tool(
        "log_event",
        event_type="weather_check",
        message=weather_msg,
        severity="info",
    )

    prompt = engine._format_prompt(
        "weather_log", NEW_L1_PROMPT_VARIANTS,
        city=city,
    )
    distractors = engine._pick_distractors(["get_weather", "log_event"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["get_weather", "log_event"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="get_weather",
                    arguments={"city": city},
                    output_key="weather", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="log_event",
                    arguments={
                        "event_type": "weather_check",
                        "message": weather_msg,
                        "severity": "info",
                    },
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=logged,
        metadata={
            "category": "chain",
            "tools": ["get_weather", "log_event"],
            "pattern": "retrieve-log",
        },
    )


def _l1_calc_round(engine: CompositionEngine) -> Task:
    """calculator -> round_number: calculate expression, round the result."""
    a = engine.rng.randint(10, 999)
    b = engine.rng.randint(2, 99)
    op = engine._pick(["/", "*", "+", "-"])
    expr = f"{a} {op} {b}"
    decimals = engine._pick([0, 1, 2, 3])

    calc_result = engine._execute_tool("calculator", expression=expr)
    result_value = calc_result.get("result", calc_result.get("value", 0))

    rounded = engine._execute_tool(
        "round_number", value=float(result_value), decimals=decimals,
    )

    prompt = engine._format_prompt(
        "calc_round", NEW_L1_PROMPT_VARIANTS,
        expr=expr, decimals=decimals,
    )
    distractors = engine._pick_distractors(["calculator", "round_number"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["calculator", "round_number"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="calculator",
                    arguments={"expression": expr},
                    output_key="calc", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="round_number",
                    arguments={
                        "value": float(result_value),
                        "decimals": decimals,
                    },
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=rounded,
        metadata={
            "category": "chain",
            "tools": ["calculator", "round_number"],
            "pattern": "compute-round",
        },
    )


def _l1_extract_nums_minmax(engine: CompositionEngine) -> Task:
    """extract_numbers -> min_max: extract numbers from text, find min and max."""
    text = engine._pick(_NUMBER_TEXTS_EXPANDED)
    extracted = engine._execute_tool("extract_numbers", text=text)
    numbers = extracted.get("numbers", [])

    if not numbers:
        numbers = [0]
    minmax = engine._execute_tool("min_max", numbers=numbers)

    prompt = engine._format_prompt(
        "extract_nums_minmax", NEW_L1_PROMPT_VARIANTS,
        text=text[:60] + "...",
    )
    distractors = engine._pick_distractors(["extract_numbers", "min_max"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["extract_numbers", "min_max"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="extract_numbers",
                    arguments={"text": text},
                    output_key="extracted", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="min_max",
                    arguments={"numbers": numbers},
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=minmax,
        metadata={
            "category": "chain",
            "tools": ["extract_numbers", "min_max"],
            "pattern": "extract-aggregate",
        },
    )


def _l1_invoice_email(engine: CompositionEngine) -> Task:
    """create_invoice -> send_email: create invoice, email it."""
    client = engine._pick(_INVOICE_CLIENTS)
    items = engine._pick(_INVOICE_ITEMS_POOL)
    email = engine._pick(EMAIL_RECIPIENTS_EXPANDED)
    currency = engine._pick(["USD", "EUR", "GBP"])

    invoice = engine._execute_tool(
        "create_invoice", client_name=client, items=items, currency=currency,
    )
    invoice_summary = (
        f"Invoice {invoice.get('invoice_id', 'N/A')} for {client}: "
        f"{invoice.get('total', 0)} {currency}"
    )
    sent = engine._execute_tool(
        "send_email",
        to=email,
        subject=f"Invoice for {client}",
        body=invoice_summary,
    )

    prompt = engine._format_prompt(
        "invoice_email", NEW_L1_PROMPT_VARIANTS,
        client=client, email=email,
    )
    distractors = engine._pick_distractors(["create_invoice", "send_email"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["create_invoice", "send_email"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="create_invoice",
                    arguments={
                        "client_name": client,
                        "items": items,
                        "currency": currency,
                    },
                    output_key="invoice", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="send_email",
                    arguments={
                        "to": email,
                        "subject": f"Invoice for {client}",
                        "body": invoice_summary,
                    },
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=sent,
        metadata={
            "category": "chain",
            "tools": ["create_invoice", "send_email"],
            "pattern": "create-send",
        },
    )


def _l1_report_save(engine: CompositionEngine) -> Task:
    """generate_report -> write_file: generate report, save to file."""
    title = engine._pick(_REPORT_TITLES)
    data = engine._pick(_REPORT_DATA_POOL)
    fmt = engine._pick(["text", "markdown"])
    path = f"/data/{title.lower().replace(' ', '_')}.{'md' if fmt == 'markdown' else 'txt'}"

    report = engine._execute_tool(
        "generate_report", title=title, data=data, format=fmt,
    )
    report_text = report.get("report", json.dumps(report, default=str))[:500]

    written = engine._execute_tool("write_file", path=path, content=report_text)

    prompt = engine._format_prompt(
        "report_save", NEW_L1_PROMPT_VARIANTS,
        title=title, path=path,
    )
    distractors = engine._pick_distractors(["generate_report", "write_file"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["generate_report", "write_file"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="generate_report",
                    arguments={"title": title, "data": data, "format": fmt},
                    output_key="report", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="write_file",
                    arguments={"path": path, "content": report_text},
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=written,
        metadata={
            "category": "chain",
            "tools": ["generate_report", "write_file"],
            "pattern": "generate-save",
        },
    )


def _l1_encrypt_save(engine: CompositionEngine) -> Task:
    """encrypt_text -> write_file: encrypt text, save encrypted version."""
    text = engine._pick(TEXTS_EXPANDED)[:200]
    method = engine._pick(_ENCRYPTION_METHODS)
    path = f"/data/encrypted_{engine.rng.randint(100, 999)}.enc"

    encrypted = engine._execute_tool("encrypt_text", text=text, method=method)
    encrypted_str = encrypted.get("encrypted", "")

    written = engine._execute_tool("write_file", path=path, content=encrypted_str)

    prompt = engine._format_prompt(
        "encrypt_save", NEW_L1_PROMPT_VARIANTS,
        text=text[:50] + "...", method=method, path=path,
    )
    distractors = engine._pick_distractors(["encrypt_text", "write_file"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["encrypt_text", "write_file"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="encrypt_text",
                    arguments={"text": text, "method": method},
                    output_key="encrypted", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="write_file",
                    arguments={"path": path, "content": encrypted_str},
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=written,
        metadata={
            "category": "chain",
            "tools": ["encrypt_text", "write_file"],
            "pattern": "transform-save",
        },
    )


def _l1_slugify_url(engine: CompositionEngine) -> Task:
    """slugify -> generate_url: create slug from title, generate URL with it."""
    title = engine._pick(_ARTICLE_TITLES_FOR_SLUG)
    base = engine._pick(_BASE_URLS)

    slug_result = engine._execute_tool("slugify", text=title)
    slug = slug_result.get("slug", "untitled")

    url_result = engine._execute_tool(
        "generate_url",
        base=base,
        path=f"/articles/{slug}",
        params="{}",
    )

    prompt = engine._format_prompt(
        "slugify_url", NEW_L1_PROMPT_VARIANTS,
        title=title, base=base,
    )
    distractors = engine._pick_distractors(["slugify", "generate_url"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["slugify", "generate_url"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="slugify",
                    arguments={"text": title},
                    output_key="slug", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="generate_url",
                    arguments={
                        "base": base,
                        "path": f"/articles/{slug}",
                        "params": "{}",
                    },
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=url_result,
        metadata={
            "category": "chain",
            "tools": ["slugify", "generate_url"],
            "pattern": "transform-construct",
        },
    )


def _l1_entity_mask(engine: CompositionEngine) -> Task:
    """extract_entities -> mask_pii: extract entities, then mask PII in the text."""
    text = engine._pick(ENTITY_TEXTS_EXPANDED)

    entities = engine._execute_tool("extract_entities", text=text)
    masked = engine._execute_tool("mask_pii", text=text)

    prompt = engine._format_prompt(
        "entity_mask", NEW_L1_PROMPT_VARIANTS,
        text=text[:80] + "...",
    )
    distractors = engine._pick_distractors(["extract_entities", "mask_pii"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["extract_entities", "mask_pii"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="extract_entities",
                    arguments={"text": text},
                    output_key="entities", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="mask_pii",
                    arguments={"text": text},
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=masked,
        metadata={
            "category": "chain",
            "tools": ["extract_entities", "mask_pii"],
            "pattern": "analyze-redact",
        },
    )


def _l1_stock_pctchange(engine: CompositionEngine) -> Task:
    """get_stock_price -> percentage_change: get price, compute % change from base."""
    symbol = engine._pick(STOCKS_EXPANDED)
    base_price = round(engine.rng.uniform(50, 400), 2)

    stock = engine._execute_tool("get_stock_price", symbol=symbol)
    current_price = stock.get("price", stock.get("price_usd", 100.0))

    pct = engine._execute_tool(
        "percentage_change",
        old_value=base_price,
        new_value=float(current_price),
    )

    prompt = engine._format_prompt(
        "stock_pctchange", NEW_L1_PROMPT_VARIANTS,
        symbol=symbol, base_price=base_price,
    )
    distractors = engine._pick_distractors(["get_stock_price", "percentage_change"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["get_stock_price", "percentage_change"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="get_stock_price",
                    arguments={"symbol": symbol},
                    output_key="stock", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="percentage_change",
                    arguments={
                        "old_value": base_price,
                        "new_value": float(current_price),
                    },
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=pct,
        metadata={
            "category": "chain",
            "tools": ["get_stock_price", "percentage_change"],
            "pattern": "retrieve-compute",
        },
    )


def _l1_weather_calendar(engine: CompositionEngine) -> Task:
    """get_weather -> create_calendar_event: get weather, create outdoor event."""
    city = engine._pick(CITIES_EXPANDED)
    event_title = engine._pick(_OUTDOOR_EVENT_TITLES)
    event_date = engine._pick([
        "2026-03-15 10:00", "2026-04-20 14:00", "2026-05-10 09:00",
        "2026-06-01 11:00", "2026-07-15 08:00", "2026-08-22 16:00",
        "2026-09-05 10:00", "2026-10-12 13:00",
    ])

    weather = engine._execute_tool("get_weather", city=city)
    condition = weather.get("condition", "clear")
    temp = weather.get("temperature_celsius", 20)
    event_desc = f"{event_title} in {city} - Forecast: {condition}, {temp}°C"

    cal_event = engine._execute_tool(
        "create_calendar_event",
        title=event_desc,
        date=event_date,
        duration_minutes=120,
    )

    prompt = engine._format_prompt(
        "weather_calendar", NEW_L1_PROMPT_VARIANTS,
        city=city, event_title=event_title,
    )
    distractors = engine._pick_distractors(["get_weather", "create_calendar_event"])
    return Task(
        task_id=engine._next_id(CompositionLevel.CHAIN),
        level=CompositionLevel.CHAIN,
        prompt=prompt,
        available_tools=["get_weather", "create_calendar_event"] + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1", tool_name="get_weather",
                    arguments={"city": city},
                    output_key="weather", depends_on=[],
                ),
                ToolCall(
                    step_id="step_2", tool_name="create_calendar_event",
                    arguments={
                        "title": event_desc,
                        "date": event_date,
                        "duration_minutes": 120,
                    },
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_2",
        ),
        expected_final_answer=cal_event,
        metadata={
            "category": "chain",
            "tools": ["get_weather", "create_calendar_event"],
            "pattern": "retrieve-schedule",
        },
    )


# ---------------------------------------------------------------------------
# Registry: name -> function reference (to be merged into _l1_generators)
# ---------------------------------------------------------------------------

NEW_L1_GENERATORS: dict[str, Any] = {
    "word_count_format": _l1_word_count_format,
    "keyword_search": _l1_keyword_search,
    "detect_translate": _l1_detect_translate,
    "pctchange_format": _l1_pctchange_format,
    "extract_nums_stats": _l1_extract_nums_stats,
    "parse_weekday": _l1_parse_weekday,
    "read_keywords": _l1_read_keywords,
    "read_wordcount": _l1_read_wordcount,
    "weather_log": _l1_weather_log,
    "calc_round": _l1_calc_round,
    "extract_nums_minmax": _l1_extract_nums_minmax,
    "invoice_email": _l1_invoice_email,
    "report_save": _l1_report_save,
    "encrypt_save": _l1_encrypt_save,
    "slugify_url": _l1_slugify_url,
    "entity_mask": _l1_entity_mask,
    "stock_pctchange": _l1_stock_pctchange,
    "weather_calendar": _l1_weather_calendar,
}

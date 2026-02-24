"""Task generator for CompToolBench.

Generates benchmark tasks at all 4 composition levels using
parameterized templates. Each task has:
- A natural language prompt
- A list of available tools
- An expected execution trace (ground truth)
- A verifiable expected answer
"""

from __future__ import annotations

import json
import random
from typing import Any

from comptoolbench.tasks.models import (
    CompositionLevel,
    ExpectedTrace,
    Task,
    TaskSuite,
    ToolCall,
)
from comptoolbench.tools import ToolMode, get_all_tools, get_tool


# ---------------------------------------------------------------------------
# Parameter pools for task generation (rotatable for contamination resistance)
# ---------------------------------------------------------------------------

CITIES = [
    "Tokyo", "Paris", "London", "New York", "Sydney",
    "Berlin", "Mumbai", "Dubai", "Toronto", "Singapore",
    "Seoul", "Beijing", "Cairo", "Moscow", "Rome",
    "San Francisco", "Los Angeles", "Chicago",
]

CURRENCIES = ["USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "CNY", "INR"]

STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "TSLA", "META"]

TEMP_UNITS = [("celsius", "fahrenheit"), ("fahrenheit", "celsius"), ("celsius", "kelvin")]

DISTANCE_UNITS = [("km", "miles"), ("miles", "km"), ("meters", "feet")]

WEIGHT_UNITS = [("kg", "lbs"), ("lbs", "kg")]

DATE_PAIRS = [
    ("2026-01-01", "2026-06-30"),
    ("2025-03-15", "2026-03-15"),
    ("2026-02-14", "2026-12-25"),
    ("2025-07-04", "2026-07-04"),
    ("2026-01-01", "2026-12-31"),
]

SEARCH_QUERIES = [
    "renewable energy trends 2026",
    "machine learning best practices",
    "climate change solutions",
    "artificial intelligence in healthcare",
    "space exploration missions",
    "quantum computing applications",
    "sustainable agriculture technology",
]

CONTINENTS = ["Asia", "Europe", "North America", "South America", "Oceania"]

SENTIMENTS_TEXTS = [
    ("This product is amazing and wonderful, I absolutely love it!", "positive"),
    ("Terrible experience, the worst service I have ever received.", "negative"),
    ("The meeting is scheduled for Tuesday at 3pm in room 204.", "neutral"),
    ("Great quality and fast shipping, highly recommend!", "positive"),
    ("Disappointing and frustrating, complete waste of money.", "negative"),
]

# Additional parameter pools for expanded templates
TIMEZONES = ["US/Eastern", "US/Pacific", "Europe/London", "Asia/Tokyo", "Australia/Sydney"]

TRAVEL_MODES = ["driving", "walking", "transit", "cycling"]

PRODUCT_CATEGORIES = ["electronics", "books", "clothing", "home", "sports"]

CLASSIFICATION_CATEGORIES = [
    (["finance", "sports", "technology"], "The stock market rallied today with tech stocks leading gains"),
    (["politics", "science", "entertainment"], "NASA announced a new Mars mission scheduled for 2028"),
    (["health", "education", "business"], "The university launched a new online MBA program this fall"),
]

ENTITY_TEXTS = [
    "Contact Dr. Sarah Johnson at sjohnson@university.edu about the meeting on 03/15/2026.",
    "Albert Einstein was born in Ulm, Germany on March 14, 1879.",
    "Please send invoices to billing@acme.com by 12/31/2026. Call 555-0123 for questions.",
]

LONG_TEXTS = [
    "Artificial intelligence has made remarkable progress in recent years. Machine learning models can now perform complex tasks that were once thought impossible. From natural language processing to computer vision, AI is transforming every industry. However, challenges remain in areas like fairness, interpretability, and alignment with human values. Researchers continue to push boundaries while also addressing these critical concerns.",
    "Climate change represents one of the greatest challenges facing humanity. Rising temperatures have led to more frequent extreme weather events, melting ice caps, and rising sea levels. Scientists worldwide are working on solutions including renewable energy, carbon capture, and sustainable agriculture. The transition to a green economy requires coordinated global action and significant investment in clean technologies.",
    "The global economy is undergoing a period of significant transformation. Digital technologies are reshaping traditional business models and creating new opportunities. Remote work has become mainstream, changing how companies operate and how employees balance work and life. Supply chain disruptions have highlighted the need for more resilient and diversified economic systems.",
]

KB_QUERIES = [
    "What is the capital of France?",
    "What is the speed of light?",
    "What is the population of Tokyo?",
    "Who invented the telephone?",
]

MEMORY_KEYS = ["user_preferences", "project_notes", "meeting_agenda", "shopping_list"]

FILE_PATHS = ["/data/report.txt", "/data/config.json", "/data/employees.csv"]

MEETING_TITLES = ["Sprint Planning", "Budget Review", "Team Standup", "Design Review"]

NOTIFICATION_MESSAGES = [
    "Your build completed successfully",
    "New pull request requires review",
    "Deployment to staging finished",
    "Weekly report is ready",
]

LANGUAGES = ["Spanish", "French", "German", "Japanese", "Chinese"]
LANGUAGE_CODES = {"Spanish": "es", "French": "fr", "German": "de", "Japanese": "ja", "Chinese": "zh"}


class TaskGenerator:
    """Generates benchmark tasks at all 4 composition levels."""

    def __init__(self, seed: int = 42, mode: ToolMode = ToolMode.SIMULATED) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.mode = mode
        self._task_counter = 0

    def _next_id(self, level: CompositionLevel) -> str:
        self._task_counter += 1
        return f"{level.value}_{self._task_counter:04d}"

    def _pick(self, items: list[Any]) -> Any:
        return self.rng.choice(items)

    def _pick_n(self, items: list[Any], n: int) -> list[Any]:
        return self.rng.sample(items, min(n, len(items)))

    def _execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool to get the ground truth answer."""
        tool_cls = get_tool(tool_name)
        tool = tool_cls(mode=self.mode)
        result = tool.execute(**kwargs)
        if result.success:
            return result.data
        raise RuntimeError(f"Tool {tool_name} failed: {result.error}")

    # -----------------------------------------------------------------------
    # Level 0: Node (single tool)
    # -----------------------------------------------------------------------

    def generate_l0_tasks(self, count: int = 50) -> list[Task]:
        """Generate single-tool tasks covering all 43 tools."""
        tasks: list[Task] = []
        generators = [
            # Original 10
            self._l0_weather,
            self._l0_calculator,
            self._l0_unit_convert,
            self._l0_stock_price,
            self._l0_exchange_rate,
            self._l0_sentiment,
            self._l0_date_diff,
            self._l0_database_query,
            self._l0_search,
            self._l0_entity_lookup,
            # New 25 (cover ALL remaining tools)
            self._l0_classify_text,
            self._l0_convert_timezone,
            self._l0_create_notification,
            self._l0_data_aggregate,
            self._l0_data_filter,
            self._l0_execute_python,
            self._l0_extract_entities,
            self._l0_get_directions,
            self._l0_get_location_info,
            self._l0_get_session_context,
            self._l0_knowledge_base_query,
            self._l0_list_files,
            self._l0_list_memories,
            self._l0_merge_data,
            self._l0_read_file,
            self._l0_retrieve_memory,
            self._l0_schedule_meeting,
            self._l0_search_products,
            self._l0_send_message,
            self._l0_store_memory,
            self._l0_summarize_text,
            self._l0_compare_texts,
            self._l0_translate_text,
            self._l0_transform_format,
            self._l0_write_file,
            self._l0_generate_image,
            self._l0_transcribe_audio,
            self._l0_web_page_fetch,
        ]

        per_generator = max(count // len(generators), 1)
        for gen in generators:
            for _ in range(per_generator):
                task = gen()
                if task:
                    tasks.append(task)
                if len(tasks) >= count:
                    break
            if len(tasks) >= count:
                break

        return tasks[:count]

    def _l0_weather(self) -> Task:
        city = self._pick(CITIES)
        answer = self._execute_tool("get_weather", city=city)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"What is the current weather in {city}?",
            available_tools=["get_weather", "calculator", "web_search"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="get_weather", arguments={"city": city})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "external_services", "tool": "get_weather"},
        )

    def _l0_calculator(self) -> Task:
        a, b = self.rng.randint(10, 999), self.rng.randint(2, 99)
        op = self._pick(["+", "-", "*"])
        expr = f"{a} {op} {b}"
        answer = self._execute_tool("calculator", expression=expr)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"What is {expr}?",
            available_tools=["calculator", "unit_convert", "get_weather"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="calculator", arguments={"expression": expr})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "computation", "tool": "calculator"},
        )

    def _l0_unit_convert(self) -> Task:
        value = round(self.rng.uniform(1, 100), 1)
        from_u, to_u = self._pick(TEMP_UNITS + DISTANCE_UNITS + WEIGHT_UNITS)
        answer = self._execute_tool("unit_convert", value=value, from_unit=from_u, to_unit=to_u)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"Convert {value} {from_u} to {to_u}.",
            available_tools=["unit_convert", "calculator", "get_weather"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="unit_convert", arguments={"value": value, "from_unit": from_u, "to_unit": to_u})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "computation", "tool": "unit_convert"},
        )

    def _l0_stock_price(self) -> Task:
        symbol = self._pick(STOCKS)
        answer = self._execute_tool("get_stock_price", symbol=symbol)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"What is the current stock price of {symbol}?",
            available_tools=["get_stock_price", "calculator", "web_search"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="get_stock_price", arguments={"symbol": symbol})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "external_services", "tool": "get_stock_price"},
        )

    def _l0_exchange_rate(self) -> Task:
        from_c, to_c = self._pick_n(CURRENCIES, 2)
        amount = self.rng.choice([1, 10, 50, 100, 500, 1000])
        answer = self._execute_tool("get_exchange_rate", from_currency=from_c, to_currency=to_c, amount=amount)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"Convert {amount} {from_c} to {to_c}.",
            available_tools=["get_exchange_rate", "calculator", "unit_convert"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="get_exchange_rate", arguments={"from_currency": from_c, "to_currency": to_c, "amount": amount})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "external_services", "tool": "get_exchange_rate"},
        )

    def _l0_sentiment(self) -> Task:
        text, expected_sentiment = self._pick(SENTIMENTS_TEXTS)
        answer = self._execute_tool("sentiment_analysis", text=text)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f'Analyze the sentiment of this text: "{text}"',
            available_tools=["sentiment_analysis", "summarize_text", "classify_text"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="sentiment_analysis", arguments={"text": text})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "text_processing", "tool": "sentiment_analysis", "expected_sentiment": expected_sentiment},
        )

    def _l0_date_diff(self) -> Task:
        d1, d2 = self._pick(DATE_PAIRS)
        answer = self._execute_tool("calculate_date_diff", date1=d1, date2=d2)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"How many days are between {d1} and {d2}?",
            available_tools=["calculate_date_diff", "calculator", "get_current_time"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="calculate_date_diff", arguments={"date1": d1, "date2": d2})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "time_scheduling", "tool": "calculate_date_diff"},
        )

    def _l0_database_query(self) -> Task:
        continent = self._pick(CONTINENTS)
        answer = self._execute_tool(
            "database_query", table="countries",
            filter_field="continent", filter_op="equals", filter_value=continent,
            sort_by="gdp_trillion", limit=5,
        )
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"List the countries in {continent} sorted by GDP.",
            available_tools=["database_query", "data_sort", "calculator"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(
                    step_id="step_1", tool_name="database_query",
                    arguments={"table": "countries", "filter_field": "continent", "filter_op": "equals", "filter_value": continent, "sort_by": "gdp_trillion", "limit": 5},
                )],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "information_retrieval", "tool": "database_query"},
        )

    def _l0_search(self) -> Task:
        query = self._pick(SEARCH_QUERIES)
        answer = self._execute_tool("web_search", query=query, max_results=3)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f'Search the web for: "{query}"',
            available_tools=["web_search", "summarize_text", "lookup_entity"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="web_search", arguments={"query": query, "max_results": 3})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "information_retrieval", "tool": "web_search"},
        )

    def _l0_entity_lookup(self) -> Task:
        entities = ["Python programming", "Albert Einstein", "Tokyo", "Machine learning", "Climate change"]
        entity = self._pick(entities)
        answer = self._execute_tool("lookup_entity", entity=entity)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"Look up information about {entity}.",
            available_tools=["lookup_entity", "web_search", "summarize_text"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="lookup_entity", arguments={"entity": entity})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "information_retrieval", "tool": "lookup_entity"},
        )

    # --- New L0 generators for expanded tool coverage ---

    def _l0_classify_text(self) -> Task:
        cats, text = self._pick(CLASSIFICATION_CATEGORIES)
        answer = self._execute_tool("classify_text", text=text, categories=cats)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f'Classify this text into one of {cats}: "{text}"',
            available_tools=["classify_text", "sentiment_analysis", "summarize_text"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="classify_text", arguments={"text": text, "categories": cats})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "text_processing", "tool": "classify_text"},
        )

    def _l0_convert_timezone(self) -> Task:
        tz_from, tz_to = self._pick_n(TIMEZONES, 2)
        time_str = self._pick(["14:00:00", "09:30:00", "18:45:00", "08:00:00"])
        datetime_str = f"2026-03-15T{time_str}"
        answer = self._execute_tool("convert_timezone", datetime_str=datetime_str, from_timezone=tz_from, to_timezone=tz_to)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"Convert {time_str} on 2026-03-15 from {tz_from} to {tz_to}.",
            available_tools=["convert_timezone", "get_current_time", "calculator"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="convert_timezone", arguments={"datetime_str": datetime_str, "from_timezone": tz_from, "to_timezone": tz_to})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "time_scheduling", "tool": "convert_timezone"},
        )

    def _l0_create_notification(self) -> Task:
        msg = self._pick(NOTIFICATION_MESSAGES)
        title = "Alert"
        answer = self._execute_tool("create_notification", title=title, message=msg, priority="normal")
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f'Create a notification titled "Alert" with message: "{msg}"',
            available_tools=["create_notification", "send_email", "send_message"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="create_notification", arguments={"title": title, "message": msg, "priority": "normal"})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "communication", "tool": "create_notification"},
        )

    def _l0_data_aggregate(self) -> Task:
        items = [{"dept": "eng", "salary": 120000}, {"dept": "eng", "salary": 130000}, {"dept": "sales", "salary": 95000}, {"dept": "sales", "salary": 105000}]
        answer = self._execute_tool("data_aggregate", items=items, group_by="dept", value_field="salary", operation="average")
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt="Calculate the average salary by department from this data: engineering (120K, 130K) and sales (95K, 105K).",
            available_tools=["data_aggregate", "calculator", "statistical_analysis"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="data_aggregate", arguments={"items": items, "group_by": "dept", "value_field": "salary", "operation": "average"})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "computation", "tool": "data_aggregate"},
        )

    def _l0_data_filter(self) -> Task:
        items = [30, 25, 35, 22, 40]
        answer = self._execute_tool("data_filter", items=items, condition="greater_than", value="28")
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt="Filter the list [30, 25, 35, 22, 40] to keep only numbers greater than 28.",
            available_tools=["data_filter", "data_sort", "statistical_analysis"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="data_filter", arguments={"items": items, "condition": "greater_than", "value": "28"})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "computation", "tool": "data_filter"},
        )

    def _l0_execute_python(self) -> Task:
        expressions = [
            ("sum(range(1, 11))", "Sum of numbers 1 to 10"),
            ("max([3, 7, 1, 9, 4])", "Find the maximum of [3, 7, 1, 9, 4]"),
            ("len('hello world')", "How many characters in 'hello world'?"),
            ("2 ** 10", "What is 2 to the power of 10?"),
        ]
        code, prompt = self._pick(expressions)
        answer = self._execute_tool("execute_python", code=code)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=prompt,
            available_tools=["execute_python", "calculator", "statistical_analysis"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="execute_python", arguments={"code": code})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "computation", "tool": "execute_python"},
        )

    def _l0_extract_entities(self) -> Task:
        text = self._pick(ENTITY_TEXTS)
        answer = self._execute_tool("extract_entities", text=text)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f'Extract all entities (names, emails, dates) from: "{text}"',
            available_tools=["extract_entities", "sentiment_analysis", "classify_text"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="extract_entities", arguments={"text": text})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "text_processing", "tool": "extract_entities"},
        )

    def _l0_get_directions(self) -> Task:
        origins = ["New York", "London", "Tokyo", "San Francisco"]
        destinations = ["Boston", "Paris", "Osaka", "Los Angeles"]
        idx = self.rng.randint(0, len(origins) - 1)
        origin, dest = origins[idx], destinations[idx]
        mode = self._pick(TRAVEL_MODES)
        answer = self._execute_tool("get_directions", origin=origin, destination=dest, mode=mode)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"Get {mode} directions from {origin} to {dest}.",
            available_tools=["get_directions", "get_location_info", "calculator"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="get_directions", arguments={"origin": origin, "destination": dest, "mode": mode})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "external_services", "tool": "get_directions"},
        )

    def _l0_get_location_info(self) -> Task:
        city = self._pick(CITIES)
        answer = self._execute_tool("get_location_info", location=city)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"Get location information (coordinates, timezone) for {city}.",
            available_tools=["get_location_info", "get_weather", "get_directions"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="get_location_info", arguments={"location": city})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "external_services", "tool": "get_location_info"},
        )

    def _l0_get_session_context(self) -> Task:
        answer = self._execute_tool("get_session_context")
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt="Get the current session context and memory count.",
            available_tools=["get_session_context", "list_memories", "retrieve_memory"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="get_session_context", arguments={})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "state_management", "tool": "get_session_context"},
        )

    def _l0_knowledge_base_query(self) -> Task:
        query = self._pick(KB_QUERIES)
        answer = self._execute_tool("knowledge_base_query", query=query)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=query,
            available_tools=["knowledge_base_query", "web_search", "lookup_entity"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="knowledge_base_query", arguments={"query": query})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "information_retrieval", "tool": "knowledge_base_query"},
        )

    def _l0_list_files(self) -> Task:
        directory = self._pick(["/", "/data"])
        answer = self._execute_tool("list_files", directory=directory)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"List all files in the {directory} directory.",
            available_tools=["list_files", "read_file", "write_file"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="list_files", arguments={"directory": directory})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "file_data", "tool": "list_files"},
        )

    def _l0_list_memories(self) -> Task:
        answer = self._execute_tool("list_memories")
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt="List all stored memories.",
            available_tools=["list_memories", "retrieve_memory", "store_memory"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="list_memories", arguments={})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "state_management", "tool": "list_memories"},
        )

    def _l0_merge_data(self) -> Task:
        ds1 = '[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]'
        ds2 = '[{"id": 1, "score": 95}, {"id": 2, "score": 87}]'
        answer = self._execute_tool("merge_data", dataset1=ds1, dataset2=ds2, join_key="id")
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt="Merge the employee names dataset with their scores on the 'id' field.",
            available_tools=["merge_data", "data_sort", "data_filter"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="merge_data", arguments={"dataset1": ds1, "dataset2": ds2, "join_key": "id"})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "file_data", "tool": "merge_data"},
        )

    def _l0_read_file(self) -> Task:
        path = self._pick(FILE_PATHS)
        answer = self._execute_tool("read_file", path=path)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"Read the contents of {path}.",
            available_tools=["read_file", "write_file", "list_files"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="read_file", arguments={"path": path})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "file_data", "tool": "read_file"},
        )

    def _l0_retrieve_memory(self) -> Task:
        key = self._pick(MEMORY_KEYS)
        answer = self._execute_tool("retrieve_memory", key=key)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"Retrieve the stored value for key '{key}'.",
            available_tools=["retrieve_memory", "store_memory", "list_memories"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="retrieve_memory", arguments={"key": key})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "state_management", "tool": "retrieve_memory"},
        )

    def _l0_schedule_meeting(self) -> Task:
        title = self._pick(MEETING_TITLES)
        dt = self._pick(["2026-03-15 10:00", "2026-04-01 14:00", "2026-05-20 09:30"])
        duration = self._pick([30, 60, 90])
        answer = self._execute_tool("schedule_meeting", title=title, datetime_str=dt, duration_minutes=duration, participants=["alice@co.com", "bob@co.com"])
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f'Schedule a {duration}-minute meeting "{title}" on {dt} with alice@co.com and bob@co.com.',
            available_tools=["schedule_meeting", "send_email", "create_task"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="schedule_meeting", arguments={"title": title, "datetime_str": dt, "duration_minutes": duration, "participants": ["alice@co.com", "bob@co.com"]})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "communication", "tool": "schedule_meeting"},
        )

    def _l0_search_products(self) -> Task:
        query = self._pick(["laptop", "wireless headphones", "running shoes", "coffee maker"])
        cat = self._pick(PRODUCT_CATEGORIES)
        answer = self._execute_tool("search_products", query=query, category=cat, max_results=3)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f'Search for "{query}" in the {cat} category.',
            available_tools=["search_products", "web_search", "calculator"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="search_products", arguments={"query": query, "category": cat, "max_results": 3})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "external_services", "tool": "search_products"},
        )

    def _l0_send_message(self) -> Task:
        recipient = self._pick(["alice", "bob", "charlie", "dev-team"])
        msg = self._pick(["Hello, how are you?", "Meeting postponed to Friday.", "Code review is ready."])
        answer = self._execute_tool("send_message", recipient=recipient, message=msg)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f'Send a message to {recipient}: "{msg}"',
            available_tools=["send_message", "send_email", "create_notification"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="send_message", arguments={"recipient": recipient, "message": msg})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "communication", "tool": "send_message"},
        )

    def _l0_store_memory(self) -> Task:
        key = f"test_{self.rng.randint(100, 999)}"
        value = self._pick(["important_data", "config_setting", "user_preference"])
        answer = self._execute_tool("store_memory", key=key, value=value)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"Store the value '{value}' under the key '{key}'.",
            available_tools=["store_memory", "retrieve_memory", "list_memories"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="store_memory", arguments={"key": key, "value": value})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "state_management", "tool": "store_memory"},
        )

    def _l0_summarize_text(self) -> Task:
        text = self._pick(LONG_TEXTS)
        answer = self._execute_tool("summarize_text", text=text, max_sentences=2)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f'Summarize this text in 2 sentences: "{text[:100]}..."',
            available_tools=["summarize_text", "extract_entities", "sentiment_analysis"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="summarize_text", arguments={"text": text, "max_sentences": 2})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "text_processing", "tool": "summarize_text"},
        )

    def _l0_compare_texts(self) -> Task:
        pairs = [
            ("The cat sat on the mat", "The cat was sitting on the mat"),
            ("Machine learning is powerful", "Deep learning transforms industries"),
            ("Hello world", "Hello world"),
        ]
        t1, t2 = self._pick(pairs)
        answer = self._execute_tool("compare_texts", text1=t1, text2=t2)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f'Compare the similarity of "{t1}" and "{t2}".',
            available_tools=["compare_texts", "sentiment_analysis", "classify_text"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="compare_texts", arguments={"text1": t1, "text2": t2})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "text_processing", "tool": "compare_texts"},
        )

    def _l0_translate_text(self) -> Task:
        text = self._pick(["Hello, how are you?", "Good morning", "Thank you very much", "Where is the train station?"])
        lang = self._pick(LANGUAGES)
        code = LANGUAGE_CODES[lang]
        answer = self._execute_tool("translate_text", text=text, from_language="en", to_language=code)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f'Translate "{text}" from English to {lang}.',
            available_tools=["translate_text", "extract_entities", "compare_texts"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="translate_text", arguments={"text": text, "from_language": "en", "to_language": code})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "external_services", "tool": "translate_text"},
        )

    def _l0_transform_format(self) -> Task:
        data = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
        answer = self._execute_tool("transform_format", data=data, from_format="json", to_format="csv")
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt="Convert this JSON data to CSV format.",
            available_tools=["transform_format", "read_file", "write_file"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="transform_format", arguments={"data": data, "from_format": "json", "to_format": "csv"})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "file_data", "tool": "transform_format"},
        )

    def _l0_write_file(self) -> Task:
        path = f"/data/output_{self.rng.randint(100, 999)}.txt"
        content = self._pick(["Report complete.", "Task finished successfully.", "Data exported."])
        answer = self._execute_tool("write_file", path=path, content=content)
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f'Write "{content}" to {path}.',
            available_tools=["write_file", "read_file", "list_files"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="write_file", arguments={"path": path, "content": content})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "file_data", "tool": "write_file"},
        )

    def _l0_generate_image(self) -> Task:
        prompts = ["a sunset over mountains", "a futuristic city skyline", "a cat wearing glasses", "abstract geometric art"]
        prompt = self._pick(prompts)
        answer = self._execute_tool("generate_image", prompt=prompt, size="512x512")
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f'Generate a 512x512 image of "{prompt}".',
            available_tools=["generate_image", "write_file", "send_email"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="generate_image", arguments={"prompt": prompt, "size": "512x512"})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "media", "tool": "generate_image"},
        )

    def _l0_transcribe_audio(self) -> Task:
        sources = ["https://example.com/audio/meeting.mp3", "https://example.com/audio/lecture.mp3", "https://example.com/audio/podcast.mp3"]
        source = self._pick(sources)
        answer = self._execute_tool("transcribe_audio", audio_source=source, language="en")
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"Transcribe the English audio at {source}.",
            available_tools=["transcribe_audio", "summarize_text", "translate_text"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="transcribe_audio", arguments={"audio_source": source, "language": "en"})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "media", "tool": "transcribe_audio"},
        )

    def _l0_web_page_fetch(self) -> Task:
        urls = ["https://example.com/article/climate-change", "https://example.com/blog/ai-trends", "https://example.com/news/space-exploration"]
        url = self._pick(urls)
        answer = self._execute_tool("web_page_fetch", url=url, extract_mode="text")
        return Task(
            task_id=self._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"Fetch and extract the text content from {url}.",
            available_tools=["web_page_fetch", "summarize_text", "extract_entities"],
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="web_page_fetch", arguments={"url": url, "extract_mode": "text"})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "information_retrieval", "tool": "web_page_fetch"},
        )

    # -----------------------------------------------------------------------
    # Level 1: Chain (2-3 tools sequential)
    # -----------------------------------------------------------------------

    def generate_l1_tasks(self, count: int = 50) -> list[Task]:
        """Generate chain (sequential multi-tool) tasks."""
        tasks: list[Task] = []
        generators = [
            # Original 7
            self._l1_weather_convert,
            self._l1_weather_email,
            self._l1_stock_convert_currency,
            self._l1_search_summarize,
            self._l1_db_stats,
            self._l1_entity_sentiment,
            self._l1_currency_calculate,
            # New 13 (expanded coverage)
            self._l1_read_file_summarize,
            self._l1_location_directions,
            self._l1_extract_entities_email,
            self._l1_search_classify,
            self._l1_products_filter,
            self._l1_store_retrieve_memory,
            self._l1_kb_translate,
            self._l1_file_transform_write,
            self._l1_weather_timezone,
            self._l1_python_aggregate,
            self._l1_merge_sort_data,
            self._l1_entity_schedule_meeting,
            self._l1_search_notify,
        ]

        per_generator = max(count // len(generators), 1)
        for gen in generators:
            for _ in range(per_generator):
                task = gen()
                if task:
                    tasks.append(task)
                if len(tasks) >= count:
                    break
            if len(tasks) >= count:
                break

        return tasks[:count]

    def _l1_weather_convert(self) -> Task:
        """Get weather → convert temperature."""
        city = self._pick(CITIES)
        weather = self._execute_tool("get_weather", city=city)
        temp_c = weather["temperature_celsius"]
        converted = self._execute_tool("unit_convert", value=temp_c, from_unit="celsius", to_unit="fahrenheit")

        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f"What is the temperature in {city} in Fahrenheit?",
            available_tools=["get_weather", "unit_convert", "calculator", "send_email"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_weather", arguments={"city": city}, output_key="weather"),
                    ToolCall(step_id="step_2", tool_name="unit_convert", arguments={"value": temp_c, "from_unit": "celsius", "to_unit": "fahrenheit"}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=converted,
            metadata={"category": "chain", "tools": ["get_weather", "unit_convert"], "chain_type": "transform"},
        )

    def _l1_weather_email(self) -> Task:
        """Get weather → email the report."""
        city = self._pick(CITIES)
        weather = self._execute_tool("get_weather", city=city)

        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f"Check the weather in {city} and email the report to user@example.com with subject 'Weather Update'.",
            available_tools=["get_weather", "send_email", "summarize_text", "calculator"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_weather", arguments={"city": city}, output_key="weather"),
                    ToolCall(step_id="step_2", tool_name="send_email", arguments={
                        "to": "user@example.com",
                        "subject": "Weather Update",
                        "body": f"Weather in {city}: {weather['temperature_celsius']}°C, {weather['condition']}",
                    }, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer={"status": "sent"},  # Simplified check
            metadata={"category": "chain", "tools": ["get_weather", "send_email"], "chain_type": "act"},
        )

    def _l1_stock_convert_currency(self) -> Task:
        """Get stock price → convert to another currency."""
        symbol = self._pick(STOCKS)
        to_currency = self._pick([c for c in CURRENCIES if c != "USD"])
        stock = self._execute_tool("get_stock_price", symbol=symbol)
        price_usd = stock["price_usd"]
        converted = self._execute_tool("get_exchange_rate", from_currency="USD", to_currency=to_currency, amount=price_usd)

        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f"What is the stock price of {symbol} in {to_currency}?",
            available_tools=["get_stock_price", "get_exchange_rate", "calculator", "send_email"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_stock_price", arguments={"symbol": symbol}, output_key="stock"),
                    ToolCall(step_id="step_2", tool_name="get_exchange_rate", arguments={"from_currency": "USD", "to_currency": to_currency, "amount": price_usd}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=converted,
            metadata={"category": "chain", "tools": ["get_stock_price", "get_exchange_rate"], "chain_type": "transform"},
        )

    def _l1_search_summarize(self) -> Task:
        """Search web → summarize results."""
        query = self._pick(SEARCH_QUERIES)
        search_results = self._execute_tool("web_search", query=query, max_results=3)
        text_to_summarize = " ".join(r["snippet"] for r in search_results["results"])
        summary = self._execute_tool("summarize_text", text=text_to_summarize, max_sentences=2)

        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f'Search for "{query}" and give me a brief summary of the results.',
            available_tools=["web_search", "summarize_text", "sentiment_analysis", "send_email"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="web_search", arguments={"query": query, "max_results": 3}, output_key="results"),
                    ToolCall(step_id="step_2", tool_name="summarize_text", arguments={"text": text_to_summarize, "max_sentences": 2}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=summary,
            metadata={"category": "chain", "tools": ["web_search", "summarize_text"], "chain_type": "analyze"},
        )

    def _l1_db_stats(self) -> Task:
        """Query database → compute statistics."""
        continent = self._pick(CONTINENTS)
        db_result = self._execute_tool(
            "database_query", table="countries",
            filter_field="continent", filter_op="equals", filter_value=continent,
            sort_by="population", limit=10,
        )
        populations = [r["population"] for r in db_result["results"]]
        stats = self._execute_tool("statistical_analysis", numbers=populations, operation="summary")

        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f"What are the population statistics for countries in {continent}?",
            available_tools=["database_query", "statistical_analysis", "calculator", "data_sort"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="database_query", arguments={
                        "table": "countries", "filter_field": "continent", "filter_op": "equals",
                        "filter_value": continent, "sort_by": "population", "limit": 10,
                    }, output_key="countries"),
                    ToolCall(step_id="step_2", tool_name="statistical_analysis", arguments={
                        "numbers": populations, "operation": "summary",
                    }, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=stats,
            metadata={"category": "chain", "tools": ["database_query", "statistical_analysis"], "chain_type": "analyze"},
        )

    def _l1_entity_sentiment(self) -> Task:
        """Look up entity → analyze sentiment of description."""
        entities = ["Machine learning", "Climate change", "Artificial intelligence"]
        entity = self._pick(entities)
        info = self._execute_tool("lookup_entity", entity=entity)
        sentiment = self._execute_tool("sentiment_analysis", text=info.get("summary", ""))

        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f"Look up {entity} and analyze the sentiment of its description.",
            available_tools=["lookup_entity", "sentiment_analysis", "summarize_text", "web_search"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="lookup_entity", arguments={"entity": entity}, output_key="info"),
                    ToolCall(step_id="step_2", tool_name="sentiment_analysis", arguments={"text": info.get("summary", "")}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=sentiment,
            metadata={"category": "chain", "tools": ["lookup_entity", "sentiment_analysis"], "chain_type": "analyze"},
        )

    def _l1_currency_calculate(self) -> Task:
        """Convert currency → do arithmetic."""
        from_c, to_c = self._pick_n(CURRENCIES, 2)
        amount = self.rng.choice([100, 500, 1000])
        converted = self._execute_tool("get_exchange_rate", from_currency=from_c, to_currency=to_c, amount=amount)
        converted_amount = converted["converted_amount"]
        tax_rate = self._pick([0.05, 0.08, 0.10, 0.15, 0.20])
        tax_expr = f"{converted_amount} * {tax_rate}"
        tax = self._execute_tool("calculator", expression=tax_expr)

        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f"Convert {amount} {from_c} to {to_c}, then calculate {int(tax_rate * 100)}% tax on the result.",
            available_tools=["get_exchange_rate", "calculator", "unit_convert", "send_email"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_exchange_rate", arguments={"from_currency": from_c, "to_currency": to_c, "amount": amount}, output_key="converted"),
                    ToolCall(step_id="step_2", tool_name="calculator", arguments={"expression": tax_expr}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=tax,
            metadata={"category": "chain", "tools": ["get_exchange_rate", "calculator"], "chain_type": "transform"},
        )

    # --- New L1 generators ---

    def _l1_read_file_summarize(self) -> Task:
        """Read file → summarize contents."""
        path = "/data/report.txt"
        file_data = self._execute_tool("read_file", path=path)
        content = file_data.get("content", "")
        summary = self._execute_tool("summarize_text", text=content, max_sentences=2)
        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f"Read {path} and summarize its contents in 2 sentences.",
            available_tools=["read_file", "summarize_text", "write_file", "extract_entities"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="read_file", arguments={"path": path}, output_key="file"),
                    ToolCall(step_id="step_2", tool_name="summarize_text", arguments={"text": content, "max_sentences": 2}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=summary,
            metadata={"category": "chain", "tools": ["read_file", "summarize_text"], "chain_type": "analyze"},
        )

    def _l1_location_directions(self) -> Task:
        """Get location info → get directions to it."""
        city = self._pick(CITIES[:6])
        loc = self._execute_tool("get_location_info", location=city)
        origin = self._pick(["current location", "downtown", "airport"])
        directions = self._execute_tool("get_directions", origin=origin, destination=city, mode="driving")
        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f"Find the location of {city}, then get driving directions from {origin} to {city}.",
            available_tools=["get_location_info", "get_directions", "calculator", "get_weather"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_location_info", arguments={"location": city}, output_key="loc"),
                    ToolCall(step_id="step_2", tool_name="get_directions", arguments={"origin": origin, "destination": city, "mode": "driving"}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=directions,
            metadata={"category": "chain", "tools": ["get_location_info", "get_directions"], "chain_type": "act"},
        )

    def _l1_extract_entities_email(self) -> Task:
        """Extract entities from text → email the results."""
        text = self._pick(ENTITY_TEXTS)
        entities = self._execute_tool("extract_entities", text=text)
        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f'Extract entities from this text and email them to admin@co.com: "{text}"',
            available_tools=["extract_entities", "send_email", "summarize_text", "classify_text"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="extract_entities", arguments={"text": text}, output_key="entities"),
                    ToolCall(step_id="step_2", tool_name="send_email", arguments={
                        "to": "admin@co.com",
                        "subject": "Extracted Entities",
                        "body": f"Found {entities['entity_count']} entities",
                    }, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer={"entities": entities, "email_sent": True},
            metadata={"category": "chain", "tools": ["extract_entities", "send_email"], "chain_type": "act"},
        )

    def _l1_search_classify(self) -> Task:
        """Search web → classify the top result."""
        query = self._pick(SEARCH_QUERIES)
        results = self._execute_tool("web_search", query=query, max_results=1)
        snippet = results["results"][0]["snippet"] if results["results"] else ""
        cats = ["science", "technology", "business", "politics"]
        classified = self._execute_tool("classify_text", text=snippet, categories=cats)
        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f'Search for "{query}" and classify the top result into: {cats}.',
            available_tools=["web_search", "classify_text", "sentiment_analysis", "summarize_text"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="web_search", arguments={"query": query, "max_results": 1}, output_key="results"),
                    ToolCall(step_id="step_2", tool_name="classify_text", arguments={"text": snippet, "categories": cats}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=classified,
            metadata={"category": "chain", "tools": ["web_search", "classify_text"], "chain_type": "analyze"},
        )

    def _l1_products_filter(self) -> Task:
        """Search products → filter by price."""
        query = self._pick(["laptop", "headphones", "smartwatch"])
        products = self._execute_tool("search_products", query=query, max_results=5)
        prices = [p["price_usd"] for p in products.get("results", [])]
        filtered = self._execute_tool("data_filter", items=prices, condition="less_than", value="500")
        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f'Search for "{query}" products and filter to those priced under $500.',
            available_tools=["search_products", "data_filter", "data_sort", "calculator"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="search_products", arguments={"query": query, "max_results": 5}, output_key="products"),
                    ToolCall(step_id="step_2", tool_name="data_filter", arguments={"items": prices, "condition": "less_than", "value": "500"}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=filtered,
            metadata={"category": "chain", "tools": ["search_products", "data_filter"], "chain_type": "transform"},
        )

    def _l1_store_retrieve_memory(self) -> Task:
        """Store a value → retrieve it back."""
        key = f"chain_test_{self.rng.randint(100, 999)}"
        value = f"data_{self.rng.randint(1, 100)}"
        store_result = self._execute_tool("store_memory", key=key, value=value)
        retrieve_result = self._execute_tool("retrieve_memory", key=key)
        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f"Store the value '{value}' under key '{key}', then retrieve it to verify.",
            available_tools=["store_memory", "retrieve_memory", "list_memories", "send_message"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="store_memory", arguments={"key": key, "value": value}, output_key="stored"),
                    ToolCall(step_id="step_2", tool_name="retrieve_memory", arguments={"key": key}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=retrieve_result,
            metadata={"category": "chain", "tools": ["store_memory", "retrieve_memory"], "chain_type": "verify"},
        )

    def _l1_kb_translate(self) -> Task:
        """Query knowledge base → translate the answer."""
        query = self._pick(KB_QUERIES)
        kb_result = self._execute_tool("knowledge_base_query", query=query)
        answer_text = kb_result.get("answer", "")
        lang = self._pick(LANGUAGES)
        code = LANGUAGE_CODES[lang]
        translated = self._execute_tool("translate_text", text=answer_text, from_language="en", to_language=code)
        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f'Look up "{query}" in the knowledge base and translate the answer to {lang}.',
            available_tools=["knowledge_base_query", "translate_text", "summarize_text", "send_email"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="knowledge_base_query", arguments={"query": query}, output_key="kb"),
                    ToolCall(step_id="step_2", tool_name="translate_text", arguments={"text": answer_text, "from_language": "en", "to_language": code}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=translated,
            metadata={"category": "chain", "tools": ["knowledge_base_query", "translate_text"], "chain_type": "transform"},
        )

    def _l1_file_transform_write(self) -> Task:
        """Read JSON data → transform to CSV → write to file."""
        data = '[{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}]'
        transformed = self._execute_tool("transform_format", data=data, from_format="json", to_format="csv")
        csv_content = transformed["transformed"]
        path = f"/data/export_{self.rng.randint(100, 999)}.csv"
        written = self._execute_tool("write_file", path=path, content=csv_content)
        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f"Convert this JSON to CSV and save it to {path}.",
            available_tools=["transform_format", "write_file", "read_file", "list_files"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="transform_format", arguments={"data": data, "from_format": "json", "to_format": "csv"}, output_key="csv"),
                    ToolCall(step_id="step_2", tool_name="write_file", arguments={"path": path, "content": csv_content}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=written,
            metadata={"category": "chain", "tools": ["transform_format", "write_file"], "chain_type": "act"},
        )

    def _l1_weather_timezone(self) -> Task:
        """Get weather → convert timezone for the city."""
        city = self._pick(CITIES[:6])
        weather = self._execute_tool("get_weather", city=city)
        from_tz = self._pick(TIMEZONES[:2])
        to_tz = self._pick(TIMEZONES[2:])
        converted = self._execute_tool("convert_timezone", datetime_str="2026-03-15T12:00:00", from_timezone=from_tz, to_timezone=to_tz)
        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f"Check the weather in {city}, then convert noon {from_tz} to {to_tz}.",
            available_tools=["get_weather", "convert_timezone", "get_current_time", "send_email"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_weather", arguments={"city": city}, output_key="weather"),
                    ToolCall(step_id="step_2", tool_name="convert_timezone", arguments={"datetime_str": "2026-03-15T12:00:00", "from_timezone": from_tz, "to_timezone": to_tz}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer={"weather": weather, "converted_time": converted},
            metadata={"category": "chain", "tools": ["get_weather", "convert_timezone"], "chain_type": "info-gather"},
        )

    def _l1_python_aggregate(self) -> Task:
        """Execute Python to generate data → aggregate it."""
        code = "list(range(1, 11))"
        py_result = self._execute_tool("execute_python", code=code)
        numbers = py_result.get("result", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        stats = self._execute_tool("statistical_analysis", numbers=numbers, operation="summary")
        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt="Generate numbers 1 to 10 using Python, then compute their statistics.",
            available_tools=["execute_python", "statistical_analysis", "calculator", "data_aggregate"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="execute_python", arguments={"code": code}, output_key="numbers"),
                    ToolCall(step_id="step_2", tool_name="statistical_analysis", arguments={"numbers": numbers, "operation": "summary"}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=stats,
            metadata={"category": "chain", "tools": ["execute_python", "statistical_analysis"], "chain_type": "analyze"},
        )

    def _l1_merge_sort_data(self) -> Task:
        """Merge two datasets → sort the result."""
        ds1 = '[{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}, {"id": 3, "name": "Charlie"}]'
        ds2 = '[{"id": 1, "score": 95}, {"id": 2, "score": 87}, {"id": 3, "score": 92}]'
        merged = self._execute_tool("merge_data", dataset1=ds1, dataset2=ds2, join_key="id")
        merged_items = merged.get("merged", [])
        sorted_data = self._execute_tool("data_sort", items=merged_items, key="score", order="descending")
        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt="Merge the names and scores datasets by id, then sort by score descending.",
            available_tools=["merge_data", "data_sort", "data_filter", "data_aggregate"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="merge_data", arguments={"dataset1": ds1, "dataset2": ds2, "join_key": "id"}, output_key="merged"),
                    ToolCall(step_id="step_2", tool_name="data_sort", arguments={"items": merged_items, "key": "score", "order": "descending"}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=sorted_data,
            metadata={"category": "chain", "tools": ["merge_data", "data_sort"], "chain_type": "transform"},
        )

    def _l1_entity_schedule_meeting(self) -> Task:
        """Extract entities from email → schedule a meeting."""
        text = "Let's meet with Dr. Smith on 03/20/2026 at 2pm to discuss the project."
        entities = self._execute_tool("extract_entities", text=text)
        meeting = self._execute_tool("schedule_meeting", title="Project Discussion", datetime_str="2026-03-20 14:00", duration_minutes=60, participants=["drsmith@co.com"])
        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f'Extract the meeting details from this text and schedule it: "{text}"',
            available_tools=["extract_entities", "schedule_meeting", "send_email", "create_task"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="extract_entities", arguments={"text": text}, output_key="entities"),
                    ToolCall(step_id="step_2", tool_name="schedule_meeting", arguments={
                        "title": "Project Discussion",
                        "datetime_str": "2026-03-20 14:00",
                        "duration_minutes": 60,
                        "participants": ["drsmith@co.com"],
                    }, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=meeting,
            metadata={"category": "chain", "tools": ["extract_entities", "schedule_meeting"], "chain_type": "act"},
        )

    def _l1_search_notify(self) -> Task:
        """Search web → send notification about results."""
        query = self._pick(SEARCH_QUERIES)
        results = self._execute_tool("web_search", query=query, max_results=2)
        count = results["total"]
        notif = self._execute_tool("create_notification", title="Search Results", message=f"Found {count} results for '{query}'", priority="normal")
        return Task(
            task_id=self._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=f'Search for "{query}" and send a notification with the result count.',
            available_tools=["web_search", "create_notification", "send_email", "send_message"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="web_search", arguments={"query": query, "max_results": 2}, output_key="results"),
                    ToolCall(step_id="step_2", tool_name="create_notification", arguments={
                        "title": "Search Results",
                        "message": f"Found {count} results for '{query}'",
                        "priority": "normal",
                    }, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=notif,
            metadata={"category": "chain", "tools": ["web_search", "create_notification"], "chain_type": "act"},
        )

    # -----------------------------------------------------------------------
    # Level 2: Parallel (concurrent tools → merge)
    # -----------------------------------------------------------------------

    def generate_l2_tasks(self, count: int = 30) -> list[Task]:
        """Generate parallel composition tasks."""
        tasks: list[Task] = []
        generators = [
            # Original 3
            self._l2_multi_city_weather,
            self._l2_multi_stock_compare,
            self._l2_weather_and_time,
            # New 7
            self._l2_multi_entity_sentiment,
            self._l2_multi_direction_compare,
            self._l2_parallel_search_kb,
            self._l2_multi_product_search,
            self._l2_parallel_file_ops,
            self._l2_weather_stock_location,
            self._l2_multi_translate,
        ]

        per_generator = max(count // len(generators), 1)
        for gen in generators:
            for _ in range(per_generator):
                task = gen()
                if task:
                    tasks.append(task)
                if len(tasks) >= count:
                    break
            if len(tasks) >= count:
                break

        return tasks[:count]

    def _l2_multi_city_weather(self) -> Task:
        """Get weather for 2-3 cities in parallel → compare."""
        cities = self._pick_n(CITIES, 3)
        weathers = {}
        steps = []
        for i, city in enumerate(cities):
            w = self._execute_tool("get_weather", city=city)
            weathers[city] = w
            steps.append(ToolCall(
                step_id=f"step_{i + 1}",
                tool_name="get_weather",
                arguments={"city": city},
                output_key=f"weather_{city.lower().replace(' ', '_')}",
            ))

        # Find warmest
        warmest = max(weathers, key=lambda c: weathers[c]["temperature_celsius"])

        return Task(
            task_id=self._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=f"Check the weather in {', '.join(cities[:-1])} and {cities[-1]}. Which city is the warmest?",
            available_tools=["get_weather", "calculator", "data_sort", "send_email", "summarize_text"],
            expected_trace=ExpectedTrace(
                steps=steps,
                final_answer_source=f"step_{cities.index(warmest) + 1}",
            ),
            expected_final_answer={"warmest_city": warmest, "temperature": weathers[warmest]["temperature_celsius"], "all_temps": {c: w["temperature_celsius"] for c, w in weathers.items()}},
            metadata={"category": "parallel", "tools": ["get_weather"], "pattern": "fan-out-compare"},
        )

    def _l2_multi_stock_compare(self) -> Task:
        """Get multiple stock prices in parallel → find highest."""
        symbols = self._pick_n(STOCKS, 3)
        stocks = {}
        steps = []
        for i, sym in enumerate(symbols):
            s = self._execute_tool("get_stock_price", symbol=sym)
            stocks[sym] = s
            steps.append(ToolCall(
                step_id=f"step_{i + 1}",
                tool_name="get_stock_price",
                arguments={"symbol": sym},
                output_key=f"stock_{sym.lower()}",
            ))

        highest = max(stocks, key=lambda s: stocks[s]["price_usd"])

        return Task(
            task_id=self._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=f"Compare the stock prices of {', '.join(symbols[:-1])} and {symbols[-1]}. Which has the highest price?",
            available_tools=["get_stock_price", "calculator", "data_sort", "get_exchange_rate"],
            expected_trace=ExpectedTrace(steps=steps, final_answer_source=f"step_{symbols.index(highest) + 1}"),
            expected_final_answer={"highest": highest, "price": stocks[highest]["price_usd"], "all_prices": {s: d["price_usd"] for s, d in stocks.items()}},
            metadata={"category": "parallel", "tools": ["get_stock_price"], "pattern": "fan-out-compare"},
        )

    def _l2_weather_and_time(self) -> Task:
        """Get weather AND current time for a city in parallel."""
        city = self._pick(CITIES)
        weather = self._execute_tool("get_weather", city=city)
        time_data = self._execute_tool("get_current_time", timezone="UTC")

        return Task(
            task_id=self._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=f"What is the current weather and time in {city}?",
            available_tools=["get_weather", "get_current_time", "convert_timezone", "send_email"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_weather", arguments={"city": city}, output_key="weather"),
                    ToolCall(step_id="step_2", tool_name="get_current_time", arguments={"timezone": "UTC"}, output_key="time"),
                ],
                final_answer_source="step_1",  # Both contribute
            ),
            expected_final_answer={"weather": weather, "time": time_data},
            metadata={"category": "parallel", "tools": ["get_weather", "get_current_time"], "pattern": "independent-merge"},
        )

    # --- New L2 generators ---

    def _l2_multi_entity_sentiment(self) -> Task:
        """Analyze sentiment of 2-3 texts in parallel."""
        texts = self._pick_n(SENTIMENTS_TEXTS, 3)
        steps = []
        sentiments = {}
        for i, (text, _expected) in enumerate(texts):
            s = self._execute_tool("sentiment_analysis", text=text)
            sentiments[text[:30]] = s
            steps.append(ToolCall(
                step_id=f"step_{i + 1}",
                tool_name="sentiment_analysis",
                arguments={"text": text},
                output_key=f"sent_{i}",
            ))
        return Task(
            task_id=self._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=f'Analyze the sentiment of these texts: "{texts[0][0][:40]}...", "{texts[1][0][:40]}...", and "{texts[2][0][:40]}..."',
            available_tools=["sentiment_analysis", "classify_text", "compare_texts", "summarize_text"],
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=sentiments,
            metadata={"category": "parallel", "tools": ["sentiment_analysis"], "pattern": "fan-out-compare"},
        )

    def _l2_multi_direction_compare(self) -> Task:
        """Get directions via 2 modes in parallel → compare travel time."""
        city1, city2 = self._pick_n(CITIES[:8], 2)
        modes = self._pick_n(TRAVEL_MODES, 2)
        results = {}
        steps = []
        for i, mode in enumerate(modes):
            d = self._execute_tool("get_directions", origin=city1, destination=city2, mode=mode)
            results[mode] = d
            steps.append(ToolCall(
                step_id=f"step_{i + 1}",
                tool_name="get_directions",
                arguments={"origin": city1, "destination": city2, "mode": mode},
                output_key=f"dir_{mode}",
            ))
        return Task(
            task_id=self._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=f"Compare {modes[0]} vs {modes[1]} directions from {city1} to {city2}. Which is faster?",
            available_tools=["get_directions", "calculator", "get_location_info", "compare_texts"],
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=results,
            metadata={"category": "parallel", "tools": ["get_directions"], "pattern": "fan-out-compare"},
        )

    def _l2_parallel_search_kb(self) -> Task:
        """Search web AND query knowledge base for the same topic in parallel."""
        topic = self._pick(["speed of light", "capital of France", "population of Tokyo"])
        search = self._execute_tool("web_search", query=topic, max_results=2)
        kb = self._execute_tool("knowledge_base_query", query=f"What is the {topic}?")
        return Task(
            task_id=self._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=f'Look up "{topic}" using both web search and the knowledge base.',
            available_tools=["web_search", "knowledge_base_query", "summarize_text", "compare_texts"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="web_search", arguments={"query": topic, "max_results": 2}, output_key="web"),
                    ToolCall(step_id="step_2", tool_name="knowledge_base_query", arguments={"query": f"What is the {topic}?"}, output_key="kb"),
                ],
                final_answer_source="step_1",
            ),
            expected_final_answer={"web": search, "kb": kb},
            metadata={"category": "parallel", "tools": ["web_search", "knowledge_base_query"], "pattern": "independent-merge"},
        )

    def _l2_multi_product_search(self) -> Task:
        """Search for products in 2 categories in parallel."""
        query = "wireless headphones"
        cats = self._pick_n(PRODUCT_CATEGORIES, 2)
        results = {}
        steps = []
        for i, cat in enumerate(cats):
            p = self._execute_tool("search_products", query=query, category=cat, max_results=3)
            results[cat] = p
            steps.append(ToolCall(
                step_id=f"step_{i + 1}",
                tool_name="search_products",
                arguments={"query": query, "category": cat, "max_results": 3},
                output_key=f"products_{cat}",
            ))
        return Task(
            task_id=self._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=f'Search for "{query}" in both {cats[0]} and {cats[1]} categories.',
            available_tools=["search_products", "data_sort", "data_filter", "calculator"],
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=results,
            metadata={"category": "parallel", "tools": ["search_products"], "pattern": "fan-out-compare"},
        )

    def _l2_parallel_file_ops(self) -> Task:
        """Read two files in parallel."""
        paths = ["/data/report.txt", "/data/config.json"]
        results = {}
        steps = []
        for i, path in enumerate(paths):
            f = self._execute_tool("read_file", path=path)
            results[path] = f
            steps.append(ToolCall(
                step_id=f"step_{i + 1}",
                tool_name="read_file",
                arguments={"path": path},
                output_key=f"file_{i}",
            ))
        return Task(
            task_id=self._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=f"Read both {paths[0]} and {paths[1]} simultaneously.",
            available_tools=["read_file", "write_file", "list_files", "compare_texts"],
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=results,
            metadata={"category": "parallel", "tools": ["read_file"], "pattern": "independent-merge"},
        )

    def _l2_weather_stock_location(self) -> Task:
        """Get weather + stock price + location info in parallel (3 independent)."""
        city = self._pick(CITIES[:6])
        symbol = self._pick(STOCKS[:4])
        w = self._execute_tool("get_weather", city=city)
        s = self._execute_tool("get_stock_price", symbol=symbol)
        l = self._execute_tool("get_location_info", location=city)
        return Task(
            task_id=self._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=f"Get the weather in {city}, the stock price of {symbol}, and location info for {city}.",
            available_tools=["get_weather", "get_stock_price", "get_location_info", "calculator", "send_email"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_weather", arguments={"city": city}, output_key="weather"),
                    ToolCall(step_id="step_2", tool_name="get_stock_price", arguments={"symbol": symbol}, output_key="stock"),
                    ToolCall(step_id="step_3", tool_name="get_location_info", arguments={"location": city}, output_key="loc"),
                ],
                final_answer_source="step_1",
            ),
            expected_final_answer={"weather": w, "stock": s, "location": l},
            metadata={"category": "parallel", "tools": ["get_weather", "get_stock_price", "get_location_info"], "pattern": "independent-merge"},
        )

    def _l2_multi_translate(self) -> Task:
        """Translate a text to 2-3 languages in parallel."""
        text = self._pick(["Hello, how are you?", "Good morning", "Thank you"])
        langs = self._pick_n(LANGUAGES, 3)
        translations = {}
        steps = []
        for i, lang in enumerate(langs):
            code = LANGUAGE_CODES[lang]
            t = self._execute_tool("translate_text", text=text, from_language="en", to_language=code)
            translations[lang] = t
            steps.append(ToolCall(
                step_id=f"step_{i + 1}",
                tool_name="translate_text",
                arguments={"text": text, "from_language": "en", "to_language": code},
                output_key=f"trans_{lang.lower()}",
            ))
        return Task(
            task_id=self._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=f'Translate "{text}" from English to {", ".join(langs[:-1])} and {langs[-1]} simultaneously.',
            available_tools=["translate_text", "compare_texts", "summarize_text", "send_email"],
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=translations,
            metadata={"category": "parallel", "tools": ["translate_text"], "pattern": "fan-out"},
        )

    # -----------------------------------------------------------------------
    # Level 3: DAG (branching + merging)
    # -----------------------------------------------------------------------

    def generate_l3_tasks(self, count: int = 20) -> list[Task]:
        """Generate DAG composition tasks (branching + merging)."""
        tasks: list[Task] = []
        generators = [
            # Original 2
            self._l3_weather_compare_email,
            self._l3_stock_currency_notify,
            # New 8 (diverse DAG patterns)
            self._l3_search_translate_email,
            self._l3_multi_weather_stats_report,
            self._l3_file_compare_summarize_write,
            self._l3_products_compare_notify,
            self._l3_kb_sentiment_store,
            self._l3_directions_weather_schedule,
            self._l3_entities_classify_task,
            self._l3_stocks_aggregate_translate,
        ]

        per_generator = max(count // len(generators), 1)
        for gen in generators:
            for _ in range(per_generator):
                task = gen()
                if task:
                    tasks.append(task)
                if len(tasks) >= count:
                    break
            if len(tasks) >= count:
                break

        return tasks[:count]

    def _l3_weather_compare_email(self) -> Task:
        """Weather(A) + Weather(B) → compare → convert winner temp → email."""
        cities = self._pick_n(CITIES, 2)
        w1 = self._execute_tool("get_weather", city=cities[0])
        w2 = self._execute_tool("get_weather", city=cities[1])
        warmer = cities[0] if w1["temperature_celsius"] > w2["temperature_celsius"] else cities[1]
        warmer_temp = max(w1["temperature_celsius"], w2["temperature_celsius"])
        converted = self._execute_tool("unit_convert", value=warmer_temp, from_unit="celsius", to_unit="fahrenheit")

        return Task(
            task_id=self._next_id(CompositionLevel.DAG),
            level=CompositionLevel.DAG,
            prompt=f"Compare the weather in {cities[0]} and {cities[1]}. Convert the warmer city's temperature to Fahrenheit and email the result to user@example.com.",
            available_tools=["get_weather", "unit_convert", "send_email", "calculator", "summarize_text", "compare_texts"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_weather", arguments={"city": cities[0]}, output_key="w1"),
                    ToolCall(step_id="step_2", tool_name="get_weather", arguments={"city": cities[1]}, output_key="w2"),
                    ToolCall(step_id="step_3", tool_name="unit_convert", arguments={"value": warmer_temp, "from_unit": "celsius", "to_unit": "fahrenheit"}, depends_on=["step_1", "step_2"], output_key="converted"),
                    ToolCall(step_id="step_4", tool_name="send_email", arguments={
                        "to": "user@example.com",
                        "subject": f"Weather Comparison: {warmer} is warmer",
                        "body": f"{warmer} is warmer at {converted['converted_value']}°F",
                    }, depends_on=["step_3"]),
                ],
                final_answer_source="step_4",
            ),
            expected_final_answer={"warmer_city": warmer, "temperature_f": converted["converted_value"]},
            metadata={"category": "dag", "tools": ["get_weather", "unit_convert", "send_email"], "pattern": "parallel-merge-chain"},
        )

    def _l3_stock_currency_notify(self) -> Task:
        """Stock(A) + Stock(B) → find higher → convert to EUR → create task."""
        symbols = self._pick_n(STOCKS, 2)
        s1 = self._execute_tool("get_stock_price", symbol=symbols[0])
        s2 = self._execute_tool("get_stock_price", symbol=symbols[1])
        higher = symbols[0] if s1["price_usd"] > s2["price_usd"] else symbols[1]
        higher_price = max(s1["price_usd"], s2["price_usd"])
        converted = self._execute_tool("get_exchange_rate", from_currency="USD", to_currency="EUR", amount=higher_price)

        return Task(
            task_id=self._next_id(CompositionLevel.DAG),
            level=CompositionLevel.DAG,
            prompt=f"Compare {symbols[0]} and {symbols[1]} stock prices. Convert the higher one to EUR and create a task to review it.",
            available_tools=["get_stock_price", "get_exchange_rate", "create_task", "calculator", "send_email", "data_sort"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_stock_price", arguments={"symbol": symbols[0]}, output_key="s1"),
                    ToolCall(step_id="step_2", tool_name="get_stock_price", arguments={"symbol": symbols[1]}, output_key="s2"),
                    ToolCall(step_id="step_3", tool_name="get_exchange_rate", arguments={"from_currency": "USD", "to_currency": "EUR", "amount": higher_price}, depends_on=["step_1", "step_2"], output_key="converted"),
                    ToolCall(step_id="step_4", tool_name="create_task", arguments={
                        "title": f"Review {higher} stock position",
                        "description": f"{higher} is at €{converted['converted_amount']}",
                        "priority": "high",
                    }, depends_on=["step_3"]),
                ],
                final_answer_source="step_4",
            ),
            expected_final_answer={"higher_stock": higher, "price_eur": converted["converted_amount"]},
            metadata={"category": "dag", "tools": ["get_stock_price", "get_exchange_rate", "create_task"], "pattern": "parallel-merge-chain"},
        )

    def _l3_search_translate_email(self) -> Task:
        """Search(query) → summarize → translate summary → email translated."""
        query = self._pick(SEARCH_QUERIES)
        lang = self._pick(LANGUAGES)
        lang_code = LANGUAGE_CODES[lang]
        search_result = self._execute_tool("web_search", query=query)
        snippet = search_result["results"][0]["snippet"]
        summary = self._execute_tool("summarize_text", text=snippet, max_sentences=2)
        translated = self._execute_tool("translate_text", text=summary["summary"], from_language="en", to_language=lang_code)

        return Task(
            task_id=self._next_id(CompositionLevel.DAG),
            level=CompositionLevel.DAG,
            prompt=f'Search for "{query}", summarize the top result in 2 sentences, translate the summary to {lang}, and email it to researcher@lab.org.',
            available_tools=["web_search", "summarize_text", "translate_text", "send_email", "extract_entities", "compare_texts"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="web_search", arguments={"query": query}, output_key="search"),
                    ToolCall(step_id="step_2", tool_name="summarize_text", arguments={"text": snippet, "max_sentences": 2}, depends_on=["step_1"], output_key="summary"),
                    ToolCall(step_id="step_3", tool_name="translate_text", arguments={"text": summary["summary"], "from_language": "en", "to_language": lang_code}, depends_on=["step_2"], output_key="translated"),
                    ToolCall(step_id="step_4", tool_name="send_email", arguments={
                        "to": "researcher@lab.org",
                        "subject": f"Research Summary ({lang})",
                        "body": translated["translated_text"],
                    }, depends_on=["step_3"]),
                ],
                final_answer_source="step_4",
            ),
            expected_final_answer={"translated_summary": translated["translated_text"], "language": lang},
            metadata={"category": "dag", "tools": ["web_search", "summarize_text", "translate_text", "send_email"], "pattern": "chain-with-dependencies"},
        )

    def _l3_multi_weather_stats_report(self) -> Task:
        """Weather(A) + Weather(B) + Weather(C) → stats on temps → write report file."""
        cities = self._pick_n(CITIES, 3)
        w_results = [self._execute_tool("get_weather", city=c) for c in cities]
        temps = [w["temperature_celsius"] for w in w_results]
        stats = self._execute_tool("statistical_analysis", numbers=temps, operation="summary")
        report = f"Avg temp: {stats['mean']}°C across {', '.join(cities)}"
        self._execute_tool("write_file", path="/data/weather_report.txt", content=report)

        return Task(
            task_id=self._next_id(CompositionLevel.DAG),
            level=CompositionLevel.DAG,
            prompt=f"Get weather for {cities[0]}, {cities[1]}, and {cities[2]}. Calculate statistics on their temperatures and write a report to /data/weather_report.txt.",
            available_tools=["get_weather", "statistical_analysis", "write_file", "calculator", "unit_convert", "summarize_text"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_weather", arguments={"city": cities[0]}, output_key="w1"),
                    ToolCall(step_id="step_2", tool_name="get_weather", arguments={"city": cities[1]}, output_key="w2"),
                    ToolCall(step_id="step_3", tool_name="get_weather", arguments={"city": cities[2]}, output_key="w3"),
                    ToolCall(step_id="step_4", tool_name="statistical_analysis", arguments={"numbers": temps, "operation": "summary"}, depends_on=["step_1", "step_2", "step_3"], output_key="stats"),
                    ToolCall(step_id="step_5", tool_name="write_file", arguments={
                        "path": "/data/weather_report.txt",
                        "content": report,
                    }, depends_on=["step_4"]),
                ],
                final_answer_source="step_5",
            ),
            expected_final_answer={"mean_temp": stats["mean"], "stdev": stats.get("stdev"), "cities": cities},
            metadata={"category": "dag", "tools": ["get_weather", "statistical_analysis", "write_file"], "pattern": "triple-parallel-merge-chain"},
        )

    def _l3_file_compare_summarize_write(self) -> Task:
        """Read(A) + Read(B) → compare_texts → summarize comparison → write file."""
        paths = self._pick_n(FILE_PATHS, 2)
        f1 = self._execute_tool("read_file", path=paths[0])
        f2 = self._execute_tool("read_file", path=paths[1])
        text1 = f1["content"] or f"Content of {paths[0]}"
        text2 = f2["content"] or f"Content of {paths[1]}"
        comparison = self._execute_tool("compare_texts", text1=text1, text2=text2)
        sim = comparison.get("similarity", 0)
        summary = self._execute_tool("summarize_text", text=f"Comparison of {paths[0]} and {paths[1]}: similarity score {sim}. The files are {'similar' if sim > 0.5 else 'different'}.", max_sentences=1)
        self._execute_tool("write_file", path="/data/comparison_report.txt", content=summary["summary"])

        return Task(
            task_id=self._next_id(CompositionLevel.DAG),
            level=CompositionLevel.DAG,
            prompt=f"Read files {paths[0]} and {paths[1]}, compare their contents, summarize the comparison in 1 sentence, and write the summary to /data/comparison_report.txt.",
            available_tools=["read_file", "compare_texts", "summarize_text", "write_file", "extract_entities", "sentiment_analysis"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="read_file", arguments={"path": paths[0]}, output_key="f1"),
                    ToolCall(step_id="step_2", tool_name="read_file", arguments={"path": paths[1]}, output_key="f2"),
                    ToolCall(step_id="step_3", tool_name="compare_texts", arguments={"text1": text1, "text2": text2}, depends_on=["step_1", "step_2"], output_key="comparison"),
                    ToolCall(step_id="step_4", tool_name="summarize_text", arguments={"text": f"Comparison of {paths[0]} and {paths[1]}: similarity score {sim}. The files are {'similar' if sim > 0.5 else 'different'}.", "max_sentences": 1}, depends_on=["step_3"], output_key="summary"),
                    ToolCall(step_id="step_5", tool_name="write_file", arguments={"path": "/data/comparison_report.txt", "content": summary["summary"]}, depends_on=["step_4"]),
                ],
                final_answer_source="step_5",
            ),
            expected_final_answer={"similarity": sim, "output_path": "/data/comparison_report.txt"},
            metadata={"category": "dag", "tools": ["read_file", "compare_texts", "summarize_text", "write_file"], "pattern": "parallel-merge-chain-chain"},
        )

    def _l3_products_compare_notify(self) -> Task:
        """Search products(A) + Search products(B) → compare prices → notify winner."""
        categories = self._pick_n(PRODUCT_CATEGORIES, 2)
        p1 = self._execute_tool("search_products", query=categories[0])
        p2 = self._execute_tool("search_products", query=categories[1])
        p1_price = p1["results"][0]["price_usd"] if p1["results"] else 0
        p2_price = p2["results"][0]["price_usd"] if p2["results"] else 0
        cheaper_cat = categories[0] if p1_price <= p2_price else categories[1]
        cheaper_price = min(p1_price, p2_price)
        self._execute_tool(
            "create_notification",
            title=f"Best Deal: {cheaper_cat}",
            message=f"Cheapest top product is ${cheaper_price:.2f} in {cheaper_cat}",
            priority="high",
        )

        return Task(
            task_id=self._next_id(CompositionLevel.DAG),
            level=CompositionLevel.DAG,
            prompt=f"Search for top products in '{categories[0]}' and '{categories[1]}'. Compare the first result's price from each and send a notification about the cheaper option.",
            available_tools=["search_products", "create_notification", "calculator", "data_sort", "compare_texts", "send_message"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="search_products", arguments={"query": categories[0]}, output_key="p1"),
                    ToolCall(step_id="step_2", tool_name="search_products", arguments={"query": categories[1]}, output_key="p2"),
                    ToolCall(step_id="step_3", tool_name="create_notification", arguments={
                        "title": f"Best Deal: {cheaper_cat}",
                        "message": f"Cheapest top product is ${cheaper_price:.2f} in {cheaper_cat}",
                        "priority": "high",
                    }, depends_on=["step_1", "step_2"]),
                ],
                final_answer_source="step_3",
            ),
            expected_final_answer={"cheaper_category": cheaper_cat, "price": cheaper_price},
            metadata={"category": "dag", "tools": ["search_products", "create_notification"], "pattern": "parallel-merge-action"},
        )

    def _l3_kb_sentiment_store(self) -> Task:
        """KB query → sentiment of answer → store result in memory."""
        query = self._pick(KB_QUERIES)
        kb_result = self._execute_tool("knowledge_base_query", query=query)
        answer_text = kb_result.get("answer", query)
        sentiment = self._execute_tool("sentiment_analysis", text=answer_text)
        store_value = json.dumps({"query": query, "sentiment": sentiment["sentiment"], "confidence": sentiment["confidence"]})
        self._execute_tool("store_memory", key="kb_sentiment_result", value=store_value)

        return Task(
            task_id=self._next_id(CompositionLevel.DAG),
            level=CompositionLevel.DAG,
            prompt=f'Query the knowledge base for "{query}", analyze the sentiment of the answer, and store the result in memory under key "kb_sentiment_result".',
            available_tools=["knowledge_base_query", "sentiment_analysis", "store_memory", "classify_text", "retrieve_memory", "summarize_text"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="knowledge_base_query", arguments={"query": query}, output_key="kb"),
                    ToolCall(step_id="step_2", tool_name="sentiment_analysis", arguments={"text": answer_text}, depends_on=["step_1"], output_key="sentiment"),
                    ToolCall(step_id="step_3", tool_name="store_memory", arguments={
                        "key": "kb_sentiment_result",
                        "value": store_value,
                    }, depends_on=["step_2"]),
                ],
                final_answer_source="step_3",
            ),
            expected_final_answer={"sentiment": sentiment["sentiment"], "confidence": sentiment["confidence"]},
            metadata={"category": "dag", "tools": ["knowledge_base_query", "sentiment_analysis", "store_memory"], "pattern": "chain-with-analysis"},
        )

    def _l3_directions_weather_schedule(self) -> Task:
        """Directions(A→B) + Weather(B) → schedule meeting based on travel + conditions."""
        cities = self._pick_n(CITIES, 2)
        mode = self._pick(TRAVEL_MODES)
        title = self._pick(MEETING_TITLES)
        directions = self._execute_tool("get_directions", origin=cities[0], destination=cities[1], mode=mode)
        weather = self._execute_tool("get_weather", city=cities[1])
        duration_mins = directions.get("duration_minutes", 60)
        condition = weather.get("condition", "clear")

        meeting_duration = max(30, duration_mins)
        self._execute_tool(
            "schedule_meeting",
            title=title,
            datetime_str="2026-03-15 14:00",
            duration_minutes=meeting_duration,
            participants=["team@company.com"],
        )

        return Task(
            task_id=self._next_id(CompositionLevel.DAG),
            level=CompositionLevel.DAG,
            prompt=f"Get {mode} directions from {cities[0]} to {cities[1]} and check {cities[1]}'s weather. Schedule a '{title}' meeting on 2026-03-15 at 14:00 with duration matching the travel time (minimum 30 min) for team@company.com.",
            available_tools=["get_directions", "get_weather", "schedule_meeting", "calculate_date_diff", "convert_timezone", "create_task"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_directions", arguments={"origin": cities[0], "destination": cities[1], "mode": mode}, output_key="directions"),
                    ToolCall(step_id="step_2", tool_name="get_weather", arguments={"city": cities[1]}, output_key="weather"),
                    ToolCall(step_id="step_3", tool_name="schedule_meeting", arguments={
                        "title": title,
                        "datetime_str": "2026-03-15 14:00",
                        "duration_minutes": meeting_duration,
                        "participants": ["team@company.com"],
                    }, depends_on=["step_1", "step_2"]),
                ],
                final_answer_source="step_3",
            ),
            expected_final_answer={"travel_duration": duration_mins, "weather_condition": condition, "meeting_duration": meeting_duration},
            metadata={"category": "dag", "tools": ["get_directions", "get_weather", "schedule_meeting"], "pattern": "parallel-merge-action"},
        )

    def _l3_entities_classify_task(self) -> Task:
        """Extract entities from text + classify text → create task with findings."""
        text = self._pick(ENTITY_TEXTS)
        cat_set = self._pick(CLASSIFICATION_CATEGORIES)
        categories = cat_set[0]
        extracted = self._execute_tool("extract_entities", text=text)
        classified = self._execute_tool("classify_text", text=text, categories=categories)
        entity_names = [e["text"] for e in extracted.get("entities", [])][:3]
        category = classified.get("predicted_category", categories[0])

        self._execute_tool("create_task", title=f"Review: {category} entities", description=f"Found entities: {', '.join(entity_names)} in {category} text", priority="medium")

        return Task(
            task_id=self._next_id(CompositionLevel.DAG),
            level=CompositionLevel.DAG,
            prompt=f'Extract entities from the text: "{text[:80]}..." and classify it into categories {categories}. Create a task summarizing the entities found and the text category.',
            available_tools=["extract_entities", "classify_text", "create_task", "sentiment_analysis", "summarize_text", "store_memory"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="extract_entities", arguments={"text": text}, output_key="entities"),
                    ToolCall(step_id="step_2", tool_name="classify_text", arguments={"text": text, "categories": categories}, output_key="classified"),
                    ToolCall(step_id="step_3", tool_name="create_task", arguments={
                        "title": f"Review: {category} entities",
                        "description": f"Found entities: {', '.join(entity_names)} in {category} text",
                        "priority": "medium",
                    }, depends_on=["step_1", "step_2"]),
                ],
                final_answer_source="step_3",
            ),
            expected_final_answer={"entities": entity_names, "category": category},
            metadata={"category": "dag", "tools": ["extract_entities", "classify_text", "create_task"], "pattern": "parallel-merge-action"},
        )

    def _l3_stocks_aggregate_translate(self) -> Task:
        """Stock(A) + Stock(B) + Stock(C) → compute mean price → translate summary."""
        symbols = self._pick_n(STOCKS, 3)
        lang = self._pick(LANGUAGES)
        lang_code = LANGUAGE_CODES[lang]
        s_results = [self._execute_tool("get_stock_price", symbol=s) for s in symbols]
        prices = [s["price_usd"] for s in s_results]
        stats = self._execute_tool("statistical_analysis", numbers=prices, operation="mean")
        mean_price = stats["result"]
        summary_text = f"Average stock price of {', '.join(symbols)}: ${mean_price:.2f} USD"
        translated = self._execute_tool("translate_text", text=summary_text, from_language="en", to_language=lang_code)

        return Task(
            task_id=self._next_id(CompositionLevel.DAG),
            level=CompositionLevel.DAG,
            prompt=f"Get stock prices for {symbols[0]}, {symbols[1]}, and {symbols[2]}. Calculate their average price and translate a summary to {lang}.",
            available_tools=["get_stock_price", "statistical_analysis", "translate_text", "calculator", "get_exchange_rate", "summarize_text"],
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_stock_price", arguments={"symbol": symbols[0]}, output_key="s1"),
                    ToolCall(step_id="step_2", tool_name="get_stock_price", arguments={"symbol": symbols[1]}, output_key="s2"),
                    ToolCall(step_id="step_3", tool_name="get_stock_price", arguments={"symbol": symbols[2]}, output_key="s3"),
                    ToolCall(step_id="step_4", tool_name="statistical_analysis", arguments={"numbers": prices, "operation": "mean"}, depends_on=["step_1", "step_2", "step_3"], output_key="stats"),
                    ToolCall(step_id="step_5", tool_name="translate_text", arguments={"text": summary_text, "from_language": "en", "to_language": lang_code}, depends_on=["step_4"]),
                ],
                final_answer_source="step_5",
            ),
            expected_final_answer={"average_price": mean_price, "translated_summary": translated["translated_text"]},
            metadata={"category": "dag", "tools": ["get_stock_price", "statistical_analysis", "translate_text"], "pattern": "triple-parallel-merge-chain"},
        )

    # -----------------------------------------------------------------------
    # Generate full suite
    # -----------------------------------------------------------------------

    def generate_suite(
        self,
        l0_count: int = 50,
        l1_count: int = 50,
        l2_count: int = 30,
        l3_count: int = 20,
    ) -> TaskSuite:
        """Generate the complete benchmark suite."""
        tasks = []
        tasks.extend(self.generate_l0_tasks(l0_count))
        tasks.extend(self.generate_l1_tasks(l1_count))
        tasks.extend(self.generate_l2_tasks(l2_count))
        tasks.extend(self.generate_l3_tasks(l3_count))

        return TaskSuite(
            name="CompToolBench",
            version="0.1.0",
            tasks=tasks,
            metadata={
                "seed": self.seed,
                "mode": self.mode.value,
                "generation_params": {
                    "l0_count": l0_count,
                    "l1_count": l1_count,
                    "l2_count": l2_count,
                    "l3_count": l3_count,
                },
            },
        )

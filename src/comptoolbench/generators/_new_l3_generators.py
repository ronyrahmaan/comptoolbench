"""New L3 (DAG) generator methods for CompToolBench composition engine.

This module contains 10 additional L3 generators with diverse DAG patterns:
  1. read_extract_stats_save       - 4-step linear chain (file->extract->stats->save)
  2. search_keyword_search_summarize - Multi-hop search (search->keywords->search->summarize)
  3. detect_translate_summarize_email - 4-step NLP chain (detect->translate->summarize->email)
  4. multi_stock_pctchange_notify   - parallel->compute->notify
  5. weather_calendar_reminder      - 3+1 step chain with context (weather->calendar->reminder)
  6. multi_entity_mask_save         - parallel->process->save
  7. read_wordcount_pctchange_log   - True DAG: parallel reads->parallel counts->merge->log
  8. search_entities_sentiment_report - chain->fan-out->merge->chain
  9. multi_currency_minmax_format   - fan-out->aggregate->format
  10. kb_translate_hash_email       - chain->fan-out (KB->translate->hash+email parallel)

Usage:
    from comptoolbench.generators._new_l3_generators import (
        NEW_L3_PROMPT_VARIANTS,
        NEW_L3_REGISTRY,
    )
    # Merge into CompositionEngine._l3_generators and L3_PROMPT_VARIANTS
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

# Re-import parameter pools used by these generators
from comptoolbench.generators.composition_engine import (
    CITIES_EXPANDED,
    CURRENCIES_EXPANDED,
    EMAIL_RECIPIENTS_EXPANDED,
    ENTITY_TEXTS_EXPANDED,
    KB_QUERIES_EXPANDED,
    LANGUAGE_CODES_EXPANDED,
    LANGUAGES_EXPANDED,
    MEETING_TITLES_EXPANDED,
    SEARCH_QUERIES_EXPANDED,
    STOCKS_EXPANDED,
    TEXTS_EXPANDED,
)

# ---------------------------------------------------------------------------
# Prompt template variants (3-5 per new L3 pattern)
# ---------------------------------------------------------------------------

NEW_L3_PROMPT_VARIANTS: dict[str, list[str]] = {
    "read_extract_stats_save": [
        "Read the file at {path}, extract all numbers, compute statistics, and save the results to {out_path}.",
        "Open {path}, pull out numeric values, run statistical analysis, and write the output to {out_path}.",
        "Load {path}, extract numbers from its contents, calculate stats (mean, median, etc.), and save to {out_path}.",
    ],
    "search_keyword_search_summarize": [
        'Search for "{query}", extract keywords from the results, search again using the top keyword, and summarize the final results.',
        'Do a web search for "{query}", find the most important keywords, perform a follow-up search with the top keyword, then summarize.',
        'Look up "{query}" online, extract keywords, use the best keyword for a deeper search, and provide a summary.',
    ],
    "detect_translate_summarize_email": [
        'Detect the language of the text "{text_preview}", translate it to English, summarize it, and email the summary to {email}.',
        'Figure out what language "{text_preview}" is in, translate to English, create a summary, then send it to {email}.',
        'Identify the language of "{text_preview}", convert it to English, summarize, and email {email} with the result.',
    ],
    "multi_stock_pctchange_notify": [
        "Compare {s1} and {s2} stock prices, calculate the percentage difference, and send a notification.",
        "Get the prices of {s1} and {s2}, compute the percent change between them, and create a notification with the result.",
        "Look up {s1} and {s2} stocks, find the percentage difference, and notify about the comparison.",
    ],
    "weather_calendar_reminder": [
        "Check the weather in {city}, create a calendar event for an outdoor {event_title} on {date}, and set a reminder.",
        "Get the weather forecast for {city}, schedule an outdoor {event_title} for {date}, and set a reminder for it.",
        "Look up {city}'s weather, create a {event_title} event on {date}, and set a reminder about the event.",
    ],
    "multi_entity_mask_save": [
        "Extract entities from two texts, combine the results, mask any PII, and save the cleaned text to {path}.",
        "Find named entities in two different texts, then mask personally identifiable information from the combined text and write it to {path}.",
        "Pull entities from two texts simultaneously, mask PII in the merged entity list, and save to {path}.",
    ],
    "read_wordcount_pctchange_log": [
        "Read {p1} and {p2} in parallel, count the words in each, calculate the percentage difference in word counts, and log the result.",
        "Open files {p1} and {p2} simultaneously, get word counts for both, compute the percent change, and log it.",
        "Load {p1} and {p2}, count words in each file, find the percentage change between the counts, and create a log entry.",
    ],
    "search_entities_sentiment_report": [
        'Search for "{query}", extract entities and analyze sentiment in parallel, generate a report, and email it to {email}.',
        'Look up "{query}", then simultaneously extract entities and determine sentiment, compile a report, and send it to {email}.',
        'Web search "{query}", run entity extraction and sentiment analysis on the results at the same time, create a report, and email {email}.',
    ],
    "multi_currency_minmax_format": [
        "Convert {amount} {base} to {c1}, {c2}, and {c3} simultaneously, find the min and max converted values, and format the result.",
        "Exchange {amount} {base} into {c1}, {c2}, and {c3} in parallel, determine which gives the highest and lowest amounts, and format the output.",
        "Convert {amount} {base} to three currencies ({c1}, {c2}, {c3}), find the best and worst rates, and format the numbers.",
    ],
    "kb_translate_hash_email": [
        'Look up "{query}" in the knowledge base, translate the answer to {lang}, then hash it and email it to {email} in parallel.',
        'Query the KB for "{query}", translate the response to {lang}, and simultaneously compute its hash and send it via email to {email}.',
        'Answer "{query}" from the knowledge base, translate to {lang}, then both hash the translated text and email {email} at the same time.',
    ],
}


# ---------------------------------------------------------------------------
# Generator static methods
# ---------------------------------------------------------------------------


def _l3_read_extract_stats_save(engine: CompositionEngine) -> Task:
    """L3 DAG: read_file -> extract_numbers -> statistical_analysis -> write_file.

    Pattern: 4-step linear chain.
    """
    # Pick a file that has numeric content
    path = engine._pick(["/data/report.txt", "/data/employees.csv"])
    out_path = engine._pick(
        [
            "/data/stats_output.json",
            "/data/analysis_results.json",
            "/data/numbers_report.json",
        ]
    )

    # Step 1: read file
    file_data = engine._execute_tool("read_file", path=path)
    content = file_data.get("content", "")

    # Step 2: extract numbers from the content
    extracted = engine._execute_tool("extract_numbers", text=content[:500])
    numbers = extracted.get("numbers", [])

    # Step 3: statistical analysis on extracted numbers
    # Ensure we have enough numbers; fallback to placeholder if empty
    if len(numbers) < 2:
        numbers = [10, 20, 30, 40, 50]
    stats = engine._execute_tool("statistical_analysis", numbers=numbers)

    # Step 4: write results
    stats_text = json.dumps(
        {"source_file": path, "numbers": numbers, "statistics": stats}, default=str
    )
    written = engine._execute_tool("write_file", path=out_path, content=stats_text)

    prompt = engine._format_prompt(
        "read_extract_stats_save",
        NEW_L3_PROMPT_VARIANTS,
        path=path,
        out_path=out_path,
    )
    used_tools = ["read_file", "extract_numbers", "statistical_analysis", "write_file"]
    distractors = engine._pick_distractors(used_tools)

    return Task(
        task_id=engine._next_id(CompositionLevel.DAG),
        level=CompositionLevel.DAG,
        prompt=prompt,
        available_tools=used_tools + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1",
                    tool_name="read_file",
                    arguments={"path": path},
                    output_key="file_data",
                    depends_on=[],
                ),
                ToolCall(
                    step_id="step_2",
                    tool_name="extract_numbers",
                    arguments={"text": content[:500]},
                    output_key="extracted",
                    depends_on=["step_1"],
                ),
                ToolCall(
                    step_id="step_3",
                    tool_name="statistical_analysis",
                    arguments={"numbers": numbers},
                    output_key="stats",
                    depends_on=["step_2"],
                ),
                ToolCall(
                    step_id="step_4",
                    tool_name="write_file",
                    arguments={"path": out_path, "content": stats_text},
                    depends_on=["step_3"],
                ),
            ],
            final_answer_source="step_4",
        ),
        expected_final_answer=written,
        metadata={
            "category": "dag",
            "tools": used_tools,
            "pattern": "linear-chain-4-file-stats",
        },
    )


def _l3_search_keyword_search_summarize(engine: CompositionEngine) -> Task:
    """L3 DAG: web_search -> keyword_extract -> web_search -> summarize_text.

    Pattern: Multi-hop search. Search, extract keywords, search again with top keyword, summarize.
    """
    query = engine._pick(SEARCH_QUERIES_EXPANDED)

    # Step 1: initial web search
    results = engine._execute_tool("web_search", query=query)
    result_text = json.dumps(results.get("results", results), default=str)[:500]

    # Step 2: extract keywords from results
    keywords = engine._execute_tool("keyword_extract", text=result_text)
    kw_list = keywords.get("keywords", [])
    top_keyword = kw_list[0] if kw_list else query.split()[0]

    # Step 3: second search with top keyword
    results2 = engine._execute_tool("web_search", query=top_keyword)
    result2_text = json.dumps(results2.get("results", results2), default=str)[:500]

    # Step 4: summarize the second search results
    summary = engine._execute_tool("summarize_text", text=result2_text)

    prompt = engine._format_prompt(
        "search_keyword_search_summarize",
        NEW_L3_PROMPT_VARIANTS,
        query=query,
    )
    used_tools = ["web_search", "keyword_extract", "summarize_text"]
    distractors = engine._pick_distractors(used_tools)

    return Task(
        task_id=engine._next_id(CompositionLevel.DAG),
        level=CompositionLevel.DAG,
        prompt=prompt,
        available_tools=used_tools + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1",
                    tool_name="web_search",
                    arguments={"query": query},
                    output_key="results1",
                    depends_on=[],
                ),
                ToolCall(
                    step_id="step_2",
                    tool_name="keyword_extract",
                    arguments={"text": result_text},
                    output_key="keywords",
                    depends_on=["step_1"],
                ),
                ToolCall(
                    step_id="step_3",
                    tool_name="web_search",
                    arguments={"query": top_keyword},
                    output_key="results2",
                    depends_on=["step_2"],
                ),
                ToolCall(
                    step_id="step_4",
                    tool_name="summarize_text",
                    arguments={"text": result2_text},
                    output_key="summary",
                    depends_on=["step_3"],
                ),
            ],
            final_answer_source="step_4",
        ),
        expected_final_answer=summary,
        metadata={
            "category": "dag",
            "tools": used_tools,
            "pattern": "multi-hop-search",
        },
    )


def _l3_detect_translate_summarize_email(engine: CompositionEngine) -> Task:
    """L3 DAG: detect_language -> translate_text -> summarize_text -> send_email.

    Pattern: 4-step linear NLP chain.
    """
    text = engine._pick(TEXTS_EXPANDED)
    email = engine._pick(EMAIL_RECIPIENTS_EXPANDED)

    # Step 1: detect language
    detected = engine._execute_tool("detect_language", text=text)
    detected_lang = detected.get("language_code", "en")

    # Step 2: translate to English using detected source language
    from_lang = detected_lang if detected_lang != "und" else "en"
    translated = engine._execute_tool(
        "translate_text", text=text, from_language=from_lang, to_language="en"
    )
    translated_text = translated.get("translated_text", text)[:500]

    # Step 3: summarize the translated text
    summary = engine._execute_tool("summarize_text", text=translated_text)
    summary_text = summary.get("summary", translated_text)[:300]

    # Step 4: email the summary
    sent = engine._execute_tool(
        "send_email", to=email, subject="Translated Summary", body=summary_text
    )

    prompt = engine._format_prompt(
        "detect_translate_summarize_email",
        NEW_L3_PROMPT_VARIANTS,
        text_preview=text[:60] + "...",
        email=email,
    )
    used_tools = ["detect_language", "translate_text", "summarize_text", "send_email"]
    distractors = engine._pick_distractors(used_tools)

    return Task(
        task_id=engine._next_id(CompositionLevel.DAG),
        level=CompositionLevel.DAG,
        prompt=prompt,
        available_tools=used_tools + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1",
                    tool_name="detect_language",
                    arguments={"text": text},
                    output_key="detected",
                    depends_on=[],
                ),
                ToolCall(
                    step_id="step_2",
                    tool_name="translate_text",
                    arguments={
                        "text": text,
                        "from_language": from_lang,
                        "to_language": "en",
                    },
                    output_key="translated",
                    depends_on=["step_1"],
                ),
                ToolCall(
                    step_id="step_3",
                    tool_name="summarize_text",
                    arguments={"text": translated_text},
                    output_key="summary",
                    depends_on=["step_2"],
                ),
                ToolCall(
                    step_id="step_4",
                    tool_name="send_email",
                    arguments={
                        "to": email,
                        "subject": "Translated Summary",
                        "body": summary_text,
                    },
                    depends_on=["step_3"],
                ),
            ],
            final_answer_source="step_4",
        ),
        expected_final_answer=sent,
        metadata={
            "category": "dag",
            "tools": used_tools,
            "pattern": "linear-chain-4-nlp",
        },
    )


def _l3_multi_stock_pctchange_notify(engine: CompositionEngine) -> Task:
    """L3 DAG: 2x get_stock_price parallel -> percentage_change -> create_notification.

    Pattern: parallel -> compute -> notify.
    """
    symbols = engine._pick_n(STOCKS_EXPANDED, 2)

    # Steps 1 & 2: get stock prices in parallel
    s1 = engine._execute_tool("get_stock_price", symbol=symbols[0])
    s2 = engine._execute_tool("get_stock_price", symbol=symbols[1])
    p1 = s1.get("price", s1.get("price_usd", 100))
    p2 = s2.get("price", s2.get("price_usd", 100))

    # Step 3: compute percentage change
    pct = engine._execute_tool("percentage_change", old_value=p1, new_value=p2)
    pct_val = pct.get("change_percent", pct.get("percentage_change", 0))
    direction = pct.get("direction", "changed")

    # Step 4: create notification
    notif_msg = f"{symbols[1]} is {pct_val:.1f}% {direction} compared to {symbols[0]}"
    notif = engine._execute_tool(
        "create_notification",
        title="Stock Comparison",
        message=notif_msg,
        priority="medium",
    )

    prompt = engine._format_prompt(
        "multi_stock_pctchange_notify",
        NEW_L3_PROMPT_VARIANTS,
        s1=symbols[0],
        s2=symbols[1],
    )
    used_tools = ["get_stock_price", "percentage_change", "create_notification"]
    distractors = engine._pick_distractors(used_tools)

    return Task(
        task_id=engine._next_id(CompositionLevel.DAG),
        level=CompositionLevel.DAG,
        prompt=prompt,
        available_tools=used_tools + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1",
                    tool_name="get_stock_price",
                    arguments={"symbol": symbols[0]},
                    output_key="s1",
                    depends_on=[],
                ),
                ToolCall(
                    step_id="step_2",
                    tool_name="get_stock_price",
                    arguments={"symbol": symbols[1]},
                    output_key="s2",
                    depends_on=[],
                ),
                ToolCall(
                    step_id="step_3",
                    tool_name="percentage_change",
                    arguments={"old_value": p1, "new_value": p2},
                    depends_on=["step_1", "step_2"],
                ),
                ToolCall(
                    step_id="step_4",
                    tool_name="create_notification",
                    arguments={
                        "title": "Stock Comparison",
                        "message": notif_msg,
                        "priority": "medium",
                    },
                    depends_on=["step_3"],
                ),
            ],
            final_answer_source="step_4",
        ),
        expected_final_answer=notif,
        metadata={
            "category": "dag",
            "tools": used_tools,
            "pattern": "parallel-compute-notify",
        },
    )


def _l3_weather_calendar_reminder(engine: CompositionEngine) -> Task:
    """L3 DAG: get_weather -> create_calendar_event -> set_reminder.

    Pattern: 3-step chain with contextual branching.
    Weather informs the calendar event description; reminder links to the event.
    We add a 4th step (word_count on weather description) to meet L3 minimum.
    """
    city = engine._pick(CITIES_EXPANDED)
    event_title = engine._pick(MEETING_TITLES_EXPANDED)
    date = engine._pick(
        [
            "2026-04-15 10:00",
            "2026-05-01 14:00",
            "2026-06-10 09:00",
            "2026-07-20 11:00",
            "2026-08-05 16:00",
            "2026-09-12 08:30",
        ]
    )

    # Step 1: get weather
    weather = engine._execute_tool("get_weather", city=city)
    temp = weather.get("temperature_celsius", 20)
    condition = weather.get("condition", "Clear")
    weather_desc = f"Weather in {city}: {temp}°C, {condition}"

    # Step 2: create calendar event informed by weather
    _event = engine._execute_tool(
        "create_calendar_event",
        title=f"{event_title} (Outdoor - {condition})",
        date=date,
        duration_minutes=60,
    )
    # Step 3: set reminder for the event
    remind_at = date  # remind at event time
    reminder = engine._execute_tool(
        "set_reminder",
        message=f"Reminder: {event_title} in {city}. {weather_desc}",
        remind_at=remind_at,
    )

    # Step 4: word_count on the weather description for reporting
    _wc = engine._execute_tool("word_count", text=weather_desc)

    prompt = engine._format_prompt(
        "weather_calendar_reminder",
        NEW_L3_PROMPT_VARIANTS,
        city=city,
        event_title=event_title,
        date=date,
    )
    used_tools = ["get_weather", "create_calendar_event", "set_reminder", "word_count"]
    distractors = engine._pick_distractors(used_tools)

    return Task(
        task_id=engine._next_id(CompositionLevel.DAG),
        level=CompositionLevel.DAG,
        prompt=prompt,
        available_tools=used_tools + distractors,
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
                    tool_name="create_calendar_event",
                    arguments={
                        "title": f"{event_title} (Outdoor - {condition})",
                        "date": date,
                        "duration_minutes": 60,
                    },
                    output_key="event",
                    depends_on=["step_1"],
                ),
                ToolCall(
                    step_id="step_3",
                    tool_name="set_reminder",
                    arguments={
                        "message": f"Reminder: {event_title} in {city}. {weather_desc}",
                        "remind_at": remind_at,
                    },
                    output_key="reminder",
                    depends_on=["step_2"],
                ),
                ToolCall(
                    step_id="step_4",
                    tool_name="word_count",
                    arguments={"text": weather_desc},
                    depends_on=["step_1"],
                ),
            ],
            final_answer_source="step_3",
        ),
        expected_final_answer=reminder,
        metadata={
            "category": "dag",
            "tools": used_tools,
            "pattern": "chain-with-branch",
        },
    )


def _l3_multi_entity_mask_save(engine: CompositionEngine) -> Task:
    """L3 DAG: 2x extract_entities parallel -> mask_pii -> write_file.

    Pattern: parallel -> process -> save.
    """
    texts = engine._pick_n(ENTITY_TEXTS_EXPANDED, 2)

    # Steps 1 & 2: extract entities from both texts in parallel
    ent1 = engine._execute_tool("extract_entities", text=texts[0])
    ent2 = engine._execute_tool("extract_entities", text=texts[1])

    # Combine both texts for PII masking
    combined = f"{texts[0]}\n---\n{texts[1]}"

    # Step 3: mask PII in combined text
    masked = engine._execute_tool("mask_pii", text=combined)
    masked_text = masked.get("masked_text", combined)

    # Step 4: write masked text to file
    out_path = engine._pick(
        [
            "/data/entities_masked.txt",
            "/data/pii_cleaned.txt",
            "/data/safe_entities.txt",
        ]
    )
    save_content = json.dumps(
        {
            "entities_text1": ent1.get("entities", []),
            "entities_text2": ent2.get("entities", []),
            "masked_text": masked_text,
        },
        default=str,
    )
    written = engine._execute_tool("write_file", path=out_path, content=save_content)

    prompt = engine._format_prompt(
        "multi_entity_mask_save",
        NEW_L3_PROMPT_VARIANTS,
        path=out_path,
    )
    used_tools = ["extract_entities", "mask_pii", "write_file"]
    distractors = engine._pick_distractors(used_tools)

    return Task(
        task_id=engine._next_id(CompositionLevel.DAG),
        level=CompositionLevel.DAG,
        prompt=prompt,
        available_tools=used_tools + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1",
                    tool_name="extract_entities",
                    arguments={"text": texts[0]},
                    output_key="ent1",
                    depends_on=[],
                ),
                ToolCall(
                    step_id="step_2",
                    tool_name="extract_entities",
                    arguments={"text": texts[1]},
                    output_key="ent2",
                    depends_on=[],
                ),
                ToolCall(
                    step_id="step_3",
                    tool_name="mask_pii",
                    arguments={"text": combined},
                    depends_on=["step_1", "step_2"],
                ),
                ToolCall(
                    step_id="step_4",
                    tool_name="write_file",
                    arguments={"path": out_path, "content": save_content},
                    depends_on=["step_3"],
                ),
            ],
            final_answer_source="step_4",
        ),
        expected_final_answer=written,
        metadata={
            "category": "dag",
            "tools": used_tools,
            "pattern": "parallel-process-save",
        },
    )


def _l3_read_wordcount_pctchange_log(engine: CompositionEngine) -> Task:
    """L3 DAG: read_file(1) + read_file(2) parallel -> word_count each -> percentage_change -> log_event.

    Pattern: True DAG with 6 steps.
    Read 2 files in parallel, count words in each (parallel), compute % change, log.
    """
    paths = engine._pick_n(
        [
            "/data/report.txt",
            "/data/notes.txt",
            "/data/employees.csv",
            "/data/config.json",
            "/data/products.json",
        ],
        2,
    )

    # Steps 1 & 2: read files in parallel
    f1 = engine._execute_tool("read_file", path=paths[0])
    f2 = engine._execute_tool("read_file", path=paths[1])
    content1 = f1.get("content", "")
    content2 = f2.get("content", "")

    # Steps 3 & 4: word count on each file's content (parallel after their respective reads)
    wc1 = engine._execute_tool("word_count", text=content1)
    wc2 = engine._execute_tool("word_count", text=content2)
    count1 = wc1.get("words", 1)
    count2 = wc2.get("words", 1)

    # Ensure non-zero for percentage_change
    if count1 == 0:
        count1 = 1
    if count2 == 0:
        count2 = 1

    # Step 5: percentage change between word counts
    pct = engine._execute_tool("percentage_change", old_value=count1, new_value=count2)
    pct_val = pct.get("change_percent", 0)

    # Step 6: log the result
    log_msg = (
        f"Word count comparison: {paths[0]} has {count1} words, "
        f"{paths[1]} has {count2} words. Change: {pct_val:.1f}%"
    )
    logged = engine._execute_tool(
        "log_event",
        event_type="word_count_comparison",
        message=log_msg,
        severity="info",
    )

    prompt = engine._format_prompt(
        "read_wordcount_pctchange_log",
        NEW_L3_PROMPT_VARIANTS,
        p1=paths[0],
        p2=paths[1],
    )
    used_tools = ["read_file", "word_count", "percentage_change", "log_event"]
    distractors = engine._pick_distractors(used_tools)

    return Task(
        task_id=engine._next_id(CompositionLevel.DAG),
        level=CompositionLevel.DAG,
        prompt=prompt,
        available_tools=used_tools + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1",
                    tool_name="read_file",
                    arguments={"path": paths[0]},
                    output_key="f1",
                    depends_on=[],
                ),
                ToolCall(
                    step_id="step_2",
                    tool_name="read_file",
                    arguments={"path": paths[1]},
                    output_key="f2",
                    depends_on=[],
                ),
                ToolCall(
                    step_id="step_3",
                    tool_name="word_count",
                    arguments={"text": content1},
                    output_key="wc1",
                    depends_on=["step_1"],
                ),
                ToolCall(
                    step_id="step_4",
                    tool_name="word_count",
                    arguments={"text": content2},
                    output_key="wc2",
                    depends_on=["step_2"],
                ),
                ToolCall(
                    step_id="step_5",
                    tool_name="percentage_change",
                    arguments={"old_value": count1, "new_value": count2},
                    depends_on=["step_3", "step_4"],
                ),
                ToolCall(
                    step_id="step_6",
                    tool_name="log_event",
                    arguments={
                        "event_type": "word_count_comparison",
                        "message": log_msg,
                        "severity": "info",
                    },
                    depends_on=["step_5"],
                ),
            ],
            final_answer_source="step_6",
        ),
        expected_final_answer=logged,
        metadata={
            "category": "dag",
            "tools": used_tools,
            "pattern": "true-dag-parallel-reads-merge",
        },
    )


def _l3_search_entities_sentiment_report(engine: CompositionEngine) -> Task:
    """L3 DAG: web_search -> (extract_entities + sentiment_analysis parallel) -> generate_report -> send_email.

    Pattern: chain -> fan-out -> merge -> chain.
    """
    query = engine._pick(SEARCH_QUERIES_EXPANDED)
    email = engine._pick(EMAIL_RECIPIENTS_EXPANDED)

    # Step 1: web search
    results = engine._execute_tool("web_search", query=query)
    result_text = json.dumps(results.get("results", results), default=str)[:500]

    # Steps 2 & 3: extract entities AND sentiment analysis in parallel
    entities = engine._execute_tool("extract_entities", text=result_text)
    sentiment = engine._execute_tool("sentiment_analysis", text=result_text)

    entity_list = entities.get("entities", [])
    sentiment_label = sentiment.get("sentiment", "neutral")
    confidence = sentiment.get("confidence", 0.5)

    # Step 4: generate report combining both
    report_data = json.dumps(
        {
            "query": query,
            "entities_found": len(entity_list),
            "top_entities": [e.get("text", "") for e in entity_list[:5]],
            "sentiment": sentiment_label,
            "sentiment_confidence": confidence,
        },
        default=str,
    )
    report = engine._execute_tool(
        "generate_report",
        title=f"Analysis Report: {query}",
        data=report_data,
        format="text",
    )
    report_text = report.get("report", "Report generated.")[:500]

    # Step 5: email the report
    sent = engine._execute_tool(
        "send_email",
        to=email,
        subject=f"Analysis: {query}",
        body=report_text,
    )

    prompt = engine._format_prompt(
        "search_entities_sentiment_report",
        NEW_L3_PROMPT_VARIANTS,
        query=query,
        email=email,
    )
    used_tools = [
        "web_search",
        "extract_entities",
        "sentiment_analysis",
        "generate_report",
        "send_email",
    ]
    distractors = engine._pick_distractors(used_tools)

    return Task(
        task_id=engine._next_id(CompositionLevel.DAG),
        level=CompositionLevel.DAG,
        prompt=prompt,
        available_tools=used_tools + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1",
                    tool_name="web_search",
                    arguments={"query": query},
                    output_key="results",
                    depends_on=[],
                ),
                ToolCall(
                    step_id="step_2",
                    tool_name="extract_entities",
                    arguments={"text": result_text},
                    output_key="entities",
                    depends_on=["step_1"],
                ),
                ToolCall(
                    step_id="step_3",
                    tool_name="sentiment_analysis",
                    arguments={"text": result_text},
                    output_key="sentiment",
                    depends_on=["step_1"],
                ),
                ToolCall(
                    step_id="step_4",
                    tool_name="generate_report",
                    arguments={
                        "title": f"Analysis Report: {query}",
                        "data": report_data,
                        "format": "text",
                    },
                    output_key="report",
                    depends_on=["step_2", "step_3"],
                ),
                ToolCall(
                    step_id="step_5",
                    tool_name="send_email",
                    arguments={
                        "to": email,
                        "subject": f"Analysis: {query}",
                        "body": report_text,
                    },
                    depends_on=["step_4"],
                ),
            ],
            final_answer_source="step_5",
        ),
        expected_final_answer=sent,
        metadata={
            "category": "dag",
            "tools": used_tools,
            "pattern": "chain-fanout-merge-chain",
        },
    )


def _l3_multi_currency_minmax_format(engine: CompositionEngine) -> Task:
    """L3 DAG: 3x get_exchange_rate parallel -> min_max -> format_number.

    Pattern: fan-out -> aggregate -> format.
    """
    base = "USD"
    amount = engine._pick([100, 250, 500, 1000, 2500])
    targets = engine._pick_n([c for c in CURRENCIES_EXPANDED if c != base], 3)

    # Steps 1-3: convert to 3 currencies in parallel
    conversions = []
    converted_amounts: list[float] = []
    for curr in targets:
        conv = engine._execute_tool(
            "get_exchange_rate",
            from_currency=base,
            to_currency=curr,
            amount=amount,
        )
        conversions.append(conv)
        converted_amounts.append(conv.get("converted_amount", amount))

    # Step 4: find min and max
    minmax = engine._execute_tool("min_max", numbers=converted_amounts)

    # Step 5: format the max value
    max_val = minmax.get("max", max(converted_amounts))
    formatted = engine._execute_tool("format_number", value=max_val, format="comma")

    prompt = engine._format_prompt(
        "multi_currency_minmax_format",
        NEW_L3_PROMPT_VARIANTS,
        amount=amount,
        base=base,
        c1=targets[0],
        c2=targets[1],
        c3=targets[2],
    )
    used_tools = ["get_exchange_rate", "min_max", "format_number"]
    distractors = engine._pick_distractors(used_tools)

    return Task(
        task_id=engine._next_id(CompositionLevel.DAG),
        level=CompositionLevel.DAG,
        prompt=prompt,
        available_tools=used_tools + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1",
                    tool_name="get_exchange_rate",
                    arguments={
                        "from_currency": base,
                        "to_currency": targets[0],
                        "amount": amount,
                    },
                    output_key="conv1",
                    depends_on=[],
                ),
                ToolCall(
                    step_id="step_2",
                    tool_name="get_exchange_rate",
                    arguments={
                        "from_currency": base,
                        "to_currency": targets[1],
                        "amount": amount,
                    },
                    output_key="conv2",
                    depends_on=[],
                ),
                ToolCall(
                    step_id="step_3",
                    tool_name="get_exchange_rate",
                    arguments={
                        "from_currency": base,
                        "to_currency": targets[2],
                        "amount": amount,
                    },
                    output_key="conv3",
                    depends_on=[],
                ),
                ToolCall(
                    step_id="step_4",
                    tool_name="min_max",
                    arguments={"numbers": converted_amounts},
                    depends_on=["step_1", "step_2", "step_3"],
                ),
                ToolCall(
                    step_id="step_5",
                    tool_name="format_number",
                    arguments={"value": max_val, "format": "comma"},
                    depends_on=["step_4"],
                ),
            ],
            final_answer_source="step_5",
        ),
        expected_final_answer=formatted,
        metadata={
            "category": "dag",
            "tools": used_tools,
            "pattern": "fan-out-aggregate-format",
        },
    )


def _l3_kb_translate_hash_email(engine: CompositionEngine) -> Task:
    """L3 DAG: knowledge_base_query -> translate_text -> (hash_text + send_email parallel).

    Pattern: chain -> fan-out. Query KB, translate answer, then hash AND email in parallel.
    """
    query = engine._pick(KB_QUERIES_EXPANDED)
    lang = engine._pick(LANGUAGES_EXPANDED)
    code = LANGUAGE_CODES_EXPANDED.get(lang, "es")
    email = engine._pick(EMAIL_RECIPIENTS_EXPANDED)

    # Step 1: query knowledge base
    kb_result = engine._execute_tool("knowledge_base_query", query=query)
    answer = kb_result.get("answer", "No answer found")[:300]

    # Step 2: translate the answer
    translated = engine._execute_tool(
        "translate_text", text=answer, from_language="en", to_language=code
    )
    translated_text = translated.get("translated_text", answer)[:300]

    # Steps 3 & 4 (parallel): hash and email
    _hash_result = engine._execute_tool(
        "hash_text", text=translated_text, algorithm="sha256"
    )
    sent = engine._execute_tool(
        "send_email",
        to=email,
        subject=f"KB Answer: {query[:50]}",
        body=translated_text,
    )

    prompt = engine._format_prompt(
        "kb_translate_hash_email",
        NEW_L3_PROMPT_VARIANTS,
        query=query,
        lang=lang,
        email=email,
    )
    used_tools = ["knowledge_base_query", "translate_text", "hash_text", "send_email"]
    distractors = engine._pick_distractors(used_tools)

    return Task(
        task_id=engine._next_id(CompositionLevel.DAG),
        level=CompositionLevel.DAG,
        prompt=prompt,
        available_tools=used_tools + distractors,
        expected_trace=ExpectedTrace(
            steps=[
                ToolCall(
                    step_id="step_1",
                    tool_name="knowledge_base_query",
                    arguments={"query": query},
                    output_key="kb_result",
                    depends_on=[],
                ),
                ToolCall(
                    step_id="step_2",
                    tool_name="translate_text",
                    arguments={
                        "text": answer,
                        "from_language": "en",
                        "to_language": code,
                    },
                    output_key="translated",
                    depends_on=["step_1"],
                ),
                ToolCall(
                    step_id="step_3",
                    tool_name="hash_text",
                    arguments={"text": translated_text, "algorithm": "sha256"},
                    depends_on=["step_2"],
                ),
                ToolCall(
                    step_id="step_4",
                    tool_name="send_email",
                    arguments={
                        "to": email,
                        "subject": f"KB Answer: {query[:50]}",
                        "body": translated_text,
                    },
                    depends_on=["step_2"],
                ),
            ],
            final_answer_source="step_4",
        ),
        expected_final_answer=sent,
        metadata={
            "category": "dag",
            "tools": used_tools,
            "pattern": "chain-fan-out",
        },
    )


# ---------------------------------------------------------------------------
# Registry: maps name -> function reference
# ---------------------------------------------------------------------------

NEW_L3_REGISTRY: dict[str, Any] = {
    "read_extract_stats_save": _l3_read_extract_stats_save,
    "search_keyword_search_summarize": _l3_search_keyword_search_summarize,
    "detect_translate_summarize_email": _l3_detect_translate_summarize_email,
    "multi_stock_pctchange_notify": _l3_multi_stock_pctchange_notify,
    "weather_calendar_reminder": _l3_weather_calendar_reminder,
    "multi_entity_mask_save": _l3_multi_entity_mask_save,
    "read_wordcount_pctchange_log": _l3_read_wordcount_pctchange_log,
    "search_entities_sentiment_report": _l3_search_entities_sentiment_report,
    "multi_currency_minmax_format": _l3_multi_currency_minmax_format,
    "kb_translate_hash_email": _l3_kb_translate_hash_email,
}

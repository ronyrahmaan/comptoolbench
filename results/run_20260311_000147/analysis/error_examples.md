# CompToolBench — Qualitative Error Examples

Concrete examples for each error type, showing task prompt, expected tool calls, and model output.

---

## E10_format_error

**Description:** Model output could not be parsed as tool calls.

### Example 1 (GPT-4o, L0_node_0003)

**Prompt:**
> Please encode a plain-text string to its Base64 representation.

**Available tools:** base64_encode, knowledge_base_query, business_days_between, read_file

**Expected tool calls:**

- `step_1`: **base64_encode**(text='The food industry is experiencing disruption from plant-based alternatives and lab-grown meat. Consumer preferences are shifting rapidly.')


**Model output (from scoring):**

- `step_1`: MISSING, tool=WRONG, args_score=0.00


**Score:** overall=0.00, tool_seq=0.00, args=0.00, completeness=0.00, data_flow=1.00

**Error classification:** `E10_format_error`

---

### Example 2 (GPT-4o, L0_node_0004)

**Prompt:**
> Please encode a plain-text string to its Base64 representation.

**Available tools:** base64_encode, standard_deviation, percentile, compare_texts

**Expected tool calls:**

- `step_1`: **base64_encode**(text='Quantum computing promises to revolutionize how we process information. Unlike classical bits, quantum bits can exist in multiple states simultaneously.')


**Model output (from scoring):**

- `step_1`: MISSING, tool=WRONG, args_score=0.00


**Score:** overall=0.00, tool_seq=0.00, args=0.00, completeness=0.00, data_flow=1.00

**Error classification:** `E10_format_error`

---

## E4_wrong_arguments

**Description:** Correct tool selected but with incorrect arguments.

### Example 1 (GPT-4o, L0_node_0021)

**Prompt:**
> Create a calendar event with a title, date, duration, and optional list of attendees. Returns a confirmation with the event ID with the following: title: Retrospective, date: 2026-02-14, duration minutes: 91.

**Available tools:** create_calendar_event, data_sort, schedule_meeting, correlation

**Expected tool calls:**

- `step_1`: **create_calendar_event**(title='Retrospective', date='2026-02-14', duration_minutes=91, attendees=['item_0', 'item_1', 'item_2'])


**Model output (from scoring):**

- `step_1`: MATCHED, tool=correct, args_score=0.50


**Score:** overall=0.00, tool_seq=1.00, args=0.50, completeness=1.00, data_flow=1.00

**Error classification:** `E4_wrong_arguments`

---

### Example 2 (GPT-4o, L0_node_0022)

**Prompt:**
> Can you create a calendar event with a title, date, duration, and optional list of attendees. Returns a confirmation with the event ID? The parameters are title: Marketing Sync, date: 2026-01-01, duration minutes: 93.

**Available tools:** create_calendar_event, merge_data, ip_geolocation, create_contact

**Expected tool calls:**

- `step_1`: **create_calendar_event**(title='Marketing Sync', date='2026-01-01', duration_minutes=93, attendees=['item_0', 'item_1', 'item_2', 'item_3', 'item_4'])


**Model output (from scoring):**

- `step_1`: MATCHED, tool=correct, args_score=0.50


**Score:** overall=0.00, tool_seq=1.00, args=0.50, completeness=1.00, data_flow=1.00

**Error classification:** `E4_wrong_arguments`

---

## E7_unnecessary_tool

**Description:** Model made extra, unnecessary tool calls.

### Example 1 (GPT-4o Mini, L0_node_0006)

**Prompt:**
> Please calculate the difference between two dates in days, weeks, months, or years — specifically, date1: 2025-11-15, date2: 2026-02-01.

**Available tools:** calculate_date_diff, string_replace, clamp_value, get_weather

**Expected tool calls:**

- `step_1`: **calculate_date_diff**(date1='2025-11-15', date2='2026-02-01')


**Model output (from scoring):**

- `step_1`: MATCHED, tool=correct, args_score=1.00


**Score:** overall=1.00, tool_seq=1.00, args=1.00, completeness=1.00, data_flow=1.00

**Error classification:** `E7_unnecessary_tool`

---

### Example 2 (GPT-4o Mini, L0_node_0007)

**Prompt:**
> I need you to calculate the difference between two dates in days, weeks, months, or years. Here are the details: date1: 2026-04-01, date2: 2026-02-14.

**Available tools:** calculate_date_diff, compare_texts, string_replace, get_exchange_rate

**Expected tool calls:**

- `step_1`: **calculate_date_diff**(date1='2026-04-01', date2='2026-02-14')


**Model output (from scoring):**

- `step_1`: MATCHED, tool=correct, args_score=1.00


**Score:** overall=1.00, tool_seq=1.00, args=1.00, completeness=1.00, data_flow=1.00

**Error classification:** `E7_unnecessary_tool`

---

## E8_partial_completion

**Description:** Model did not complete all required steps.

### Example 1 (GPT-4o, L1_chain_0049)

**Prompt:**
> Check the weather in Berlin and convert the temperature to Fahrenheit.

**Available tools:** get_weather, unit_convert, word_count, normalize_data, transform_format

**Expected tool calls:**

- `step_1`: **get_weather**(city='Berlin')

- `step_2`: **unit_convert**(value=36, from_unit='celsius', to_unit='fahrenheit') (depends on: step_1)


**Model output (from scoring):**

- `step_1`: MATCHED, tool=correct, args_score=1.00

- `step_2`: MISSING, tool=WRONG, args_score=0.00


**Score:** overall=0.68, tool_seq=0.50, args=1.00, completeness=0.50, data_flow=0.00

**Error classification:** `E8_partial_completion`

---

### Example 2 (GPT-4o, L1_chain_0050)

**Prompt:**
> Check the weather in Cairo and convert the temperature to Fahrenheit.

**Available tools:** get_weather, unit_convert, create_task, calculator, database_query

**Expected tool calls:**

- `step_1`: **get_weather**(city='Cairo')

- `step_2`: **unit_convert**(value=33, from_unit='celsius', to_unit='fahrenheit') (depends on: step_1)


**Model output (from scoring):**

- `step_1`: MATCHED, tool=correct, args_score=1.00

- `step_2`: MISSING, tool=WRONG, args_score=0.00


**Score:** overall=0.68, tool_seq=0.50, args=1.00, completeness=0.50, data_flow=0.00

**Error classification:** `E8_partial_completion`

---

## Error Frequency Summary

| error_type | Claude Sonnet 4 | GPT-4o | GPT-4o Mini |
|---|---|---|---|
| E10_format_error | 34 | 36 | 22 |
| E4_wrong_arguments | 32 | 21 | 33 |
| E7_unnecessary_tool | 0 | 0 | 2 |
| E8_partial_completion | 85 | 95 | 96 |


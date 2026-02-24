"""Composition Engine for CompToolBench.

Automatically generates valid multi-tool compositions by defining
semantic input/output types for each tool and building a compatibility
graph. This enables scaling to 2,500+ tasks across 100+ tools without
manually coding each composition pattern.

Architecture:
  1. ToolTypeSpec defines semantic I/O types for each tool
  2. Compatibility graph links tools whose outputs feed others' inputs
  3. CompositionEngine generates L0/L1/L2/L3 tasks from the graph
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
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
# Semantic type system
# ---------------------------------------------------------------------------


class SemanticType:
    """Semantic types for tool inputs/outputs.

    These go beyond JSON types (string/number) to capture WHAT the
    value represents, enabling meaningful composition.
    """

    TEXT = "text"                      # Any text string
    NUMBER = "number"                  # Any numeric value
    BOOLEAN = "boolean"               # True/False
    TEMPERATURE = "temperature"       # Temperature value (needs unit context)
    CURRENCY_AMOUNT = "currency_amount"
    LOCATION = "location"             # City/place name
    URL = "url"                       # Web URL
    EMAIL_ADDRESS = "email_address"
    DATE = "date"                     # Date string (ISO or human-readable)
    DATETIME = "datetime"             # Date + time
    TIMEZONE = "timezone"             # Timezone identifier
    LIST_TEXT = "list_text"           # List of strings
    LIST_NUMBER = "list_number"       # List of numbers
    STRUCTURED_DATA = "structured_data"  # JSON/dict
    FILE_PATH = "file_path"           # Path to a file
    FILE_CONTENT = "file_content"     # File contents (text)
    LANGUAGE = "language"             # Language name
    LANGUAGE_CODE = "language_code"
    STOCK_SYMBOL = "stock_symbol"
    STOCK_PRICE = "stock_price"       # Stock price data
    CURRENCY_CODE = "currency_code"
    SEARCH_RESULTS = "search_results"
    WEATHER_DATA = "weather_data"
    DIRECTION_DATA = "direction_data"
    SENTIMENT = "sentiment"           # Sentiment label
    ENTITIES = "entities"             # Extracted entities
    SUMMARY = "summary"              # Summarized text
    KEYWORDS = "keywords"            # Extracted keywords
    HASH_VALUE = "hash_value"
    FORMATTED_TEXT = "formatted_text"
    REPORT = "report"
    CONFIRMATION = "confirmation"     # Action confirmation
    PRODUCT_LIST = "product_list"


@dataclass
class OutputField:
    """A single field in a tool's output."""

    name: str                # Field name in the output dict
    semantic_type: str       # SemanticType value
    description: str = ""    # What this field contains


@dataclass
class InputSlot:
    """A single input parameter with its semantic type."""

    param_name: str          # Parameter name
    semantic_type: str       # SemanticType value
    description: str = ""


@dataclass
class ToolTypeSpec:
    """Semantic type specification for a tool."""

    tool_name: str
    inputs: list[InputSlot]
    outputs: list[OutputField]
    primary_output_field: str = ""  # Which output field is the "main" result
    category: str = ""


@dataclass
class Connection:
    """How tool A's output connects to tool B's input."""

    from_tool: str
    to_tool: str
    from_field: str          # Output field name from source tool
    to_param: str            # Input parameter name on target tool
    # Optional fixed params for the target tool (e.g., from_unit="celsius")
    fixed_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositionPattern:
    """A specific multi-tool composition with prompt templates."""

    pattern_id: str
    level: CompositionLevel
    connections: list[Connection]
    prompt_templates: list[str]
    # Tool names in execution order
    tool_sequence: list[str]
    # Additional param pools for variation
    param_pools: dict[str, list[Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Expanded parameter pools (30+ items each for diversity)
# ---------------------------------------------------------------------------

CITIES_EXPANDED = [
    "Tokyo", "Paris", "London", "New York", "Sydney", "Berlin", "Mumbai",
    "Dubai", "Toronto", "Singapore", "Seoul", "Beijing", "Cairo", "Moscow",
    "Rome", "San Francisco", "Los Angeles", "Chicago", "Bangkok", "Istanbul",
    "Lagos", "Jakarta", "Mexico City", "Buenos Aires", "Lima", "Nairobi",
    "Johannesburg", "Stockholm", "Oslo", "Copenhagen", "Helsinki", "Warsaw",
    "Prague", "Vienna", "Zurich", "Amsterdam", "Brussels", "Lisbon",
    "Madrid", "Barcelona", "Athens", "Dublin", "Edinburgh", "Montreal",
    "Vancouver", "Osaka", "Shanghai", "Delhi", "Kuala Lumpur", "Manila",
]

CURRENCIES_EXPANDED = [
    "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "CNY", "INR",
    "KRW", "BRL", "MXN", "SGD", "HKD", "NZD", "SEK", "NOK", "DKK",
    "PLN", "THB", "ZAR", "TRY", "RUB", "AED", "SAR",
]

STOCKS_EXPANDED = [
    "AAPL", "GOOGL", "MSFT", "AMZN", "NVDA", "TSLA", "META", "NFLX",
    "AMD", "INTC", "CRM", "ORCL", "ADBE", "PYPL", "SQ", "SHOP",
    "UBER", "ABNB", "SNAP", "PINS", "ROKU", "ZM", "PLTR", "COIN",
    "V", "MA", "JPM", "BAC", "WMT", "DIS",
]

SEARCH_QUERIES_EXPANDED = [
    "renewable energy trends 2026", "machine learning best practices",
    "climate change solutions", "artificial intelligence in healthcare",
    "space exploration missions", "quantum computing applications",
    "sustainable agriculture technology", "electric vehicle market",
    "remote work productivity tools", "cybersecurity best practices",
    "blockchain in supply chain", "gene therapy breakthroughs",
    "ocean conservation efforts", "smart city infrastructure",
    "mental health technology", "autonomous vehicle regulations",
    "5G network deployment", "circular economy business models",
    "digital twin technology", "personalized medicine advances",
    "green building standards", "food technology innovations",
    "water purification methods", "drone delivery systems",
    "augmented reality education", "carbon credit trading",
    "microplastics research", "fusion energy progress",
    "neurotechnology developments", "vertical farming techniques",
]

LANGUAGES_EXPANDED = [
    "Spanish", "French", "German", "Japanese", "Chinese", "Korean",
    "Portuguese", "Italian", "Russian", "Arabic", "Hindi", "Turkish",
    "Dutch", "Swedish", "Polish", "Thai", "Vietnamese", "Indonesian",
    "Greek", "Czech", "Romanian", "Hungarian", "Finnish", "Norwegian",
    "Danish", "Hebrew", "Malay", "Filipino", "Ukrainian", "Bengali",
]

LANGUAGE_CODES_EXPANDED = {
    "Spanish": "es", "French": "fr", "German": "de", "Japanese": "ja",
    "Chinese": "zh", "Korean": "ko", "Portuguese": "pt", "Italian": "it",
    "Russian": "ru", "Arabic": "ar", "Hindi": "hi", "Turkish": "tr",
    "Dutch": "nl", "Swedish": "sv", "Polish": "pl", "Thai": "th",
    "Vietnamese": "vi", "Indonesian": "id", "Greek": "el", "Czech": "cs",
    "Romanian": "ro", "Hungarian": "hu", "Finnish": "fi", "Norwegian": "no",
    "Danish": "da", "Hebrew": "he", "Malay": "ms", "Filipino": "tl",
    "Ukrainian": "uk", "Bengali": "bn",
}

TIMEZONES_EXPANDED = [
    "US/Eastern", "US/Pacific", "US/Central", "US/Mountain",
    "Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Moscow",
    "Asia/Tokyo", "Asia/Shanghai", "Asia/Singapore", "Asia/Dubai",
    "Asia/Kolkata", "Asia/Seoul", "Asia/Bangkok",
    "Australia/Sydney", "Australia/Melbourne",
    "America/Sao_Paulo", "America/Mexico_City", "America/Toronto",
    "Africa/Lagos", "Africa/Cairo", "Africa/Johannesburg",
    "Pacific/Auckland", "Pacific/Honolulu",
]

PRODUCT_QUERIES_EXPANDED = [
    "laptop", "wireless headphones", "running shoes", "coffee maker",
    "mechanical keyboard", "standing desk", "webcam", "monitor",
    "backpack", "water bottle", "phone case", "tablet", "smartwatch",
    "bluetooth speaker", "desk lamp", "office chair", "mouse pad",
    "usb hub", "power bank", "noise canceling earbuds", "microphone",
    "external ssd", "router", "fitness tracker", "e-reader",
    "portable charger", "webcam cover", "screen protector",
    "cable organizer", "desk mat",
]

PRODUCT_CATEGORIES_EXPANDED = [
    "electronics", "books", "clothing", "home", "sports", "toys",
    "beauty", "health", "automotive", "garden", "office", "food",
    "pet", "tools", "music",
]

TEXTS_EXPANDED = [
    "Artificial intelligence has made remarkable progress in recent years. Machine learning models can now perform complex tasks that were once thought impossible.",
    "Climate change represents one of the greatest challenges facing humanity. Rising temperatures have led to more frequent extreme weather events.",
    "The global economy is undergoing a period of significant transformation. Digital technologies are reshaping traditional business models.",
    "Quantum computing promises to revolutionize how we process information. Unlike classical bits, quantum bits can exist in multiple states simultaneously.",
    "The rise of remote work has fundamentally changed workplace dynamics. Companies are rethinking office spaces and collaboration tools.",
    "Biodiversity loss is accelerating at an unprecedented rate. Conservation efforts must be scaled up significantly to protect endangered species.",
    "Space exploration has entered a new era with private companies joining national agencies. Mars colonization is being actively planned.",
    "The healthcare industry is being transformed by precision medicine. Genetic testing enables personalized treatment plans for patients.",
    "Renewable energy adoption is accelerating globally. Solar and wind power costs have decreased dramatically over the past decade.",
    "Cybersecurity threats are becoming more sophisticated. Organizations must invest in robust security infrastructure and training.",
    "The food industry is experiencing disruption from plant-based alternatives and lab-grown meat. Consumer preferences are shifting rapidly.",
    "Autonomous vehicles are getting closer to widespread deployment. Safety regulations and infrastructure updates are key challenges.",
    "The education sector is embracing technology-enhanced learning. Online platforms and AI tutors are making education more accessible.",
    "Ocean acidification threatens marine ecosystems worldwide. Coral reefs are particularly vulnerable to changes in water chemistry.",
    "The gig economy continues to grow, raising questions about worker protections and benefits. Policy makers are debating new frameworks.",
    "Advancements in battery technology are driving the electric vehicle revolution. Solid-state batteries promise longer range and faster charging.",
    "Social media platforms are facing increased scrutiny over misinformation and privacy. New regulations are being proposed globally.",
    "Urban farming and vertical agriculture are addressing food security in cities. These innovations reduce transportation costs and carbon emissions.",
    "Gene editing technologies like CRISPR offer enormous potential for treating genetic diseases. Ethical considerations remain important.",
    "The metaverse concept is evolving from gaming to business applications. Virtual reality meetings and digital workspaces are becoming mainstream.",
    "Water scarcity is a growing global concern affecting billions of people. Desalination and water recycling technologies are crucial.",
    "Microplastics have been found in every environment on Earth. Research into their health effects is intensifying.",
    "The circular economy model is gaining traction as businesses seek sustainable practices. Waste reduction and recycling are key priorities.",
    "Neuroscience breakthroughs are improving our understanding of brain function. Brain-computer interfaces could transform healthcare.",
    "The future of transportation includes hyperloop systems and urban air mobility. These technologies aim to reduce travel times significantly.",
    "Cryptocurrency regulation is evolving rapidly across different jurisdictions. Central bank digital currencies are being developed worldwide.",
    "3D printing technology is advancing into new materials and applications. From medical implants to housing construction, possibilities expand.",
    "The Internet of Things connects billions of devices worldwide. Smart home technology is becoming standard in new construction.",
    "Fusion energy research has achieved significant milestones. Commercial fusion power could provide virtually unlimited clean energy.",
    "Biodegradable materials are replacing traditional plastics in packaging. Consumer demand is driving innovation in sustainable materials.",
]

SENTIMENT_TEXTS_EXPANDED = [
    ("This product is amazing and wonderful, I absolutely love it!", "positive"),
    ("Terrible experience, the worst service I have ever received.", "negative"),
    ("The meeting is scheduled for Tuesday at 3pm in room 204.", "neutral"),
    ("Great quality and fast shipping, highly recommend!", "positive"),
    ("Disappointing and frustrating, complete waste of money.", "negative"),
    ("The weather today is partly cloudy with temperatures around 72°F.", "neutral"),
    ("Outstanding customer support, they went above and beyond!", "positive"),
    ("Product arrived broken and customer service was unhelpful.", "negative"),
    ("The report contains data from Q1 through Q3 of this year.", "neutral"),
    ("Absolutely brilliant, exceeded all my expectations!", "positive"),
    ("Slow delivery, poor packaging, and the item was damaged.", "negative"),
    ("The office will be closed on Monday for the holiday.", "neutral"),
    ("Best purchase I've made this year, works perfectly!", "positive"),
    ("Overpriced and underperforming, would not recommend.", "negative"),
    ("The conference starts at 9am and ends at 5pm.", "neutral"),
    ("Incredible value for money, five stars without hesitation!", "positive"),
    ("Nothing works as described, total disappointment.", "negative"),
    ("Please submit your expense reports by end of month.", "neutral"),
    ("Fantastic experience from start to finish!", "positive"),
    ("Waited three weeks for delivery and item was wrong.", "negative"),
    ("The quarterly earnings report will be released on Friday.", "neutral"),
    ("Love the new design, much better than the previous version!", "positive"),
    ("Horrible quality control, multiple defects found.", "negative"),
    ("The building maintenance is scheduled for this weekend.", "neutral"),
    ("Perfect gift, my family was thrilled!", "positive"),
    ("Rude staff and long wait times, very unpleasant visit.", "negative"),
    ("The next team meeting will be held in conference room B.", "neutral"),
    ("Superb craftsmanship and attention to detail!", "positive"),
    ("Misleading advertising, product looks nothing like the photos.", "negative"),
    ("The parking lot will be repaved starting next Monday.", "neutral"),
]

ENTITY_TEXTS_EXPANDED = [
    "Contact Dr. Sarah Johnson at sjohnson@university.edu about the meeting on 03/15/2026.",
    "Albert Einstein was born in Ulm, Germany on March 14, 1879.",
    "Please send invoices to billing@acme.com by 12/31/2026. Call 555-0123 for questions.",
    "Marie Curie discovered radium in Paris, France in 1898.",
    "The conference keynote by Prof. James Chen is on January 20, 2026 at MIT.",
    "Reach out to support@techcorp.io for any issues. Our office is in San Francisco, CA.",
    "Elon Musk founded SpaceX in 2002 and Tesla headquarters is in Austin, Texas.",
    "The report was authored by Dr. Lisa Wang from Stanford University on 05/10/2025.",
    "Contact hr@globalinc.com to schedule an interview. The position is based in London.",
    "Isaac Newton published Principia Mathematica in 1687 while at Cambridge University.",
    "Book your appointment with Dr. Michael Brown at 555-9876 or email mbrown@clinic.org.",
    "Ada Lovelace wrote the first algorithm in 1843, often considered the first programmer.",
    "Send feedback to feedback@startup.co by March 1st. We are located in Berlin, Germany.",
    "Nikola Tesla invented the alternating current motor in 1888 in Pittsburgh, Pennsylvania.",
    "Registration closes on 06/30/2026. Contact events@conference.org for more information.",
    "Tim Berners-Lee invented the World Wide Web in 1989 at CERN in Geneva, Switzerland.",
    "The project deadline is set for April 15, 2026. Email lead@project.dev for updates.",
    "Alexander Fleming discovered penicillin in 1928 at St. Mary's Hospital in London.",
    "Quarterly review with VP of Sales Mark Thompson is on 02/28/2026 in the NYC office.",
    "Rosalind Franklin's X-ray crystallography work was done at King's College London in 1952.",
    "New hire orientation: Contact onboarding@company.com before your start date of 03/01/2026.",
    "Leonardo da Vinci painted the Mona Lisa beginning in 1503 in Florence, Italy.",
    "Submit travel requests to travel@corp.org at least 14 days before departure.",
    "Charles Darwin published On the Origin of Species in 1859 after his voyage on the Beagle.",
    "Customer success meeting: reach out to cs@saas.io or call 800-555-0199.",
    "Galileo Galilei made his first telescope observations in 1609 in Padua, Italy.",
    "Final exam schedule: email registrar@university.edu or visit room 301 in Admin Building.",
    "Grace Hopper developed the first compiler in 1952 while working at Remington Rand.",
    "Bug reports go to bugs@opensource.dev. The maintainer is based in Tokyo, Japan.",
    "Alan Turing published his seminal paper on computability in 1936 at Cambridge.",
]

DATE_PAIRS_EXPANDED = [
    ("2026-01-01", "2026-06-30"), ("2025-03-15", "2026-03-15"),
    ("2026-02-14", "2026-12-25"), ("2025-07-04", "2026-07-04"),
    ("2026-01-01", "2026-12-31"), ("2025-01-01", "2025-12-31"),
    ("2026-03-01", "2026-09-30"), ("2025-06-15", "2026-06-15"),
    ("2026-04-01", "2026-10-31"), ("2025-11-01", "2026-05-01"),
    ("2026-01-15", "2026-07-15"), ("2025-08-20", "2026-02-20"),
    ("2026-05-01", "2026-11-30"), ("2025-09-01", "2026-03-01"),
    ("2026-02-01", "2026-08-31"), ("2025-12-25", "2026-12-25"),
    ("2026-06-01", "2026-12-01"), ("2025-04-15", "2026-04-15"),
    ("2026-01-10", "2026-04-10"), ("2025-10-01", "2026-10-01"),
    ("2026-03-20", "2026-06-20"), ("2025-07-15", "2026-01-15"),
    ("2026-08-01", "2026-12-31"), ("2025-02-14", "2026-02-14"),
    ("2026-01-01", "2027-01-01"), ("2025-05-05", "2026-05-05"),
    ("2026-04-15", "2026-08-15"), ("2025-11-15", "2026-11-15"),
    ("2026-07-04", "2026-12-31"), ("2025-03-01", "2026-09-01"),
]

KB_QUERIES_EXPANDED = [
    "What is the capital of France?", "What is the speed of light?",
    "What is the population of Tokyo?", "Who invented the telephone?",
    "What is the boiling point of water?", "Who wrote Romeo and Juliet?",
    "What is the largest ocean on Earth?", "Who discovered penicillin?",
    "What is the distance from Earth to the Moon?", "What year was the internet invented?",
    "What is photosynthesis?", "Who painted the Sistine Chapel?",
    "What is the chemical formula for water?", "What is the tallest mountain?",
    "Who was the first person to walk on the Moon?", "What is the Pythagorean theorem?",
    "What is the largest desert in the world?", "Who developed the theory of relativity?",
    "What is the main component of the Sun?", "What is the speed of sound?",
    "How many bones are in the human body?", "What causes rainbows?",
    "What is the deepest ocean trench?", "Who wrote The Art of War?",
    "What is the smallest country in the world?", "How does DNA replication work?",
    "What is the greenhouse effect?", "Who invented the printing press?",
    "What is the Fibonacci sequence?", "What is the coldest place on Earth?",
]

FILE_PATHS_EXPANDED = [
    "/data/report.txt", "/data/config.json", "/data/employees.csv",
    "/data/sales_q1.json", "/data/inventory.csv", "/data/readme.md",
    "/data/budget_2026.json", "/data/contacts.csv", "/data/meeting_notes.txt",
    "/data/project_plan.json", "/data/performance_review.txt", "/data/api_log.json",
    "/data/customer_feedback.csv", "/data/product_catalog.json", "/data/team_roster.csv",
]

MEETING_TITLES_EXPANDED = [
    "Sprint Planning", "Budget Review", "Team Standup", "Design Review",
    "Quarterly OKR Review", "Architecture Discussion", "Client Demo",
    "Retrospective", "Product Roadmap", "Security Audit Review",
    "Performance Review", "Hiring Committee", "Launch Planning",
    "Training Session", "Vendor Evaluation", "All-Hands Meeting",
    "Code Review Session", "Strategy Brainstorm", "User Research Debrief",
    "Incident Postmortem", "Data Pipeline Review", "Marketing Sync",
    "Sales Pipeline Review", "Onboarding Orientation", "Board Presentation Prep",
    "Customer Success Check-in", "Engineering Sync", "Cross-Team Alignment",
    "Release Planning", "Innovation Sprint Kickoff",
]

NOTIFICATION_MESSAGES_EXPANDED = [
    "Your build completed successfully", "New pull request requires review",
    "Deployment to staging finished", "Weekly report is ready",
    "Test suite passed with 100% coverage", "Database backup completed",
    "New user registration from enterprise.com", "API rate limit warning: 80% used",
    "Security scan found 0 vulnerabilities", "Monthly billing invoice generated",
    "Server CPU usage above 90%", "New feature flag enabled in production",
    "CI/CD pipeline completed in 3m 42s", "Package update available: security patch",
    "Disk usage alert: 85% capacity reached", "New support ticket from VIP customer",
    "Cache cleared successfully", "SSL certificate expires in 30 days",
    "Load balancer health check passed", "Data migration completed successfully",
    "New team member added to project", "Sprint velocity report available",
    "Automated backup verification passed", "API endpoint deprecated: v1/users",
    "Memory usage normalized after spike", "New integration webhook configured",
    "Penetration test report available", "Feature branch merged to main",
    "Code coverage increased to 94%", "Performance benchmark completed",
]

MEMORY_KEYS_EXPANDED = [
    "user_preferences", "project_notes", "meeting_agenda", "shopping_list",
    "travel_plans", "book_recommendations", "recipe_collection", "exercise_log",
    "learning_goals", "budget_tracker", "contact_notes", "todo_list",
    "ideas_backlog", "password_hints", "subscription_list", "gift_ideas",
    "meal_plan", "reading_list", "habit_tracker", "dream_journal",
    "research_notes", "music_playlist", "movie_watchlist", "workout_routine",
    "daily_reflection", "gratitude_list", "vocabulary_words", "code_snippets",
    "interview_prep", "project_timeline",
]

EMAIL_RECIPIENTS_EXPANDED = [
    "alice@company.com", "bob@techcorp.io", "charlie@startup.dev",
    "diana@university.edu", "eve@research.org", "frank@consulting.biz",
    "grace@hospital.med", "henry@finance.bank", "iris@media.news",
    "james@logistics.ship", "karen@legal.law", "leon@engineering.build",
    "maria@design.studio", "nick@marketing.brand", "olivia@sales.deal",
    "peter@support.help", "quinn@analytics.data", "rachel@product.mgmt",
    "sam@operations.ops", "tina@hr.people", "ursula@security.safe",
    "victor@infra.cloud", "wendy@quality.test", "xander@devops.ci",
    "yuki@international.global", "zara@executive.ceo",
]

CONTINENTS_EXPANDED = [
    "Asia", "Europe", "North America", "South America", "Oceania",
    "Africa", "Antarctica",
]

TRAVEL_MODES_EXPANDED = ["driving", "walking", "transit", "cycling"]


# ---------------------------------------------------------------------------
# Prompt template variants (3-5 per pattern for diversity)
# ---------------------------------------------------------------------------

L0_PROMPT_VARIANTS = {
    "get_weather": [
        "What is the current weather in {city}?",
        "Check the weather conditions for {city}.",
        "Tell me the temperature and weather in {city} right now.",
        "What's the forecast like in {city} today?",
        "Look up the current weather report for {city}.",
    ],
    "calculator": [
        "What is {expr}?",
        "Calculate {expr}.",
        "Compute the value of {expr}.",
        "Evaluate the expression {expr}.",
    ],
    "unit_convert": [
        "Convert {value} {from_u} to {to_u}.",
        "What is {value} {from_u} in {to_u}?",
        "How many {to_u} is {value} {from_u}?",
        "Transform {value} from {from_u} to {to_u}.",
    ],
    "get_stock_price": [
        "What is the current stock price of {symbol}?",
        "Look up the latest price for {symbol}.",
        "Check the stock price for {symbol}.",
        "How much is {symbol} trading at right now?",
    ],
    "get_exchange_rate": [
        "Convert {amount} {from_c} to {to_c}.",
        "What is {amount} {from_c} worth in {to_c}?",
        "How much {to_c} can I get for {amount} {from_c}?",
        "Exchange {amount} {from_c} to {to_c}.",
    ],
    "sentiment_analysis": [
        "Analyze the sentiment of: \"{text}\"",
        "What is the sentiment of this text: \"{text}\"",
        "Determine whether this is positive, negative, or neutral: \"{text}\"",
        "Classify the sentiment: \"{text}\"",
    ],
    "web_search": [
        "Search the web for: {query}",
        "Find information about: {query}",
        "Look up: {query}",
        "Search for: {query}",
    ],
    "translate_text": [
        "Translate \"{text}\" to {lang}.",
        "Convert this text to {lang}: \"{text}\"",
        "How do you say \"{text}\" in {lang}?",
        "Translate the following into {lang}: \"{text}\"",
    ],
    "summarize_text": [
        "Summarize this text: \"{text}\"",
        "Give me a brief summary of: \"{text}\"",
        "Provide a concise summary: \"{text}\"",
        "What are the key points of: \"{text}\"",
    ],
    "extract_entities": [
        "Extract named entities from: \"{text}\"",
        "Identify the people, places, and dates in: \"{text}\"",
        "Find all named entities in this text: \"{text}\"",
        "What entities are mentioned in: \"{text}\"",
    ],
    "hash_text": [
        "Generate a {algo} hash of \"{text}\".",
        "Compute the {algo} hash for: \"{text}\"",
        "What is the {algo} hash of \"{text}\"?",
    ],
    "word_count": [
        "Count the words in: \"{text}\"",
        "How many words are in: \"{text}\"?",
        "Get the word count for: \"{text}\"",
    ],
    "detect_language": [
        "What language is this: \"{text}\"?",
        "Detect the language of: \"{text}\"",
        "Identify the language: \"{text}\"",
    ],
    "percentage_change": [
        "What is the percentage change from {old} to {new}?",
        "Calculate the percent change between {old} and {new}.",
        "How much did the value change from {old} to {new} in percent?",
    ],
    "format_date": [
        "Format the date {date} in {fmt} format.",
        "Convert {date} to {fmt} format.",
        "Show the date {date} in {fmt} style.",
    ],
}

L1_PROMPT_VARIANTS = {
    "weather_convert": [
        "What is the temperature in {city} in Fahrenheit?",
        "Check the weather in {city} and convert the temperature to Fahrenheit.",
        "Get the weather for {city}, then convert celsius to fahrenheit.",
        "Look up {city}'s temperature and express it in Fahrenheit.",
    ],
    "search_summarize": [
        "Search for \"{query}\" and summarize the results.",
        "Find information about \"{query}\" and give me a summary.",
        "Look up \"{query}\" and provide a brief summary of what you find.",
        "Search the web for \"{query}\", then summarize the findings.",
    ],
    "stock_convert": [
        "What is {symbol}'s stock price in {currency}?",
        "Get the price of {symbol} and convert it to {currency}.",
        "Look up {symbol}'s price, then convert the amount to {currency}.",
    ],
    "entity_sentiment": [
        "Extract entities from \"{text}\" and analyze the sentiment.",
        "Find named entities in \"{text}\" and determine the sentiment.",
        "Identify entities in \"{text}\", then classify its sentiment.",
    ],
    "search_translate": [
        "Search for \"{query}\" and translate the results to {lang}.",
        "Find information about \"{query}\" and translate it to {lang}.",
        "Look up \"{query}\", then translate the findings into {lang}.",
    ],
    "weather_email": [
        "Get the weather in {city} and email the report to {email}.",
        "Check {city}'s weather and send it via email to {email}.",
        "Look up weather in {city}, then email the results to {email}.",
    ],
    "read_summarize": [
        "Read the file at {path} and summarize its contents.",
        "Open {path} and provide a summary of what's in it.",
        "Read {path}, then give me a concise summary.",
    ],
    "kb_translate": [
        "Answer \"{query}\" and translate the answer to {lang}.",
        "Look up \"{query}\" in the knowledge base, then translate to {lang}.",
        "Find the answer to \"{query}\" and provide it in {lang}.",
    ],
    "text_hash": [
        "Get the {algo} hash of the text: \"{text}\"",
        "Hash this text with {algo}: \"{text}\"",
        "Compute the {algo} checksum of: \"{text}\"",
    ],
    "search_classify": [
        "Search for \"{query}\" and classify the results into categories.",
        "Find information about \"{query}\" and categorize it.",
        "Look up \"{query}\", then classify the topic of the results.",
    ],
}

L2_PROMPT_VARIANTS = {
    "multi_weather": [
        "Compare the weather in {cities}. Which city is warmest?",
        "Check the temperature in {cities} and tell me which is hottest.",
        "Get weather for {cities} simultaneously and compare temperatures.",
    ],
    "multi_stock": [
        "Compare the stock prices of {symbols}. Which is highest?",
        "Look up prices for {symbols} and tell me which costs most.",
        "Get current prices of {symbols} and compare them.",
    ],
    "multi_sentiment": [
        "Analyze the sentiment of these texts and compare: {texts}",
        "Determine the sentiment for each of these: {texts}",
        "Classify the sentiment of multiple texts: {texts}",
    ],
    "parallel_search_kb": [
        "Search the web for \"{q1}\" and also look up \"{q2}\" in the knowledge base.",
        "Simultaneously search for \"{q1}\" and find \"{q2}\" in the KB.",
        "In parallel: web search \"{q1}\" and knowledge base query \"{q2}\".",
    ],
    "multi_translate": [
        "Translate \"{text}\" into {langs} simultaneously.",
        "Provide translations of \"{text}\" in {langs}.",
        "Convert \"{text}\" to {langs} at the same time.",
    ],
}

L3_PROMPT_VARIANTS = {
    "weather_compare_email": [
        "Compare weather in {c1} and {c2}, convert the warmer temp to Fahrenheit, and email the result to {email}.",
        "Get weather for {c1} and {c2}, find which is warmer, convert to °F, then email {email}.",
        "Check weather in both {c1} and {c2}. Convert the higher temperature to Fahrenheit and send it to {email}.",
    ],
    "stock_compare_notify": [
        "Compare {s1} and {s2} stock prices, convert the higher one to {currency}, and send a notification.",
        "Get prices for {s1} and {s2}, find the more expensive one, convert to {currency}, then notify.",
        "Look up {s1} and {s2}, convert the winner's price to {currency}, and create a notification.",
    ],
    "search_summarize_translate_email": [
        "Search for \"{query}\", summarize the results, translate to {lang}, and email to {email}.",
        "Find \"{query}\" online, create a summary, translate it to {lang}, then email {email}.",
        "Web search \"{query}\", summarize findings, convert to {lang}, and send via email to {email}.",
    ],
    "multi_weather_stats": [
        "Get weather for {cities}, compute temperature statistics, and save the report.",
        "Check weather in {cities}, calculate average/min/max temperature, and write results to a file.",
        "Fetch weather data for {cities}, run statistical analysis on temperatures, and generate a report.",
    ],
}


# ---------------------------------------------------------------------------
# Composition Engine
# ---------------------------------------------------------------------------


class CompositionEngine:
    """Generates valid multi-tool compositions from semantic type compatibility.

    The engine maintains a registry of tool type specifications and a
    compatibility graph. It uses these to generate diverse, valid
    compositions at all 4 complexity levels.
    """

    def __init__(self, seed: int = 42, mode: ToolMode = ToolMode.SIMULATED) -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.mode = mode
        self._task_counter = 0
        self._generated_hashes: set[str] = set()  # For deduplication
        self._all_tool_names = list(get_all_tools().keys())

    def _next_id(self, level: CompositionLevel) -> str:
        self._task_counter += 1
        return f"{level.value}_{self._task_counter:04d}"

    def _pick(self, items: list[Any]) -> Any:
        return self.rng.choice(items)

    def _pick_n(self, items: list[Any], n: int) -> list[Any]:
        return self.rng.sample(items, min(n, len(items)))

    def _execute_tool(self, tool_name: str, **kwargs: Any) -> Any:
        """Execute a tool in simulated mode to get ground truth."""
        tool_cls = get_tool(tool_name)
        tool = tool_cls(mode=self.mode)
        result = tool.execute(**kwargs)
        if result.success:
            return result.data
        raise RuntimeError(f"Tool {tool_name} failed: {result.error}")

    def _pick_distractors(self, used_tools: list[str], count: int = 3) -> list[str]:
        """Pick random distractor tools not in the used set."""
        available = [t for t in self._all_tool_names if t not in used_tools]
        return self._pick_n(available, min(count, len(available)))

    def _task_hash(self, level: str, tools: list[str], key_params: dict[str, Any]) -> str:
        """Generate a hash for deduplication."""
        raw = json.dumps({"level": level, "tools": sorted(tools), "params": key_params}, sort_keys=True, default=str)
        return hashlib.md5(raw.encode()).hexdigest()

    def _is_duplicate(self, level: str, tools: list[str], key_params: dict[str, Any]) -> bool:
        """Check if this task is a duplicate."""
        h = self._task_hash(level, tools, key_params)
        if h in self._generated_hashes:
            return True
        self._generated_hashes.add(h)
        return False

    def _format_prompt(self, template_key: str, variants_dict: dict[str, list[str]], **kwargs: Any) -> str:
        """Pick a random prompt variant and format it."""
        if template_key in variants_dict:
            template = self._pick(variants_dict[template_key])
        else:
            # Fallback: use first variant if key not found
            template = template_key
        try:
            return template.format(**kwargs)
        except KeyError:
            return template

    # -----------------------------------------------------------------------
    # L0 Task Generation
    # -----------------------------------------------------------------------

    def generate_l0_tasks(self, count: int = 600) -> list[Task]:
        """Generate L0 (single tool) tasks for ALL registered tools."""
        tasks: list[Task] = []
        all_tools = get_all_tools()
        tool_names = sorted(all_tools.keys())
        per_tool = max(count // len(tool_names), 2)

        for tool_name in tool_names:
            for _ in range(per_tool):
                task = self._generate_single_l0(tool_name)
                if task and not self._is_duplicate("L0", [tool_name], task.expected_trace.steps[0].arguments):
                    tasks.append(task)
                if len(tasks) >= count:
                    break
            if len(tasks) >= count:
                break

        # Fill remaining with random tools
        while len(tasks) < count:
            tool_name = self._pick(tool_names)
            task = self._generate_single_l0(tool_name)
            if task:
                h = self._task_hash("L0", [tool_name], task.expected_trace.steps[0].arguments)
                if h not in self._generated_hashes:
                    self._generated_hashes.add(h)
                    tasks.append(task)

        return tasks[:count]

    def _generate_single_l0(self, tool_name: str) -> Task | None:
        """Generate a single L0 task for any tool."""
        gen = self._l0_generators.get(tool_name)
        if gen:
            try:
                return gen(self)
            except Exception:
                return None
        # Fallback: try to generate a generic L0 task
        return self._generic_l0(tool_name)

    # Natural language prompt templates for generic L0 tasks (keyed by tool description keyword)
    _GENERIC_PROMPTS: list[str] = [
        "I need to {action}. Can you help me with that?",
        "Could you {action} for me?",
        "Please {action}.",
        "{action_cap} and tell me what you find.",
        "I'd like you to {action}.",
    ]

    def _generic_l0(self, tool_name: str) -> Task | None:
        """Generate a generic L0 task from a tool's schema with natural language prompts."""
        try:
            tool_cls = get_tool(tool_name)
            schema = tool_cls.schema
            # Generate random arguments from schema
            args = self._generate_args_from_schema(schema)
            if args is None:
                return None
            answer = self._execute_tool(tool_name, **args)
            distractors = self._pick_distractors([tool_name])

            # Build a natural language prompt from the tool description and arguments
            prompt = self._build_natural_prompt(schema, args)

            return Task(
                task_id=self._next_id(CompositionLevel.NODE),
                level=CompositionLevel.NODE,
                prompt=prompt,
                available_tools=[tool_name] + distractors,
                expected_trace=ExpectedTrace(
                    steps=[ToolCall(step_id="step_1", tool_name=tool_name, arguments=args)],
                    final_answer_source="step_1",
                ),
                expected_final_answer=answer,
                metadata={"category": schema.category.value, "tool": tool_name, "generated_by": "generic"},
            )
        except Exception:
            return None

    def _build_natural_prompt(self, schema: ToolSchema, args: dict[str, Any]) -> str:  # noqa: F821
        """Build a natural language prompt from a tool's schema and arguments."""
        # Build action description and parameter details
        action_parts = []
        for param in schema.parameters:
            if param.name in args:
                val = args[param.name]
                human_name = param.name.replace("_", " ")
                if isinstance(val, str) and len(val) < 80:
                    action_parts.append(f"{human_name}: {val}")
                elif isinstance(val, (int, float)):
                    action_parts.append(f"{human_name}: {val}")

        # Use the tool description as the action
        action = schema.description.rstrip(".")
        if action_parts:
            detail = ", ".join(action_parts[:3])
            prompts = [
                f"{action} with the following: {detail}.",
                f"I need you to {action[0].lower()}{action[1:]}. Here are the details: {detail}.",
                f"Please {action[0].lower()}{action[1:]} — specifically, {detail}.",
                f"Can you {action[0].lower()}{action[1:]}? The parameters are {detail}.",
            ]
        else:
            prompts = [
                f"{action}.",
                f"I need you to {action[0].lower()}{action[1:]}.",
                f"Please {action[0].lower()}{action[1:]}.",
                f"Can you {action[0].lower()}{action[1:]}?",
            ]

        return self._pick(prompts)

    def _generate_args_from_schema(self, schema: ToolSchema) -> dict[str, Any] | None:  # noqa: F821
        """Generate random arguments for a tool based on its schema."""
        from comptoolbench.tools.base import ToolSchema  # noqa: F811

        args: dict[str, Any] = {}
        for param in schema.parameters:
            if not param.required and self.rng.random() < 0.3:
                continue
            if param.enum:
                args[param.name] = self._pick(param.enum)
            elif param.type == "string":
                args[param.name] = self._generate_string_for_param(param.name)
            elif param.type == "number":
                args[param.name] = round(self.rng.uniform(1, 100), 2)
            elif param.type == "integer":
                args[param.name] = self.rng.randint(1, 100)
            elif param.type == "boolean":
                args[param.name] = self.rng.choice([True, False])
            elif param.type == "array":
                args[param.name] = [f"item_{i}" for i in range(self.rng.randint(2, 5))]
            else:
                return None  # Can't generate for this type
        return args

    def _generate_string_for_param(self, param_name: str) -> str:
        """Generate a contextually appropriate string based on parameter name."""
        param_pools = {
            "city": CITIES_EXPANDED,
            "location": CITIES_EXPANDED,
            "text": TEXTS_EXPANDED,
            "query": SEARCH_QUERIES_EXPANDED,
            "url": ["https://example.com", "https://news.com", "https://blog.dev"],
            "email": EMAIL_RECIPIENTS_EXPANDED,
            "recipient": EMAIL_RECIPIENTS_EXPANDED,
            "to": EMAIL_RECIPIENTS_EXPANDED,
            "path": FILE_PATHS_EXPANDED,
            "file_path": FILE_PATHS_EXPANDED,
            "directory": ["/", "/data", "/data/reports", "/data/exports"],
            "language": LANGUAGES_EXPANDED,
            "target_language": LANGUAGES_EXPANDED,
            "from_currency": CURRENCIES_EXPANDED,
            "to_currency": CURRENCIES_EXPANDED,
            "currency": CURRENCIES_EXPANDED,
            "symbol": STOCKS_EXPANDED,
            "key": MEMORY_KEYS_EXPANDED,
            "title": MEETING_TITLES_EXPANDED,
            "message": NOTIFICATION_MESSAGES_EXPANDED,
            "expression": [f"{self.rng.randint(10, 999)} {self._pick(['+', '-', '*'])} {self.rng.randint(2, 99)}"],
            "timezone": TIMEZONES_EXPANDED,
            "from_timezone": TIMEZONES_EXPANDED,
            "to_timezone": TIMEZONES_EXPANDED,
            "date": [dp[0] for dp in DATE_PAIRS_EXPANDED],
            "start_date": [dp[0] for dp in DATE_PAIRS_EXPANDED],
            "end_date": [dp[1] for dp in DATE_PAIRS_EXPANDED],
            "datetime_str": ["2026-03-15 10:00", "2026-04-01 14:00", "2026-05-20 09:30", "2026-06-10 11:00"],
            "from_unit": ["celsius", "fahrenheit", "km", "miles", "kg", "lbs", "meters", "feet"],
            "to_unit": ["fahrenheit", "celsius", "miles", "km", "lbs", "kg", "feet", "meters"],
            "algorithm": ["md5", "sha256", "sha512"],
            "format": ["short", "long", "iso"],
            "method": ["word", "sentence"],
            "target_case": ["upper", "lower", "title", "snake"],
            "action": ["encode", "decode"],
            "severity": ["info", "warning", "error"],
            "category": PRODUCT_CATEGORIES_EXPANDED,
            "content": TEXTS_EXPANDED,
            "subject": ["Meeting Update", "Report Ready", "Action Required", "FYI"],
        }

        # Try to find a matching pool
        name_lower = param_name.lower()
        for key, pool in param_pools.items():
            if key in name_lower or name_lower in key:
                return self._pick(pool)

        # Default: return a generic string
        return f"sample_{param_name}_{self.rng.randint(1, 1000)}"

    # -----------------------------------------------------------------------
    # L1 Chain Generation
    # -----------------------------------------------------------------------

    def generate_l1_tasks(self, count: int = 800) -> list[Task]:
        """Generate L1 (2-tool chain) tasks."""
        tasks: list[Task] = []
        chain_generators = list(self._l1_generators.items())

        per_gen = max(count // len(chain_generators), 3)
        for gen_name, gen_fn in chain_generators:
            for _ in range(per_gen):
                try:
                    task = gen_fn(self)
                    if task:
                        tasks.append(task)
                except Exception:
                    continue
                if len(tasks) >= count:
                    break
            if len(tasks) >= count:
                break

        # Fill remaining with random chain generators
        while len(tasks) < count:
            gen_name, gen_fn = self._pick(chain_generators)
            try:
                task = gen_fn(self)
                if task:
                    tasks.append(task)
            except Exception:
                continue

        return tasks[:count]

    # -----------------------------------------------------------------------
    # L2 Parallel Generation
    # -----------------------------------------------------------------------

    def generate_l2_tasks(self, count: int = 500) -> list[Task]:
        """Generate L2 (parallel fork-join) tasks."""
        tasks: list[Task] = []
        parallel_generators = list(self._l2_generators.items())

        per_gen = max(count // len(parallel_generators), 3)
        for gen_name, gen_fn in parallel_generators:
            for _ in range(per_gen):
                try:
                    task = gen_fn(self)
                    if task:
                        tasks.append(task)
                except Exception:
                    continue
                if len(tasks) >= count:
                    break
            if len(tasks) >= count:
                break

        while len(tasks) < count:
            gen_name, gen_fn = self._pick(parallel_generators)
            try:
                task = gen_fn(self)
                if task:
                    tasks.append(task)
            except Exception:
                continue

        return tasks[:count]

    # -----------------------------------------------------------------------
    # L3 DAG Generation
    # -----------------------------------------------------------------------

    def generate_l3_tasks(self, count: int = 500) -> list[Task]:
        """Generate L3 (DAG) tasks."""
        tasks: list[Task] = []
        dag_generators = list(self._l3_generators.items())

        per_gen = max(count // len(dag_generators), 3)
        for gen_name, gen_fn in dag_generators:
            for _ in range(per_gen):
                try:
                    task = gen_fn(self)
                    if task:
                        tasks.append(task)
                except Exception:
                    continue
                if len(tasks) >= count:
                    break
            if len(tasks) >= count:
                break

        while len(tasks) < count:
            gen_name, gen_fn = self._pick(dag_generators)
            try:
                task = gen_fn(self)
                if task:
                    tasks.append(task)
            except Exception:
                continue

        return tasks[:count]

    # -----------------------------------------------------------------------
    # Full suite generation
    # -----------------------------------------------------------------------

    def generate_suite(
        self,
        l0_count: int = 600,
        l1_count: int = 800,
        l2_count: int = 500,
        l3_count: int = 600,
    ) -> TaskSuite:
        """Generate a complete benchmark suite (default: 2,500 tasks)."""
        tasks = []
        tasks.extend(self.generate_l0_tasks(l0_count))
        tasks.extend(self.generate_l1_tasks(l1_count))
        tasks.extend(self.generate_l2_tasks(l2_count))
        tasks.extend(self.generate_l3_tasks(l3_count))

        return TaskSuite(
            name="CompToolBench",
            version="2.0.0",
            tasks=tasks,
            metadata={
                "seed": self.seed,
                "total_tools": len(self._all_tool_names),
                "tool_names": sorted(self._all_tool_names),
                "generation_config": {
                    "l0_count": l0_count,
                    "l1_count": l1_count,
                    "l2_count": l2_count,
                    "l3_count": l3_count,
                },
            },
        )

    # -----------------------------------------------------------------------
    # L0 Generator Registry
    # -----------------------------------------------------------------------

    @staticmethod
    def _l0_weather(engine: CompositionEngine) -> Task:
        city = engine._pick(CITIES_EXPANDED)
        answer = engine._execute_tool("get_weather", city=city)
        prompt = engine._format_prompt("get_weather", L0_PROMPT_VARIANTS, city=city)
        distractors = engine._pick_distractors(["get_weather"])
        return Task(
            task_id=engine._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=prompt,
            available_tools=["get_weather"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="get_weather", arguments={"city": city})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "external_services", "tool": "get_weather"},
        )

    @staticmethod
    def _l0_calculator(engine: CompositionEngine) -> Task:
        a, b = engine.rng.randint(10, 999), engine.rng.randint(2, 99)
        op = engine._pick(["+", "-", "*"])
        expr = f"{a} {op} {b}"
        answer = engine._execute_tool("calculator", expression=expr)
        prompt = engine._format_prompt("calculator", L0_PROMPT_VARIANTS, expr=expr)
        distractors = engine._pick_distractors(["calculator"])
        return Task(
            task_id=engine._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=prompt,
            available_tools=["calculator"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="calculator", arguments={"expression": expr})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "computation", "tool": "calculator"},
        )

    @staticmethod
    def _l0_unit_convert(engine: CompositionEngine) -> Task:
        value = round(engine.rng.uniform(1, 100), 1)
        all_units = [("celsius", "fahrenheit"), ("fahrenheit", "celsius"), ("celsius", "kelvin"),
                     ("km", "miles"), ("miles", "km"), ("meters", "feet"), ("kg", "lbs"), ("lbs", "kg")]
        from_u, to_u = engine._pick(all_units)
        answer = engine._execute_tool("unit_convert", value=value, from_unit=from_u, to_unit=to_u)
        prompt = engine._format_prompt("unit_convert", L0_PROMPT_VARIANTS, value=value, from_u=from_u, to_u=to_u)
        distractors = engine._pick_distractors(["unit_convert"])
        return Task(
            task_id=engine._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=prompt,
            available_tools=["unit_convert"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="unit_convert", arguments={"value": value, "from_unit": from_u, "to_unit": to_u})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "computation", "tool": "unit_convert"},
        )

    @staticmethod
    def _l0_stock_price(engine: CompositionEngine) -> Task:
        symbol = engine._pick(STOCKS_EXPANDED)
        answer = engine._execute_tool("get_stock_price", symbol=symbol)
        prompt = engine._format_prompt("get_stock_price", L0_PROMPT_VARIANTS, symbol=symbol)
        distractors = engine._pick_distractors(["get_stock_price"])
        return Task(
            task_id=engine._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=prompt,
            available_tools=["get_stock_price"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="get_stock_price", arguments={"symbol": symbol})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "external_services", "tool": "get_stock_price"},
        )

    @staticmethod
    def _l0_exchange_rate(engine: CompositionEngine) -> Task:
        from_c, to_c = engine._pick_n(CURRENCIES_EXPANDED, 2)
        amount = engine._pick([1, 10, 50, 100, 500, 1000])
        answer = engine._execute_tool("get_exchange_rate", from_currency=from_c, to_currency=to_c, amount=amount)
        prompt = engine._format_prompt("get_exchange_rate", L0_PROMPT_VARIANTS, amount=amount, from_c=from_c, to_c=to_c)
        distractors = engine._pick_distractors(["get_exchange_rate"])
        return Task(
            task_id=engine._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=prompt,
            available_tools=["get_exchange_rate"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="get_exchange_rate", arguments={"from_currency": from_c, "to_currency": to_c, "amount": amount})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "external_services", "tool": "get_exchange_rate"},
        )

    @staticmethod
    def _l0_sentiment(engine: CompositionEngine) -> Task:
        text, _label = engine._pick(SENTIMENT_TEXTS_EXPANDED)
        answer = engine._execute_tool("sentiment_analysis", text=text)
        prompt = engine._format_prompt("sentiment_analysis", L0_PROMPT_VARIANTS, text=text[:80])
        distractors = engine._pick_distractors(["sentiment_analysis"])
        return Task(
            task_id=engine._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=prompt,
            available_tools=["sentiment_analysis"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="sentiment_analysis", arguments={"text": text})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "text_processing", "tool": "sentiment_analysis"},
        )

    @staticmethod
    def _l0_web_search(engine: CompositionEngine) -> Task:
        query = engine._pick(SEARCH_QUERIES_EXPANDED)
        answer = engine._execute_tool("web_search", query=query)
        prompt = engine._format_prompt("web_search", L0_PROMPT_VARIANTS, query=query)
        distractors = engine._pick_distractors(["web_search"])
        return Task(
            task_id=engine._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=prompt,
            available_tools=["web_search"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="web_search", arguments={"query": query})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "information_retrieval", "tool": "web_search"},
        )

    @staticmethod
    def _l0_translate(engine: CompositionEngine) -> Task:
        text = engine._pick(["Hello, how are you?", "Good morning", "Thank you very much",
                             "Where is the nearest restaurant?", "I need help please",
                             "The weather is beautiful today", "Nice to meet you",
                             "Have a great day", "See you tomorrow", "What time is it?"])
        lang = engine._pick(LANGUAGES_EXPANDED)
        code = LANGUAGE_CODES_EXPANDED.get(lang, "es")
        answer = engine._execute_tool("translate_text", text=text, target_language=code)
        prompt = engine._format_prompt("translate_text", L0_PROMPT_VARIANTS, text=text, lang=lang)
        distractors = engine._pick_distractors(["translate_text"])
        return Task(
            task_id=engine._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=prompt,
            available_tools=["translate_text"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="translate_text", arguments={"text": text, "target_language": code})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "external_services", "tool": "translate_text"},
        )

    @staticmethod
    def _l0_summarize(engine: CompositionEngine) -> Task:
        text = engine._pick(TEXTS_EXPANDED)
        answer = engine._execute_tool("summarize_text", text=text)
        prompt = engine._format_prompt("summarize_text", L0_PROMPT_VARIANTS, text=text[:80] + "...")
        distractors = engine._pick_distractors(["summarize_text"])
        return Task(
            task_id=engine._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"Summarize this text: \"{text}\"",
            available_tools=["summarize_text"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="summarize_text", arguments={"text": text})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "text_processing", "tool": "summarize_text"},
        )

    @staticmethod
    def _l0_extract_entities(engine: CompositionEngine) -> Task:
        text = engine._pick(ENTITY_TEXTS_EXPANDED)
        answer = engine._execute_tool("extract_entities", text=text)
        prompt = engine._format_prompt("extract_entities", L0_PROMPT_VARIANTS, text=text[:80] + "...")
        distractors = engine._pick_distractors(["extract_entities"])
        return Task(
            task_id=engine._next_id(CompositionLevel.NODE),
            level=CompositionLevel.NODE,
            prompt=f"Extract named entities from: \"{text}\"",
            available_tools=["extract_entities"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[ToolCall(step_id="step_1", tool_name="extract_entities", arguments={"text": text})],
                final_answer_source="step_1",
            ),
            expected_final_answer=answer,
            metadata={"category": "text_processing", "tool": "extract_entities"},
        )

    # -----------------------------------------------------------------------
    # L1 Chain Generator Registry
    # -----------------------------------------------------------------------

    @staticmethod
    def _l1_weather_convert(engine: CompositionEngine) -> Task:
        city = engine._pick(CITIES_EXPANDED)
        weather = engine._execute_tool("get_weather", city=city)
        temp = weather["temperature_celsius"]
        converted = engine._execute_tool("unit_convert", value=temp, from_unit="celsius", to_unit="fahrenheit")
        prompt = engine._format_prompt("weather_convert", L1_PROMPT_VARIANTS, city=city)
        distractors = engine._pick_distractors(["get_weather", "unit_convert"])
        return Task(
            task_id=engine._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=prompt,
            available_tools=["get_weather", "unit_convert"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_weather", arguments={"city": city}, output_key="weather", depends_on=[]),
                    ToolCall(step_id="step_2", tool_name="unit_convert", arguments={"value": temp, "from_unit": "celsius", "to_unit": "fahrenheit"}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=converted,
            metadata={"category": "chain", "tools": ["get_weather", "unit_convert"], "pattern": "retrieve-transform"},
        )

    @staticmethod
    def _l1_search_summarize(engine: CompositionEngine) -> Task:
        query = engine._pick(SEARCH_QUERIES_EXPANDED)
        results = engine._execute_tool("web_search", query=query)
        result_text = json.dumps(results.get("results", results), default=str)[:500]
        summary = engine._execute_tool("summarize_text", text=result_text)
        prompt = engine._format_prompt("search_summarize", L1_PROMPT_VARIANTS, query=query)
        distractors = engine._pick_distractors(["web_search", "summarize_text"])
        return Task(
            task_id=engine._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=prompt,
            available_tools=["web_search", "summarize_text"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="web_search", arguments={"query": query}, output_key="results", depends_on=[]),
                    ToolCall(step_id="step_2", tool_name="summarize_text", arguments={"text": result_text}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=summary,
            metadata={"category": "chain", "tools": ["web_search", "summarize_text"], "pattern": "retrieve-process"},
        )

    @staticmethod
    def _l1_stock_convert(engine: CompositionEngine) -> Task:
        symbol = engine._pick(STOCKS_EXPANDED)
        currency = engine._pick([c for c in CURRENCIES_EXPANDED if c != "USD"])
        stock = engine._execute_tool("get_stock_price", symbol=symbol)
        price = stock.get("price", stock.get("price_usd", 100.0))
        converted = engine._execute_tool("get_exchange_rate", from_currency="USD", to_currency=currency, amount=price)
        prompt = engine._format_prompt("stock_convert", L1_PROMPT_VARIANTS, symbol=symbol, currency=currency)
        distractors = engine._pick_distractors(["get_stock_price", "get_exchange_rate"])
        return Task(
            task_id=engine._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=prompt,
            available_tools=["get_stock_price", "get_exchange_rate"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_stock_price", arguments={"symbol": symbol}, output_key="stock", depends_on=[]),
                    ToolCall(step_id="step_2", tool_name="get_exchange_rate", arguments={"from_currency": "USD", "to_currency": currency, "amount": price}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=converted,
            metadata={"category": "chain", "tools": ["get_stock_price", "get_exchange_rate"], "pattern": "retrieve-convert"},
        )

    @staticmethod
    def _l1_entity_sentiment(engine: CompositionEngine) -> Task:
        text = engine._pick(ENTITY_TEXTS_EXPANDED)
        entities = engine._execute_tool("extract_entities", text=text)
        sentiment = engine._execute_tool("sentiment_analysis", text=text)
        prompt = engine._format_prompt("entity_sentiment", L1_PROMPT_VARIANTS, text=text[:80] + "...")
        distractors = engine._pick_distractors(["extract_entities", "sentiment_analysis"])
        return Task(
            task_id=engine._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=prompt,
            available_tools=["extract_entities", "sentiment_analysis"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="extract_entities", arguments={"text": text}, output_key="entities", depends_on=[]),
                    ToolCall(step_id="step_2", tool_name="sentiment_analysis", arguments={"text": text}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=sentiment,
            metadata={"category": "chain", "tools": ["extract_entities", "sentiment_analysis"], "pattern": "analyze-classify"},
        )

    @staticmethod
    def _l1_search_translate(engine: CompositionEngine) -> Task:
        query = engine._pick(SEARCH_QUERIES_EXPANDED)
        lang = engine._pick(LANGUAGES_EXPANDED)
        code = LANGUAGE_CODES_EXPANDED.get(lang, "es")
        results = engine._execute_tool("web_search", query=query)
        result_text = json.dumps(results.get("results", results), default=str)[:300]
        translated = engine._execute_tool("translate_text", text=result_text, target_language=code)
        prompt = engine._format_prompt("search_translate", L1_PROMPT_VARIANTS, query=query, lang=lang)
        distractors = engine._pick_distractors(["web_search", "translate_text"])
        return Task(
            task_id=engine._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=prompt,
            available_tools=["web_search", "translate_text"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="web_search", arguments={"query": query}, output_key="results", depends_on=[]),
                    ToolCall(step_id="step_2", tool_name="translate_text", arguments={"text": result_text, "target_language": code}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=translated,
            metadata={"category": "chain", "tools": ["web_search", "translate_text"], "pattern": "retrieve-translate"},
        )

    @staticmethod
    def _l1_weather_email(engine: CompositionEngine) -> Task:
        city = engine._pick(CITIES_EXPANDED)
        email = engine._pick(EMAIL_RECIPIENTS_EXPANDED)
        weather = engine._execute_tool("get_weather", city=city)
        weather_str = json.dumps(weather, default=str)
        sent = engine._execute_tool("send_email", to=email, subject=f"Weather in {city}", body=weather_str)
        prompt = engine._format_prompt("weather_email", L1_PROMPT_VARIANTS, city=city, email=email)
        distractors = engine._pick_distractors(["get_weather", "send_email"])
        return Task(
            task_id=engine._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=prompt,
            available_tools=["get_weather", "send_email"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_weather", arguments={"city": city}, output_key="weather", depends_on=[]),
                    ToolCall(step_id="step_2", tool_name="send_email", arguments={"to": email, "subject": f"Weather in {city}", "body": weather_str}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=sent,
            metadata={"category": "chain", "tools": ["get_weather", "send_email"], "pattern": "retrieve-send"},
        )

    @staticmethod
    def _l1_read_summarize(engine: CompositionEngine) -> Task:
        path = engine._pick(FILE_PATHS_EXPANDED)
        content = engine._execute_tool("read_file", path=path)
        content_str = content.get("content", json.dumps(content, default=str))[:500]
        summary = engine._execute_tool("summarize_text", text=content_str)
        prompt = engine._format_prompt("read_summarize", L1_PROMPT_VARIANTS, path=path)
        distractors = engine._pick_distractors(["read_file", "summarize_text"])
        return Task(
            task_id=engine._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=prompt,
            available_tools=["read_file", "summarize_text"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="read_file", arguments={"path": path}, output_key="content", depends_on=[]),
                    ToolCall(step_id="step_2", tool_name="summarize_text", arguments={"text": content_str}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=summary,
            metadata={"category": "chain", "tools": ["read_file", "summarize_text"], "pattern": "read-process"},
        )

    @staticmethod
    def _l1_kb_translate(engine: CompositionEngine) -> Task:
        query = engine._pick(KB_QUERIES_EXPANDED)
        lang = engine._pick(LANGUAGES_EXPANDED)
        code = LANGUAGE_CODES_EXPANDED.get(lang, "es")
        kb_result = engine._execute_tool("knowledge_base_query", query=query)
        answer_text = json.dumps(kb_result, default=str)[:300]
        translated = engine._execute_tool("translate_text", text=answer_text, target_language=code)
        prompt = engine._format_prompt("kb_translate", L1_PROMPT_VARIANTS, query=query, lang=lang)
        distractors = engine._pick_distractors(["knowledge_base_query", "translate_text"])
        return Task(
            task_id=engine._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=prompt,
            available_tools=["knowledge_base_query", "translate_text"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="knowledge_base_query", arguments={"query": query}, output_key="answer", depends_on=[]),
                    ToolCall(step_id="step_2", tool_name="translate_text", arguments={"text": answer_text, "target_language": code}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=translated,
            metadata={"category": "chain", "tools": ["knowledge_base_query", "translate_text"], "pattern": "retrieve-translate"},
        )

    @staticmethod
    def _l1_search_classify(engine: CompositionEngine) -> Task:
        query = engine._pick(SEARCH_QUERIES_EXPANDED)
        categories = engine._pick([
            ["science", "technology", "politics"],
            ["health", "finance", "entertainment"],
            ["sports", "education", "environment"],
        ])
        results = engine._execute_tool("web_search", query=query)
        result_text = json.dumps(results.get("results", results), default=str)[:300]
        classified = engine._execute_tool("classify_text", text=result_text, categories=categories)
        prompt = engine._format_prompt("search_classify", L1_PROMPT_VARIANTS, query=query)
        distractors = engine._pick_distractors(["web_search", "classify_text"])
        return Task(
            task_id=engine._next_id(CompositionLevel.CHAIN),
            level=CompositionLevel.CHAIN,
            prompt=prompt,
            available_tools=["web_search", "classify_text"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="web_search", arguments={"query": query}, output_key="results", depends_on=[]),
                    ToolCall(step_id="step_2", tool_name="classify_text", arguments={"text": result_text, "categories": categories}, depends_on=["step_1"]),
                ],
                final_answer_source="step_2",
            ),
            expected_final_answer=classified,
            metadata={"category": "chain", "tools": ["web_search", "classify_text"], "pattern": "retrieve-classify"},
        )

    # -----------------------------------------------------------------------
    # L2 Parallel Generator Registry
    # -----------------------------------------------------------------------

    @staticmethod
    def _l2_multi_weather(engine: CompositionEngine) -> Task:
        cities = engine._pick_n(CITIES_EXPANDED, 3)
        weathers = {}
        steps = []
        for i, city in enumerate(cities):
            w = engine._execute_tool("get_weather", city=city)
            weathers[city] = w
            steps.append(ToolCall(
                step_id=f"step_{i + 1}", tool_name="get_weather",
                arguments={"city": city},
                output_key=f"weather_{city.lower().replace(' ', '_')}",
                depends_on=[],
            ))
        warmest = max(weathers, key=lambda c: weathers[c]["temperature_celsius"])
        prompt = engine._format_prompt("multi_weather", L2_PROMPT_VARIANTS, cities=", ".join(cities))
        distractors = engine._pick_distractors(["get_weather"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["get_weather"] + distractors,
            expected_trace=ExpectedTrace(steps=steps, final_answer_source=f"step_{cities.index(warmest) + 1}"),
            expected_final_answer={"warmest_city": warmest, "temperature": weathers[warmest]["temperature_celsius"]},
            metadata={"category": "parallel", "tools": ["get_weather"], "pattern": "fan-out-compare"},
        )

    @staticmethod
    def _l2_multi_stock(engine: CompositionEngine) -> Task:
        symbols = engine._pick_n(STOCKS_EXPANDED, 3)
        stocks = {}
        steps = []
        for i, sym in enumerate(symbols):
            s = engine._execute_tool("get_stock_price", symbol=sym)
            stocks[sym] = s
            steps.append(ToolCall(
                step_id=f"step_{i + 1}", tool_name="get_stock_price",
                arguments={"symbol": sym},
                output_key=f"stock_{sym.lower()}",
                depends_on=[],
            ))
        highest = max(stocks, key=lambda s: stocks[s].get("price", stocks[s].get("price_usd", 0)))
        prompt = engine._format_prompt("multi_stock", L2_PROMPT_VARIANTS, symbols=", ".join(symbols))
        distractors = engine._pick_distractors(["get_stock_price"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["get_stock_price"] + distractors,
            expected_trace=ExpectedTrace(steps=steps, final_answer_source=f"step_{symbols.index(highest) + 1}"),
            expected_final_answer={"highest_stock": highest, "price": stocks[highest].get("price", stocks[highest].get("price_usd", 0))},
            metadata={"category": "parallel", "tools": ["get_stock_price"], "pattern": "fan-out-compare"},
        )

    @staticmethod
    def _l2_multi_sentiment(engine: CompositionEngine) -> Task:
        items = engine._pick_n(SENTIMENT_TEXTS_EXPANDED, 3)
        sentiments = {}
        steps = []
        for i, (text, _label) in enumerate(items):
            s = engine._execute_tool("sentiment_analysis", text=text)
            sentiments[text[:40]] = s
            steps.append(ToolCall(
                step_id=f"step_{i + 1}", tool_name="sentiment_analysis",
                arguments={"text": text},
                output_key=f"sentiment_{i + 1}",
                depends_on=[],
            ))
        prompt = f"Analyze the sentiment of these texts: 1) \"{items[0][0][:60]}...\" 2) \"{items[1][0][:60]}...\" 3) \"{items[2][0][:60]}...\""
        distractors = engine._pick_distractors(["sentiment_analysis"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["sentiment_analysis"] + distractors,
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=list(sentiments.values()),
            metadata={"category": "parallel", "tools": ["sentiment_analysis"], "pattern": "fan-out-analyze"},
        )

    @staticmethod
    def _l2_parallel_search_kb(engine: CompositionEngine) -> Task:
        q1 = engine._pick(SEARCH_QUERIES_EXPANDED)
        q2 = engine._pick(KB_QUERIES_EXPANDED)
        r1 = engine._execute_tool("web_search", query=q1)
        r2 = engine._execute_tool("knowledge_base_query", query=q2)
        prompt = engine._format_prompt("parallel_search_kb", L2_PROMPT_VARIANTS, q1=q1, q2=q2)
        distractors = engine._pick_distractors(["web_search", "knowledge_base_query"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["web_search", "knowledge_base_query"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="web_search", arguments={"query": q1}, output_key="search", depends_on=[]),
                    ToolCall(step_id="step_2", tool_name="knowledge_base_query", arguments={"query": q2}, output_key="kb", depends_on=[]),
                ],
                final_answer_source="step_1",
            ),
            expected_final_answer={"search_results": r1, "kb_answer": r2},
            metadata={"category": "parallel", "tools": ["web_search", "knowledge_base_query"], "pattern": "independent-merge"},
        )

    @staticmethod
    def _l2_multi_translate(engine: CompositionEngine) -> Task:
        text = engine._pick(["Hello, how are you?", "Good morning", "Thank you very much",
                             "The weather is nice today", "See you tomorrow"])
        langs = engine._pick_n(LANGUAGES_EXPANDED, 3)
        translations = {}
        steps = []
        for i, lang in enumerate(langs):
            code = LANGUAGE_CODES_EXPANDED.get(lang, "es")
            t = engine._execute_tool("translate_text", text=text, target_language=code)
            translations[lang] = t
            steps.append(ToolCall(
                step_id=f"step_{i + 1}", tool_name="translate_text",
                arguments={"text": text, "target_language": code},
                output_key=f"translation_{lang.lower()}",
                depends_on=[],
            ))
        prompt = engine._format_prompt("multi_translate", L2_PROMPT_VARIANTS, text=text, langs=", ".join(langs))
        distractors = engine._pick_distractors(["translate_text"])
        return Task(
            task_id=engine._next_id(CompositionLevel.PARALLEL),
            level=CompositionLevel.PARALLEL,
            prompt=prompt,
            available_tools=["translate_text"] + distractors,
            expected_trace=ExpectedTrace(steps=steps, final_answer_source="step_1"),
            expected_final_answer=translations,
            metadata={"category": "parallel", "tools": ["translate_text"], "pattern": "fan-out-translate"},
        )

    # -----------------------------------------------------------------------
    # L3 DAG Generator Registry
    # -----------------------------------------------------------------------

    @staticmethod
    def _l3_weather_compare_email(engine: CompositionEngine) -> Task:
        cities = engine._pick_n(CITIES_EXPANDED, 2)
        email = engine._pick(EMAIL_RECIPIENTS_EXPANDED)
        w1 = engine._execute_tool("get_weather", city=cities[0])
        w2 = engine._execute_tool("get_weather", city=cities[1])
        warmer_idx = 0 if w1["temperature_celsius"] > w2["temperature_celsius"] else 1
        warmer_temp = max(w1["temperature_celsius"], w2["temperature_celsius"])
        converted = engine._execute_tool("unit_convert", value=warmer_temp, from_unit="celsius", to_unit="fahrenheit")
        email_body = f"The warmer city is {cities[warmer_idx]} at {converted['converted_value']}°F"
        sent = engine._execute_tool("send_email", to=email, subject="Weather Comparison", body=email_body)
        prompt = engine._format_prompt("weather_compare_email", L3_PROMPT_VARIANTS,
                                       c1=cities[0], c2=cities[1], email=email)
        distractors = engine._pick_distractors(["get_weather", "unit_convert", "send_email"])
        return Task(
            task_id=engine._next_id(CompositionLevel.DAG),
            level=CompositionLevel.DAG,
            prompt=prompt,
            available_tools=["get_weather", "unit_convert", "send_email"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_weather", arguments={"city": cities[0]}, output_key="w1", depends_on=[]),
                    ToolCall(step_id="step_2", tool_name="get_weather", arguments={"city": cities[1]}, output_key="w2", depends_on=[]),
                    ToolCall(step_id="step_3", tool_name="unit_convert", arguments={"value": warmer_temp, "from_unit": "celsius", "to_unit": "fahrenheit"}, depends_on=["step_1", "step_2"]),
                    ToolCall(step_id="step_4", tool_name="send_email", arguments={"to": email, "subject": "Weather Comparison", "body": email_body}, depends_on=["step_3"]),
                ],
                final_answer_source="step_4",
            ),
            expected_final_answer=sent,
            metadata={"category": "dag", "tools": ["get_weather", "unit_convert", "send_email"], "pattern": "parallel-merge-chain"},
        )

    @staticmethod
    def _l3_stock_compare_notify(engine: CompositionEngine) -> Task:
        symbols = engine._pick_n(STOCKS_EXPANDED, 2)
        currency = engine._pick([c for c in CURRENCIES_EXPANDED if c != "USD"])
        s1 = engine._execute_tool("get_stock_price", symbol=symbols[0])
        s2 = engine._execute_tool("get_stock_price", symbol=symbols[1])
        p1 = s1.get("price", s1.get("price_usd", 100))
        p2 = s2.get("price", s2.get("price_usd", 100))
        higher_price = max(p1, p2)
        higher_sym = symbols[0] if p1 >= p2 else symbols[1]
        converted = engine._execute_tool("get_exchange_rate", from_currency="USD", to_currency=currency, amount=higher_price)
        notif = engine._execute_tool("create_notification", message=f"{higher_sym} is higher at {converted.get('converted_amount', higher_price)} {currency}", priority="high")
        prompt = engine._format_prompt("stock_compare_notify", L3_PROMPT_VARIANTS,
                                       s1=symbols[0], s2=symbols[1], currency=currency)
        distractors = engine._pick_distractors(["get_stock_price", "get_exchange_rate", "create_notification"])
        return Task(
            task_id=engine._next_id(CompositionLevel.DAG),
            level=CompositionLevel.DAG,
            prompt=prompt,
            available_tools=["get_stock_price", "get_exchange_rate", "create_notification"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="get_stock_price", arguments={"symbol": symbols[0]}, output_key="s1", depends_on=[]),
                    ToolCall(step_id="step_2", tool_name="get_stock_price", arguments={"symbol": symbols[1]}, output_key="s2", depends_on=[]),
                    ToolCall(step_id="step_3", tool_name="get_exchange_rate", arguments={"from_currency": "USD", "to_currency": currency, "amount": higher_price}, depends_on=["step_1", "step_2"]),
                    ToolCall(step_id="step_4", tool_name="create_notification", arguments={"message": f"{higher_sym} is higher at {converted.get('converted_amount', higher_price)} {currency}", "priority": "high"}, depends_on=["step_3"]),
                ],
                final_answer_source="step_4",
            ),
            expected_final_answer=notif,
            metadata={"category": "dag", "tools": ["get_stock_price", "get_exchange_rate", "create_notification"], "pattern": "parallel-merge-chain"},
        )

    @staticmethod
    def _l3_search_summarize_translate_email(engine: CompositionEngine) -> Task:
        query = engine._pick(SEARCH_QUERIES_EXPANDED)
        lang = engine._pick(LANGUAGES_EXPANDED)
        code = LANGUAGE_CODES_EXPANDED.get(lang, "es")
        email = engine._pick(EMAIL_RECIPIENTS_EXPANDED)
        results = engine._execute_tool("web_search", query=query)
        result_text = json.dumps(results.get("results", results), default=str)[:500]
        summary = engine._execute_tool("summarize_text", text=result_text)
        summary_text = summary.get("summary", json.dumps(summary, default=str))[:300]
        translated = engine._execute_tool("translate_text", text=summary_text, target_language=code)
        translated_text = translated.get("translated_text", json.dumps(translated, default=str))[:300]
        sent = engine._execute_tool("send_email", to=email, subject=f"Research: {query}", body=translated_text)
        prompt = engine._format_prompt("search_summarize_translate_email", L3_PROMPT_VARIANTS,
                                       query=query, lang=lang, email=email)
        distractors = engine._pick_distractors(["web_search", "summarize_text", "translate_text", "send_email"])
        return Task(
            task_id=engine._next_id(CompositionLevel.DAG),
            level=CompositionLevel.DAG,
            prompt=prompt,
            available_tools=["web_search", "summarize_text", "translate_text", "send_email"] + distractors,
            expected_trace=ExpectedTrace(
                steps=[
                    ToolCall(step_id="step_1", tool_name="web_search", arguments={"query": query}, output_key="results", depends_on=[]),
                    ToolCall(step_id="step_2", tool_name="summarize_text", arguments={"text": result_text}, output_key="summary", depends_on=["step_1"]),
                    ToolCall(step_id="step_3", tool_name="translate_text", arguments={"text": summary_text, "target_language": code}, output_key="translated", depends_on=["step_2"]),
                    ToolCall(step_id="step_4", tool_name="send_email", arguments={"to": email, "subject": f"Research: {query}", "body": translated_text}, depends_on=["step_3"]),
                ],
                final_answer_source="step_4",
            ),
            expected_final_answer=sent,
            metadata={"category": "dag", "tools": ["web_search", "summarize_text", "translate_text", "send_email"], "pattern": "linear-chain-4"},
        )

    @staticmethod
    def _l3_multi_weather_stats(engine: CompositionEngine) -> Task:
        cities = engine._pick_n(CITIES_EXPANDED, 4)
        weathers = {}
        steps = []
        for i, city in enumerate(cities):
            w = engine._execute_tool("get_weather", city=city)
            weathers[city] = w
            steps.append(ToolCall(
                step_id=f"step_{i + 1}", tool_name="get_weather",
                arguments={"city": city}, output_key=f"w_{i + 1}", depends_on=[],
            ))

        temps = [weathers[c]["temperature_celsius"] for c in cities]
        stats = engine._execute_tool("statistical_analysis", values=temps)
        depends = [f"step_{i + 1}" for i in range(len(cities))]
        steps.append(ToolCall(
            step_id=f"step_{len(cities) + 1}", tool_name="statistical_analysis",
            arguments={"values": temps}, output_key="stats",
            depends_on=depends,
        ))

        stats_text = json.dumps({"cities": cities, "temperatures": temps, "stats": stats}, default=str)
        written = engine._execute_tool("write_file", path="/data/weather_report.txt", content=stats_text)
        steps.append(ToolCall(
            step_id=f"step_{len(cities) + 2}", tool_name="write_file",
            arguments={"path": "/data/weather_report.txt", "content": stats_text},
            depends_on=[f"step_{len(cities) + 1}"],
        ))

        prompt = engine._format_prompt("multi_weather_stats", L3_PROMPT_VARIANTS, cities=", ".join(cities))
        distractors = engine._pick_distractors(["get_weather", "statistical_analysis", "write_file"])
        return Task(
            task_id=engine._next_id(CompositionLevel.DAG),
            level=CompositionLevel.DAG,
            prompt=prompt,
            available_tools=["get_weather", "statistical_analysis", "write_file"] + distractors,
            expected_trace=ExpectedTrace(steps=steps, final_answer_source=f"step_{len(cities) + 2}"),
            expected_final_answer=written,
            metadata={"category": "dag", "tools": ["get_weather", "statistical_analysis", "write_file"], "pattern": "fan-out-aggregate-save"},
        )

    # -----------------------------------------------------------------------
    # Generator registries (maps name -> static method)
    # -----------------------------------------------------------------------

    _l0_generators: dict[str, Any] = {
        "get_weather": _l0_weather.__func__,
        "calculator": _l0_calculator.__func__,
        "unit_convert": _l0_unit_convert.__func__,
        "get_stock_price": _l0_stock_price.__func__,
        "get_exchange_rate": _l0_exchange_rate.__func__,
        "sentiment_analysis": _l0_sentiment.__func__,
        "web_search": _l0_web_search.__func__,
        "translate_text": _l0_translate.__func__,
        "summarize_text": _l0_summarize.__func__,
        "extract_entities": _l0_extract_entities.__func__,
    }

    _l1_generators: dict[str, Any] = {
        "weather_convert": _l1_weather_convert.__func__,
        "search_summarize": _l1_search_summarize.__func__,
        "stock_convert": _l1_stock_convert.__func__,
        "entity_sentiment": _l1_entity_sentiment.__func__,
        "search_translate": _l1_search_translate.__func__,
        "weather_email": _l1_weather_email.__func__,
        "read_summarize": _l1_read_summarize.__func__,
        "kb_translate": _l1_kb_translate.__func__,
        "search_classify": _l1_search_classify.__func__,
    }

    _l2_generators: dict[str, Any] = {
        "multi_weather": _l2_multi_weather.__func__,
        "multi_stock": _l2_multi_stock.__func__,
        "multi_sentiment": _l2_multi_sentiment.__func__,
        "parallel_search_kb": _l2_parallel_search_kb.__func__,
        "multi_translate": _l2_multi_translate.__func__,
    }

    _l3_generators: dict[str, Any] = {
        "weather_compare_email": _l3_weather_compare_email.__func__,
        "stock_compare_notify": _l3_stock_compare_notify.__func__,
        "search_summarize_translate_email": _l3_search_summarize_translate_email.__func__,
        "multi_weather_stats": _l3_multi_weather_stats.__func__,
    }


# ---------------------------------------------------------------------------
# Wire in new generators from external modules
# ---------------------------------------------------------------------------

def _integrate_new_generators() -> None:
    """Import and register all new generators into the CompositionEngine."""
    from comptoolbench.generators._new_l1_generators import (
        NEW_L1_GENERATORS,
        NEW_L1_PROMPT_VARIANTS,
    )
    from comptoolbench.generators._new_l2_generators import (
        NEW_L2_PROMPT_VARIANTS,
        NEW_L2_REGISTRY,
    )
    from comptoolbench.generators._new_l3_generators import (
        NEW_L3_PROMPT_VARIANTS,
        NEW_L3_REGISTRY,
    )

    # Merge prompt variants
    L1_PROMPT_VARIANTS.update(NEW_L1_PROMPT_VARIANTS)
    L2_PROMPT_VARIANTS.update(NEW_L2_PROMPT_VARIANTS)
    L3_PROMPT_VARIANTS.update(NEW_L3_PROMPT_VARIANTS)

    # Merge generator registries
    CompositionEngine._l1_generators.update(NEW_L1_GENERATORS)
    CompositionEngine._l2_generators.update(NEW_L2_REGISTRY)
    CompositionEngine._l3_generators.update(NEW_L3_REGISTRY)


_integrate_new_generators()

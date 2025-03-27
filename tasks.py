from crewai import Task
from crewai.tasks.conditional_task import ConditionalTask
from tools import csv_search_tool, pdf_search_tool, file_writer_tool, scrapper_tool, serper_tool
from agents import csv_search_agent, pdf_search_agent, merger_agent, file_uploader_agent, scrapper_agent, serper_agent
from typing import List
from pydantic import BaseModel

# Helper function to extract text from an output.
def get_text(output) -> str:
    # If output has a 'raw_output' attribute, use it; otherwise, assume it's a string.
    return output.raw_output if hasattr(output, "raw_output") else str(output)

# Define conditions using the extracted text.
def pdf_cond(output) -> bool:
    text = get_text(output)
    # Consider output insufficient if it's empty or its length is less than 200 characters.
    return (text is None) or (len(text.strip()) == 0) or (len(text.strip()) < 200)

def serper_cond(output) -> bool:
    text = get_text(output)
    # Consider output insufficient if it's empty or its length is less than 20 characters.
    return (text is None) or (len(text.strip()) == 0) or (len(text.strip()) < 20)

def scrapper_cond(output) -> bool:
    # For scrapper, we expect output to have a pydantic attribute with an 'events' field,
    # or to be a list. We'll try to extract events accordingly.
    events = []
    if hasattr(output, "pydantic") and hasattr(output.pydantic, "events"):
        events = output.pydantic.events
    elif isinstance(output, list):
        events = output
    return len(events) < 10

# Define a simple Pydantic model to wrap serper output.
class EventOutput(BaseModel):
    events: List[str]

# Task for CSV RAG Agent:
csv_rag_task = Task(
    description=(
        "Search the CSV file for detailed information about {topic} and extract all relevant records. "
        "The report should include key details such as career information, contact details, or other structured data. "
        "Make sure to include any URLs found in the records."
    ),
    expected_output=(
        "A comprehensive summary report of {topic} extracted from the CSV file, highlighting critical data points."
    ),
    tools=[csv_search_tool],
    agent=csv_search_agent,
)

# Task for PDF RAG Agent:
pdf_rag_task = ConditionalTask(
    description=(
        "If the data from the CSV is missing or insufficient, extract and summarize relevant content about {topic} from the PDF document. "
        "Focus on capturing detailed insights and context from the unstructured text. "
        "Use a simple string query with the PDF search tool."
    ),
    expected_output=(
        "A concise summary report of {topic} extracted from the PDF file, as plain text."
    ),
    tools=[pdf_search_tool],
    agent=pdf_search_agent,
    condition=pdf_cond,
)

# Task for Serper Agent:
serper_task = ConditionalTask(
    description=(
        "If the CSV and PDF outputs are insufficient, search online for {topic} using the Serper tool "
        "and return a list of relevant URLs."
    ),
    expected_output=(
        "A list of URLs relevant to the query about {topic}."
    ),
    tools=[serper_tool],
    agent=serper_agent,
    allow_delegation=False,
    condition=serper_cond,
    output_pydantic=EventOutput,
)

# Task for Scrapper Agent:
scrapper_task = ConditionalTask(
    description=(
        "If the output from the Serper agent is insufficient (less than 10 events), "
        "scrape the webpages from the provided URLs to extract additional information about {topic} "
        "and return a comprehensive summary."
    ),
    expected_output=(
        "A comprehensive summary report of {topic} extracted from the webpages, in clear plain text."
    ),
    tools=[scrapper_tool],
    agent=scrapper_agent,
    context=[serper_task],
    allow_delegation=False,
    condition=scrapper_cond,
)

# Task for Merging Agent:
merging_task = Task(
    description=(
        "Merge the available outputs into a unified markdown report for {topic}. "
        "Include data from CSV, PDF, and/or web scraping sources as available."
    ),
    expected_output=(
        "A comprehensive markdown report combining all extracted information about {topic}."
    ),
    agent=merger_agent,
    allow_delegation=False,
    tools=[],
    context=[csv_rag_task, pdf_rag_task, scrapper_task],
)

# Task for File Uploader Agent:
file_writer_task = Task(
    description=(
        "Write the final report for {topic} to a markdown file based on the merged data."
    ),
    expected_output=(
        "A markdown file containing the final report in the specified output directory."
    ),
    tools=[file_writer_tool],
    agent=file_uploader_agent,
    allow_delegation=False,
    context=[merging_task],
    output_file="outputs/{topic}_report.md",
)

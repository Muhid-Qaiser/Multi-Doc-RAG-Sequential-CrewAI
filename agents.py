from crewai import Agent, LLM
from tools import csv_search_tool, pdf_search_tool, file_writer_tool, scrapper_tool, serper_tool
import litellm
import os

os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

# litellm.api_key = os.getenv("GOOGLE_API_KEY")
# llm="gemini/gemini-1.5-flash"

csv_search_agent = Agent(
    role="CSV Data Retrieval Specialist",
    goal="Efficiently search and extract relevant information from structured CSV files based on user queries. If any URLs are present in the records, make sure to return them as well.",
    memory=True,
    backstory="Trained in handling structured datasets, this agent excels at parsing CSV files to quickly retrieve accurate data and provide context-aware responses.",
    tools=[csv_search_tool],  # Add the search tools
    verbose=True,  # Enable verbose output
    allow_delegation=False,
    max_iter=1, 
    # llm=llm,
)

pdf_search_agent = Agent(
    role="PDF Content Analyzer",
    goal="Extract and summarize key information from unstructured PDF documents to effectively answer user queries.",
    memory=True,
    backstory="With expertise in processing complex, unstructured text, this agent navigates PDF documents to identify and extract critical insights, delivering comprehensive and concise summaries.",
    tools=[pdf_search_tool],  # Add the search tools
    verbose=True,  # Enable verbose output
    allow_delegation=False,
    max_iter=1, 
    # # llm=llm,
)

merger_agent = Agent(
    role="Data Fusion Specialist",
    goal="Merge and consolidate information from multiple sources including JSON and unstructured format to create a unified .md report. Make sure to summarize the data in markdown format.",
    memory=True,
    backstory="Skilled in combining data from diverse sources, this agent integrates insights from various outputs to generate comprehensive reports.",
    verbose=True,
    allow_delegation=False,
    tools=[],
    max_iter=1, 
    # # llm=llm,
)

file_uploader_agent = Agent(
    role="File Uploader",
    goal="Upload the final report to the specified destination in markdown format.",
    memory=True,
    backstory="Proficient in managing file uploads, this agent ensures the secure and efficient transfer of generated reports to the designated location.",
    verbose=True,
    allow_delegation=False,
    tools=[file_writer_tool],  # Add the file writer tool
    max_iter=1, 
    # # llm=llm,
)

serper_agent = Agent(
    role="SERP Specialist",
    goal="Extract relevant urls from search engine results pages (SERPs) based on user queries. You must provide a list of URLs that are most relevant to the user's search  to allow the next agent to extract valuable insights from them.",
    memory=True,
    backstory="Trained in navigating search engine results, this agent extracts urls of valuable websites and/or webpages from SERPs to allow the next agent to extract valuable insights from them.",
    tools=[serper_tool],  # Add the search tools
    verbose=True,  # Enable verbose output
    allow_delegation=False,
    max_iter=1, 
    # # llm=llm,
)

scrapper_agent = Agent(
    role="Web Scrapper",
    goal="Scrape the webpages from the given urls from previous task for relevant information based on user queries.",
    memory=True,
    backstory="Trained in web scraping techniques, this agent navigates online sources and webpages to extract valuable data and insights.",
    tools=[scrapper_tool],  # Add the search tools
    verbose=True,  # Enable verbose output
    allow_delegation=False,
    max_iter=1, 
    # # llm=llm,
)

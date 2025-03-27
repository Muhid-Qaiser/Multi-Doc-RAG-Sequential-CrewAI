import os
import json
from crewai import Crew, Agent, Task
from crewai import crew, task, agent
from crewai_tools import CSVSearchTool, PDFSearchTool, LinkedInScrapeTool  # Assumed available

# File paths
CSV_PATH = "data/famous_people.csv"
PDF_PATH = "data/information.pdf"

@agent
def decision_agent() -> Agent:
    return Agent(
        llm="openai/gpt-4",
        max_iter=3,
        verbose=True
    )

@agent
def csv_agent() -> Agent:
    return Agent(
        llm="openai/gpt-4",
        tools=[CSVSearchTool(csv=CSV_PATH)],
        max_iter=3,
        verbose=True
    )

@agent
def pdf_agent() -> Agent:
    return Agent(
        llm="openai/gpt-4",
        tools=[PDFSearchTool(pdf=PDF_PATH)],
        max_iter=3,
        verbose=True
    )

@agent
def summarization_agent() -> Agent:
    return Agent(
        llm="openai/gpt-4",
        max_iter=3,
        verbose=True
    )

def create_scraping_agent(linkedin_url: str) -> Agent:
    # Create a LinkedIn scraping tool for the provided URL.
    linkedin_tool = LinkedInScrapeTool(url=linkedin_url)
    return Agent(
        llm="openai/gpt-4",
        tools=[linkedin_tool],
        max_iter=3,
        verbose=True
    )

@task
def multi_agent_task(inputs: dict) -> Task:
    query = inputs.get("query")
    
    # Step 1: Decide which source to use.
    decision_prompt = (
        f"Given the query: '{query}', decide if the answer should be retrieved from a CSV "
        "of famous people or from a PDF document with general information. Answer with 'csv' or 'pdf'."
    )
    decision = decision_agent().run(decision_prompt).strip().lower()
    
    if decision == "csv":
        # Step 2: Search CSV for matching entity.
        csv_prompt = f"Search for an entity matching: '{query}'. Return the CSV record in JSON format."
        csv_response = csv_agent().run(csv_prompt)
        try:
            csv_record = json.loads(csv_response)
        except Exception:
            csv_record = None
        
        if csv_record and csv_record.get("linkedin_url"):
            linkedin_url = csv_record["linkedin_url"]
            # Step 3: Scrape LinkedIn profile.
            scraping_agent = create_scraping_agent(linkedin_url)
            scrape_prompt = (
                f"Scrape the LinkedIn profile at {linkedin_url} and extract key details "
                "such as experience, education, and achievements."
            )
            scraped_info = scraping_agent.run(scrape_prompt)
            # Step 4: Summarize scraped info into a markdown report.
            summary_prompt = (
                f"Summarize the following scraped LinkedIn profile information into a markdown report. "
                "Include sections for Experience, Education, and Achievements:\n\n{0}"
            ).format(scraped_info)
            final_report = summarization_agent().run(summary_prompt)
            result = final_report
        else:
            # No LinkedIn URL: simply report CSV information.
            summary_prompt = (
                f"Summarize the following CSV record into a markdown formatted report:\n\n{csv_response}"
            )
            final_report = summarization_agent().run(summary_prompt)
            result = final_report

    elif decision == "pdf":
        # Step 2 (alternative): Use PDF to answer.
        pdf_prompt = f"Extract and summarize relevant information from the PDF regarding: '{query}'."
        pdf_info = pdf_agent().run(pdf_prompt)
        # Step 3: Summarize PDF info into a markdown report.
        summary_prompt = (
            f"Summarize the following PDF extracted information into a markdown report:\n\n{pdf_info}"
        )
        final_report = summarization_agent().run(summary_prompt)
        result = final_report
    else:
        result = "Unable to determine the appropriate data source."

    return Task(raw_output=result, description="Multi-agent report task completed.")

@crew
def my_crew(inputs: dict) -> Crew:
    return Crew(
        tasks=[multi_agent_task(inputs=inputs)],
        verbose=True
    )

if __name__ == "__main__":
    # Provide the query as input.
    crew_inputs = {"query": "Tell me about Bill Gates"}
    crew_instance = my_crew(inputs=crew_inputs)
    final_output = crew_instance.kickoff()
    print("Final Markdown Report:\n", final_output)

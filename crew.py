from dotenv import load_dotenv
load_dotenv()
from crewai import Crew,Process
from agents import csv_search_agent, pdf_search_agent, merger_agent, file_uploader_agent, serper_agent, scrapper_agent
from tasks import csv_rag_task, pdf_rag_task, merging_task, file_writer_task, serper_task, scrapper_task


# Forming the tech-focused crew with some enhanced configurations
docs_crew = Crew(
  agents=[csv_search_agent, pdf_search_agent, serper_agent, scrapper_agent, merger_agent, file_uploader_agent],
  tasks=[csv_rag_task, pdf_rag_task, serper_task, scrapper_task, merging_task, file_writer_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  cache=True,
  max_rpm=100,
  share_crew=False,
  planning=True,
)

## start the task execution process with enhanced feedback
result=docs_crew.kickoff(inputs={'topic':'Satya Nadella'})
print(result)



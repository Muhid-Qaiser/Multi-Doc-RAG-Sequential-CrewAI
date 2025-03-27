from crewai_tools import CSVSearchTool, PDFSearchTool, FileWriterTool, FirecrawlScrapeWebsiteTool, ScrapeWebsiteTool, SerperDevTool
# Initialize the CSV and PDF search tools with the respective file paths
csv_search_tool = CSVSearchTool(csv='data/famous_people.csv')
pdf_search_tool = PDFSearchTool(pdf='data/Pakistan_pdf.pdf')
file_writer_tool = FileWriterTool()
# scrapper_tool = FirecrawlScrapeWebsiteTool()
scrapper_tool = ScrapeWebsiteTool()
serper_tool = SerperDevTool()
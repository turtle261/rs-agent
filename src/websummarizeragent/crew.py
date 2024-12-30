import os
from dotenv import load_dotenv
import yaml
from langchain_google_genai import ChatGoogleGenerativeAI

from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import (
    WebsiteSearchTool,
    EXASearchTool,
    GithubSearchTool,
    DirectorySearchTool
)
import logging
import sys
import chromadb
from typing import Dict, Optional

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def clear_chroma_collections():
    client = chromadb.Client()
    for name in client.list_collections():
        client.delete_collection(name)

class WebSummarizer:
    """WebSummarizer crew that handles web summarization and research tasks"""

    def __init__(self, crew_inputs: Optional[Dict] = None):
        self.crew_inputs = crew_inputs or {}

        # Load agent configurations
        with open('src/websummarizeragent/config/agents.yaml', 'r') as f:
            agents_config = yaml.safe_load(f)

        agent_config = agents_config.get('web_summarizer_agent', {})
        llm_config = agent_config.get('llm', {})

        # Initialize the base LLM
        base_llm = ChatGoogleGenerativeAI(
            model="gemini-pro",
            verbose=True,
            temperature=0.1,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            convert_system_message_to_human=True
        )

        # Initialize CrewAI's LLM wrapper
        self.llm = LLM(
            provider="google",
            model="gemini-pro",
            temperature=0.1,
            api_key=os.getenv("GEMINI_API_KEY")
        )

        # Initialize Tools with the base LLM for better compatibility
        self.tools = [
            WebsiteSearchTool(llm=base_llm),
            EXASearchTool(),
            GithubSearchTool(),
            DirectorySearchTool()
        ]

        # Initialize Agent
        self.agent = Agent(
            role=agent_config.get('role', 'Web Content Summarizer'),
            goal=agent_config.get('goal', 'Create concise and accurate summaries of web content based on user queries'),
            backstory=agent_config.get('backstory', ''),
            llm=self.llm,
            tools=self.tools,
            verbose=True
        )

    def handle_request(self, url: Optional[str], query: str) -> Dict:
        try:
            tasks = [
                Task(
                    description=query,
                    agent=self.agent,
                    expected_output="A concise and relevant summary."
                )
                # Add other tasks as needed
            ]

            # Create Crew
            crew = Crew(
                agents=[self.agent],
                tasks=tasks,
                process=Process.hierarchical,
                manager=self.agent,
                verbose=True,
                max_rpm=20,  # Rate limit to 20 requests per minute
                task_timeout=600  # 10 minutes timeout per task
            )

            # Execute the tasks
            result = crew.kickoff()

            return {
                'success': True,
                'summary': result
            }

        except Exception as e:
            logger.error(f"Service error: {str(e)}")
            return {
                'error': 'Service error',
                'details': str(e)
            }

    def run(self) -> Dict:
        """Run the crew with the provided inputs."""
        query = self.crew_inputs.get('query')
        if not query:
            raise ValueError("Query is required in crew_inputs")

        url = self.crew_inputs.get('url')  # URL is now optional
        return self.handle_request(url, query) 
#!/usr/bin/env python
import sys
import warnings
import argparse
from .crew import WebSummarizer
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress pysbd warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def get_research_query():
    """Get the research query from the user."""
    print("\n🤖 Research Assistant AI")
    print("------------------------")
    print("I can help you research any topic, analyze websites, or answer complex questions.")
    print("Examples:")
    print("- What are the latest developments in quantum computing?")
    print("- Analyze https://example.com and tell me about its main points")
    print("- Compare different approaches to transformer architecture")
    print("\nType 'exit' to quit.")
    
    while True:
        query = input("\n🔍 What would you like to research? ").strip()
        if query.lower() == 'exit':
            sys.exit(0)
        if query:
            return query
        print("Please enter a valid query.")

def extract_urls(query):
    """Extract URLs from the query if present."""
    import re
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+'
    urls = re.findall(url_pattern, query)
    
    # Remove URLs from query to get clean research question
    clean_query = query
    for url in urls:
        clean_query = clean_query.replace(url, '').strip()
    
    return urls[0] if urls else None, clean_query

def run():
    """Run the research assistant crew."""
    # Load environment variables
    load_dotenv()

    try:
        while True:
            # Get research query from user
            query = get_research_query()
            
            # Extract URL if present in query
            url, clean_query = extract_urls(query)
            
            logger.info(f"Processing research query: {clean_query}")
            if url:
                logger.info(f"With reference URL: {url}")

            crew = WebSummarizer(crew_inputs={
                'url': url,
                'query': clean_query
            })
            result = crew.run()

            if result.get('success'):
                print("\n📊 Research Results:")
                print("-------------------")
                print(result['summary'])
            else:
                print("\n❌ Error:")
                print(result.get('error', 'Unknown error'))
                print("\nDetails:")
                print(result.get('details', 'No details available'))

            print("\n------------------------")
            print("Ready for another query!")

    except KeyboardInterrupt:
        print("\nThank you for using Research Assistant AI!")
        return 0
    except Exception as e:
        logger.error(f"Error running research assistant: {str(e)}")
        print(f"\n❌ Error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(run()) 
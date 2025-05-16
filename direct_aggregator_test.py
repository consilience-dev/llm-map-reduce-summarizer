"""
Direct test of the aggregation functionality for debugging purposes.
This uses pre-defined content to test just the aggregation step.
"""

import asyncio
import json
import logging
import os
from dotenv import load_dotenv
from typing import Dict, Any, List

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DirectTest")

# Sample summaries for testing
SAMPLE_SUMMARIES = [
    """
    ### 1. Concise Summary
    In this segment, the speaker discusses their plans for a website that allows users to record voices over beats with audio processing features. They express uncertainty about the project and deployment process.

    ### 2. Key Topics Discussed
    - Website for recording vocals over beats
    - Audio processing features
    - Deployment strategies and costs
    - Role of AI in coding

    ### 3. Notable Quotes or Statements
    - "I have no idea how to build any of that stuff, right? So it's really AI dependent if it can do it for me."
    - "Right now, your grandma can straight up code a website now."
    """,
    
    """
    ### 1. Concise Summary
    In this segment, the speaker explores various AI tools for backend and frontend development, highlighting platforms like Google Cloud Functions and Replay for rapid deployment.

    ### 2. Key Topics Discussed
    - AI tools for application development
    - Serverless architecture
    - Deployment platforms and options
    - Evolving technology landscape

    ### 3. Notable Quotes or Statements
    - "I just want to click a button and have my super duper complex backend setup."
    - "This is a quickly evolving area, and so I'm trying to figure out what to use."
    """
]

async def call_openai_api(prompt: str, system_message: str) -> Dict[str, Any]:
    """
    Direct call to OpenAI API for testing.
    
    Args:
        prompt: User message content
        system_message: System message to control behavior
        
    Returns:
        API response as dictionary
    """
    import aiohttp
    import os
    
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": 1000
    }
    
    logger.info("Calling OpenAI API directly")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=60
        ) as response:
            result = await response.json()
            
            if response.status != 200:
                error_message = result.get("error", {}).get("message", "Unknown error")
                raise Exception(f"OpenAI API Error: {error_message}")
            
            return result

async def test_direct_aggregation():
    """Test direct aggregation with OpenAI API."""
    logger.info("Starting direct aggregation test")
    
    # Format the summaries
    formatted_summaries = ""
    for i, summary in enumerate(SAMPLE_SUMMARIES):
        formatted_summaries += f"SUMMARY {i+1}:\n"
        formatted_summaries += "=" * 40 + "\n"
        formatted_summaries += f"{summary}\n"
        formatted_summaries += "=" * 40 + "\n\n"
    
    # Prepare a system message
    system_message = """
    You are a professional transcript summarizer. Your ONLY job is to combine multiple summaries into a coherent final summary.
    
    IMPORTANT RULES:
    1. DO NOT introduce yourself
    2. DO NOT include any greeting or pleasantries
    3. DO NOT ask how you can help
    4. ONLY output the requested summary in the specified format
    5. Start your response with the heading "# Transcript Summary"
    """
    
    # Prepare the prompt
    prompt = f"""
    I need you to combine these transcript summaries into a cohesive final summary.
    
    Here are the summaries from different segments of the transcript:
    
    {formatted_summaries}
    
    Format your response with these sections:
    
    # Transcript Summary
    
    ## Overview
    [2-3 sentence high-level overview]
    
    ## Main Topics
    [Bullet list of key themes]
    
    ## Key Points
    [Important details and insights]
    
    ## Notable Quotes
    [Significant quotes from the transcript]
    """
    
    try:
        # Call the API directly
        response = await call_openai_api(prompt, system_message)
        
        if response and response.get("choices") and response["choices"][0].get("message"):
            summary = response["choices"][0]["message"]["content"]
            
            print("\n" + "=" * 50)
            print("DIRECT AGGREGATION RESULT")
            print("=" * 50 + "\n")
            print(summary)
            
            # Output token usage
            usage = response.get("usage", {})
            print("\nToken Usage:")
            print(f"- Prompt tokens: {usage.get('prompt_tokens', 'unknown')}")
            print(f"- Completion tokens: {usage.get('completion_tokens', 'unknown')}")
            print(f"- Total tokens: {usage.get('total_tokens', 'unknown')}")
        else:
            print("Failed to get a proper response")
            print(response)
    except Exception as e:
        logger.error(f"Error during direct aggregation: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(test_direct_aggregation())

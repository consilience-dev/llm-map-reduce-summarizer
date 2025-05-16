"""
Simple Result Aggregator Module for Transcript Summarization.

This module provides a simplified approach to combining individual chunk summaries
into a coherent final summary, focusing on reliability and accuracy.
"""

import asyncio
import logging
import json
from typing import List, Dict, Any, Optional
import aiohttp
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SimpleAggregator")

class SimpleAggregator:
    """
    A simplified aggregator that combines summaries from transcript chunks.
    This implementation focuses on reliable, direct API calls with clear prompting.
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        Initialize the Simple Aggregator.
        
        Args:
            api_key: OpenAI API key (uses environment variable if None)
            model: OpenAI model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        self.model = model
        logger.info(f"Simple Aggregator initialized with model {model}")
    
    async def aggregate(self, chunk_summaries: List[str], metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Aggregate multiple chunk summaries into a cohesive final summary.
        
        Args:
            chunk_summaries: List of summaries from transcript chunks
            metadata: Optional metadata to include in the prompt
            
        Returns:
            Final aggregated summary
        """
        logger.info(f"Aggregating {len(chunk_summaries)} summaries")
        
        # Format the summaries for the prompt
        formatted_summaries = self._format_summaries(chunk_summaries)
        
        # Format metadata string if provided
        metadata_str = ""
        if metadata:
            metadata_str = "Additional Information:\n"
            for key, value in metadata.items():
                metadata_str += f"- {key}: {value}\n"
        
        # Create the system message
        system_message = """
You are a professional transcript summarizer that ONLY creates summaries.

IMPORTANT RULES:
1. DO NOT include any greeting in your response
2. DO NOT introduce yourself or explain what you're doing
3. DO NOT ask how you can help
4. ONLY output the requested summary in the specified format
5. Your response MUST start with "# Transcript Summary"
6. DO NOT make up information - use ONLY what's in the provided summaries
"""
        
        # Create the user prompt
        user_prompt = f"""
I need you to combine these transcript segment summaries into a final summary.

{metadata_str}

Here are the summaries from different parts of the transcript:

{formatted_summaries}

Your summary must accurately reflect ONLY the content in these summaries.

Format your response with these exact headings:

# Transcript Summary

## Overview
[2-3 sentence high-level description of the transcript content]

## Main Topics
[Bullet list of key themes and topics discussed]

## Key Points
[Bullet list of important details and takeaways]

## Notable Quotes
[Direct quotes from the transcript that were mentioned in the summaries]
"""
        
        # Call the OpenAI API directly
        try:
            summary = await self._call_openai_api(system_message, user_prompt)
            logger.info("Successfully generated aggregated summary")
            return summary
        except Exception as e:
            logger.error(f"Error generating aggregated summary: {e}")
            return f"Error generating summary: {str(e)}"
    
    def _format_summaries(self, summaries: List[str]) -> str:
        """Format summaries with clear separators for the prompt."""
        formatted_text = ""
        for i, summary in enumerate(summaries):
            formatted_text += f"SUMMARY {i+1}:\n"
            formatted_text += "=" * 40 + "\n"
            formatted_text += summary.strip() + "\n"
            formatted_text += "=" * 40 + "\n\n"
        return formatted_text
    
    async def _call_openai_api(self, system_message: str, user_prompt: str) -> str:
        """
        Call the OpenAI API directly to generate a summary.
        
        Args:
            system_message: System message to control model behavior
            user_prompt: User prompt with the summaries
            
        Returns:
            Generated summary text
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,  # Lower temperature for more consistent output
            "max_tokens": 1000
        }
        
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
                
                # Extract and return the summary text
                if result.get("choices") and result["choices"][0].get("message"):
                    return result["choices"][0]["message"]["content"]
                else:
                    raise Exception("No content in API response")

# Synchronous wrapper for easier usage
def aggregate_summaries(chunk_summaries: List[str], metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Aggregate chunk summaries into a final summary (synchronous wrapper).
    
    Args:
        chunk_summaries: List of summaries from transcript chunks
        metadata: Optional metadata to include in the prompt
        
    Returns:
        Final aggregated summary
    """
    aggregator = SimpleAggregator()
    return asyncio.run(aggregator.aggregate(chunk_summaries, metadata))

# Example usage
if __name__ == "__main__":
    import json
    
    async def test_simple_aggregator():
        # Sample summaries (simulated output from chunk processing)
        sample_summaries = [
            """
### 1. Concise Summary
The speaker discusses their plans for a website that allows users to record vocals over beats, with audio processing features. They express uncertainty about implementation and deployment.

### 2. Key Topics Discussed
- Website for recording vocals over beats
- Audio processing features
- Deployment strategies
- Role of AI in coding
            
### 3. Notable Quotes or Statements
- "I have no idea how to build any of that stuff, right? So it's really AI dependent if it can do it for me."
- "Right now, your grandma can straight up code a website now."
            """,
            
            """
### 1. Concise Summary
The speaker explores various AI tools for backend and frontend development, highlighting platforms like Google Cloud Functions and Replay for rapid deployment.

### 2. Key Topics Discussed
- AI tools for application development
- Serverless architecture
- Deployment platforms
            
### 3. Notable Quotes or Statements
- "I just want to click a button and have my super duper complex backend setup."
- "This is a quickly evolving area, and so I'm trying to figure out what to use."
            """
        ]
        
        # Sample metadata
        metadata = {
            "Title": "Coding Stream Planning",
            "Speaker": "SPEAKER_00",
            "Duration": "25:30"
        }
        
        # Create aggregator and test
        aggregator = SimpleAggregator()
        summary = await aggregator.aggregate(sample_summaries, metadata)
        
        print("\n" + "=" * 50)
        print("SIMPLE AGGREGATOR TEST RESULT")
        print("=" * 50 + "\n")
        print(summary)
    
    # Run the test
    asyncio.run(test_simple_aggregator())

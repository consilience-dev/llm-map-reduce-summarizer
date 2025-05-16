"""
Test script to verify OpenAI API integration with our transcript summarizer.
This will process a small portion of the transcript using the actual OpenAI API.
"""

import asyncio
import json
import logging
from preprocessor import preprocess_transcript
from big_chunkeroosky import BigChunkeroosky
from llm_executor import LLMExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLMTest")

async def test_real_api_call():
    """Test the LLM executor with a real OpenAI API call"""
    
    logger.info("Starting LLM API test with real OpenAI API call")
    
    # Load example transcript
    with open('transcript-example.json', 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # Preprocess a small portion of the transcript (first 200 segments)
    # This keeps the test focused and reduces API costs
    logger.info("Preprocessing the first 200 segments of the transcript")
    limited_segments = transcript_data['segments'][:200]
    processed_segments = preprocess_transcript(limited_segments)
    
    # Create chunks
    logger.info("Creating chunks with the Big Chunkeroosky")
    chunker = BigChunkeroosky(max_tokens_per_chunk=3000)  # Smaller chunks for testing
    chunks = chunker.chunk_transcript(processed_segments)
    chunks = chunker.postprocess_chunks(chunks)
    
    # Use only the first chunk to minimize API costs for this test
    test_chunks = [chunks[0]]
    logger.info(f"Testing with {len(test_chunks)} chunk")
    
    # Create a more detailed prompt
    prompt_template = """
    You're analyzing a transcript segment. Please provide:
    
    1. A concise summary (3-5 sentences) of the main points
    2. Key topics discussed
    3. Any notable quotes or statements
    
    Here's the transcript segment:
    
    {transcript}
    
    Please format your response clearly with headings for each section.
    """
    
    # Process the chunk with the LLM Executor
    try:
        logger.info("Calling OpenAI API with gpt-4o-mini model")
        # Use the model specified in the .env file
        executor = LLMExecutor()
        processed_chunks = await executor.process_chunks(test_chunks, prompt_template)
        
        # Print the results
        print("\n==== OPENAI API TEST RESULTS ====\n")
        print(f"API Call Successful: Yes")
        print(f"Model Used: {executor.model}")
        print(f"Tokens Used: {executor.total_tokens_used}")
        print(f"Estimated Cost: ${executor.total_cost:.4f}")
        
        # Display the summary
        for i, chunk in enumerate(processed_chunks):
            print(f"\nChunk {i+1} Summary:\n")
            print(f"Time Range: {chunker._format_time(chunk['start_time'])} - {chunker._format_time(chunk['end_time'])}")
            print(f"Tokens Used: {chunk.get('tokens_used', 'N/A')}")
            print(f"\n{chunk.get('summary', 'No summary generated')}")
            print("\n" + "-" * 50)
        
        logger.info("Test completed successfully")
        return True
    
    except Exception as e:
        logger.error(f"API call failed: {e}")
        print(f"\n==== ERROR DURING API TEST ====\n")
        print(f"API Call Failed: {str(e)}")
        return False

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_real_api_call())

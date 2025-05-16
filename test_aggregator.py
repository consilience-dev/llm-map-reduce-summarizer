"""
Test script for the Result Aggregator module.

This script demonstrates the use of the Result Aggregator to combine
multiple chunk summaries into a final summary.
"""

import asyncio
import json
import logging
import os
import time
from typing import List, Dict, Any

from preprocessor import preprocess_transcript
from big_chunkeroosky import BigChunkeroosky
from llm_executor import LLMExecutor
from result_aggregator import ResultAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AggregatorTest")

async def test_aggregator(
    transcript_path: str = 'transcript-example.json',
    use_hierarchical: bool = True,
    num_segments: int = 300,
    max_chunks: int = 3
):
    """
    Test the Result Aggregator with real data.
    
    Args:
        transcript_path: Path to the transcript JSON file
        use_hierarchical: Whether to use hierarchical aggregation
        num_segments: Number of transcript segments to process
        max_chunks: Maximum number of chunks to process (to limit API costs)
    """
    start_time = time.time()
    logger.info(f"Starting Result Aggregator test (hierarchical={use_hierarchical})")
    
    # Load transcript
    try:
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        logger.info(f"Loaded transcript from {transcript_path}")
    except Exception as e:
        logger.error(f"Failed to load transcript: {e}")
        return
    
    # Step 1: Preprocess the transcript (use limited segments to save time)
    logger.info(f"Preprocessing the first {num_segments} segments of the transcript")
    limited_segments = transcript_data['segments'][:num_segments]
    processed_segments = preprocess_transcript(limited_segments)
    logger.info(f"Preprocessed {len(limited_segments)} segments into {len(processed_segments)} segments")
    
    # Step 2: Create chunks
    logger.info("Creating chunks with the Big Chunkeroosky")
    chunker = BigChunkeroosky(max_tokens_per_chunk=3000)
    chunks = chunker.chunk_transcript(processed_segments)
    chunks = chunker.postprocess_chunks(chunks)
    logger.info(f"Created {len(chunks)} chunks")
    
    # Limit to specified number of chunks for testing
    test_chunks = chunks[:max_chunks]
    logger.info(f"Testing with {len(test_chunks)} chunks to limit API costs")
    
    # Step 3: Process chunks with LLM to get summaries
    logger.info("Calling LLM to generate summaries for each chunk")
    prompt_template = """
    Please summarize the following transcript segment:
    
    {transcript}
    
    Provide:
    
    ### 1. Concise Summary
    [3-5 sentence overview of the main content]
    
    ### 2. Key Topics Discussed
    [Bullet list of main topics]
    
    ### 3. Notable Quotes or Statements
    [2-3 important or representative quotes]
    """
    
    executor = LLMExecutor()
    processed_chunks = await executor.process_chunks(test_chunks, prompt_template)
    logger.info(f"Generated summaries for {len(processed_chunks)} chunks")
    
    # Step 4: Create the aggregator
    logger.info(f"Initializing Result Aggregator (hierarchical={use_hierarchical})")
    aggregator = ResultAggregator(
        executor=executor,
        hierarchical=use_hierarchical
    )
    
    # Metadata about the transcript for context
    metadata = {
        "Title": "Transcript Example",
        "Speakers": ", ".join(set([seg.get('speaker', 'Unknown') for seg in processed_segments])),
        "Total Duration": chunker._format_time(test_chunks[-1]['end_time']),
        "Number of Chunks": len(test_chunks)
    }
    
    # Step 5: Aggregate the summaries
    logger.info("Aggregating summaries into a final summary")
    result = await aggregator.aggregate(processed_chunks, metadata=metadata)
    
    # Print the summaries and final result
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*50)
    print("INDIVIDUAL CHUNK SUMMARIES")
    print("="*50)
    
    for i, chunk in enumerate(processed_chunks):
        print(f"\nChunk {i+1} ({chunk.get('start_time_formatted', '?')} - {chunk.get('end_time_formatted', '?')})")
        print("-"*50)
        print(chunk.get('summary', 'No summary generated'))
    
    print("\n" + "="*50)
    print("AGGREGATED SUMMARY")
    print("="*50)
    print(f"\nAggregation Method: {'Hierarchical' if use_hierarchical else 'Single-Pass'}")
    print(f"Processed {result['chunks_aggregated']} chunks in {result['processing_time']:.2f} seconds")
    print(f"Total process time: {elapsed_time:.2f} seconds")
    print("\nFinal Summary:")
    print("-"*50)
    print(result['summary'])

if __name__ == "__main__":
    # Run the test
    asyncio.run(test_aggregator(
        use_hierarchical=True,  # Test with hierarchical aggregation
        num_segments=300,       # Process 300 segments
        max_chunks=3            # Only use 3 chunks to limit API costs
    ))

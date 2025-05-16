"""
Result Aggregator Module for Transcript Summarization.

This module combines individual chunk summaries into a coherent final summary.
It implements hierarchical summarization to handle large numbers of summaries
and preserve important information from all chunks.
"""

import asyncio
import os
import logging
import time
import math
import aiohttp
from typing import List, Dict, Any, Optional, Union
import tiktoken
from llm_executor import LLMExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ResultAggregator")

class ResultAggregator:
    """
    Aggregator that combines individual chunk summaries into a coherent final summary.
    Implements hierarchical summarization for handling large numbers of summaries.
    """
    
    def __init__(
        self,
        executor: Optional[LLMExecutor] = None,
        max_tokens_per_batch: int = 6000,
        tokenizer_name: str = "cl100k_base",
        hierarchical: bool = True
    ):
        """
        Initialize the Result Aggregator.
        
        Args:
            executor: LLM executor to use (creates a new one if not provided)
            max_tokens_per_batch: Maximum tokens per aggregation batch
            tokenizer_name: Tokenizer to use for counting tokens
            hierarchical: Whether to use hierarchical summarization for large inputs
        """
        self.executor = executor or LLMExecutor()
        self.max_tokens_per_batch = max_tokens_per_batch
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.hierarchical = hierarchical
        
        logger.info(f"Result Aggregator initialized (hierarchical={hierarchical})")
    
    async def aggregate(
        self,
        processed_chunks: List[Dict[Any, Any]],
        prompt_template: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate multiple chunk summaries into a coherent final summary.
        
        Args:
            processed_chunks: List of processed chunks with summaries
            prompt_template: Custom prompt template for aggregation (uses default if None)
            metadata: Additional metadata to include in the prompt
            
        Returns:
            Dictionary with final summary and metadata
        """
        start_time = time.time()
        
        if not processed_chunks:
            logger.warning("No chunks provided for aggregation")
            return {"summary": "", "error": "No chunks provided for aggregation"}
        
        # Sort chunks by index to ensure correct order
        processed_chunks.sort(key=lambda x: x.get('chunk_index', 0))
        
        # Extract summaries
        summaries = []
        for chunk in processed_chunks:
            if 'summary' in chunk and chunk['summary']:
                # Add position information
                position_info = f"[Time: {self._format_time(chunk.get('start_time', 0))} - {self._format_time(chunk.get('end_time', 0))}]"
                chunk_summary = f"{position_info}\n{chunk['summary']}"
                summaries.append(chunk_summary)
            else:
                logger.warning(f"Chunk {chunk.get('chunk_index', '?')} missing summary")
        
        logger.info(f"Aggregating {len(summaries)} summaries")
        
        # Determine whether to use hierarchical summarization
        if not self.hierarchical or self._total_tokens(summaries) <= self.max_tokens_per_batch:
            # Simple case: all summaries fit in one batch
            final_summary = await self._single_aggregation(summaries, prompt_template, metadata)
        else:
            # Complex case: hierarchical summarization for many summaries
            final_summary = await self._hierarchical_aggregation(summaries, prompt_template, metadata)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Aggregation completed in {elapsed_time:.2f} seconds")
        
        return {
            "summary": final_summary,
            "chunks_aggregated": len(processed_chunks),
            "processing_time": elapsed_time
        }
    
    async def _single_aggregation(
        self,
        summaries: List[str],
        prompt_template: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Perform a single-pass aggregation of summaries using direct API calls.
        
        Args:
            summaries: List of summaries to aggregate
            prompt_template: Custom prompt template (uses default if None)
            metadata: Additional metadata to include in the prompt
            
        Returns:
            Final aggregated summary
        """
        logger.info(f"Performing single aggregation of {len(summaries)} summaries")
        
        # Prepare metadata string
        metadata_str = ""
        if metadata:
            metadata_str = "Additional Information:\n"
            for key, value in metadata.items():
                metadata_str += f"- {key}: {value}\n"
        
        # Format summaries with clear visual separators
        formatted_summaries = ""
        for i, summary in enumerate(summaries):
            formatted_summaries += f"SUMMARY {i+1}:\n"
            formatted_summaries += "=" * 40 + "\n"
            formatted_summaries += f"{summary}\n"
            formatted_summaries += "=" * 40 + "\n\n"
        
        # System message to control model behavior
        if prompt_template and "TIMELINE SUMMARY" in prompt_template:
            # Use a more flexible system message for custom format prompts like video editor
            system_message = """
            You are a professional transcript summarizer specializing in video editing formats. Your job is to create a 
            structured summary that combines information from multiple transcript segment summaries.
            
            IMPORTANT RULES:
            1. DO NOT include any greeting or introduction
            2. DO NOT ask how you can help
            3. Follow EXACTLY the format specified in the user prompt
            4. Preserve ALL timestamps in [HH:MM:SS] format
            5. The summary MUST ONLY contain information from the provided summaries
            6. DO NOT make up information not contained in the summaries
            7. DO NOT discuss general impacts of technology - stay focused on the transcript content
            """
        else:
            # Use the standard system message for default format
            system_message = """
            You are a professional transcript summarizer. Your ONLY job is to create a structured summary that 
            combines information from multiple transcript segment summaries.
            
            IMPORTANT RULES:
            1. DO NOT include any greeting or introduction
            2. DO NOT ask how you can help
            3. ONLY produce the summary in the requested format
            4. START your response with "# Transcript Summary"
            5. The summary MUST ONLY contain information from the provided summaries
            6. DO NOT make up information not contained in the summaries
            7. DO NOT discuss general impacts of technology - stay focused on the transcript content
            """
        
        # Check if we're using the video editor aggregator prompt
        is_video_editor = prompt_template and "TIMELINE SUMMARY" in prompt_template
        
        # User prompt with explicit instructions
        if is_video_editor:
            # Use the exact prompt template for video editor format
            # Replace the placeholder for summaries with the actual formatted summaries
            user_prompt = prompt_template.replace("{summaries}", formatted_summaries)
            
            # Add metadata if we have any
            if metadata_str:
                user_prompt = f"{metadata_str}\n\n{user_prompt}"
                
            logger.info("Using video editor format with custom template")
        else:
            # Use the default format for regular summaries
            user_prompt = f"""
            I need you to combine multiple transcript summaries into a single coherent summary.
            
            {metadata_str}
            
            Here are the summaries from different segments of the transcript:
            
            {formatted_summaries}
            
            Your summary must accurately reflect ONLY the content in these summaries.
            
            Format your response with these exact headings:
            
            # Transcript Summary
            
            ## Overview
            [2-3 sentence high-level description of what the transcript contains]
            
            ## Main Topics
            [Bullet list of key themes and topics discussed]
            
            ## Key Points
            [Bullet list of important details and takeaways]
            
            ## Notable Quotes
            [Direct quotes from the transcript that were mentioned in the summaries]
            """
        
        # Make direct API call
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.executor._get_api_key()}"
            }
            
            # Add organization ID if present
            if hasattr(self.executor.config, 'OPENAI_ORG_ID') and self.executor.config.OPENAI_ORG_ID:
                headers["OpenAI-Organization"] = self.executor.config.OPENAI_ORG_ID
            
            data = {
                "model": self.executor.model,
                "messages": [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.2,  # Low temperature for consistency
                "max_tokens": self.executor.config.MAX_TOKENS
            }
            
            # For testing, we can use mock responses
            if not self.executor._get_api_key() and self.executor.provider == "openai":
                logger.warning("Using mock response for aggregation (no API key provided)")
                return "# Transcript Summary\n\n## Overview\nThis is a mock summary for testing without an API key.\n\n## Main Topics\n- Topic 1\n- Topic 2\n\n## Key Points\n- Key point 1\n- Key point 2\n\n## Notable Quotes\n- 'This is a mock quote.'"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=self.executor.config.REQUEST_TIMEOUT
                ) as response:
                    result = await response.json()
                    
                    if response.status != 200:
                        error_message = result.get("error", {}).get("message", "Unknown error")
                        logger.error(f"API Error: {error_message}")
                        return f"Error generating summary: {error_message}"
                    
                    # Extract the summary content
                    if result.get("choices") and result["choices"][0].get("message"):
                        summary = result["choices"][0]["message"]["content"]
                        
                        # Update token usage tracking
                        if result.get("usage"):
                            usage = result["usage"]
                            logger.info(f"Aggregation tokens - Prompt: {usage.get('prompt_tokens', 0)}, Completion: {usage.get('completion_tokens', 0)}, Total: {usage.get('total_tokens', 0)}")
                            
                            # Update executor token tracking
                            if hasattr(self.executor, 'total_tokens_used'):
                                self.executor.total_tokens_used += usage.get('total_tokens', 0)
                            
                            # Update cost tracking (estimated based on model)
                            if hasattr(self.executor, 'total_cost'):
                                cost_per_1k = 0.03  # Default estimate for gpt-4o-mini
                                cost = (usage.get('total_tokens', 0) / 1000) * cost_per_1k
                                self.executor.total_cost += cost
                        
                        return summary
                    else:
                        logger.error("No content in API response")
                        return "Error: No content in API response"
        except Exception as e:
            logger.error(f"Error in direct API call: {e}")
            return f"Error generating summary: {str(e)}"
    
    async def _hierarchical_aggregation(
        self,
        summaries: List[str],
        prompt_template: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Perform hierarchical aggregation for many summaries.
        
        This approach:
        1. Divides summaries into batches
        2. Summarizes each batch
        3. Combines batch summaries into a final summary
        
        Args:
            summaries: List of summaries to aggregate
            prompt_template: Custom prompt template (uses default if None)
            metadata: Additional metadata to include in the prompt
            
        Returns:
            Final aggregated summary
        """
        logger.info(f"Performing hierarchical aggregation of {len(summaries)} summaries")
        
        # Use provided prompt template or default
        if not prompt_template:
            prompt_template = self._get_default_aggregation_prompt()
            
        # Calculate batch size based on token limit
        batch_size = self._calculate_batch_size(summaries)
        logger.info(f"Using batch size of {batch_size} summaries per batch")
        
        # Create batches
        batches = [summaries[i:i+batch_size] for i in range(0, len(summaries), batch_size)]
        logger.info(f"Created {len(batches)} batches")
        
        # First level: Process each batch to get intermediate summaries
        intermediate_tasks = []
        for i, batch in enumerate(batches):
            # Create a batch-specific prompt
            batch_prompt = self._get_batch_prompt()
            batch_metadata = metadata.copy() if metadata else {}
            batch_metadata.update({
                "Batch": f"{i+1}/{len(batches)}",
                "Position": f"Covering approximately {100 * i / len(batches):.0f}% - {100 * (i + 1) / len(batches):.0f}% of the transcript"
            })
            
            # Schedule the batch processing
            task = asyncio.create_task(
                self._single_aggregation(batch, batch_prompt, batch_metadata)
            )
            intermediate_tasks.append(task)
        
        # Wait for all batches to be processed
        intermediate_summaries = await asyncio.gather(*intermediate_tasks)
        logger.info(f"Completed {len(intermediate_summaries)} intermediate summaries")
        
        # Second level: Combine intermediate summaries
        if len(intermediate_summaries) == 1:
            # Only one batch, no need for second level
            return intermediate_summaries[0]
        else:
            # Multiple batches, combine intermediate summaries
            final_prompt = self._get_final_prompt()
            final_summary = await self._single_aggregation(
                intermediate_summaries, final_prompt, metadata
            )
            return final_summary
    
    def _calculate_batch_size(self, summaries: List[str]) -> int:
        """
        Calculate optimal batch size based on token limits.
        
        Args:
            summaries: List of summaries to process
            
        Returns:
            Number of summaries per batch
        """
        if not summaries:
            return 1
            
        # Get average tokens per summary
        avg_tokens = sum(len(self.tokenizer.encode(s)) for s in summaries) / len(summaries)
        
        # Reserve tokens for prompt, metadata, formatting
        reserved_tokens = 1000
        
        # Calculate how many summaries we can fit in a batch
        summaries_per_batch = max(1, int((self.max_tokens_per_batch - reserved_tokens) / avg_tokens))
        
        # Ensure reasonable batch sizes
        return min(summaries_per_batch, 10)  # Max 10 summaries per batch
    
    def _total_tokens(self, texts: List[str]) -> int:
        """
        Calculate total tokens for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Total token count
        """
        return sum(len(self.tokenizer.encode(text)) for text in texts)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS format."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def _get_default_aggregation_prompt(self) -> str:
        """Get the default prompt for aggregation."""
        return """
        I need you to combine these transcript summaries into a cohesive final summary.
        
        {metadata}
        
        Here are {num_summaries} summaries from transcript segments:
        
        {summaries}
        
        Format your response with these specific headings:
        
        # Transcript Summary
        
        ## Overview
        [2-3 sentence high-level overview]
        
        ## Main Topics
        [Bullet list of key themes discussed]
        
        ## Key Points
        [Important details and takeaways]
        
        ## Notable Quotes
        [Direct quotes from the transcript]
        """
    
    def _get_batch_prompt(self) -> str:
        """Get the prompt for batch-level summarization."""
        return """
        Create an intermediate summary for this section of a transcript.
        
        {metadata}
        
        Here are {num_summaries} summaries from consecutive segments:
        
        {summaries}
        
        IMPORTANT INSTRUCTIONS:
        1. DO NOT introduce yourself or add any greeting
        2. DO NOT ask how you can help
        3. ONLY provide the summary in the format below
        4. START your response with "# Intermediate Summary"
        
        Your intermediate summary must:
        - Combine key information from these segment summaries
        - Preserve important details, quotes, and themes
        - Maintain chronological order and context
        - Be thorough rather than brief at this stage
        
        Format your summary as:
        # Intermediate Summary
        
        [Your detailed summary content here]
        """
    
    def _get_final_prompt(self) -> str:
        """Get the prompt for final-level summarization."""
        return """
        Create the FINAL SUMMARY of a complete transcript by combining these section summaries.
        
        {metadata}
        
        Here are {num_summaries} section summaries covering the entire transcript:
        
        {summaries}
        
        IMPORTANT INSTRUCTIONS:
        1. DO NOT introduce yourself or add any greeting
        2. DO NOT ask how you can help
        3. ONLY provide the summary in the format below
        4. START your response with "# Transcript Summary"
        
        Your final summary must:
        - Synthesize key information from all sections
        - Present a cohesive narrative of the entire transcript
        - Highlight important themes, insights, and quotes
        - Organize information in a logical structure
        
        Format your summary with these headings:
        # Transcript Summary
        
        ## Overview
        [2-3 sentence high-level description]
        
        ## Main Topics
        [Bullet list of key themes discussed]
        
        ## Important Points
        [Key details and takeaways]
        
        ## Notable Quotes
        [Direct quotes from the transcript]
        """

# Synchronous wrapper function for easier usage
def aggregate_results(
    processed_chunks: List[Dict[Any, Any]],
    prompt_template: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    hierarchical: bool = True,
    executor: Optional[LLMExecutor] = None
) -> str:
    """
    Aggregate multiple chunk summaries into a coherent final summary.
    This is a synchronous wrapper around the async function.
    
    Args:
        processed_chunks: List of processed chunks with summaries
        prompt_template: Custom prompt template for aggregation
        metadata: Additional metadata to include in the prompt
        hierarchical: Whether to use hierarchical summarization
        executor: LLM executor to use (creates a new one if not provided)
        
    Returns:
        Final aggregated summary
    """
    aggregator = ResultAggregator(executor=executor, hierarchical=hierarchical)
    result = asyncio.run(aggregator.aggregate(processed_chunks, prompt_template, metadata))
    return result["summary"]

# Example usage (when run directly)
if __name__ == "__main__":
    import json
    from preprocessor import preprocess_transcript
    from big_chunkeroosky import BigChunkeroosky
    from llm_executor import LLMExecutor
    
    async def test_aggregator():
        # Load example transcript
        with open('transcript-example.json', 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Preprocess the transcript (just use part of it to save time)
        limited_segments = transcript_data['segments'][:300]
        processed_segments = preprocess_transcript(limited_segments)
        
        # Create chunks
        chunker = BigChunkeroosky(max_tokens_per_chunk=3000)
        chunks = chunker.chunk_transcript(processed_segments)
        chunks = chunker.postprocess_chunks(chunks)
        
        # Limit to just a few chunks for testing
        test_chunks = chunks[:3]
        
        # Process chunks with LLM to get summaries
        prompt_template = """
        Please summarize the following transcript segment:
        
        {transcript}
        
        Provide a concise summary focusing on key points and main ideas.
        """
        
        executor = LLMExecutor()
        processed_chunks = await executor.process_chunks(test_chunks, prompt_template)
        
        # Create the aggregator
        aggregator = ResultAggregator(executor=executor)
        
        # Metadata about the transcript
        metadata = {
            "Title": "Transcript Example",
            "Speakers": "SPEAKER_00",
            "Total Duration": chunker._format_time(test_chunks[-1]['end_time']),
            "Number of Chunks": len(test_chunks)
        }
        
        # Aggregate the summaries
        result = await aggregator.aggregate(processed_chunks, metadata=metadata)
        
        # Print the result
        print("\n==== RESULT AGGREGATOR TEST ====\n")
        print(f"Aggregated {result['chunks_aggregated']} chunks in {result['processing_time']:.2f} seconds")
        print("\nFinal Summary:")
        print(result['summary'])
    
    # Run the test
    asyncio.run(test_aggregator())

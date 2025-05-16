#!/usr/bin/env python3
"""
Transcript Summarizer - Main Script

This script ties together all components of the transcript summarization system:
1. Preprocessor - Cleans and formats transcript data
2. Big Chunkeroosky - Splits transcript into LLM-friendly chunks
3. LLM Executor - Processes chunks in parallel with LLMs
4. Result Aggregator - Combines chunk summaries into a final summary

Usage:
    python main.py --input transcript.json --output summary.txt [options]
"""

import argparse
import asyncio
import datetime
import json
import logging
import os
import sys
import time
from dotenv import load_dotenv
from pathlib import Path
from typing import Dict, Any, Optional, List

from preprocessor import preprocess_transcript
from big_chunkeroosky import BigChunkeroosky
from llm_executor import LLMExecutor
from result_aggregator import ResultAggregator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TranscriptSummarizer")

# Load environment variables from .env file
load_dotenv()

class TranscriptSummarizer:
    """
    Main class for the transcript summarization workflow.
    Coordinates preprocessing, chunking, LLM processing, and aggregation.
    """
    
    def __init__(
        self,
        provider: str = "openai",
        model: str = None,
        max_tokens_per_chunk: int = 4000,
        max_concurrent_requests: int = 5,
        hierarchical_aggregation: bool = True
    ):
        """
        Initialize the transcript summarizer.
        
        Args:
            provider: LLM provider to use ('openai' or 'anthropic')
            model: Model name to use (if None, uses default from .env)
            max_tokens_per_chunk: Maximum tokens per chunk for chunking
            max_concurrent_requests: Maximum concurrent API requests
            hierarchical_aggregation: Whether to use hierarchical aggregation
        """
        self.provider = provider
        self.model = model
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.max_concurrent_requests = max_concurrent_requests
        self.hierarchical_aggregation = hierarchical_aggregation
        
        # Initialize components when needed
        self.executor = None
        self.chunker = None
        self.aggregator = None
        
        logger.info(f"Transcript Summarizer initialized with provider={provider}")
    
    async def summarize(
        self,
        transcript_data: Dict[str, Any],
        merge_same_speaker: bool = True,
        max_segment_duration: int = 120,
        prompt_template: Optional[str] = None,
        prompt_file: Optional[str] = None,
        system_prompt: Optional[str] = None,
        system_prompt_file: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        limit_segments: Optional[int] = None,
        save_intermediate_chunks: Optional[str] = None,
        aggregator_prompt_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a transcript through the full pipeline.
        
        Args:
            transcript_data: Raw transcript data
            merge_same_speaker: Whether to merge segments from same speaker
            max_segment_duration: Maximum segment duration in seconds
            prompt_template: Custom prompt template for the LLM
            metadata: Additional metadata to include in final summary
            limit_segments: Optional limit on number of segments to process
            
        Returns:
            Dictionary containing the final summary and processing statistics
        """
        start_time = time.time()
        
        # Initialize components if not already initialized
        if self.executor is None:
            self.executor = LLMExecutor(
                provider=self.provider, 
                model=self.model,
                max_concurrent_requests=self.max_concurrent_requests
            )
        
        if self.chunker is None:
            self.chunker = BigChunkeroosky(max_tokens_per_chunk=self.max_tokens_per_chunk)
        
        if self.aggregator is None:
            self.aggregator = ResultAggregator(
                executor=self.executor,
                hierarchical=self.hierarchical_aggregation
            )
        
        # Step 1: Extract segments from transcript data
        segments = transcript_data.get('segments', [])
        if limit_segments:
            logger.info(f"Limiting to first {limit_segments} segments")
            segments = segments[:limit_segments]
        
        logger.info(f"Starting summarization of transcript with {len(segments)} segments")
        
        # Step 2: Preprocess transcript
        logger.info("Preprocessing transcript...")
        processed_segments = preprocess_transcript(
            segments,
            merge_same_speaker=merge_same_speaker,
            max_segment_duration=max_segment_duration
        )
        logger.info(f"Preprocessed {len(segments)} segments into {len(processed_segments)} segments")
        
        # Step 3: Create chunks
        logger.info("Splitting transcript into chunks...")
        chunks = self.chunker.chunk_transcript(processed_segments)
        chunks = self.chunker.postprocess_chunks(chunks)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 4: Process chunks with LLM
        logger.info(f"Processing chunks with {self.provider} LLM...")
        
        # Get prompt template - prioritize explicit template over file over default
        if not prompt_template:
            prompt_template = self._get_prompt_template(prompt_file)
            
        # Get system prompt if provided
        system_prompt_content = None
        if system_prompt:
            system_prompt_content = system_prompt
            logger.info("Using provided system prompt")
        elif system_prompt_file:
            system_prompt_content = self._get_system_prompt(system_prompt_file)
            if system_prompt_content:
                logger.info("Using system prompt from file")
        
        processed_chunks = await self.executor.process_chunks(
            chunks, 
            prompt_template, 
            system_prompt=system_prompt_content
        )
        
        logger.info(f"Generated summaries for {len(processed_chunks)} chunks")
        
        # Save intermediate chunks if requested
        if save_intermediate_chunks:
            try:
                chunk_output_data = {
                    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "chunks": []
                }
                
                for chunk in processed_chunks:
                    # Create a clean version with just essential info
                    clean_chunk = {
                        "chunk_index": chunk.get("chunk_index", -1),
                        "start_time": chunk.get("start_time", ""),
                        "end_time": chunk.get("end_time", ""),
                        "summary": chunk.get("summary", ""),
                        "tokens_used": chunk.get("tokens_used", 0)
                    }
                    chunk_output_data["chunks"].append(clean_chunk)
                
                with open(save_intermediate_chunks, "w", encoding="utf-8") as f:
                    json.dump(chunk_output_data, f, indent=2)
                
                logger.info(f"Saved {len(processed_chunks)} intermediate chunk summaries to {save_intermediate_chunks}")
            except Exception as e:
                logger.error(f"Failed to save intermediate chunks to {save_intermediate_chunks}: {e}")
        
        # Step 5: Aggregate results
        logger.info("Aggregating chunk summaries...")
        # Extract summaries from processed chunks
        summaries = [chunk["summary"] for chunk in processed_chunks]
        
        # Load custom aggregator prompt if provided
        aggregator_prompt = None
        if aggregator_prompt_file:
            try:
                with open(aggregator_prompt_file, "r", encoding="utf-8") as f:
                    aggregator_prompt = f.read().strip()
                    logger.info(f"Using custom aggregator prompt from {aggregator_prompt_file}")
            except Exception as e:
                logger.error(f"Failed to load aggregator prompt from {aggregator_prompt_file}: {e}")
                logger.info("Using default aggregator prompt")
        
        # Add file metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Try to get file info from transcript_data if available
        file_info = "Unknown"
        if hasattr(transcript_data, "get") and transcript_data.get("file_info"):
            file_info = transcript_data.get("file_info")
        
        metadata.update({
            "File": file_info,
            "Total Duration": self._format_duration(chunks[-1]["end_time"] if chunks else 0)
        })
        
        # Aggregate results using custom prompt if provided
        result = await self.aggregator.aggregate(processed_chunks, 
                                              prompt_template=aggregator_prompt,
                                              metadata=metadata)
        
        # Calculate processing stats
        elapsed_time = time.time() - start_time
        tokens_used = self.executor.total_tokens_used
        cost = self.executor.total_cost
        
        logger.info(f"Summarization completed in {elapsed_time:.2f} seconds")
        logger.info(f"Total tokens used: {tokens_used}")
        logger.info(f"Estimated cost: ${cost:.4f}")
        
        # Return results with metadata
        return {
            "summary": result["summary"],
            "processing_time": elapsed_time,
            "tokens_used": tokens_used,
            "cost": cost,
            "segments": len(segments),
            "chunks": len(chunks),
            "provider": self.provider,
            "model": self.executor.model
        }
    
    def _get_prompt_template(self, prompt_file: Optional[str] = None) -> str:
        """Get the prompt template, either from a file or the default.
        
        Args:
            prompt_file: Optional path to a prompt template file
            
        Returns:
            Prompt template string
        """
        if prompt_file:
            try:
                with open(prompt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    logger.info(f"Loaded custom prompt template from {prompt_file}")
                    
                    # Verify the prompt has the {transcript} placeholder
                    if "{transcript}" not in content:
                        logger.warning(f"Prompt template in {prompt_file} does not contain {{transcript}} placeholder. Adding it.")
                        content += "\n\n{transcript}"
                    
                    return content
            except Exception as e:
                logger.error(f"Error loading prompt template from {prompt_file}: {e}")
                logger.info("Falling back to default prompt template")
        
        # Return the default prompt template
        return """
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
        
    def _get_system_prompt(self, system_prompt_file: Optional[str] = None) -> Optional[str]:
        """Get the system prompt, either from a file or None if not provided.
        
        Args:
            system_prompt_file: Optional path to a system prompt file
            
        Returns:
            System prompt string or None
        """
        if system_prompt_file:
            try:
                with open(system_prompt_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    logger.info(f"Loaded system prompt from {system_prompt_file}")
                    return content
            except Exception as e:
                logger.error(f"Error loading system prompt from {system_prompt_file}: {e}")
                logger.info("Not using a system prompt due to error")
        
        # Return None if no system prompt file provided
        return None
    
    def _format_duration(self, seconds: float) -> str:
        """Format seconds into hours:minutes:seconds."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        else:
            return f"{minutes}m {seconds}s"

async def async_main(args):
    """Async entry point for the script."""
    # Load transcript
    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        logger.info(f"Loaded transcript from {args.input}")
    except Exception as e:
        logger.error(f"Failed to load transcript: {e}")
        return 1
    
    # Create summarizer
    summarizer = TranscriptSummarizer(
        provider=args.provider,
        model=args.model,
        max_tokens_per_chunk=args.max_tokens_per_chunk,
        max_concurrent_requests=args.max_concurrent_requests,
        hierarchical_aggregation=not args.no_hierarchical
    )
    
    # Process transcript
    result = await summarizer.summarize(
        transcript_data,
        merge_same_speaker=not args.no_merge,
        max_segment_duration=args.max_segment_duration,
        prompt_file=args.prompt_file,
        system_prompt_file=args.system_prompt_file,
        limit_segments=args.limit_segments,
        save_intermediate_chunks=args.save_chunks,
        aggregator_prompt_file=args.aggregator_prompt_file
    )
    
    # Save or print results
    summary = result["summary"]
    
    # Print summary to console
    if not args.quiet:
        print("\n" + "=" * 80)
        print("TRANSCRIPT SUMMARY")
        print("=" * 80 + "\n")
        print(summary)
        print("\n" + "=" * 80)
        print(f"Processing time: {result['processing_time']:.2f} seconds")
        print(f"Tokens used: {result['tokens_used']}")
        print(f"Estimated cost: ${result['cost']:.4f}")
        print("=" * 80 + "\n")
    
    # Save to file if output path is provided
    if args.output:
        try:
            # Create output directory if it doesn't exist
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save summary to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(summary)
            
            # Save detailed report if requested
            if args.report:
                report_path = output_path.with_suffix('.report.json')
                with open(report_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)
                logger.info(f"Saved detailed report to {report_path}")
            
            logger.info(f"Saved summary to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save output: {e}")
            return 1
    
    return 0

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Summarize a transcript using multiple LLMs in parallel"
    )
    
    parser.add_argument("--input", "-i", 
                        required=True,
                        help="Path to the input transcript JSON file")
    
    parser.add_argument("--output", "-o", 
                        help="Path to the output summary file (default: print to console)")
    
    parser.add_argument("--provider", 
                        choices=["openai", "anthropic"],
                        default="openai",
                        help="LLM provider to use (default: openai)")
    
    parser.add_argument("--model",
                        help="Model to use (default: from .env file)")
    
    parser.add_argument("--max-tokens-per-chunk", 
                        type=int,
                        default=4000,
                        help="Maximum tokens per chunk (default: 4000)")
    
    parser.add_argument("--max-concurrent-requests", 
                        type=int,
                        default=5,
                        help="Maximum concurrent API requests (default: 5)")
    
    parser.add_argument("--max-segment-duration", 
                        type=int,
                        default=120,
                        help="Maximum segment duration in seconds (default: 120)")
    
    parser.add_argument("--no-merge", 
                        action="store_true",
                        help="Disable merging of consecutive segments from the same speaker")
    
    parser.add_argument("--no-hierarchical", 
                        action="store_true",
                        help="Disable hierarchical aggregation for large transcripts")
    
    parser.add_argument("--limit-segments", 
                        type=int,
                        help="Limit the number of segments to process (for testing)")
    
    parser.add_argument("--report", 
                        action="store_true",
                        help="Generate a detailed report JSON file")
    
    parser.add_argument("--prompt-file",
                        help="Path to a file containing a custom prompt template")
                        
    parser.add_argument("--system-prompt-file",
                        help="Path to a file containing a system prompt for the LLM")
                        
    parser.add_argument("--save-chunks",
                        help="Path to save intermediate chunk summaries before aggregation")
                        
    parser.add_argument("--aggregator-prompt-file",
                        help="Path to a file containing a custom prompt template for the result aggregator")
                        
    parser.add_argument("--quiet", "-q", 
                        action="store_true",
                        help="Suppress console output")
    
    args = parser.parse_args()
    
    # Run the async main
    return asyncio.run(async_main(args))

if __name__ == "__main__":
    sys.exit(main())

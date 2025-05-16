"""
LLM Executor Module for Transcript Summarization.

This module handles parallel processing of transcript chunks using multiple LLM instances.
It includes functions for:
- Configuring LLM API access
- Processing chunks in parallel with rate limiting
- Handling errors and retries
- Tracking processing status
"""

import os
import asyncio
import logging
import time
import json
from typing import List, Dict, Any, Optional, Union, Callable
import aiohttp
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("LLMExecutor")

# Load environment variables
load_dotenv()

class LLMConfig:
    """Configuration for LLM API access"""
    
    # OpenAI configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_ORG_ID = os.getenv("OPENAI_ORG_ID", "")
    OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    
    # Anthropic configuration
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229")
    
    # Other configuration
    MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "60"))
    RETRY_ATTEMPTS = int(os.getenv("RETRY_ATTEMPTS", "3"))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", "5"))
    
    # Provider selection
    DEFAULT_PROVIDER = os.getenv("DEFAULT_PROVIDER", "openai")  # openai, anthropic

class LLMExecutor:
    """
    LLM Executor for processing transcript chunks in parallel.
    """
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        max_concurrent_requests: Optional[int] = None
    ):
        """
        Initialize the LLM Executor.
        
        Args:
            config: LLM configuration (uses default if not provided)
            provider: LLM provider to use (overrides config)
            model: LLM model to use (overrides config)
            max_concurrent_requests: Maximum concurrent requests (overrides config)
        """
        self.config = config or LLMConfig()
        self.provider = provider or self.config.DEFAULT_PROVIDER
        self.model = model or (
            self.config.OPENAI_MODEL if self.provider == "openai" 
            else self.config.ANTHROPIC_MODEL
        )
        self.max_concurrent_requests = max_concurrent_requests or self.config.MAX_CONCURRENT_REQUESTS
        
        # Validate configuration
        self._validate_config()
        
        # Initialize tracking
        self.total_tokens_used = 0
        self.total_cost = 0.0
        self.total_requests = 0
        self.failed_requests = 0
        
        logger.info(f"LLM Executor initialized with provider: {self.provider}, model: {self.model}")
    
    def _validate_config(self) -> None:
        """Validate the configuration and provide helpful error messages"""
        if self.provider == "openai" and not self.config.OPENAI_API_KEY:
            logger.warning("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        elif self.provider == "anthropic" and not self.config.ANTHROPIC_API_KEY:
            logger.warning("Anthropic API key not found. Set the ANTHROPIC_API_KEY environment variable.")
    
    def _get_api_key(self) -> str:
        """Get the appropriate API key based on the selected provider"""
        if self.provider == "openai":
            return self.config.OPENAI_API_KEY
        elif self.provider == "anthropic":
            return self.config.ANTHROPIC_API_KEY
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def process_chunks(
        self,
        chunks: List[Dict[Any, Any]],
        prompt_template: str,
        summary_type: str = "summary"
    ) -> List[Dict[Any, Any]]:
        """
        Process multiple chunks in parallel with rate limiting.
        
        Args:
            chunks: List of transcript chunks to process
            prompt_template: Template for the LLM prompt
            summary_type: Type of summary to generate
            
        Returns:
            List of processed chunks with summaries
        """
        # Start timing
        start_time = time.time()
        logger.info(f"Starting parallel processing of {len(chunks)} chunks")
        
        # Create a semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # Process each chunk concurrently
        tasks = [
            self.process_chunk(chunk, prompt_template, summary_type, semaphore, index)
            for index, chunk in enumerate(chunks)
        ]
        
        # Wait for all tasks to complete
        processed_chunks = await asyncio.gather(*tasks)
        
        # End timing
        elapsed_time = time.time() - start_time
        logger.info(f"Completed processing {len(chunks)} chunks in {elapsed_time:.2f} seconds")
        logger.info(f"Total tokens used: {self.total_tokens_used}")
        logger.info(f"Estimated cost: ${self.total_cost:.4f}")
        logger.info(f"Failed requests: {self.failed_requests}/{self.total_requests}")
        
        # Sort chunks by index to maintain original order
        processed_chunks.sort(key=lambda x: x['chunk_index'])
        
        return processed_chunks
    
    async def process_chunk(
        self,
        chunk: Dict[Any, Any],
        prompt_template: str,
        summary_type: str,
        semaphore: asyncio.Semaphore,
        index: int
    ) -> Dict[Any, Any]:
        """
        Process a single chunk with rate limiting.
        
        Args:
            chunk: Transcript chunk to process
            prompt_template: Template for the LLM prompt
            summary_type: Type of summary to generate
            semaphore: Semaphore for rate limiting
            index: Processing index for tracking
            
        Returns:
            Processed chunk with summary
        """
        # Make a copy to avoid modifying the original
        result_chunk = chunk.copy()
        result_chunk['processing_index'] = index
        
        # Fill the prompt template
        prompt = prompt_template.format(
            transcript=chunk['text_with_context'],
            summary_type=summary_type
        )
        
        # Extract system prompt if provided in chunk
        system_prompt = chunk.get('system_prompt')
        
        # Call the LLM API with rate limiting and retries
        async with semaphore:
            self.total_requests += 1
            for attempt in range(1, self.config.RETRY_ATTEMPTS + 1):
                try:
                    logger.debug(f"Processing chunk {index+1}/{chunk['total_chunks']} (attempt {attempt})")
                    
                    response = await self._call_llm_api(prompt, system_prompt)
                    
                    # Store the response in the result
                    result_chunk['summary'] = response['content']
                    result_chunk['tokens_used'] = response.get('tokens_used', 0)
                    result_chunk['cost'] = response.get('cost', 0)
                    
                    # Update tracking
                    self.total_tokens_used += result_chunk['tokens_used']
                    self.total_cost += result_chunk['cost']
                    
                    logger.debug(f"Successfully processed chunk {index+1}/{chunk['total_chunks']}")
                    break
                
                except Exception as e:
                    logger.warning(f"Error processing chunk {index+1}/{chunk['total_chunks']} (attempt {attempt}): {e}")
                    
                    # On final attempt, mark as failed and continue
                    if attempt == self.config.RETRY_ATTEMPTS:
                        logger.error(f"Failed to process chunk {index+1}/{chunk['total_chunks']} after {attempt} attempts")
                        result_chunk['summary'] = f"[Error processing chunk: {str(e)}]"
                        result_chunk['error'] = str(e)
                        self.failed_requests += 1
                        break
                    
                    # Wait before retrying
                    await asyncio.sleep(self.config.RETRY_DELAY)
        
        return result_chunk
    
    async def _call_llm_api(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Call the LLM API based on the configured provider.
        
        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt to control model behavior
            
        Returns:
            Dictionary with response content and metadata
        """
        if self.provider == "openai":
            return await self._call_openai_api(prompt, system_prompt)
        elif self.provider == "anthropic":
            return await self._call_anthropic_api(prompt, system_prompt)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    async def _call_openai_api(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Call the OpenAI API.
        
        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt to control model behavior
            
        Returns:
            Dictionary with response content and metadata
        """
        if not self.config.OPENAI_API_KEY:
            logger.warning("Using mock response for OpenAI API call (no API key provided)")
            return self._get_mock_response("openai")
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.OPENAI_API_KEY}"
        }
        
        if self.config.OPENAI_ORG_ID:
            headers["OpenAI-Organization"] = self.config.OPENAI_ORG_ID
        
        # Prepare messages list with optional system message
        messages = []
        
        # Add system message if provided
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.TEMPERATURE,
            "max_tokens": self.config.MAX_TOKENS
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=self.config.REQUEST_TIMEOUT
            ) as response:
                result = await response.json()
                
                if response.status != 200:
                    error_message = result.get("error", {}).get("message", "Unknown error")
                    raise Exception(f"OpenAI API Error: {error_message}")
                
                # Calculate tokens used and estimated cost
                prompt_tokens = result['usage']['prompt_tokens']
                completion_tokens = result['usage']['completion_tokens']
                total_tokens = result['usage']['total_tokens']
                
                # Approximate cost calculation 
                # (update pricing as needed: https://openai.com/pricing)
                if "gpt-4" in self.model:
                    prompt_cost = prompt_tokens * 0.00003  # $0.03 per 1K tokens
                    completion_cost = completion_tokens * 0.00006  # $0.06 per 1K tokens
                else:  # GPT-3.5
                    prompt_cost = prompt_tokens * 0.000001  # $0.001 per 1K tokens
                    completion_cost = completion_tokens * 0.000002  # $0.002 per 1K tokens
                
                total_cost = prompt_cost + completion_cost
                
                return {
                    "content": result["choices"][0]["message"]["content"],
                    "tokens_used": total_tokens,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "cost": total_cost,
                    "model": self.model
                }
    
    async def _call_anthropic_api(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Call the Anthropic API.
        
        Args:
            prompt: The prompt to send
            system_prompt: Optional system prompt to control model behavior
            
        Returns:
            Dictionary with response content and metadata
        """
        if not self.config.ANTHROPIC_API_KEY:
            logger.warning("Using mock response for Anthropic API call (no API key provided)")
            return self._get_mock_response("anthropic")
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.config.ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01"  # Anthropic API version
        }
        
        # For newer Claude models (claude-3), we use the messages API
        if self.model.startswith("claude-3"):
            messages = []
            
            # Add system message if provided
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add user message
            messages.append({"role": "user", "content": prompt})
            
            data = {
                "model": self.model,
                "messages": messages,
                "temperature": self.config.TEMPERATURE,
                "max_tokens": self.config.MAX_TOKENS
            }
        else:
            # Legacy format for older Claude models
            system_text = f"{system_prompt}\n\n" if system_prompt else ""
            data = {
                "model": self.model,
                "prompt": f"{system_text}\n\nHuman: {prompt}\n\nAssistant:",
                "temperature": self.config.TEMPERATURE,
                "max_tokens_to_sample": self.config.MAX_TOKENS
            }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers=headers,
                json=data,
                timeout=self.config.REQUEST_TIMEOUT
            ) as response:
                result = await response.json()
                
                if response.status != 200:
                    error_message = result.get("error", {}).get("message", "Unknown error")
                    raise Exception(f"Anthropic API Error: {error_message}")
                
                # Anthropic doesn't provide token counts directly, estimate based on characters
                # This is a rough approximation, adjust as needed
                char_count = len(prompt) + len(result["content"][0]["text"])
                estimated_tokens = char_count // 4  # Rough estimate: ~4 chars per token
                
                # Estimate cost (update pricing as needed)
                # Claude 3 Sonnet: $3.00 per million input tokens, $15.00 per million output tokens
                input_tokens = len(prompt) // 4
                output_tokens = len(result["content"][0]["text"]) // 4
                input_cost = input_tokens * 0.000003  # $0.003 per 1K tokens
                output_cost = output_tokens * 0.000015  # $0.015 per 1K tokens
                total_cost = input_cost + output_cost
                
                return {
                    "content": result["content"][0]["text"],
                    "tokens_used": estimated_tokens,
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "cost": total_cost,
                    "model": self.model
                }
    
    def _get_mock_response(self, provider: str) -> Dict[str, Any]:
        """
        Generate a mock response for development/testing without API keys.
        
        Args:
            provider: The provider to mock
            
        Returns:
            Mock response dictionary
        """
        model = self.model
        mock_content = f"[Mock {provider.capitalize()} Response using {model}]\n\nThis is a simulated summary generated because no API key was provided. In a real scenario, this would contain a summary of the transcript chunk."
        
        return {
            "content": mock_content,
            "tokens_used": 100,
            "prompt_tokens": 75,
            "completion_tokens": 25,
            "cost": 0.0,
            "model": model,
            "is_mock": True
        }

# Helper function for easier usage
async def process_chunks_parallel(
    chunks: List[Dict[Any, Any]],
    prompt_template: str,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    summary_type: str = "summary"
) -> List[Dict[Any, Any]]:
    """
    Process multiple chunks in parallel using the LLM API.
    Helper function that creates an executor and processes chunks.
    
    Args:
        chunks: List of transcript chunks to process
        prompt_template: Template for the LLM prompt
        provider: LLM provider to use (default from config)
        model: LLM model to use (default from config)
        summary_type: Type of summary to generate
        
    Returns:
        List of processed chunks with summaries
    """
    executor = LLMExecutor(provider=provider, model=model)
    return await executor.process_chunks(chunks, prompt_template, summary_type)

# Example usage (when run directly)
if __name__ == "__main__":
    import json
    from preprocessor import preprocess_transcript
    from big_chunkeroosky import BigChunkeroosky
    
    async def test_llm_executor():
        # Load example transcript
        with open('transcript-example.json', 'r', encoding='utf-8') as f:
            transcript_data = json.load(f)
        
        # Preprocess the transcript
        processed_segments = preprocess_transcript(transcript_data['segments'])
        
        # Create and apply the chunker
        chunker = BigChunkeroosky(max_tokens_per_chunk=4000)
        chunks = chunker.chunk_transcript(processed_segments)
        chunks = chunker.postprocess_chunks(chunks)
        
        # Only process the first 2 chunks for testing
        test_chunks = chunks[:2]
        
        # Create a simple prompt template
        prompt_template = """
        Please summarize the following transcript segment:
        
        {transcript}
        
        Provide a concise {summary_type} that captures the main points, ideas, and flow of the conversation.
        """
        
        # Process chunks
        executor = LLMExecutor()
        processed_chunks = await executor.process_chunks(test_chunks, prompt_template)
        
        # Print results
        print(f"\n==== LLM EXECUTOR TEST RESULTS ====\n")
        print(f"Processed {len(processed_chunks)} chunks using {executor.provider} with model {executor.model}")
        print(f"Total tokens used: {executor.total_tokens_used}")
        print(f"Estimated cost: ${executor.total_cost:.4f}")
        
        # Show sample summaries
        for i, chunk in enumerate(processed_chunks):
            print(f"\nChunk {i+1} Summary:")
            print(f"Time Range: {chunker._format_time(chunk['start_time'])} - {chunker._format_time(chunk['end_time'])}")
            print(f"Tokens Used: {chunk.get('tokens_used', 'N/A')}")
            print(f"Summary: {chunk.get('summary', 'No summary generated')[:200]}...")
            print("-" * 50)
    
    # Run the test
    asyncio.run(test_llm_executor())

# Transcript Summarizer

A distributed system for summarizing long transcripts using multiple LLMs in parallel. This project handles large transcripts by breaking them into manageable chunks while preserving context and timing information, then processes these chunks with LLMs to generate comprehensive summaries.

![Status](https://img.shields.io/badge/status-in%20development-yellow)
![Python Version](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Architecture](#architecture)
- [Modules](#modules)
- [Data Flow](#data-flow)
- [Usage](#usage)
- [Configuration](#configuration)
- [Extensibility](#extensibility)

## Problem Statement

Summarizing very long transcripts (like interviews, speeches, or meetings) is challenging because:

1. Large language models (LLMs) have token limits that prevent processing entire transcripts at once
2. Simply splitting the text arbitrarily can lose important context and coherence
3. Timestamps and speaker information need to be preserved for accurate summaries

This system addresses these challenges by intelligently preprocessing transcripts, chunking them based on linguistic and semantic boundaries, processing chunks in parallel with multiple LLMs, and then aggregating the results into a cohesive summary.

## Architecture

The system follows a modular pipeline architecture:

```
Input Transcript → Preprocessor → Chunker → LLM Executor (parallel) → Result Aggregator → Final Summary
                                   ↑               ↑
                            Sentence Detection   Prompt Manager
```

### Core Components

1. ✅ **Preprocessor**: Cleans and prepares transcript data, combines segments, handles timestamps
2. ✅ **Big Chunkeroosky**: Splits preprocessed transcript into chunks respecting sentence boundaries and token limits
3. ✅ **LLM Executor**: Distributes chunks to multiple LLMs in parallel with support for OpenAI and Anthropic
4. 🚧 **Result Aggregator**: Combines individual chunk summaries into a coherent whole
5. 🚧 **Prompt Manager**: Provides customizable prompts for different summarization tasks

**Legend:** ✅ Implemented | 🚧 Coming Soon

## Modules

### Preprocessor (`preprocessor.py`)

Handles the initial processing of raw transcript data:

- Cleans text and normalizes formatting
- Converts timestamps to readable format (HH:MM:SS)
- Combines consecutive segments from the same speaker (with configurable limits)
- Aggregates segments into time intervals if needed
- Preserves detailed timing information for all processing steps

**Output Format**: A list of dictionaries, where each dictionary represents a processed segment with:
- Start/end times (both raw seconds and formatted timestamps)
- Speaker information
- Text content (with embedded timestamps)
- Metadata about original segments

### Big Chunkeroosky (`big_chunkeroosky.py`)

Divides preprocessed segments into chunks suitable for LLM processing:

- Splits content based on token limits of target LLMs
- Respects sentence boundaries to avoid cutting sentences in half
- Handles extremely long segments by splitting at natural points
- Adds contextual information to each chunk including timing, speaker, and position data

**Output Format**: A list of chunk dictionaries, where each contains:
- Multiple segments with timing information
- Token count tracking
- Position metadata (chunk index, percentage through transcript)
- Context headers for LLM processing

### LLM Executor (`llm_executor.py`)

Handles parallel processing of chunks:

- Supports multiple LLM providers (OpenAI and Anthropic)
- Manages async requests with rate limiting and semaphores
- Implements robust error handling and automatic retries
- Tracks token usage and estimated costs
- Provides mock responses for development without API keys

**Output Format**: Processed chunks with added summary data including:
- Generated summary content
- Token usage statistics
- Processing metadata (model used, cost, etc.)

### Result Aggregator (in development)

Will combine individual chunk summaries:

- Merges summaries using additional LLM passes
- Eliminates redundancy
- Ensures coherent narrative flow
- Preserves key insights from all chunks

### Prompt Manager (planned)

Will manage prompts for different summarization tasks:

- Provides default prompt templates
- Allows customization of prompts
- Supports loading prompts from files
- Enables prompt switching for different use cases

## Data Flow

1. **Input**: JSON transcript with segments containing:
   ```json
   {
     "segments": [
       {
         "start": 0.0,
         "end": 25.52,
         "text": "Example text",
         "speaker": "SPEAKER_00"
       },
       ...
     ]
   }
   ```

2. **Preprocessing**: Segments are cleaned, combined, and enriched with timestamp information

3. **Chunking**: Preprocessed segments are divided into chunks based on token limits and sentence boundaries

4. **LLM Processing**: Each chunk is processed by an LLM to generate a summary

5. **Aggregation**: Individual summaries are combined into a final, coherent summary

## Usage

### Basic Usage

```python
import asyncio
import json
from preprocessor import preprocess_transcript
from big_chunkeroosky import BigChunkeroosky
from llm_executor import LLMExecutor
from result_aggregator import aggregate_results  # Coming soon

async def summarize_transcript(transcript_path, output_path=None):
    # Load transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # Preprocess transcript
    processed_segments = preprocess_transcript(
        transcript_data['segments'],
        merge_same_speaker=True,
        max_segment_duration=120,  # 2 minutes max per segment
        preserve_timestamps=True
    )
    
    # Split into chunks based on token limit
    chunker = BigChunkeroosky(max_tokens_per_chunk=4000)
    chunks = chunker.chunk_transcript(processed_segments)
    chunks = chunker.postprocess_chunks(chunks)
    
    # Process chunks in parallel with LLMs
    prompt_template = """
    Please summarize the following transcript segment:
    
    {transcript}
    
    Provide a concise summary that captures the main points, ideas, and flow of the conversation.
    """
    
    executor = LLMExecutor()
    processed_chunks = await executor.process_chunks(chunks, prompt_template)
    
    # Aggregate results into a final summary (coming soon)
    # final_summary = aggregate_results(processed_chunks)
    
    # For now, just combine the summaries
    summaries = [chunk.get('summary', '') for chunk in processed_chunks]
    final_summary = '\n\n'.join(summaries)
    
    # Save summary if output path is provided
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(final_summary)
    
    return final_summary

# Run the summarizer
# asyncio.run(summarize_transcript('transcript.json', 'summary.txt'))
```

### Preprocessing Only

```python
from preprocessor import preprocess_transcript
import json

# Load transcript
with open('transcript.json', 'r', encoding='utf-8') as f:
    transcript_data = json.load(f)

# Preprocess with different options
processed_segments = preprocess_transcript(
    transcript_data['segments'],
    merge_same_speaker=True,
    max_segment_duration=120  # 2 minutes max
)

# Time-interval aggregation (1-minute intervals)
time_intervals = preprocess_transcript(
    transcript_data['segments'],
    time_interval_seconds=60
)
```

## Configuration

### API Keys and Environment

The system uses a `.env` file to manage API keys and configuration. A template is provided in `.env.template`:

```
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORG_ID=your_openai_org_id_here_if_applicable
OPENAI_MODEL=gpt-3.5-turbo  # Options: gpt-3.5-turbo, gpt-4, gpt-4o-mini, etc.

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Request Configuration
MAX_CONCURRENT_REQUESTS=5  # Maximum parallel requests to LLM API
TEMPERATURE=0.3  # Temperature parameter for LLM responses (0.0-1.0)
MAX_TOKENS=1000  # Maximum tokens in LLM responses
REQUEST_TIMEOUT=60  # Timeout for API requests in seconds

# Default Provider Selection
DEFAULT_PROVIDER=openai  # Options: openai, anthropic
```

### Module-Specific Configuration

#### Preprocessor Configuration

- `merge_same_speaker`: Whether to combine consecutive segments from same speaker
- `time_interval_seconds`: If set, aggregates into fixed time intervals
- `max_segment_duration`: Maximum duration for combined segments (in seconds)
- `preserve_timestamps`: Whether to include original timestamps in text

#### Big Chunkeroosky Configuration

- `max_tokens_per_chunk`: Maximum tokens per chunk for the target LLM
- `overlap_tokens`: Number of tokens to overlap between chunks (for context)
- `context_tokens`: Tokens reserved for metadata and context headers

#### LLM Executor Configuration

- Provider selection: Choose between OpenAI and Anthropic
- Model selection: Specify which model to use for each provider
- Concurrency controls: Limit parallel API requests
- Error handling: Configure retry attempts and delay

## Extensibility

The system is designed to be extended in several ways:

### Custom Preprocessing

You can customize how segments are processed by modifying the preprocessor parameters or extending the module with new functions. The preprocessor supports different aggregation strategies:

```python
# Custom preprocessing with specific parameters
processed_segments = preprocess_transcript(
    transcript_data['segments'],
    merge_same_speaker=True,  # Combine consecutive segments from same speaker
    max_segment_duration=180,  # 3 minutes max per segment
    preserve_timestamps=True,  # Include original timestamps in text
    time_interval_seconds=None  # Don't use time-interval aggregation
)
```

### Multiple LLM Providers

The LLM executor supports different LLM providers through a standardized interface:

```python
# Use OpenAI
executor_openai = LLMExecutor(provider="openai", model="gpt-4o-mini")

# Use Anthropic
executor_anthropic = LLMExecutor(provider="anthropic", model="claude-3-sonnet-20240229")
```

### Custom Chunking Strategies

The Big Chunkeroosky class can be configured with different chunking parameters:

```python
# Configure chunking with specific parameters
chunker = BigChunkeroosky(
    max_tokens_per_chunk=4000,  # Max tokens per chunk
    overlap_tokens=200,  # Overlap between chunks for context
    context_tokens=150  # Reserved tokens for metadata
)
```

### Custom Prompts

You can create custom prompts for different summarization needs:

```python
# Different prompt templates for different purposes
summary_prompt = """
    Please summarize the following transcript segment,
    focusing on the main points and key ideas.
    
    {transcript}
"""

analysis_prompt = """
    Please analyze the following transcript segment,
    identifying themes, insights, and notable statements.
    
    {transcript}
"""
```

### Integration with Other Systems

The modular design allows for easy integration with other systems and pipelines.

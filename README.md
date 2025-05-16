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
4. ✅ **Result Aggregator**: Combines individual chunk summaries into a coherent whole
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

### Result Aggregator (`result_aggregator.py`)

Combines individual chunk summaries into a coherent final summary:

- Makes direct API calls to LLMs for reliable summaries
- Supports both single-pass and hierarchical aggregation for large documents
- Eliminates redundancy and ensures coherent narrative flow
- Preserves key insights, quotes, and themes from all chunks
- Structures output with consistent sections (Overview, Main Topics, Key Points, Notable Quotes)

### Prompt Management

Manages prompts for different summarization tasks:

- ✅ Provides default prompt templates
- ✅ Allows loading custom prompts from files
- ✅ Supports dedicated system prompts for controlling AI behavior
- ✅ Includes sample prompts for different use cases (analytical, video editing, etc.)
- ✅ Enables command-line selection of prompt files
- ✅ Supports custom aggregator prompts for tailored final summaries
- ✅ Preserves intermediate chunk summaries for detailed analysis

The `/prompts` directory contains sample prompt templates for different use cases:

- `analytical_prompt.txt`: Focuses on critical analysis of arguments and evidence
- `video_editor_prompt.txt`: Specialized for video editing with detailed timestamps
- `video_editor_system.txt`: System prompt that sets the AI's persona as a video editor
- `video_editor_aggregator.txt`: Aggregator prompt that preserves timestamps in the final summary
- `academic_system.txt`: System prompt for scholarly, academic-style analysis
- `accessibility_system.txt`: System prompt for clear, accessible summaries

#### Prompt Types and Hierarchy

| Prompt Type | Purpose | When Applied | CLI Argument |
|-------------|---------|--------------|----------------|
| Regular Prompt | Main instructions for processing each chunk | Individual chunks | `--prompt-file` |
| System Prompt | Sets the tone, personality and style | Individual chunks | `--system-prompt-file` |
| Aggregator Prompt | Controls how chunks are combined | Final aggregation | `--aggregator-prompt-file` |

#### Prompt Placeholder Variables

- Regular prompts: Use `{transcript}` as a placeholder for the transcript content
- Aggregator prompts: Use `{summaries}` as a placeholder for the list of summaries to combine

#### Intermediate Chunk Summaries

Saving intermediate chunks provides detailed summaries with timestamps before they're aggregated into the final summary. This is especially useful for video editing workflows where detailed timestamp information is critical.

```json
{
  "timestamp": "2025-05-15 20:25:25",
  "chunks": [
    {
      "chunk_index": 0,
      "start_time": 0.0,
      "end_time": 992.4,
      "summary": "### TIMELINE SUMMARY\n[00:00] - Speaker introduces...",
      "tokens_used": 4947
    }
  ]
}
```

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

### Command-Line Usage

The simplest way to use the transcript summarizer is through the `main.py` script, which provides a command-line interface to the entire pipeline:

```bash
# Basic usage
python main.py --input transcript.json --output summary.txt

# Limit segments (for testing or cost control)
python main.py --input transcript.json --output summary.txt --limit-segments 100

# Use a different model or provider
python main.py --input transcript.json --output summary.txt --provider anthropic --model claude-3-sonnet-20240229

# Generate detailed report with processing stats
python main.py --input transcript.json --output summary.txt --report

# Customize chunking parameters
python main.py --input transcript.json --output summary.txt --max-tokens-per-chunk 3000 --max-segment-duration 90

# Use custom prompt templates
python main.py --input transcript.json --output summary.txt --prompt-file prompts/analytical_prompt.txt

# Use both custom prompt and system prompt
python main.py --input transcript.json --output summary.txt --prompt-file prompts/video_editor_prompt.txt --system-prompt-file prompts/video_editor_system.txt

# Use custom aggregator prompt
python main.py --input transcript.json --output summary.txt --prompt-file prompts/video_editor_prompt.txt --aggregator-prompt-file prompts/video_editor_aggregator.txt

# Save intermediate chunk summaries (before aggregation)
python main.py --input transcript.json --output summary.txt --save-chunks chunks_output.json

# Full video editor workflow with all features
python main.py --input transcript.json --output summary.txt --prompt-file prompts/video_editor_prompt.txt --system-prompt-file prompts/video_editor_system.txt --aggregator-prompt-file prompts/video_editor_aggregator.txt --save-chunks chunks_output.json
```

Run `python main.py --help` for a full list of options.

### Python API Usage

You can also use the transcript summarizer programmatically in your Python code:

```python
import asyncio
import json
from main import TranscriptSummarizer

async def summarize_my_transcript(transcript_path, output_path):
    # Load transcript
    with open(transcript_path, 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # Create summarizer with desired config
    summarizer = TranscriptSummarizer(
        provider="openai",
        model="gpt-4o-mini",  # Specify model or leave as None to use default from .env
        max_tokens_per_chunk=4000,
        max_concurrent_requests=5,
        hierarchical_aggregation=True
    )
    
    # Process transcript
    result = await summarizer.summarize(
        transcript_data,
        merge_same_speaker=True,
        max_segment_duration=120,  # 2 minutes max per segment
        # Custom prompt template (optional)
        prompt_template="""Please summarize this transcript segment with attention to key points and important quotes:
        
        {transcript}
        
        Provide your summary in this format:
        1. Main Points:
        2. Key Details:
        3. Notable Quotes:""",
        # Additional metadata to include in summary
        metadata={"title": "My Transcript", "speaker": "John Doe"}
    )
    
    # Extract summary and write to file
    summary = result["summary"]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(summary)
    
    return result

# Run the summarizer
# asyncio.run(summarize_my_transcript('transcript.json', 'summary.txt'))
```

### Working with Different Prompt Templates

While a dedicated prompt manager is still in development, you can already customize the prompts used for summarization in several ways:

#### 1. CLI with Custom Prompt File

Create a prompt file:
```bash
# Create a prompt file
echo '
Please analyze the following transcript segment:

{transcript}

Analyze with these sections:
1. Key Topics
2. Main Arguments
3. Evidence Presented
4. Notable Quotes
' > analytical_prompt.txt

# Use the prompt with main script
python main.py --input transcript.json --output summary.txt --prompt-file analytical_prompt.txt
```

#### 2. Programming with Custom Prompt

With the TranscriptSummarizer class:
```python
summarizer = TranscriptSummarizer()
result = await summarizer.summarize(
    transcript_data,
    prompt_template="""Summarize this transcript focusing on the emotional tone:
    
    {transcript}
    
    Include sections on:
    1. Overall emotional themes
    2. Key emotional moments
    3. Relationship dynamics"""
)
```

#### 3. Working Directly with Components

For advanced customization:
```python
# Process with custom prompt
executor = LLMExecutor(provider="openai", model="gpt-4o-mini")
processed_chunks = await executor.process_chunks(
    chunks,
    """Create a creative summary of this transcript segment as if it were a movie scene:
    
    {transcript}
    
    Include: setting, characters, dialogue highlights, and mood."""
)
```

## Configuration

Environment variables can be configured in a `.env` file. See `.env.template` for an example with all available options.

```dotenv
# Provider Selection
DEFAULT_PROVIDER=openai  # Options: openai, anthropic

# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_ORG_ID=your_openai_org_id_here  # Optional
OPENAI_MODEL=gpt-4o-mini  # Options: gpt-3.5-turbo, gpt-4, gpt-4o-mini, etc.

# Anthropic Configuration
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Request Configuration
MAX_CONCURRENT_REQUESTS=10  # Maximum parallel requests to LLM API
TEMPERATURE=1.0           # Controls creativity of outputs (0.0-2.0)
MAX_TOKENS=4000           # Maximum tokens in LLM responses
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

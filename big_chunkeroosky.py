"""
The Big Chunkeroosky - Sentence-Aware Transcript Chunker.

This module divides preprocessed transcript segments into chunks suitable for LLM processing,
while respecting sentence boundaries and ensuring efficient token utilization.
"""

import re
import nltk
import tiktoken
from typing import List, Dict, Any, Optional, Tuple
from collections import deque

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class BigChunkeroosky:
    """The chunker class that splits transcript segments into LLM-friendly chunks."""
    
    def __init__(
        self,
        max_tokens_per_chunk: int = 4000,
        overlap_tokens: int = 200,
        tokenizer_name: str = "cl100k_base",
        context_tokens: int = 150  # Reserve tokens for context/metadata
    ):
        """
        Initialize the chunker.
        
        Args:
            max_tokens_per_chunk: Maximum tokens per chunk for the target LLM
            overlap_tokens: Number of tokens to overlap between chunks
            tokenizer_name: Name of the tokenizer to use
            context_tokens: Tokens reserved for context/metadata
        """
        self.max_tokens_per_chunk = max_tokens_per_chunk
        self.overlap_tokens = overlap_tokens
        self.context_tokens = context_tokens
        self.effective_max_tokens = max_tokens_per_chunk - context_tokens
        self.tokenizer = tiktoken.get_encoding(tokenizer_name)
        self.sentence_tokenizer = nltk.tokenize.PunktSentenceTokenizer()
        
    def chunk_transcript(
        self,
        processed_segments: List[Dict[Any, Any]],
        add_context: bool = True
    ) -> List[Dict[Any, Any]]:
        """
        Split processed transcript segments into chunks suitable for LLM processing.
        
        Args:
            processed_segments: Preprocessed transcript segments
            add_context: Whether to add context information to chunks
            
        Returns:
            A list of chunk dictionaries
        """
        chunks = []
        
        # Edge case: no segments
        if not processed_segments:
            return []
        
        print(f"Chunkeroosky: Processing {len(processed_segments)} segments")
        
        # Initialize the first chunk
        current_chunk = {
            'segments': [],
            'text': "",
            'token_count': 0,
            'start_time': processed_segments[0]['start'],
            'end_time': None,
            'speakers': set()
        }
        
        # Iterate through all segments
        for segment_index, segment in enumerate(processed_segments):
            # Calculate token counts before adding this segment
            segment_text = self._format_segment_for_chunk(segment)
            segment_tokens = len(self.tokenizer.encode(segment_text))
            
            # Check if adding this segment would exceed the token limit
            if current_chunk['token_count'] + segment_tokens > self.effective_max_tokens and current_chunk['segments']:
                # Finalize the current chunk before starting a new one
                self._finalize_chunk(current_chunk, chunks, segment_index, len(processed_segments), add_context)
                
                # Start a new chunk
                current_chunk = {
                    'segments': [],
                    'text': "",
                    'token_count': 0,
                    'start_time': segment['start'],
                    'end_time': None,
                    'speakers': set()
                }
            
            # Handle segments that are too large on their own
            if segment_tokens > self.effective_max_tokens:
                # Process this large segment into sentence-aware pieces
                sub_chunks = self._chunk_large_segment(segment)
                
                for sub_chunk in sub_chunks:
                    # If the current chunk already has content and adding this sub-chunk would exceed the limit
                    if current_chunk['token_count'] > 0 and current_chunk['token_count'] + sub_chunk['token_count'] > self.effective_max_tokens:
                        # Finalize the current chunk
                        self._finalize_chunk(current_chunk, chunks, segment_index, len(processed_segments), add_context)
                        
                        # Start a new chunk with this sub-chunk
                        current_chunk = {
                            'segments': [sub_chunk['segment']],
                            'text': sub_chunk['text'],
                            'token_count': sub_chunk['token_count'],
                            'start_time': sub_chunk['segment']['start'],
                            'end_time': sub_chunk['segment']['end'],
                            'speakers': {sub_chunk['segment']['speaker']}
                        }
                    else:
                        # Add this sub-chunk to the current chunk
                        current_chunk['segments'].append(sub_chunk['segment'])
                        if current_chunk['text']:
                            current_chunk['text'] += "\n\n"
                        current_chunk['text'] += sub_chunk['text']
                        current_chunk['token_count'] += sub_chunk['token_count']
                        current_chunk['end_time'] = sub_chunk['segment']['end']
                        current_chunk['speakers'].add(sub_chunk['segment']['speaker'])
            else:
                # Normal case: add this segment to the current chunk
                current_chunk['segments'].append(segment)
                if current_chunk['text']:
                    current_chunk['text'] += "\n\n"
                current_chunk['text'] += segment_text
                current_chunk['token_count'] += segment_tokens
                current_chunk['end_time'] = segment['end']
                current_chunk['speakers'].add(segment['speaker'])
        
        # Add the final chunk if it has content
        if current_chunk['segments']:
            self._finalize_chunk(current_chunk, chunks, len(processed_segments), len(processed_segments), add_context)
        
        print(f"Chunkeroosky: Created {len(chunks)} chunks from {len(processed_segments)} segments")
        
        return chunks
    
    def _finalize_chunk(
        self,
        chunk: Dict[Any, Any],
        chunks: List[Dict[Any, Any]],
        current_segment_index: int,
        total_segments: int,
        add_context: bool
    ) -> None:
        """
        Finalize a chunk and add it to the chunks list.
        
        Args:
            chunk: The chunk to finalize
            chunks: The list of chunks to add to
            current_segment_index: Current position in the segment list
            total_segments: Total number of segments
            add_context: Whether to add context information
        """
        # Convert speakers set to a sorted list
        chunk['speakers'] = sorted(list(chunk['speakers']))
        
        # Add chunk index and position metadata
        chunk['chunk_index'] = len(chunks)
        chunk['total_chunks'] = None  # Will be filled in later
        
        # Add position indicators (percentage through the transcript)
        first_segment = chunk['segments'][0]
        last_segment = chunk['segments'][-1]
        first_segment_time = first_segment['start']
        last_segment_time = last_segment['end']
        
        # Get the very first and last segment times from the full transcript
        transcript_start_time = chunks[0]['segments'][0]['start'] if chunks else first_segment_time
        
        # Estimate position
        chunk['position_percentage'] = ((first_segment_time - transcript_start_time) / 
                                       (last_segment_time - transcript_start_time) * 100 
                                       if last_segment_time > transcript_start_time else 0)
        
        # Add context if requested
        if add_context:
            # Add a context header with metadata
            context = self._create_context_header(chunk, current_segment_index, total_segments)
            chunk['text_with_context'] = context + "\n\n" + chunk['text']
        else:
            chunk['text_with_context'] = chunk['text']
            
        # Add to chunks list
        chunks.append(chunk)
    
    def _create_context_header(
        self,
        chunk: Dict[Any, Any],
        current_segment_index: int,
        total_segments: int
    ) -> str:
        """
        Create a context header for a chunk.
        
        Args:
            chunk: The chunk to create context for
            current_segment_index: Current position in segment list
            total_segments: Total number of segments
            
        Returns:
            Context header string
        """
        # Format time range
        time_range = f"{self._format_time(chunk['start_time'])} - {self._format_time(chunk['end_time'])}"
        
        # Format speakers
        speakers_text = ", ".join(chunk['speakers'])
        
        # Create position description
        position_text = f"Chunk {chunk['chunk_index'] + 1} (approximately {chunk['position_percentage']:.1f}% through the transcript)"
        
        # Create context header
        context = (
            f"--- TRANSCRIPT CHUNK INFORMATION ---\n"
            f"Time Range: {time_range}\n"
            f"Speakers: {speakers_text}\n"
            f"Position: {position_text}\n"
            f"--- TRANSCRIPT CHUNK CONTENT ---"
        )
        
        return context
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into HH:MM:SS format."""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
        
    def _format_segment_for_chunk(self, segment: Dict[Any, Any]) -> str:
        """
        Format a segment for inclusion in a chunk.
        
        Args:
            segment: The segment to format
            
        Returns:
            Formatted segment text
        """
        # Format basic information
        speaker = segment['speaker']
        start_time = self._format_time(segment['start'])
        
        # For already combined segments, the text already includes timestamps
        if segment.get('is_combined', False):
            text = segment['text']
            return f"[{start_time}] {speaker}: {text}"
        else:
            # For regular segments
            text = segment['text']
            return f"[{start_time}] {speaker}: {text}"
    
    def _chunk_large_segment(self, segment: Dict[Any, Any]) -> List[Dict[Any, Any]]:
        """
        Break a large segment into multiple smaller chunks based on sentences.
        
        Args:
            segment: The large segment to chunk
            
        Returns:
            List of sub-chunks
        """
        sub_chunks = []
        
        # If this is a combined segment, it already has component parts we can use
        if 'segment_timestamps' in segment and segment.get('is_combined', False):
            # Group the component segments into sub-chunks based on token count
            current_sub_chunk = {
                'segment': {
                    'start': segment['segment_timestamps'][0]['start'],
                    'end': None,
                    'speaker': segment['speaker'],
                    'text': "",
                    'is_sub_chunk': True,
                    'parent_segment_start': segment['start'],
                    'parent_segment_end': segment['end']
                },
                'text': "",
                'token_count': 0
            }
            
            for ts in segment['segment_timestamps']:
                ts_text = f"[{self._format_time(ts['start'])}] {ts['text']}"
                ts_tokens = len(self.tokenizer.encode(ts_text))
                
                # If adding this segment would exceed the token limit, start a new sub-chunk
                if current_sub_chunk['token_count'] + ts_tokens > self.effective_max_tokens and current_sub_chunk['token_count'] > 0:
                    sub_chunks.append(current_sub_chunk)
                    
                    # Start a new sub-chunk
                    current_sub_chunk = {
                        'segment': {
                            'start': ts['start'],
                            'end': None,
                            'speaker': segment['speaker'],
                            'text': "",
                            'is_sub_chunk': True,
                            'parent_segment_start': segment['start'],
                            'parent_segment_end': segment['end']
                        },
                        'text': "",
                        'token_count': 0
                    }
                
                # Add this timestamp to the current sub-chunk
                if current_sub_chunk['text']:
                    current_sub_chunk['text'] += " "
                current_sub_chunk['text'] += ts_text
                current_sub_chunk['token_count'] += ts_tokens
                current_sub_chunk['segment']['end'] = ts['end']
                current_sub_chunk['segment']['text'] = current_sub_chunk['text']
            
            # Add the final sub-chunk if it has content
            if current_sub_chunk['token_count'] > 0:
                sub_chunks.append(current_sub_chunk)
                
        else:
            # For a regular segment, split by sentences
            text = segment['text']
            sentences = self.sentence_tokenizer.tokenize(text)
            
            current_sub_chunk = {
                'segment': {
                    'start': segment['start'],
                    'end': None,
                    'speaker': segment['speaker'],
                    'text': "",
                    'is_sub_chunk': True,
                    'parent_segment_start': segment['start'],
                    'parent_segment_end': segment['end']
                },
                'text': "",
                'token_count': 0
            }
            
            # Calculate approximate time per character for time estimation
            time_span = segment['end'] - segment['start']
            chars = len(text)
            time_per_char = time_span / chars if chars > 0 else 0
            chars_processed = 0
            
            for sentence in sentences:
                sentence_text = sentence.strip()
                if not sentence_text:
                    continue
                    
                # Calculate approximate time for this sentence
                sent_chars = len(sentence_text)
                sent_time = time_per_char * sent_chars
                sent_start = segment['start'] + (time_per_char * chars_processed)
                sent_end = sent_start + sent_time
                chars_processed += sent_chars
                
                # Format with timestamp
                formatted_sentence = f"[{self._format_time(sent_start)}] {sentence_text}"
                sent_tokens = len(self.tokenizer.encode(formatted_sentence))
                
                # If this sentence alone exceeds the limit, we need to split it further
                if sent_tokens > self.effective_max_tokens:
                    # Split very long sentences by clauses or punctuation
                    clause_chunks = self._split_long_sentence(sentence_text, sent_start, sent_end)
                    
                    # If we have an existing sub-chunk, finalize it first
                    if current_sub_chunk['token_count'] > 0:
                        current_sub_chunk['segment']['end'] = sent_start
                        current_sub_chunk['segment']['text'] = current_sub_chunk['text']
                        sub_chunks.append(current_sub_chunk)
                    
                    # Add each clause as its own sub-chunk
                    sub_chunks.extend(clause_chunks)
                    
                    # Start fresh for the next sentence
                    current_sub_chunk = {
                        'segment': {
                            'start': sent_end,
                            'end': None,
                            'speaker': segment['speaker'],
                            'text': "",
                            'is_sub_chunk': True,
                            'parent_segment_start': segment['start'],
                            'parent_segment_end': segment['end']
                        },
                        'text': "",
                        'token_count': 0
                    }
                    
                # If adding this sentence would exceed the token limit, start a new sub-chunk
                elif current_sub_chunk['token_count'] + sent_tokens > self.effective_max_tokens and current_sub_chunk['token_count'] > 0:
                    # Finalize the current sub-chunk
                    current_sub_chunk['segment']['end'] = sent_start
                    current_sub_chunk['segment']['text'] = current_sub_chunk['text']
                    sub_chunks.append(current_sub_chunk)
                    
                    # Start a new sub-chunk with this sentence
                    current_sub_chunk = {
                        'segment': {
                            'start': sent_start,
                            'end': sent_end,
                            'speaker': segment['speaker'],
                            'text': formatted_sentence,
                            'is_sub_chunk': True,
                            'parent_segment_start': segment['start'],
                            'parent_segment_end': segment['end']
                        },
                        'text': formatted_sentence,
                        'token_count': sent_tokens
                    }
                else:
                    # Add this sentence to the current sub-chunk
                    if current_sub_chunk['text']:
                        current_sub_chunk['text'] += " "
                    current_sub_chunk['text'] += formatted_sentence
                    current_sub_chunk['token_count'] += sent_tokens
                    current_sub_chunk['segment']['end'] = sent_end
                    current_sub_chunk['segment']['text'] = current_sub_chunk['text']
            
            # Add the final sub-chunk if it has content
            if current_sub_chunk['token_count'] > 0:
                sub_chunks.append(current_sub_chunk)
        
        return sub_chunks
    
    def _split_long_sentence(
        self, 
        sentence: str, 
        start_time: float, 
        end_time: float
    ) -> List[Dict[Any, Any]]:
        """
        Split a very long sentence into smaller chunks.
        
        Args:
            sentence: The long sentence to split
            start_time: Start time of the sentence
            end_time: End time of the sentence
            
        Returns:
            List of sub-chunks
        """
        # Split on clause boundaries (commas, semicolons, etc.)
        clause_pattern = r'([^,.;:?!]+[,.;:?!]+)'
        clauses = re.findall(clause_pattern, sentence)
        
        # If no clause boundaries found, split arbitrarily
        if not clauses:
            # Just split into words and group them
            words = sentence.split()
            clauses = []
            current_clause = []
            
            for word in words:
                current_clause.append(word)
                if len(current_clause) >= 20:  # About 20 words per clause
                    clauses.append(' '.join(current_clause))
                    current_clause = []
            
            if current_clause:
                clauses.append(' '.join(current_clause))
        
        # Calculate time for each clause
        time_span = end_time - start_time
        total_chars = len(sentence)
        time_per_char = time_span / total_chars if total_chars > 0 else 0
        
        # Create sub-chunks for each clause
        sub_chunks = []
        current_sub_chunk = {
            'segment': {
                'start': start_time,
                'end': None,
                'speaker': "",  # Will be filled in later
                'text': "",
                'is_sub_chunk': True,
                'is_clause': True
            },
            'text': "",
            'token_count': 0
        }
        
        chars_processed = 0
        
        for clause in clauses:
            clause_text = clause.strip()
            if not clause_text:
                continue
                
            # Calculate approximate time for this clause
            clause_chars = len(clause_text)
            clause_time = time_per_char * clause_chars
            clause_start = start_time + (time_per_char * chars_processed)
            clause_end = clause_start + clause_time
            chars_processed += clause_chars
            
            # Format with timestamp
            formatted_clause = f"[{self._format_time(clause_start)}] {clause_text}"
            clause_tokens = len(self.tokenizer.encode(formatted_clause))
            
            # If adding this clause would exceed the token limit, start a new sub-chunk
            if current_sub_chunk['token_count'] + clause_tokens > self.effective_max_tokens and current_sub_chunk['token_count'] > 0:
                sub_chunks.append(current_sub_chunk)
                
                # Start a new sub-chunk
                current_sub_chunk = {
                    'segment': {
                        'start': clause_start,
                        'end': clause_end,
                        'speaker': "",  # Will be filled in later
                        'text': formatted_clause,
                        'is_sub_chunk': True,
                        'is_clause': True
                    },
                    'text': formatted_clause,
                    'token_count': clause_tokens
                }
            else:
                # Add this clause to the current sub-chunk
                if current_sub_chunk['text']:
                    current_sub_chunk['text'] += " "
                current_sub_chunk['text'] += formatted_clause
                current_sub_chunk['token_count'] += clause_tokens
                current_sub_chunk['segment']['end'] = clause_end
                current_sub_chunk['segment']['text'] = current_sub_chunk['text']
        
        # Add the final sub-chunk if it has content
        if current_sub_chunk['token_count'] > 0:
            sub_chunks.append(current_sub_chunk)
        
        return sub_chunks

    def postprocess_chunks(self, chunks: List[Dict[Any, Any]]) -> List[Dict[Any, Any]]:
        """
        Perform post-processing on the chunks list.
        
        Args:
            chunks: List of chunks to post-process
            
        Returns:
            Post-processed chunks
        """
        # Fill in missing total_chunks
        total_chunks = len(chunks)
        for chunk in chunks:
            chunk['total_chunks'] = total_chunks
            
        # Fill in speakers for any clause sub-chunks
        for chunk in chunks:
            for segment in chunk['segments']:
                if segment.get('is_clause', False) and not segment['speaker']:
                    # Find the parent segment's speaker
                    if 'parent_segment_start' in segment:
                        segment['speaker'] = chunk['speakers'][0] if chunk['speakers'] else "UNKNOWN"
        
        return chunks

# Example usage (when run directly)
if __name__ == "__main__":
    import json
    from preprocessor import preprocess_transcript
    
    # Load example transcript
    with open('transcript-example.json', 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # Preprocess the transcript
    processed_segments = preprocess_transcript(transcript_data['segments'])
    
    # Create the chunker
    chunker = BigChunkeroosky(max_tokens_per_chunk=4000)
    
    # Create chunks
    chunks = chunker.chunk_transcript(processed_segments)
    
    # Post-process chunks
    chunks = chunker.postprocess_chunks(chunks)
    
    # Print chunk information
    print(f"\n==== THE BIG CHUNKEROOSKY RESULTS ====\n")
    print(f"Created {len(chunks)} chunks with max token limit of {chunker.max_tokens_per_chunk}")
    
    # Show sample chunks
    for chunk_idx in [0, 1, len(chunks)//2, len(chunks)-1]:
        if chunk_idx < len(chunks):
            chunk = chunks[chunk_idx]
            print(f"\nChunk {chunk_idx+1}/{len(chunks)}:")
            print(f"Time Range: {chunker._format_time(chunk['start_time'])} - {chunker._format_time(chunk['end_time'])}")
            print(f"Token Count: {chunk['token_count']}/{chunker.effective_max_tokens} tokens")
            print(f"Speakers: {', '.join(chunk['speakers'])}")
            print(f"Position: {chunk['position_percentage']:.1f}% through the transcript")
            print(f"Contains {len(chunk['segments'])} segments")
            print(f"First 200 chars with context: {chunk['text_with_context'][:200]}...")
            print("-" * 50)

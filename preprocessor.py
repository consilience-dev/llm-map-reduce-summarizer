"""
Preprocessor module for transcript summarization.

This module handles the preprocessing of transcript data, including:
- Converting timestamps to readable format
- Cleaning text data
- Aggregating segments by time intervals or speaker
- Combining consecutive segments from the same speaker
"""

from typing import List, Dict, Any, Tuple, Optional
import re
import math

def preprocess_transcript(
    segments: List[Dict[Any, Any]], 
    merge_same_speaker: bool = True,
    time_interval_seconds: Optional[int] = None,
    max_segment_duration: Optional[int] = 120,  # Max 2 minutes per segment by default
    preserve_timestamps: bool = True  # Whether to preserve detailed timing in merged segments
) -> List[Dict[Any, Any]]:
    """
    Preprocess transcript segments to clean and prepare them for chunking.
    
    Args:
        segments: List of transcript segments from the JSON file
        merge_same_speaker: Whether to merge consecutive segments from the same speaker
        time_interval_seconds: If provided, aggregate segments into intervals of this many seconds
        
    Returns:
        Processed segments ready for chunking
    """
    processed_segments = []
    
    # Process each segment
    for segment in segments:
        # Skip empty segments
        if not segment.get('text', '').strip():
            continue
        
        # Clean up text
        cleaned_text = clean_text(segment.get('text', ''))
        
        # Create processed segment with formatted timestamps
        processed_segment = {
            'start': segment.get('start', 0),
            'end': segment.get('end', 0),
            'start_formatted': format_timestamp(segment.get('start', 0)),
            'end_formatted': format_timestamp(segment.get('end', 0)),
            'speaker': segment.get('speaker', ''),
            'text': cleaned_text
        }
        
        processed_segments.append(processed_segment)
    
    # Apply transformations
    result = processed_segments
    
    # Combine consecutive segments from the same speaker if requested
    if merge_same_speaker and result:
        result = combine_same_speaker_segments(result, max_segment_duration, preserve_timestamps)
    
    # Aggregate segments by time interval if requested
    if time_interval_seconds and result:
        result = aggregate_by_time_interval(result, time_interval_seconds)
    
    return result

def clean_text(text: str) -> str:
    """
    Clean up text by removing excessive spaces, normalizing punctuation, etc.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    # Replace multiple spaces with a single space
    cleaned = ' '.join(text.split())
    
    # Fix common transcription artifacts
    # Remove repeated words (like "the the")
    cleaned = re.sub(r'\b(\w+)( \1\b)+', r'\1', cleaned)
    
    # Fix missing spaces after punctuation
    cleaned = re.sub(r'([.!?])([A-Za-z])', r'\1 \2', cleaned)
    
    return cleaned

def format_timestamp(seconds: float) -> str:
    """
    Format seconds into HH:MM:SS format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours, remainder = divmod(int(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"

def combine_same_speaker_segments(segments: List[Dict[Any, Any]], max_duration: Optional[int] = 120, preserve_timestamps: bool = True) -> List[Dict[Any, Any]]:
    """
    Combine consecutive segments from the same speaker, respecting maximum duration constraints.
    When preserve_timestamps is True, maintains detailed timing information for each part of the text.
    
    Args:
        segments: List of preprocessed segments
        max_duration: Maximum duration in seconds for a combined segment
        preserve_timestamps: Whether to preserve detailed timing for each segment in the combined text
        
    Returns:
        List of segments with consecutive same-speaker segments combined
    """
    if not segments:
        return []
    
    # Check if we have multiple speakers - if only one speaker, we need to be careful
    # not to combine everything into one massive segment
    unique_speakers = set(segment['speaker'] for segment in segments)
    is_single_speaker = len(unique_speakers) == 1
    
    print(f"Preprocessing: Found {len(unique_speakers)} unique speakers in transcript")
    
    combined_segments = []
    current_group = [segments[0]]
    current_duration = segments[0]['end'] - segments[0]['start']
    current_speaker = segments[0]['speaker']
    
    # Group segments by speaker and duration
    for segment in segments[1:]:
        # Check if we should start a new group
        if (segment['speaker'] != current_speaker or 
            (max_duration is not None and current_duration + (segment['end'] - segment['start']) > max_duration)):
            # Create a combined segment from the current group
            combined_segment = create_combined_segment(current_group, preserve_timestamps)
            combined_segments.append(combined_segment)
            
            # Start a new group with the current segment
            current_group = [segment]
            current_duration = segment['end'] - segment['start']
            current_speaker = segment['speaker']
        else:
            # Add to the current group
            current_group.append(segment)
            current_duration += (segment['end'] - segment['start'])
    
    # Add the final group
    if current_group:
        combined_segment = create_combined_segment(current_group, preserve_timestamps)
        combined_segments.append(combined_segment)
    
    # Print stats about the compression achieved
    print(f"Preprocessing: Combined {len(segments)} segments into {len(combined_segments)} segments")
    if len(segments) > 0:
        print(f"Preprocessing: Compression ratio: {len(combined_segments) / len(segments):.2f}")
    
    return combined_segments

def create_combined_segment(segments: List[Dict[Any, Any]], preserve_timestamps: bool = True) -> Dict[Any, Any]:
    """
    Create a combined segment from a group of segments from the same speaker.
    
    Args:
        segments: List of segments to combine (all should be from the same speaker)
        preserve_timestamps: Whether to preserve timing info in the combined text
        
    Returns:
        A combined segment
    """
    if not segments:
        return {}
    
    if len(segments) == 1:
        # Just return the single segment as is
        return segments[0]
    
    # Extract information from the segment group
    start_time = segments[0]['start']
    end_time = segments[-1]['end']
    speaker = segments[0]['speaker']
    
    if preserve_timestamps:
        # Create a text representation that includes timing information
        text_parts = []
        for seg in segments:
            # Format: [MM:SS] Text
            timestamp = format_timestamp(seg['start'])
            text_parts.append(f"[{timestamp}] {seg['text']}")
        combined_text = " ".join(text_parts)
    else:
        # Simple concatenation
        combined_text = " ".join(seg['text'] for seg in segments)
    
    # Create the combined segment
    combined_segment = {
        'start': start_time,
        'end': end_time,
        'start_formatted': format_timestamp(start_time),
        'end_formatted': format_timestamp(end_time),
        'speaker': speaker,
        'text': combined_text,
        'is_combined': True,
        'original_segments': len(segments),
        'segment_timestamps': [{'start': seg['start'], 'end': seg['end'], 'text': seg['text']} for seg in segments]
    }
    
    return combined_segment

def aggregate_by_time_interval(segments: List[Dict[Any, Any]], interval_seconds: int) -> List[Dict[Any, Any]]:
    """
    Aggregate segments into fixed time intervals.
    
    Args:
        segments: List of preprocessed segments
        interval_seconds: Length of each interval in seconds
        
    Returns:
        List of aggregated segments by time interval
    """
    if not segments:
        return []
    
    # Get the overall time range
    start_time = segments[0]['start']
    end_time = segments[-1]['end']
    
    # Calculate intervals
    num_intervals = math.ceil((end_time - start_time) / interval_seconds)
    aggregated = []
    
    # Debug information
    print(f"Creating {num_intervals} time intervals of {interval_seconds} seconds each")
    print(f"Total time range: {format_timestamp(start_time)} - {format_timestamp(end_time)} ({end_time - start_time:.2f}s)")
    
    for i in range(num_intervals):
        interval_start = start_time + (i * interval_seconds)
        interval_end = min(interval_start + interval_seconds, end_time)
        
        # Find segments that fall within this interval
        interval_segments = []
        for segment in segments:
            # We need a more precise overlap detection
            # A segment belongs in this interval if either:
            # 1. Its start time is within the interval
            # 2. It spans across the interval (starts before and ends after)
            segment_in_interval = (
                (interval_start <= segment['start'] < interval_end) or
                (segment['start'] <= interval_start and segment['end'] > interval_start)
            )
            
            if segment_in_interval:
                # For segments that span multiple intervals, we'll only include
                # the portion of text that falls within this interval by creating
                # a copy with adjusted properties
                overlap_segment = segment.copy()
                
                # If this is a combined segment, we need to filter its component segments too
                if 'segment_timestamps' in segment:
                    # Only include component segments that overlap with this interval
                    filtered_timestamps = []
                    for ts in segment['segment_timestamps']:
                        ts_in_interval = (
                            (interval_start <= ts['start'] < interval_end) or
                            (ts['start'] <= interval_start and ts['end'] > interval_start)
                        )
                        if ts_in_interval:
                            filtered_timestamps.append(ts)
                    
                    if filtered_timestamps:
                        overlap_segment['segment_timestamps'] = filtered_timestamps
                        # Recalculate text with only the segments in this interval
                        text_parts = []
                        for ts in sorted(filtered_timestamps, key=lambda x: x['start']):
                            timestamp = format_timestamp(ts['start'])
                            text_parts.append(f"[{timestamp}] {ts['text']}")
                        overlap_segment['text'] = " ".join(text_parts)
                        interval_segments.append(overlap_segment)
                else:
                    # For regular segments, just add them as is
                    interval_segments.append(overlap_segment)
        
        if interval_segments:
            # Create an aggregated segment
            speakers = set(segment['speaker'] for segment in interval_segments)
            
            # Preserve timing within the interval
            text_parts = []
            for seg in sorted(interval_segments, key=lambda x: x['start']):
                # Format: [MM:SS Speaker] Text
                timestamp = format_timestamp(seg['start'])
                text_parts.append(f"[{timestamp} {seg['speaker']}] {seg['text']}")
            
            formatted_text = "\n\n".join(text_parts)  # Use double newlines for better readability
            
            aggregated_segment = {
                'start': interval_start,
                'end': interval_end,
                'start_formatted': format_timestamp(interval_start),
                'end_formatted': format_timestamp(interval_end),
                'speaker': ', '.join(speakers) if len(speakers) > 1 else next(iter(speakers)),
                'text': formatted_text,
                'is_aggregated': True,
                'interval_index': i,  # Store the interval index for reference
                'original_segments': len(interval_segments),
                'segment_timestamps': [{
                    'start': seg['start'], 
                    'end': seg['end'], 
                    'speaker': seg['speaker'],
                    'text': seg['text']
                } for seg in sorted(interval_segments, key=lambda x: x['start'])]
            }
            
            aggregated.append(aggregated_segment)
    
    print(f"Created {len(aggregated)} time-interval segments (some intervals may be empty)")
    return aggregated

def extract_speakers(segments: List[Dict[Any, Any]]) -> List[str]:
    """
    Extract a unique list of speakers from the transcript.
    
    Args:
        segments: List of transcript segments
        
    Returns:
        List of unique speaker identifiers
    """
    speakers = set()
    
    for segment in segments:
        if 'speaker' in segment and segment['speaker']:
            speakers.add(segment['speaker'])
    
    return sorted(list(speakers))

def get_transcript_duration(segments: List[Dict[Any, Any]]) -> Tuple[float, str]:
    """
    Calculate the total duration of the transcript.
    
    Args:
        segments: List of transcript segments
        
    Returns:
        Tuple of (duration_in_seconds, formatted_duration)
    """
    if not segments:
        return 0.0, "00:00"
    
    start_time = segments[0]['start']
    end_time = segments[-1]['end']
    duration = end_time - start_time
    
    return duration, format_timestamp(duration)

# Example usage for testing
if __name__ == "__main__":
    import json
    
    # Load example transcript
    with open('transcript-example.json', 'r', encoding='utf-8') as f:
        transcript_data = json.load(f)
    
    # Process with different options
    
    # Basic processing
    processed_segments = preprocess_transcript(transcript_data['segments'])
    print(f"Original segments: {len(transcript_data['segments'])}")
    print(f"Processed segments (with same-speaker merging): {len(processed_segments)}")
    
    # With time interval aggregation (1-minute intervals)
    aggregated_segments = preprocess_transcript(
        transcript_data['segments'], 
        merge_same_speaker=True,
        time_interval_seconds=60,
        max_segment_duration=120,  # Max 2 minutes per segment
        preserve_timestamps=True
    )
    print(f"Aggregated segments (1-minute intervals): {len(aggregated_segments)}")
    
    # Print detailed information about processed segments
    if processed_segments:
        print("\n==== SAMPLE PROCESSED SEGMENTS ====\n")
        # Show more segments in detail (first 5)
        for i, segment in enumerate(processed_segments[:5]):
            print(f"Segment {i+1}:")
            print(f"Time: {segment['start_formatted']} - {segment['end_formatted']} (Duration: {segment['end'] - segment['start']:.2f}s)")
            print(f"Speaker: {segment['speaker']}")
            print(f"Is Combined: {segment.get('is_combined', False)}")
            if segment.get('is_combined', False):
                print(f"Original Segments: {segment.get('original_segments', 0)}")
            print(f"Text: {segment['text'][:200]}" + ("..." if len(segment['text']) > 200 else ""))
            # Show original segment timestamps if available
            if 'segment_timestamps' in segment:
                print("\nOriginal segment timestamps:")
                for j, ts in enumerate(segment['segment_timestamps'][:3]):
                    print(f"  {j+1}. {format_timestamp(ts['start'])} - {format_timestamp(ts['end'])}: {ts['text'][:50]}" + 
                          ("..." if len(ts['text']) > 50 else ""))
                if len(segment['segment_timestamps']) > 3:
                    print(f"  ... and {len(segment['segment_timestamps']) - 3} more segments")
            print("\n" + "-"*50)
        
        # Also show a couple segments from the middle and end
        if len(processed_segments) > 10:
            middle_idx = len(processed_segments) // 2
            print(f"\nSegment from middle (#{middle_idx}):")
            segment = processed_segments[middle_idx]
            print(f"Time: {segment['start_formatted']} - {segment['end_formatted']} (Duration: {segment['end'] - segment['start']:.2f}s)")
            print(f"Speaker: {segment['speaker']}")
            print(f"Text: {segment['text'][:200]}" + ("..." if len(segment['text']) > 200 else ""))
            print("\n" + "-"*50)
            
            end_idx = len(processed_segments) - 1
            print(f"\nLast segment (#{end_idx + 1}):")
            segment = processed_segments[end_idx]
            print(f"Time: {segment['start_formatted']} - {segment['end_formatted']} (Duration: {segment['end'] - segment['start']:.2f}s)")
            print(f"Speaker: {segment['speaker']}")
            print(f"Text: {segment['text'][:200]}" + ("..." if len(segment['text']) > 200 else ""))
            print("\n" + "-"*50)
    
    # Print detailed information about time-interval segments
    if aggregated_segments:
        print("\n==== SAMPLE TIME-INTERVAL AGGREGATED SEGMENTS ====\n")
        
        # Show various time intervals
        for interval_idx in [0, 1, 5, 10, 20, 30]:
            if interval_idx < len(aggregated_segments):
                segment = aggregated_segments[interval_idx]
                print(f"Time Interval #{segment.get('interval_index', interval_idx) + 1}: {segment['start_formatted']} - {segment['end_formatted']} (Duration: {segment['end'] - segment['start']:.2f}s)")
                print(f"Speaker(s): {segment['speaker']}")
                print(f"Original Segments: {segment.get('original_segments', 0)}")
                print(f"Text Format:\n{segment['text'][:300]}" + ("..." if len(segment['text']) > 300 else ""))
                print("\n" + "-"*50)

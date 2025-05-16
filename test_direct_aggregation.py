"""
Direct test of the aggregation functionality with pre-defined summaries.
This bypasses the chunking and LLM summarization steps to focus on just the aggregation.
"""

import asyncio
import logging
from llm_executor import LLMExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("DirectAggregationTest")

# Sample summaries (pre-defined to avoid API calls for the chunk summarization)
SAMPLE_SUMMARIES = [
    """
    ### 1. Concise Summary
    In this segment, the speaker discusses their plans for a live coding stream focused on creating a website that allows users to record their voices over beats, with basic audio processing features. They express uncertainty about the project and the deployment process, highlighting concerns about costs associated with various deployment strategies.

    ### 2. Key Topics Discussed
    - Concept of a website for recording vocals over beats
    - Basic audio processing features for improved sound quality
    - Challenges and costs associated with deploying web applications
    - The role of AI in coding and learning

    ### 3. Notable Quotes or Statements
    - "I have no idea how to build any of that stuff, right? So it's really AI dependent if it can do it for me."
    - "Right now, your grandma can code, you know, your grandma can straight up code a website now."
    """,
    
    """
    ### 1. Concise Summary
    In this segment, the speaker discusses the evolving landscape of AI tools for backend and frontend application development, highlighting various platforms and their functionalities. They express a desire for simpler, more efficient deployment solutions and critique the current offerings.

    ### 2. Key Topics Discussed
    - New AI concepts and their integration into development models
    - Popular AI tools for rapid application development (Google Cloud Functions, Replay)
    - Backend deployment strategies and serverless architecture
    - The role of AI in assisting with coding and issue resolution

    ### 3. Notable Quotes or Statements
    - "I just want to click a button and have my super duper complex backend setup."
    - "This is a quickly evolving area, and so I'm trying to figure out, like, do I really need to go all in on Replet?"
    """,
    
    """
    ### 1. Concise Summary
    In this segment, the speaker discusses the use of AI systems in project planning, emphasizing the need for clear project structures and effective comparisons of different technologies. They highlight the varying characteristics of these technologies and propose creating a matrix to compare features.

    ### 2. Key Topics Discussed
    - Generation of product requirement documents and user flow diagrams
    - Best practices for using AI systems
    - Comparison of different technologies and their features
    - Creation of a comparison matrix for analysis

    ### 3. Notable Quotes or Statements
    - "I think there's something to consider here, right? Something I'm noticing is that all of these options have different characteristics."
    - "Let's start by laying out a matrix."
    """
]

async def direct_aggregation_test():
    """Test direct aggregation with pre-defined summaries."""
    logger.info("Starting direct aggregation test")
    
    # Create the LLM Executor
    executor = LLMExecutor()
    
    # Prepare the aggregation prompt
    aggregation_prompt = """
    TASK: You are creating a comprehensive summary of a transcript by combining multiple segment summaries.
    
    Here are 3 summaries from different segments of the transcript:
    
    {summaries}
    
    IMPORTANT RULES:
    1. DO NOT include any greeting or introduction
    2. DO NOT ask how you can help
    3. ONLY provide the summary in the exact format below
    4. You MUST use the provided headings
    
    Your response MUST follow this format:
    
    # Transcript Summary
    
    ## Overview
    [Write 2-3 sentences that provide a high-level overview of the content]
    
    ## Main Topics
    [List the key themes and topics as bullet points]
    
    ## Important Points
    [Present key findings and takeaways as bullet points]
    
    ## Notable Quotes
    [Include the most significant quotes from the transcript]
    """
    
    # Format the summaries
    formatted_summaries = ""
    for i, summary in enumerate(SAMPLE_SUMMARIES):
        formatted_summaries += f"SUMMARY {i+1}:\n"
        formatted_summaries += "=" * 40 + "\n"
        formatted_summaries += f"{summary}\n"
        formatted_summaries += "=" * 40 + "\n\n"
    
    # Insert the summaries into the prompt
    final_prompt = aggregation_prompt.format(summaries=formatted_summaries)
    
    # Prepare the chunk for the executor
    # Use a custom system prompt to prevent greetings
    mock_chunk = {
        'text_with_context': final_prompt,
        'chunk_index': 0,
        'total_chunks': 1,
        'system_prompt': """You are a transcript summarizer that ONLY creates summaries of transcripts. NEVER respond with 
        greetings like 'Hello' or 'Hi'. NEVER ask how you can help. ALWAYS start your response with '# Transcript Summary'. 
        ALWAYS follow the exact format requested in the prompt. Your sole purpose is to summarize transcript content."""
    }
    
    logger.info("Calling LLM for direct aggregation")
    
    # Call the LLM with special parameters
    try:
        # Process with special system prompt
        response = await executor.process_chunks(
            [mock_chunk], 
            "", 
            summary_type="final"
        )
        
        if response and response[0].get('summary'):
            summary = response[0]['summary']
            
            # Print the result
            print("\n" + "="*50)
            print("DIRECT AGGREGATION RESULT")
            print("="*50 + "\n")
            print(summary)
            
            # If we detect a greeting, try to clean it
            if summary.lower().startswith("hello") or summary.lower().startswith("hi"):
                print("\nWARNING: Greeting detected in response.")
                
                # Simple method to remove greeting
                lines = summary.split('\n')
                clean_summary = []
                found_start = False
                
                for line in lines:
                    # Skip greeting lines
                    if not found_start:
                        if line.strip() and not any(line.lower().startswith(g) for g in ["hello", "hi", "hey"]):
                            found_start = True
                            clean_summary.append(line)
                    else:
                        clean_summary.append(line)
                
                if clean_summary:
                    print("\n" + "="*50)
                    print("CLEANED RESULT")
                    print("="*50 + "\n")
                    print('\n'.join(clean_summary))
        else:
            print("Failed to get a response from the LLM")
            
    except Exception as e:
        logger.error(f"Error during direct aggregation: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(direct_aggregation_test())

import os
import logging
import requests
import fitz
import asyncio
import json
import httpx
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.text_splitter import RecursiveCharacterTextSplitter
import ollama
from fastapi.responses import JSONResponse
from typing import Optional, List
import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

class URLRequest(BaseModel):
    url: str


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "FastAPI backend is running!"}


from fastapi import File, UploadFile, Form
from tempfile import NamedTemporaryFile

@app.post("/summarize_local/")
async def summarize_local(file: UploadFile = File(...), 
                          persona:str = Form(...),
                          word_limit:int =Form(...)):
    try:
        logger.info(f"Received PDF file: {file.filename}")
        input_context = {}
        # Save uploaded PDF file bytes
        file_bytes = await file.read()
        # Save to temp file for processing
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        # Extract text from the saved PDF
        text = extract_text_from_pdf(tmp_path)
        # Summarize the extracted text, passing input_context
        summary = await summarize_text_parallel(text, input_context,persona,word_limit)
        
        # Automatically trigger generate_output after summary is complete
        # Call /generate_output/ endpoint after summary is complete
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post("http://localhost:8000/generate_output/")
                if response.status_code == 200:
                    logger.info("✅ output.json successfully updated after summarization.")
                else:
                    logger.warning(f"⚠️ Failed to update output.json: {response.status_code} - {response.text}")
        except Exception as e:
            logger.error(f"❌ Error while calling /generate_output/: {str(e)}")
        return {"summary": summary}
    
    except Exception as e:
        logger.error(f"Error in summarize_local endpoint: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
    
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

async def summarize_chunk_with_retry(chunk, chunk_id, total_chunks, max_retries=2):
    """Retry mechanism wrapper for summarize_chunk_wrapper."""
    retries = 0
    while retries <= max_retries:
        try:
            if retries > 0:
                logger.info(f"Retry attempt {retries}/{max_retries} for chunk {chunk_id}/{total_chunks}")
            
            result = await summarize_chunk_wrapper(chunk, chunk_id, total_chunks)
            
            # If the result starts with "Error", it means there was an error but no exception was thrown
            if isinstance(result, str) and result.startswith("Error"):
                logger.warning(f"Soft error on attempt {retries+1}/{max_retries+1} for chunk {chunk_id}: {result}")
                retries += 1
                if retries <= max_retries:
                    # Exponential backoff: 5s, 10s, 20s, etc.
                    wait_time = 5 * (2 ** (retries - 1))
                    logger.info(f"Waiting {wait_time}s before retry for chunk {chunk_id}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All retry attempts failed for chunk {chunk_id}")
                    return result
            else:
                # Success
                if retries > 0:
                    logger.info(f"Successfully processed chunk {chunk_id} after {retries} retries")
                return result
                
        except Exception as e:
            retries += 1
            logger.error(f"Exception on attempt {retries}/{max_retries+1} for chunk {chunk_id}: {str(e)}")
            
            if retries <= max_retries:
                # Exponential backoff
                wait_time = 5 * (2 ** (retries - 1))
                logger.info(f"Waiting {wait_time}s before retry for chunk {chunk_id}")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All retry attempts exhausted for chunk {chunk_id}")
                return f"Error processing chunk {chunk_id} after {max_retries+1} attempts: {str(e)}"
    
    # This should never be reached, but just in case
    return f"Error: Unexpected end of retry loop for chunk {chunk_id}"


async def summarize_text_parallel(text,input_context=None,persona="user",word_limit=150):
    """Process text in chunks optimized for Gemma 3's 128K context window with full parallelism and retry logic."""
    token_estimate = len(text) // 4
    logger.info(f"Token Estimate: {token_estimate}")

    # Use larger chunks since Gemma 3 can handle 128K tokens
    chunk_size = 10000 * 4  # Approximately 32K tokens per chunk
    chunk_overlap = 100   # Larger overlap to maintain context
    
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ". ", " ", ""]
      )
    chunks = splitter.split_text(text)

    logger.info("---------------------------------------------------------")
    logger.info(f"Split text into {len(chunks)} chunks")

    
    # Log chunk details
    for i, chunk in enumerate(chunks, 1):
        chunk_length = len(chunk)
        logger.info(f"Length: {chunk_length} characters ({chunk_length // 4} estimated tokens)")

    logger.info("---------------------------------------------------------")
    logger.info(f"Processing {len(chunks)} chunks in parallel with retry mechanism...")

    # Create tasks for each chunk with retry mechanism
    tasks = [summarize_chunk_with_retry(chunk, i+1, len(chunks), max_retries=2) for i, chunk in enumerate(chunks)]
    
    # Process chunks with proper error handling at the gather level
    try:
        # Using return_exceptions=True to prevent one failure from canceling all tasks
        summaries = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process the results, handling any exceptions
        processed_summaries = []
        for i, result in enumerate(summaries):
            if isinstance(result, Exception):
                # An exception was returned
                logger.error(f"Task for chunk {i+1} returned an exception: {str(result)}")
                processed_summaries.append(f"Error processing chunk {i+1}: {str(result)}")
            else:
                # Normal result
                processed_summaries.append(result)
                
        summaries = processed_summaries
        
    except Exception as e:
        logger.error(f"Critical error in gather operation: {str(e)}")
        return f"Critical error during processing: {str(e)}"

    logger.info("All chunks processed (with or without errors)")

    # Check if we have at least some successful results
    successful_summaries = [s for s in summaries if not (isinstance(s, str) and s.startswith("Error"))]
    if not successful_summaries:
        logger.warning("No successful summaries were generated.")
        return "No meaningful summary could be generated. All chunks failed processing."

    # Combine summaries with section markers, including error messages for failed chunks
    combined_chunk_summaries = "\n\n".join(f"Section {i+1}:\n{summary}" for i, summary in enumerate(summaries))
    logger.info(f"Combined summaries length: {len(combined_chunk_summaries)} characters")
    logger.info("Generating final summary...")

    # Create final summary with system message
      # You can now access persona and word_limit here
    # persona = logger.info(f"Persona: {persona}")
    # word_limit = logger.info(f"Word limit: {word_limit}")
    task = input_context.get("job_to_be_done", {}).get("task", "analyze the document for insights")
    challenge = input_context.get("challenge_info", {}).get("description", "")
       # You can now access persona and word_limit here
    
    final_messages = [
    {
        "role": "system",
        "content": (
            f"You are a document intelligence assistant helping {persona}s complete tasks. "
            "You respond clearly, concisely, and professionally."
        )
    },
    {
        "role": "user",
        "content": f"""
Generate a technical analysis and summary tailored for a {persona} based on the user's context:

- Persona: {persona}
- Task: {task}
- Challenge Description: {challenge}

Use ONLY relevant document content to solve this challenge.
DO NOT start your response with any greetings, introductions, or words like "Okay", "Sure", or "Here is".
Begin immediately with the requested summary.
Provide a clear, actionable summary in exactly {word_limit} words.
DO NOT repeat these instructions or mention you are an AI.
Ignore chunks that have errors.

Here’s the content to analyze:
{combined_chunk_summaries}
"""
    }
]


    # Use async http client for the final summary with retry logic
    max_retries = 2
    retry_count = 0
    final_response = None
    
    while retry_count <= max_retries:
        try:
            # Use async http client for the final summary as well
            payload = {
                "model": "gemma3:1b",
                "messages": final_messages,
                "stream": False
            }
            
            logger.info(f"Sending final summary request (attempt {retry_count+1}/{max_retries+1})")
            # Make async HTTP request with increased timeout for final summary
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://localhost:11434/api/chat",
                    json=payload,
                    timeout=httpx.Timeout(connect=60, read=3600, write=60, pool=60)  # 15-minute read timeout
                )
                
                logger.info(f"Received final summary response, status code: {response.status_code}")
                
                if response.status_code != 200:
                    raise Exception(f"API returned non-200 status code: {response.status_code} - {response.text}")
                    
                final_response = response.json()
                break  # Success, exit retry loop
                
        except Exception as e:
            retry_count += 1
            logger.error(f"Error generating final summary (attempt {retry_count}/{max_retries+1}): {str(e)}")
            
            if retry_count <= max_retries:
                # Exponential backoff
                wait_time = 10 * (2 ** (retry_count - 1))
                logger.info(f"Waiting {wait_time}s before retrying final summary generation")
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"All retry attempts for final summary failed")
                return "Failed to generate final summary after multiple attempts. Please check the logs for details."
    
    if not final_response:
        return "Failed to generate final summary. Please check the logs for details."
    
    logger.info("Final summary generated")
    logger.info(f"Final summary length: {len(final_response['message']['content'])} characters")
    return final_response['message']['content']

async def summarize_chunk_wrapper(chunk, chunk_id, total_chunks):
    """Asynchronous wrapper for summarizing a single chunk using Ollama via async httpx."""
    logger.info("---------------------------------------------------------")
    logger.info(f"Starting processing of chunk {chunk_id}/{total_chunks}")
    try:
        # Add system message to better control output
        messages = [
            {"role": "system", "content": "Extract only technical details. No citations or references."},
            {"role": "user", "content": f"Extract technical content: {chunk}"}
        ]
        
        # Use httpx for truly parallel API calls
        payload = {
            "model": "gemma3:1b",
            "messages": messages,
            "stream": False
        }
        
        # Add better timeout and error handling
        try:
            # Make async HTTP request directly to Ollama API
            async with httpx.AsyncClient(timeout=3600) as client:  # Increased timeout to 10 minutes
                logger.info(f"Sending request for chunk {chunk_id}/{total_chunks} to Ollama API - Gemma3 ")
                response = await client.post(
                    "http://localhost:11434/api/chat",  # Default Ollama API endpoint
                    json=payload,
                    # Adding connection timeout and timeout parameters
                    timeout=httpx.Timeout(connect=60, read=3600, write=60, pool=60)
                )
                logger.info("---------------------------------------------------------")
                logger.info(f"Received response for chunk {chunk_id}/{total_chunks}, status code: {response.status_code}")
                
                if response.status_code != 200:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    return f"Error processing chunk {chunk_id}: API returned status code {response.status_code}"
                    
                response_data = response.json()
                summary = response_data['message']['content']
            
            logger.info(f"Completed chunk {chunk_id}/{total_chunks}")
            logger.info(f"Summary length: {len(summary)} characters")
            logger.info("---------------------------------------------------------")
            return summary
            
        except httpx.TimeoutException as te:
            error_msg = f"Timeout error for chunk {chunk_id}: {str(te)}"
            logger.error(error_msg)
            return f"Error in chunk {chunk_id}: Request timed out after 30 minutes. Consider increasing the timeout or reducing chunk size."
            
        except httpx.ConnectionError as ce:
            error_msg = f"Connection error for chunk {chunk_id}: {str(ce)}"
            logger.error(error_msg)
            return f"Error in chunk {chunk_id}: Could not connect to Ollama API. Check if Ollama is running correctly."
            
    except Exception as e:
        # Capture and log the full exception details
        import traceback
        error_details = traceback.format_exc()
        logger.error(f"Error processing chunk {chunk_id}: {str(e)}")
        logger.error(f"Traceback: {error_details}")
        return f"Error processing chunk {chunk_id}: {str(e)}"


if __name__ == "__main__":
    import uvicorn

    logger.info("Starting FastAPI server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")


@app.post("/generate_output/")
async def generate_structured_output():
    try:
        # Load input.json
        input_path = os.path.join(os.path.dirname(__file__), "Collections", "input.json")
        with open(input_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        # Fix for output.json schema: input_document, extracted_section, subsection_analysis
        documents = input_data.get("documents", [])
        persona = input_data.get("persona", "")
        job_to_be_done = input_data.get("job_to_be_done", "")
        timestamp = datetime.datetime.now().isoformat()

        extracted_section = []
        subsection_analysis = []

        for doc in documents:
            filename = doc.get("filename", "") if isinstance(doc, dict) else ""
            print("running for",filename)
            title = doc.get("title", "") if isinstance(doc, dict) else ""
            pdf_path = os.path.join(os.path.dirname(__file__), "Collections", "PDFs", filename)

            # Extract PDF text
            if not filename or not os.path.exists(pdf_path):
                logger.warning(f"File not found: {filename}")
                continue

            text = extract_text_from_pdf(pdf_path)

            # Prompt Ollama for structured extraction
            messages = [
                {"role": "system", "content": f"You are helping a persona who needs to: {job_to_be_done}. Extract the most important section titles with importance ranking (High/Medium/Low) and page numbers."},
                {"role": "user", "content": f"Document: {title}\n\n{text}\n\nReturn 1 important section with section_title, importance_rank, page_number, and a short refined_text. Format:\nSection Title: ...\nImportance: ...\nPage: ...\nText: ...\n"}
            ]

            payload = {
                "model": "gemma3:1b",
                "messages": messages,
                "stream": False
            }

            async with httpx.AsyncClient(timeout=600) as client:
                response = await client.post("http://localhost:11434/api/chat", json=payload)
                response.raise_for_status()
                result = response.json()

            content = result['message']['content']

            # Extract structured entry using regex
            import re
            entry = re.search(r"Section Title: (.*?)\nImportance: (.*?)\nPage: (.*?)\nText: (.*?)\n", content, re.DOTALL)
            if entry:
                section_title, importance, page, refined = entry.groups()
                extracted_section.append({
                    "document": filename,
                    "section_title": section_title.strip(),
                    "importance_rank": importance.strip(),
                    "page_number": page.strip()
                })
                subsection_analysis.append({
                    "document": filename,
                    "refined_text": refined.strip(),
                    "page_number": page.strip()
                })
            else:
                # If no match, append empty structure
                extracted_section.append({
                    "document": filename,
                    "section_title": "",
                    "importance_rank": "",
                    "page_number": ""
                })
                subsection_analysis.append({
                    "document": filename,
                    "refined_text": "",
                    "page_number": ""
                })

        # Build final output.json object
        output_data = {
            "metadata": {
                "input_document": [d["filename"] if isinstance(d, dict) else "" for d in documents],
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": timestamp
            },
            "extracted_section": extracted_section,
            "subsection_analysis": subsection_analysis
        }

        
        # Optional: Read final summary to include in output.json
        summary_path = os.path.join(os.path.dirname(__file__), "Collections", "final_summary.txt")
        final_summary = ""
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as fs:
                final_summary = fs.read()

        output_data["metadata"]["final_summary"] = final_summary

        output_path = os.path.join(os.path.dirname(__file__), "Collections", "output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=4)

        return {"status": "success","json":output_data, "message": "output.json updated!"}

    except Exception as e:
        logger.error(f"Error generating structured output: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)
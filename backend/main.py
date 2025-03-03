import asyncio, hashlib, io, json, logging, os, random, re, threading, time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from typing import List, Optional, Any
from uuid import uuid4

import openai, pyodbc, uvicorn
from aisearch import search
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import ResourceExistsError
from azure.data.tables import TableServiceClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes.models import SimpleField, SearchIndex
from azure.storage.blob import BlobServiceClient, ContentSettings, generate_blob_sas, BlobSasPermissions
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Header, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langchain_openai import AzureOpenAIEmbeddings
from openai import AzureOpenAI
from pdf2image import convert_from_bytes
from pydantic import BaseModel
from PyPDF2 import PdfReader
from pytesseract import image_to_string

from azuretables import Entity, TableManager
from models import chunks, FileList, AISearchRequest, FeedbackRequest, ChunkResponse, DeleteFileRequest, MemoryQueryResult, MemoryRecordMetadata, Message, SearchRequest, SearchResponse, VectorQuery
from curie_agent.backend.settings import get_settings
from template import generate_template


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)



# table_client = TableManager.get_client(endpoint=settings.storage_table_endpoint)
# table_manager = TableManager(client=table_client)
# table_manager.create_table()

app = fastapi.FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"]
)

token_provider = get_bearer_token_provider(
    DefaultAzureCredential(), 
    "https://cognitiveservices.azure.com/.default"
)

oiclient = AzureOpenAI(
    azure_endpoint=settings.Endpoint,
    azure_ad_token_provider=token_provider,
    api_version=settings.version
)

search_client = SearchClient(
    endpoint=settings.ai_search_endpoint,
    index_name="vector-health",
    credential=DefaultAzureCredential()
)

def get_blob_service_client() -> BlobServiceClient:
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url="https://datafusionaiaisearchstg.blob.core.windows.net", 
        credential=credential
    )
    return blob_service_client

blob_service_client = get_blob_service_client()

def get_embedding_function():
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment="text-embedding-ada-002",
        azure_endpoint=settings.Endpoint,
        openai_api_version=settings.version,
        api_key="67b2a1d0374848ce9e255961267995f7"
    ) 
    return embeddings

def generate_embedding(text):
    embeddings = get_embedding_function()
    return embeddings.embed_documents([text])[0]


def split_text_into_chunks(text, max_length=800, overlap =150):
    """
    Split text into chunks while preserving sentence and word boundaries.
    
    Args:
        text (str): The input text to be split
        max_length (int): Maximum length of each chunk
        
    Returns:
        list: List of tuples containing (chunk_text, start_char, end_char)
    """
    chunks = []
    current_pos = 0
    text_length = len(text)
    
    while current_pos < text_length:
        # If remaining text is shorter than max_length, take it all
        if current_pos + max_length >= text_length:
            chunk = text[current_pos:]
            chunks.append((chunk, current_pos, current_pos + len(chunk) - 1))
            break
            
        # Find the last sentence boundary within max_length
        chunk_end = current_pos + max_length
        
        # Look for sentence boundaries (., !, ?)
        last_sentence = -1
        for sep in ['. ', '! ', '? ']:
            pos = text.rfind(sep, current_pos, chunk_end)
            if pos > last_sentence:
                last_sentence = pos + len(sep)
        
        # If no sentence boundary found, look for other punctuation
        if last_sentence == -1:
            for sep in ['; ', ': ', ', ']:
                pos = text.rfind(sep, current_pos, chunk_end)
                if pos > last_sentence:
                    last_sentence = pos + len(sep)
        
        # If still no boundary found, look for the last space
        if last_sentence == -1:
            last_space = text.rfind(' ', current_pos, chunk_end)
            if last_space != -1:
                last_sentence = last_space + 1
        
        # If no natural boundaries found, force break at max_length
        if last_sentence == -1:
            last_sentence = chunk_end
            
        # Extract the chunk
        chunk = text[current_pos:last_sentence].strip()
        
        # Only add non-empty chunks
        if chunk:
            chunks.append((chunk, current_pos, current_pos + len(chunk) - 1))
        
        # current_pos = last_sentence
        current_pos = last_sentence - overlap if last_sentence - overlap > current_pos else last_sentence
        
    return chunks

def generate_sas_urls(blob_name: str, container_name: str) -> str:
    try:
        blob_service_client = get_blob_service_client()
        user_delegation_key = blob_service_client.get_user_delegation_key(
            key_start_time=datetime.utcnow(),
            key_expiry_time=datetime.utcnow() + timedelta(hours=1)
        )
        
        sas_token = generate_blob_sas(
            account_name=settings.azure_storage_account_name,
            container_name=container_name,
            blob_name=blob_name,
            user_delegation_key=user_delegation_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )
        
        return f"https://{settings.azure_storage_account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating SAS URL: {str(e)}")


async def index_chunks_in_azure_search(chunks, blob_name, debug):
    try:
        updated_documents = []
        parent_id = str(uuid4())
        container_name = settings.azure_blob_container_name_2
        
        # Increase batch size and process embeddings in parallel
        batch_size = 20  # Doubled from 10 to 20
        
        async def process_chunk_batch(batch_chunks):
            try:
                batch_documents = []
                
                # Generate embeddings in parallel
                with ThreadPoolExecutor(max_workers=4) as executor:
                    embedding_futures = []
                    for chunk, start_char, end_char in batch_chunks:
                        future = executor.submit(generate_embedding, chunk)
                        embedding_futures.append((future, chunk, start_char, end_char))
                    
                    # Collect results
                    for future, chunk, start_char, end_char in embedding_futures:
                        try:
                            embedding = future.result()
                            document = {
                                "chunk_id": str(uuid4()),
                                "parent_id": parent_id,
                                "chunk": chunk,
                                "title": blob_name,
                                "metadata_storage_path": container_name,
                                "start_char": start_char,
                                "end_char": end_char,
                                "text_vector": embedding
                            }
                            batch_documents.append(document)
                        except Exception as e:
                            logging.error(f"Error processing chunk: {str(e)}")
                            continue
                
                if batch_documents:
                    # Upload batch to search index
                    search_client.upload_documents(documents=batch_documents)
                    return batch_documents
                return []
                
            except Exception as e:
                logging.error(f"Error processing batch: {str(e)}")
                return []

        # Process batches concurrently
        tasks = []
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            tasks.append(process_chunk_batch(batch_chunks))
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks)
        for result in batch_results:
            updated_documents.extend(result)
        
        logging.info(f"Successfully indexed {len(updated_documents)} chunks from {blob_name}")
        return len(updated_documents)
        
    except Exception as e:
        logging.error(f"Error in index_chunks_in_azure_search: {str(e)}")
        raise

def preprocess_text_for_search(text):
    """
    Preprocess text before sending to Azure AI Search.
    Removes special formatting and normalizes text.
    
    Args:
        text (str): Input text with special characters
        
    Returns:
        str: Cleaned and normalized text
    """
    import re
    
    # Remove the asterisk formatting
    text = re.sub(r'\*{3}(.*?)\*{3}', r'\1', text)
    
    # Basic text normalization
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove non-breaking spaces and other special whitespace
    text = text.replace('\xa0', ' ')
    
    # Normalize quotes and apostrophes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove other special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\-\'"]', ' ', text)
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove multiple consecutive newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def preprocess_text_for_llm(text):
    """
    Preprocess text before sending to LLM.
    Maintains some structure but removes unnecessary formatting.
    
    Args:
        text (str): Input text with special characters
        
    Returns:
        str: Cleaned text suitable for LLM
    """
    import re
    
    # Remove the asterisk formatting but maintain line separation
    text = re.sub(r'\*{3}(.*?)\*{3}', r'\1', text)
    
    # Normalize quotes and apostrophes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    # Remove non-breaking spaces
    text = text.replace('\xa0', ' ')
    
    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Remove excessive newlines but maintain paragraph structure
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # Clean up extra spaces
    text = re.sub(r' {2,}', ' ', text)
    
    return text.strip()



def process_page(page):
    """
    Process a PDF page by preserving line breaks and adding special characters.
    
    Args:
        page: PDF page object
    Returns:
        str: Processed text with special characters around each line
    """
    try:
        # Extract raw text from page
        raw_text = page.extract_text()
        if not raw_text:
            return ""
        
        # Split into lines and process each line
        processed_lines = []
        current_line = []
        
        # Iterate through each character to handle line breaks properly
        for char in raw_text:
            if char == '\n':
                if current_line:
                    line = ''.join(current_line).strip()
                    if line:  # Only process non-empty lines
                        processed_lines.append(f"***{line}***")
                    current_line = []
            else:
                current_line.append(char)
        
        # Handle the last line if it exists
        if current_line:
            line = ''.join(current_line).strip()
            if line:
                processed_lines.append(f"***{line}***")
        
        # Join all processed lines with newlines
        return '\n'.join(processed_lines)
    except Exception as e:
        logging.error(f"Error processing page: {str(e)}")
        return ""  # Return empty string on error to continue processing other pages

async def process_blob(pdf_bytes, blob_name, debug=False):
    """
    Process a PDF blob by extracting and formatting text from all pages.
    
    Args:
        pdf_bytes: Bytes of the PDF file
        blob_name: Name of the blob being processed
        debug: Boolean flag for debug logging
    
    Returns:
        int: Number of indexed chunks
    """
    try:
        if debug:
            logging.info(f"Starting to process PDF blob: {blob_name}")
        
        pdf_file_like = io.BytesIO(pdf_bytes)
        pdf_reader = PyPDF2.PdfReader(pdf_file_like)
        total_pages = len(pdf_reader.pages)
        
        if debug:
            logging.info(f"PDF has {total_pages} pages")
        
        # Determine optimal chunk size based on total pages
        chunk_size = min(10, max(1, total_pages // 4))  # Adaptive chunk size
        
        async def process_page_batch(start_idx, end_idx):
            """Process a batch of pages using ThreadPoolExecutor."""
            if debug:
                logging.info(f"Processing batch from page {start_idx} to {end_idx}")
            
            with ThreadPoolExecutor() as executor:
                batch_pages = pdf_reader.pages[start_idx:end_idx]
                batch_results = list(executor.map(process_page, batch_pages))
                
                # Additional verification of processed content
                verified_results = []
                for i, content in enumerate(batch_results):
                    if content.strip():
                        # Verify that each line has the special characters
                        lines = content.split('\n')
                        verified_lines = []
                        for line in lines:
                            if line.strip() and not (line.startswith('***') and line.endswith('***')):
                                line = f"***{line.strip()}***"
                            verified_lines.append(line)
                        verified_results.append('\n'.join(verified_lines))
                    else:
                        if debug:
                            logging.warning(f"Empty content from page {start_idx + i}")
                
                return verified_results
        
        # Create tasks for processing batches
        tasks = []
        for i in range(0, total_pages, chunk_size):
            end_idx = min(i + chunk_size, total_pages)
            tasks.append(process_page_batch(i, end_idx))
        
        # Wait for all batches to complete
        batch_results = await asyncio.gather(*tasks)
        all_content = []
        for batch in batch_results:
            all_content.extend(batch)
        
        # Final verification of content
        full_content = '\n'.join(filter(None, all_content))
        if not full_content.strip():
            raise ValueError("No content extracted from PDF")
        
        # Split into chunks and index
        chunks = split_text_into_chunks(full_content)
        if not chunks:
            raise ValueError("No valid chunks generated from document")
        
        # Index chunks with optimized batch processing
        indexed_count = await index_chunks_in_azure_search(chunks, blob_name, debug)
        if indexed_count == 0:
            raise ValueError("No chunks were successfully indexed")
        
        if debug:
            logging.info(f"Successfully processed and indexed {indexed_count} chunks")
        
        return indexed_count
        
    except Exception as e:
        error_msg = f"Error processing PDF blob {blob_name}: {str(e)}"
        logging.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)


@app.post("/api/v1/upload")
async def upload_file_to_blob(
    req: Request,
    file: UploadFile,  
    file_name: str, 
    debug: bool
):
    user_id = req.headers.get("user-id")
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is required")
        
    try:
        # Early validation
        allowed_types = [
            "application/pdf",
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ]
        
        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail="Invalid file type. Only PDF and Word documents are allowed.")
        
        # Read file content once
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file")

        # Prepare blob storage parameters
        container = container_name1 if debug else container_name
        modified_file_name = f"{user_id}_{file_name}"
        content_settings = ContentSettings(content_type=file.content_type or "application/octet-stream")

        # Execute upload and processing concurrently
        async def upload_to_blob():
            blob_client = blob_service_client.get_blob_client(container=container, blob=modified_file_name)
            await asyncio.to_thread(
                blob_client.upload_blob,
                file_content,
                overwrite=True,
                content_settings=content_settings,
                max_concurrency=4  # Enable concurrent uploads
            )
            return f"https://{blob_service_client.account_name}.blob.core.windows.net/{container}/{modified_file_name}"

        # Run upload and processing in parallel
        upload_task = asyncio.create_task(upload_to_blob())
        process_task = asyncio.create_task(process_blob(file_content, modified_file_name, debug))

        # Wait for both tasks to complete
        blob_url, indexed_count = await asyncio.gather(upload_task, process_task)

        return {
            "url": blob_url,
            "name": file_name,
            "chunks_indexed": indexed_count
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error in upload_file_to_blob: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/check", response_model=ChunkResponse)
async def receive_selected_files(req: Request, file_list: FileList):     
    user_id = req.headers.get("user-id")
    
    if not user_id:
        raise HTTPException(status_code=400, detail="User ID is missing in headers.")

    if not file_list.files:         
        return ChunkResponse(file_filter=[], text_related_to_chunk_id=[]) 
    
    # Add user-id prefix to each file name
    files_with_user_id = [f"{user_id}_{file}" for file in file_list.files]

    # print("file list need to search", files_with_user_id)

    # Pass the modified file names to the find_chunk_ids function
    file_filter, text_related_to_chunk_id = find_chunk_ids(files_with_user_id)
    
    return ChunkResponse(file_filter=file_filter, text_related_to_chunk_id=text_related_to_chunk_id)



container_name=settings.azure_blob_container_name_1 #"docs"
container_name1=settings.azure_blob_container_name_2 #"input"


@app.get("/api/v1/getfiles")
async def get_files(req:Request):
    try:
        user_id=req.headers.get("user-id")
        files = {}
        # Get file names from both containers
        for container in [container_name, container_name1]:
            container_client = blob_service_client.get_container_client(container)
            for blob in container_client.list_blobs():
                if user_id in blob.name:
                    print("user id ",blob)
                    file_name= blob.name.split("_", 1)[-1]

                    files[file_name] = container
        # print("Files retrieved from Blob Storage:", files)
        return files


    except Exception as e:
        print("Error fetching files from Blob Storage:", e)
        raise HTTPException(status_code=500, detail="Error fetching files")

async def delete_document_from_search(filename: str):
    try:
        # Search for documents with this filename
        results = search_client.search(
            search_text=filename,
            select=["chunk_id", "title"],
            filter=f"title eq '{filename}'"
        )
        
        chunk_ids = []
        for result in results:
            if "chunk_id" in result:
                chunk_ids.append(result["chunk_id"])

        if chunk_ids:
            # Delete all documents by chunk_id
            search_client.delete_documents(
                documents=[{"chunk_id": chunk_id} for chunk_id in chunk_ids]
            )
            logger.info(f"Deleted {len(chunk_ids)} chunks from search index")
            return True
        else:
            logger.warning(f"No chunks found for document: {filename}")
            return False
            
    except Exception as e:
        logger.error(f"Search index deletion failed: {e}")
        return False

@app.delete("/api/v1/deletefiles")
async def delete_file(req: Request, request: DeleteFileRequest):
    try:
        filename1 = request.filename
        user_id = req.headers.get("user-id")
        
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")
            
        filename = f"{user_id}_{filename1}"
        logger.info(f"Starting delete process for file: {filename}")

        # First, list blobs to check exact filename
        container_client = blob_service_client.get_container_client(container_name1)
        blobs = container_client.list_blobs(name_starts_with=f"{user_id}_")
        
        # Log available files for debugging
        logger.info("Available files in container:")
        matching_blob = None
        for blob in blobs:
            logger.info(f"Found blob: {blob.name}")
            # Check for exact match or if only difference is spaces/encoding
            if blob.name.replace(" ", "") == filename.replace(" ", ""):
                matching_blob = blob.name
                logger.info(f"Found matching blob: {matching_blob}")
                break

        if not matching_blob:
            logger.warning(f"No matching file found for: {filename}")
            raise HTTPException(status_code=404, detail=f"File not found: {filename1}")

        # Delete the blob using exact name from storage
        try:
            blob_client = blob_service_client.get_blob_client(
                container=container_name1, 
                blob=matching_blob
            )
            # Remove await since delete_blob is not async
            blob_client.delete_blob()
            logger.info(f"Successfully deleted blob: {matching_blob}")

            # Delete from search index
            try:
                results = search_client.search(
                    search_text=matching_blob,
                    select=["chunk_id"],
                    filter=f"title eq '{matching_blob}'"
                )
                
                chunk_ids = [result["chunk_id"] for result in results if "chunk_id" in result]
                
                if chunk_ids:
                    search_client.delete_documents(
                        documents=[{"chunk_id": id} for id in chunk_ids]
                    )
                    logger.info(f"Deleted {len(chunk_ids)} chunks from search index")
                
            except Exception as e:
                logger.error(f"Error deleting from search index: {str(e)}")
                # Continue even if search index deletion fails

            return JSONResponse(
                content={
                    "success": True,
                    "message": "File deleted successfully"
                },
                status_code=200
            )

        except Exception as e:
            logger.error(f"Error deleting blob: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error deleting file: {str(e)}"
            )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
 

def find_chunk_ids(titles):
    """
    Find all chunk_ids associated with given titles.
    Handles both a single string or a list of strings.
    """
    try:
        print("Finding chunks for files:", titles)
        if isinstance(titles, str):
            titles = [titles]

        chunk_ids = []
        text_related_to_chunk_id = []

        for title in titles:
            try:
                results = search_client.search(
                    search_text=f'"{title}"',
                    filter=f"title eq '{title}'"
                )
                results_list = list(results)
                print(f"Search results for title '{title}':", results_list)

                for result in results_list:
                    if "chunk_id" in result:
                        print(f"Found chunk_id: {result['chunk_id']} for title: {title}")
                        chunk_ids.append(result["chunk_id"])
                        text_related_to_chunk_id.append(result["chunk"])
                    else:
                        print(f"No chunk_id found in result: {result}")

            except Exception as search_error:
                print(f"Error searching for title '{title}': {search_error}")

        if not chunk_ids:
            print(f"No documents found with titles: {titles}")

        return chunk_ids, text_related_to_chunk_id

    except Exception as e:
        print(f"Unexpected error in find_chunk_ids: {e}")
        raise Exception("Failed to find documents in search index.")


 
def llm_model_generate(prompt):
    """
    Sends a prompt to OpenAI's GPT model and returns the generated output.
    """
    try:
        # Make the API call to generate the text
        response = oiclient.chat.completions.create(
            model="gpt-4o",  # Or "gpt-4" depending on the version you're using
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,  # Control the length of the response
            temperature=0.3,  # Adjust creativity
        )
        # Return the generated question
        return response.choices[0].message.content

    except Exception as e:
        print(f"Error generating text: {e}")
        return None


# def generate_questions_from_texts(text_related_to_chunk_id):
#     """
#     Generates questions and ensures proper chunk ID tracking.
#     """
#     num_texts_to_select = min(3, len(text_related_to_chunk_id))
#     start_indices = random.sample(range(len(text_related_to_chunk_id)), num_texts_to_select)
#     generated_questions = []

#     for index in start_indices:
#         try:
#             # Reconstruct text context for coherence
#             reconstructed_text = text_related_to_chunk_id[index]
#             chunk_ids = [index]  # Store the current chunk index
            
#             # Add next chunk if available for context
#             if index + 1 < len(text_related_to_chunk_id):
#                 reconstructed_text += " " + text_related_to_chunk_id[index + 1]
#                 chunk_ids.append(index + 1)

#             question_prompt = (
#                 "Based on the following text from a document, generate a relevant "
#                 "and specific question that can be answered using this content:\n\n"
#                 f"Text: {reconstructed_text}\n\n"
#                 "Generate a clear, focused question that can be answered using the provided text."
#             )

#             # Generate question
#             question = llm_model_generate(question_prompt)
            
#             # Store question with its source chunk IDs and context
#             generated_questions.append({
#                 "question": question,
#                 "sourceChunks": chunk_ids,
#                 "context": reconstructed_text,
#                 "chunkIds": chunk_ids  # Make sure these IDs match the original chunks
#             })

#         except Exception as e:
#             logger.error(f"Error generating question for text chunk {index}: {e}")
#             continue

#     return generated_questions

def generate_questions_from_texts(text_related_to_chunk_id):
    num_texts_to_select = min(3, len(text_related_to_chunk_id))
    start_indices = random.sample(range(len(text_related_to_chunk_id)), num_texts_to_select)
    generated_questions = []

    for index in start_indices:
        try:
            # Reconstruct text context and preprocess for LLM
            reconstructed_text = text_related_to_chunk_id[index]
            if index + 1 < len(text_related_to_chunk_id):
                reconstructed_text += " " + text_related_to_chunk_id[index + 1]
                
            # Preprocess text before sending to LLM
            preprocessed_text = preprocess_text_for_llm(reconstructed_text)
            
            question_prompt = (
                "Based on the following text from a document, generate a relevant "
                "and specific question that can be answered using this content:\n\n"
                f"Text: {preprocessed_text}\n\n"
                "Generate a clear, focused question that can be answered using the provided text."
            )

            # Generate question
            question = llm_model_generate(question_prompt)
            
            # Store question with its source chunk IDs and context
            generated_questions.append({
                "question": question,
                "sourceChunks": [index, index + 1] if index + 1 < len(text_related_to_chunk_id) else [index],
                "context": preprocessed_text,
                "chunkIds": [index] if index + 1 >= len(text_related_to_chunk_id) else [index, index + 1]
            })

        except Exception as e:
            logger.error(f"Error generating question for text chunk {index}: {e}")
            continue

    return generated_questions


@app.post("/api/v1/suggestions")
async def suggestions(chunks: chunks):
    try:
        questions = generate_questions_from_texts(chunks.chunks)
        
        # If no questions generated, return empty but valid response
        if not questions:
            logger.warning("No questions generated from the provided chunks")
            return {
                "questions": [],
                "questionData": []
            }
            
        # Extract questions for backward compatibility
        question_texts = [q["question"] for q in questions if q.get("question")]
        
        return {
            "questions": question_texts,
            "questionData": questions
        }
        
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate suggestions: {str(e)}"
        )


def prepare_final_answer(url_mapping, answer):
    def replace_with_anchor(match):
        numbers = match.group(1).split(",")
        anchor_tags = []
        for number in numbers:
            number = number.strip()
            url = url_mapping.get(number, "#")
            # Get the text from result.chunk (which contains the ! marks)
            formatted_text = url[1]  # This assumes 'text' is available in this scope
            anchor_tags.append(f'<span data-link="{url[0]}" data-search="{formatted_text}">{number}</span>')
        return f"[{', '.join(anchor_tags)}]"

    encoded_text = re.sub(r'\[([0-9, ]+)\]', replace_with_anchor, answer)
    return encoded_text

@app.post('/api/v1/aisearch', response_model=SearchResponse)
async def post_aisearch(req: Request, request: SearchRequest, session_id: Optional[str] = Query(None, alias="session-id")):
    user_id = req.headers.get("user-id")
    logger.info(f"Processing AI search request with input: {request.input}")
    logger.info(f"Chunk IDs received: {request.chunkIds}")
    logger.info(f"User input: {request.input}")
    

    # Helper function to update database
    async def update_db_session(user_query: str, system_response: str, session_id: Optional[str] = None):
        try:
            # Only update existing sessions or create new ones for non-error messages
            if session_id and session_id != "null":
                # Update existing session
                app.state.db.update_session_chat(
                    session_name='',
                    user_query=user_query,
                    system_response=system_response,
                    unique_chat_text=user_query + " " + system_response,
                    session_id=session_id,
                    user_id=user_id
                )
                logger.info("Successfully updated existing chat session")
            else:
                # Skip creating new session for error messages
                if "Please upload or select a document" not in system_response:
                    app.state.db.update_session_chat(
                        session_name=user_query,
                        user_query=user_query,
                        system_response=system_response,
                        unique_chat_text=user_query + " " + system_response,
                        user_id=user_id
                    )
                    logger.info("Created new chat session")
                else:
                    logger.info("Skipped creating new session for document upload message")
        except Exception as db_error:
            logger.error(f"Database update failed: {str(db_error)}")

    # Helper function to create error response
    async def create_error_response(message: str, limit: int = 10, results_used: int = 0):
        
        return SearchResponse(
        limit=request.limit or limit,
        relevance=request.relevance,
        resultsUsed=results_used,
        content=message,  # Using the passed message parameter
        citations=[],
        memories=[],
        lastPrompt="",
        chunkids=[],
        chunk=[]
    )

    # Input validation
    if not request.input or not request.input.strip():
        error_msg = "Input cannot be empty"
        await update_db_session(request.input, error_msg)
        logger.warning("Empty input received in search request")
        raise HTTPException(status_code=400, detail=error_msg)

    # Initialize request parameters
    chunkids = request.chunkIds if request.chunkIds else []
    chunk = request.chunks if request.chunks else []
    filtered_results = []
    citations = []
    memories = []
    prompt = ""

    # Early return if no chunk IDs provided
    if not chunkids:
        error_msg="Please upload or select a document to proceed."
        if session_id and session_id != "null":
            await update_db_session(request.input, error_msg, session_id)
        elif session_id == "null":
            await update_db_session(request.input, error_msg)
        else:
            logger.info("No session to update for document upload message")
        
        logger.info("No chunk IDs provided in request")
        return await create_error_response(error_msg)
        

       

    try:
        # Prepare AI search request
        user_filter =f"search.ismatchscoring(title, '{user_id}_*')"
        aiSearchRequest = AISearchRequest(
            search=request.input.strip(),
            semanticConfiguration=settings.aiSearchSemanticConfig,
            filter=user_filter,
            top=request.limit,  # Increase top results to ensure we don't miss relevant ones
            vectorQueries=[
                VectorQuery(
                    text=request.input.strip(),
                    kind="text",
                    fields="text_vector"
                )
            ]
            # select="chunk_id,title,chunk,metadata_storage_path"  # Comma-separated string instead of list
        )

        # Execute search
        results, error = await search(settings, aiSearchRequest)

        logger.info(f"Initial search results count: {len(results) if results else 0}")
        if results:
            logger.info("Sample of initial results:")
            for idx, result in enumerate(results[:3]):  # Log first 3 results
                logger.info(f"Result {idx + 1}:")
                logger.info(f"  Title: {result.title}")
                logger.info(f"  Chunk ID: {result.chunk_id}")
                logger.info(f"  User ID in title: {result.title.split('_')[0]}")
                logger.info(f"  Score: {result.searchRerankerScore}")

        if error:
            error_msg = "I apologize, but I encountered an issue processing your request. Please try asking your question again."
            await update_db_session(request.input, error_msg)
            logger.error(f"Search operation failed: {error}")
            return create_error_response(error_msg)

        result_chunk_ids = [result.chunk_id for result in results]
        # logger.info(f"Chunk IDs from Azure Search: {result_chunk_ids}")
        # Compare cleaned IDs
        allowed_chunk_ids = set(cid.strip().lower() for cid in request.chunkIds)
        result_chunk_ids_cleaned = [cid.strip().lower() for cid in result_chunk_ids]

        logger.info(f"Cleaned Chunk IDs from request: {allowed_chunk_ids}")
        logger.info(f"Cleaned Chunk IDs from Azure Search: {result_chunk_ids_cleaned}")

        # Filter results by chunk IDs
        filtered_results = results
        if chunkids:
            allowed_chunk_ids = set(chunkids)
            filtered_results = [
                result for result in results 
                if result.chunk_id in allowed_chunk_ids 
                and result.title.startswith(f"{user_id}_")  # Double-check user_id in title
            ]

        # Filter results by chunk IDs
        # allowed_chunk_ids = set(request.chunkIds)
        # filtered_results = [result for result in results if result.chunk_id in allowed_chunk_ids]

        logger.info(f"Filtered Results Count: {len(filtered_results)}")
        logger.info(f"Filtered Results: {[res.chunk_id for res in filtered_results]}")

        if not filtered_results:
            error_msg = "I'm sorry, I cannot answer this question based on the available information."
            
            # Only update session if we have a valid session_id
            error_response = SearchResponse(
            limit=request.limit or 10,
            relevance=request.relevance,
            resultsUsed=0,
            content=error_msg,
            citations=[],
            memories=[],
            lastPrompt="",
            chunkids=[],
            chunk=[]
            )
            
            # Update session based on session_id state
            if session_id and session_id != "null":
                await update_db_session(request.input, error_msg, session_id)
            elif session_id == "null":
                await update_db_session(request.input, error_msg)
            else:
                logger.warning("No results found after filtering")

            return error_response

        # Process search results
        ref_dict = {}
        lines = []
        
        for idx, result in enumerate(filtered_results, start=1):
            label = str(idx)
            blob_name = result.title
            container_name = result.metadata_storage_path
            
            # Generate secure SAS URL
            # link = generate_sas_urls(blob_name=blob_name, container_name=container_name)
            link = blob_name.split("_", 1)[-1]
            text = result.chunk
            # sentences = re.split(r'(\. |\? |\*** )', text)  # Splitting on punctuation while keeping it
            sentences = text.split('\n')
            # Format the text with ! marks
            formatted_text = "".join(f"***{s.strip()}***" if s.strip() else s for s in sentences)
            
            citations.append({"label": label, "link": link})
            # cchunk = highlighted_text.replace("\n", " ") + f" {label}"
            raw_text = text.replace("***", "")  # Remove formatting for context
            cchunk = raw_text.replace("\n", " ") + f" {label}"
            ref_dict[label] = [link, text]
            lines.append(f"Source Text: +++\n{cchunk}\n\nDocument Source Location:\n{link}\n+++\n\n")

            # Build memory results
            memories.append(MemoryQueryResult(
                metadata=MemoryRecordMetadata(
                    isReference=False,
                    id=result.chunk_id,
                    text=text,
                    description=result.title,
                    externalSourceName=link,
                    additionalMetadata=""
                ),
                relevance=result.searchRerankerScore,
                embedding=None,
            ))

        # Generate context and prompt
        context = " ".join(lines)
        prompt = generate_template(request, context)

        # Get completion from OpenAI
        try:
            logger.info("Requesting completion from OpenAI")
            completion = oiclient.chat.completions.create(
                model=settings.model,
                messages=[{"role": "user", "content": prompt}],
            )
            
            if not completion.choices:
                logger.error("No completion choices received from OpenAI")
                return SearchResponse(
                    limit=request.limit or 10,
                    relevance=request.relevance,
                    resultsUsed=0,
                    content="I apologize, but I'm having trouble generating a response. Please try asking your question again.",
                    citations=citations,
                    memories=memories,
                    lastPrompt=prompt
                )

            # jbody = json.loads(completion.choices[0].message.content)
            # jbody['response'] = prepare_final_answer(ref_dict, jbody['response'])

            completion_content = completion.choices[0].message.content
            try:
                if '```json' in completion_content:
                    # Remove ```json and ``` markers
                    completion_content = completion_content.replace('```json', '').replace('```', '').strip()
                # First attempt to parse as JSON
                jbody = json.loads(completion_content)
            except json.JSONDecodeError:
                # If JSON parsing fails, create a simple JSON structure
                jbody = {
                    "response": completion_content
                }
            
            # Add logging to help debug
            logger.info(f"Processed response body: {jbody}")
            
            # If response exists in jbody, prepare it, otherwise use the full content
            if 'response' in jbody:
                jbody['response'] = prepare_final_answer(ref_dict, jbody['response'])
            else:
                jbody['response'] = prepare_final_answer(ref_dict, completion_content)

            # Handle empty or uncertain responses
            if not jbody['response'].strip() or "I don't know" in jbody['response']:
                logger.warning("Empty or uncertain response received")
                response = SearchResponse(
                    limit=request.limit,
                    relevance=request.relevance,
                    resultsUsed=len(filtered_results),
                    content="I'm sorry, I cannot answer this question based on the available information.",
                    citations=citations,
                    memories=memories,
                    lastPrompt=prompt
                )
            else:
                # Prepare successful response
                response = SearchResponse(
                    limit=request.limit,
                    relevance=request.relevance,
                    resultsUsed=len(filtered_results),
                    content=jbody['response'],
                    citations=citations,
                    memories=memories,
                    lastPrompt=prompt,
                    chunkids=chunkids,
                    chunk=chunk
                )

            # Update database with chat session
            logger.info("Updating chat session in database")
            try:
                if session_id == "null" or session_id is None:
                    app.state.db.update_session_chat(
                        session_name=request.input,
                        user_query=request.input,
                        system_response=response.content,
                        unique_chat_text=request.input + " " + response.content,
                        user_id = user_id
                    )
                else:
                    app.state.db.update_session_chat(
                        session_name='',
                        user_query=request.input,
                        system_response=response.content,
                        unique_chat_text=request.input + " " + response.content,
                        session_id=session_id,
                        user_id = user_id
                    )
                logger.info("Successfully updated chat session")
            except Exception as db_error:
                logger.error(f"Database update failed: {str(db_error)}")
                print("Database update failed")
                # Continue with response even if database update fails
                
            return response

        except json.JSONDecodeError as json_error:
            error_msg = "Error processing the AI response. Please try again."
            if session_id and session_id != "null":
                await update_db_session(request.input, error_msg, session_id)
            logger.error(f"JSON parsing error: {str(json_error)}")
            raise HTTPException(status_code=500, detail="Error parsing AI response")
            
        except Exception as completion_error:
            error_msg = "I apologize, but I encountered an unexpected error. Please try rephrasing your question or try again."
            if session_id and session_id != "null":
                await update_db_session(request.input, error_msg, session_id)
            logger.error(f"Completion error: {str(completion_error)}")
            return SearchResponse(
                limit=request.limit or 10,
                relevance=request.relevance,
                resultsUsed=len(filtered_results),
                content=error_msg,
                citations=citations,
                memories=memories,
                lastPrompt=prompt
            )

    except Exception as e:
        error_msg = "I apologize, but something went wrong on our end. Please try your request again in a moment."
        if session_id and session_id != "null":
            await update_db_session(request.input, error_msg, session_id)
        logger.error(f"Global error in post_aisearch: {str(e)}")
        return await create_error_response(error_msg)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConnectionError(Exception):
    """Raised when database connection fails"""
    pass

class DatabaseOperationError(Exception):
    """Raised when database operations fail"""
    pass

class DataBaseHandler:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super(DataBaseHandler, cls).__new__(cls)
                    instance._connection = None
                    instance._cursor = None
                    cls._instance = instance
        return cls._instance

    def __init__(self):
        """Initialize connection if not already initialized"""
        try:
            if not self._connection or not self._cursor:
                self._initialize_connection()
        except Exception as e:
            logger.error(f"Failed to initialize connection in __init__: {e}")
            raise DatabaseConnectionError(f"Failed to initialize connection: {e}")


    def create_table_if_not_exists(self):
        """Create the sessionchat table if it doesn't exist and add user_id if missing"""
        # First, create table if it doesn't exist
        create_table_query = """
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='sessionchat' AND xtype='U')
            BEGIN
                CREATE TABLE sessionchat (
                    session_id NVARCHAR(1000),
                    session_name NVARCHAR(MAX) NOT NULL,
                    chat_id NVARCHAR(MAX) NOT NULL,
                    user_query NVARCHAR(MAX) NOT NULL,
                    system_response NVARCHAR(MAX) NOT NULL,
                    timestamp DATETIME DEFAULT GETDATE(),
                    user_id NVARCHAR(100) NULL  -- Added with NULL to allow existing records
                );
            END
        """
        
        # Then, check and add user_id column if it doesn't exist
        alter_table_query = """
            IF NOT EXISTS (
                SELECT * FROM sys.columns 
                WHERE object_id = OBJECT_ID('sessionchat') AND name = 'user_id'
            )
            BEGIN
                ALTER TABLE sessionchat
                ADD user_id NVARCHAR(100) NULL;
            END
        """
        
        try:
            self._cursor.execute(create_table_query)
            self._cursor.execute(alter_table_query)
            self._connection.commit()
            logger.info("Session chat table and user_id column verified/created successfully.")
        except Exception as error:
            # logger.error(f"Failed to update session chat table: {error}")
            raise

    def _initialize_connection(self):
        """Initialize database connection with detailed logging"""
        attempt = 1
        wait_time = 2
        
        while True:
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"Connection Attempt {attempt}")
                logger.info(f"{'='*50}")
                
                # Log connection parameters (be careful not to log actual credentials)
                logger.info(f"Connecting to server: {settings.server}")
                logger.info(f"Database: {settings.database}")
                logger.info(f"Username: {settings.username}")
                logger.info("Connection string being used:")
                conn_string = (
                    f'DRIVER={{ODBC Driver 18 for SQL Server}};'
                    f'SERVER={settings.server};'
                    f'DATABASE={settings.database};'
                    f'UID={settings.username};'
                    f'PWD=********;'  # Password masked for security
                    f'Encrypt=yes;'
                    f'TrustServerCertificate=no;'
                    f'Connection Timeout=30;'
                )
                logger.info(conn_string)
                
                # Clean up any existing connections
                logger.info("Cleaning up existing connections...")
                self._cleanup_connections()
                
                # Create new connection
                logger.info("Attempting to create new connection...")
                try:

                    self._connection = pyodbc.connect(
                        f'DRIVER={{ODBC Driver 18 for SQL Server}};'
                        f'SERVER={settings.server};'
                        f'DATABASE={settings.database};'
                        f'UID={settings.username};'
                        f'PWD={settings.password};'
                        f'Encrypt=yes;'
                        f'TrustServerCertificate=no;'
                        f'Connection Timeout=30;'
                    )
                except Exception as e:
                    logger.warning(f" SQL database connection issue : {e}")
                    raise Exception (" unable to connect the SQL database ")
                
                if not self._connection:
                    raise Exception("Connection object is None after connect call")
                
                logger.info("Connection object created successfully")
                
                # Create cursor
                logger.info("Creating cursor...")
                self._cursor = self._connection.cursor()
                
                if not self._cursor:
                    raise Exception("Cursor object is None after creation")
                
                logger.info("Cursor created successfully")
                
                # Test connection
                logger.info("Testing connection with SELECT 1...")
                self._cursor.execute("SELECT 1")
                result = self._cursor.fetchone()
                
                if not result:
                    raise Exception("No result returned from test query")
                
                if result[0] != 1:
                    raise Exception(f"Unexpected test query result: {result[0]}")
                
                logger.info("Connection test successful")
                
                # Create table if needed
                logger.info("Creating/verifying table structure...")
                self.create_table_if_not_exists()
                
                logger.info(f"Database connection fully initialized after {attempt} attempts")
                logger.info(f"{'='*50}\n")
                return True
                
            except Exception as error:
                logger.error(f"\nConnection attempt {attempt} failed")
                logger.error(f"Error type: {type(error).__name__}")
                logger.error(f"Error message: {str(error)}")
                
                # Get detailed error info if available
                if hasattr(error, 'args'):
                    logger.error(f"Error args: {error.args}")
                
                # Clean up failed connection
                self._cleanup_connections()
                
                # Log wait time and increment attempt counter
                logger.info(f"Waiting {wait_time} seconds before next attempt...")
                logger.info(f"{'='*50}\n")
                
                time.sleep(wait_time)
                attempt += 1

    def _ensure_connection(self):
        """Ensure database connection with detailed logging"""
        try:
            if self._cursor is None or self._connection is None:
                logger.warning("Connection or cursor is None, initializing new connection")
                self._initialize_connection()
                return

            logger.info("Testing existing connection...")
            self._cursor.execute("SELECT 1")
            result = self._cursor.fetchone()
            
            if not result or result[0] != 1:
                logger.warning("Connection test failed, reinitializing...")
                self._initialize_connection()
            else:
                logger.info("Connection test successful")
                
        except Exception as e:
            logger.error(f"Connection check failed: {str(e)}")
            self._initialize_connection()


    def _cleanup_connections(self):
        """Clean up database connections"""
        if hasattr(self, '_cursor') and self._cursor:
            try:
                self._cursor.close()
            except Exception as e:
                logger.warning(f"Error closing cursor: {e}")
            finally:
                self._cursor = None
            
        if hasattr(self, '_connection') and self._connection:
            try:
                self._connection.close()
            except Exception as e:
                logger.warning(f"Error closing connection: {e}")
            finally:
                self._connection = None

    def _ensure_connection(self):
        """Ensure the database connection is alive and reconnect if necessary"""
        try:
            # First check if connection objects exist
            if not self._connection or not self._cursor:
                logger.warning("Connection or cursor is None, initializing new connection")
                self._initialize_connection()
                return

            # Test if connection is alive
            try:
                self._cursor.execute("SELECT 1")
                result = self._cursor.fetchone()
                if not result or result[0] != 1:
                    raise DatabaseConnectionError("Connection test failed")
            except Exception:
                logger.warning("Connection test failed, reinitializing connection")
                self._initialize_connection()
                
        except Exception as e:
            logger.error(f"Error in _ensure_connection: {e}")
            self._initialize_connection()

    def get_checksum_value(self, session_name):
        """Generate MD5 checksum for session name"""
        try:
            md5_hash = hashlib.md5()
            md5_hash.update(session_name.encode('utf-8'))
            return md5_hash.hexdigest()
        except Exception as error:
            # logger.error(f"Error generating checksum: {error}")
            raise

    def update_session_chat(self, session_name, user_query, system_response, unique_chat_text, user_id, session_id=None):
        """Update session chat with new messages including user_id"""
        self._ensure_connection()
        try:
            cur_time_stamp = datetime.now()
            
            if not session_id:
                session_id = self.get_checksum_value(session_name + cur_time_stamp.strftime("%Y-%m-%d %H:%M:%S"))
            else:
                session_name_qry = "SELECT DISTINCT session_name FROM sessionchat WHERE session_id = ?"
                result = self._cursor.execute(session_name_qry, (session_id,)).fetchone()
                if result:
                    session_name = result[0]
                else:
                    raise ValueError(f"No session found with session_id: {session_id}")

            chat_id = self.get_checksum_value(unique_chat_text)
            
            insert_query = """
                INSERT INTO sessionchat (session_id, session_name, chat_id, user_query, system_response, timestamp, user_id)
                VALUES (?, ?, ?, ?, ?, ?, ?);
            """
            self._cursor.execute(insert_query, (session_id, session_name, chat_id, user_query, system_response, cur_time_stamp, user_id))
            self._connection.commit()
            # logger.info(f"Session chat updated successfully for session_id: {session_id}, user_id: {user_id}")
            
        except Exception as error:
            # logger.error(f"Error updating session chat: {error}")
            raise



    def get_all_sessions_chat(self, user_id):
        """Get all chat sessions for a specific user"""
        if not user_id:
            raise ValueError("User ID is required")
        
        while True:
            try:
                self._ensure_connection()
                query = "SELECT session_id, session_name FROM sessionchat WHERE user_id = ? ORDER BY timestamp ASC"
                
                results = self._cursor.execute(query, (user_id,)).fetchall()
                
                # Since we modified the query to include timestamp, we need to return just the fields we want
                return [(row[0], row[1]) for row in results]  # Return only session_id and session_name
                
            except Exception as error:
                logger.error(f"Error fetching sessions for user {user_id}: {error}")
                self._initialize_connection()

    def delete_session_chat(self, session_id):
        """Delete a specific chat session"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self._ensure_connection()
                count_query = "SELECT COUNT(*) FROM sessionchat WHERE session_id = ?"
                delete_query = "DELETE FROM sessionchat WHERE session_id = ?"
                
                row_count = self._cursor.execute(count_query, (session_id,)).fetchone()[0]
                self._cursor.execute(delete_query, (session_id,))
                self._connection.commit()
                
                logger.info(f"Deleted {row_count} records for session_id: {session_id}")
                return row_count
                
            except Exception as error:
                retry_count += 1
                logger.error(f"Attempt {retry_count}: Error deleting session chat: {error}")
                
                if retry_count >= max_retries:
                    raise DatabaseOperationError(f"Failed to delete session after {max_retries} attempts: {error}")
                
                time.sleep(1)

    def update_session_name(self, session_id, new_session_name):
        """Update session name"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # self._ensure_connection()
                update_query = "UPDATE sessionchat SET session_name = ? WHERE session_id = ?"
                self._cursor.execute(update_query, (new_session_name, session_id))
                self._connection.commit()
                
                if self._cursor.rowcount > 0:
                    logger.info(f"Session name updated successfully for session_id: {session_id}")
                    return {"response": "Session name updated successfully"}
                else:
                    logger.warning(f"No session found with session_id: {session_id}")
                    return {"response": "No session found with the given ID"}
                
            except Exception as error:
                retry_count += 1
                logger.error(f"Attempt {retry_count}: Error updating session name: {error}")
                
                if retry_count >= max_retries:
                    raise DatabaseOperationError(f"Failed to update session name after {max_retries} attempts: {error}")
                
                time.sleep(1)

    def get_unique_session_records(self, session_id):
        """Get unique session records"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self._ensure_connection()
                query = "SELECT DISTINCT session_id, session_name FROM sessionchat WHERE session_id = ?"
                result = self._cursor.execute(query, (session_id,)).fetchone()
                
                if result:
                    return {
                        "response": {
                            "session_id": result[0],
                            "session_name": result[1]
                        }
                    }
                return {"response": "Session not found"}
                
            except Exception as error:
                retry_count += 1
                logger.error(f"Attempt {retry_count}: Error fetching unique session records: {error}")
                
                if retry_count >= max_retries:
                    raise DatabaseOperationError(f"Failed to fetch session records after {max_retries} attempts: {error}")
                
                time.sleep(1)

    def get_all_session_details(self, session_id):
        """Get all session details"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                self._ensure_connection()
                query = """
                    SELECT session_name, chat_id, user_query, system_response, timestamp
                    FROM sessionchat
                    WHERE session_id = ?
                    ORDER BY timestamp ASC
                """
                result = self._cursor.execute(query, (session_id,)).fetchall()
                
                if result:
                    session_data = {
                        session_id: [
                            {
                                "session_name": row[0],
                                "chat_id": row[1],
                                "user_query": row[2],
                                "system_response": row[3],
                                "timestamp": row[4].strftime("%Y-%m-%d %H:%M:%S")
                            }
                            for row in result
                        ]
                    }
                    return {"response": session_data}
                return {"response": "No details found for the given session ID"}
                
            except Exception as error:
                retry_count += 1
                logger.error(f"Attempt {retry_count}: Error fetching session details: {error}")
                
                if retry_count >= max_retries:
                    raise DatabaseOperationError(f"Failed to fetch session details after {max_retries} attempts: {error}")
                
                time.sleep(1)

    def __del__(self):
        """Cleanup database connections"""
        self._cleanup_connections()

@app.get("/api/v1/sessions")
async def get_all_sessions_with_names(request: Request):
    """API to fetch all sessions with their session_id and session_name for a specific user."""
    user_id = request.headers.get("user-id")

    if not user_id:
        return JSONResponse(
            status_code=400,
            content={"response": "User ID is required in headers"}
        )

    try:
        # Ensure db handler exists
        if not hasattr(app.state, 'db') or not app.state.db:
            logger.error("Database handler not initialized")
            return JSONResponse(
                status_code=500,
                content={"response": "Database connection not initialized"}
            )
            
        sessions = app.state.db.get_all_sessions_chat(user_id)
        
        if not sessions:
            return {"response": []}
        
        session_list = [
            {"session_id": row[0], "session_name": row[1]} 
            for row in sessions
        ]
        
        return {"response": session_list}

    except DatabaseConnectionError as e:
        logger.error(f"Database connection error: {e}")
        return JSONResponse(
            status_code=503,
            content={"response": "Database connection error, please try again"}
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return JSONResponse(
            status_code=500,
            content={"response": "An unexpected error occurred"}
        )

# Initialize database handler with proper error handling
try:
    logger.info("Initializing database handler...")
    db_handler = DataBaseHandler()
    # Test connection immediately
    db_handler._ensure_connection()
    app.state.db = db_handler
    logger.info("Database handler successfully initialized")
except Exception as e:
    logger.critical(f"Failed to initialize database handler: {e}")
    raise


@app.get("/api/v1/sessions/{session_id}")
def get_unique_sessions(session_id: str):
    """
    API to fetch a particular session's information based on session_id.
    """
    logger.info(f"Fetching unique session details for session_id: {session_id}.")
    try:
        response = app.state.db.get_unique_session_records(session_id)
        logger.info(f"Successfully fetched session details for session_id: {session_id}.")
        return response
    except Exception as e:
        logger.error(f"Error occurred while fetching session details: {e}")
        return {"response": f"Encountered Error: {e}"}

@app.get("/api/v1/sessions/details/{session_id}")
def get_session_details(session_id: str):
    """
    API to fetch all details of a session by session_id.
    """
    logger.info(f"Fetching details for session_id: {session_id}.")
    try:
        response = app.state.db.get_all_session_details(session_id)
        logger.info(f"Successfully fetched details for session_id: {session_id}.")
        return response
    except Exception as e:
        logger.error(f"Error occurred while fetching session details: {e}")
        return {"response": f"Encountered Error: {e}"}

@app.put("/api/v1/sessions/update_session_name/{session_id}")
def update_session_name(session_id: str, new_session_name: str):
    """
    API to update the session name for a given session_id.
    """
    logger.info(f"Updating session name for session_id: {session_id}.")
    try:
        response = app.state.db.update_session_name(session_id, new_session_name)
        logger.info(f"Successfully updated session name for session_id: {session_id}.")
        return response
    except Exception as e:
        logger.error(f"Error occurred while updating session name: {e}")
        return {"response": f"Encountered Error: {e}"}

@app.delete("/api/v1/session_chats/delete")
def delete_sessions(request: Request,session_id: str):
    """
    API to delete all chats for a specific session_id.
    """
    logger.info(f"Deleting all chats for session_id: {session_id}.")
    try:
        rows = app.state.db.delete_session_chat(session_id)
        logger.info(f"Successfully deleted {rows} records for session_id: {session_id}.")
        return {"response": f"Success. Deleted {rows} records"}
    except Exception as e:
        logger.error(f"Error occurred while deleting session chats: {e}")
        return {"response": f"Encountered Error: {e}"}


class NewSessionResponse(BaseModel):
    session_id: str
    timestamp: str

@app.post("/api/v1/new-session", response_model=NewSessionResponse)
async def create_new_session(request: Request):
    """
    Creates a new session with a unique session ID.
    Returns the session ID and timestamp.
    """
    user_id = request.headers.get("user-id", "jay.reddy")
    
    try:
        # Generate current timestamp
        timestamp = datetime.now()
        
        # Create a unique string combining user_id and timestamp
        unique_string = f"{user_id}_{timestamp.strftime('%Y%m%d%H%M%S%f')}"
        
        # Generate MD5 hash for the session ID
        md5_hash = hashlib.md5()
        md5_hash.update(unique_string.encode('utf-8'))
        session_id = md5_hash.hexdigest()
        
        return NewSessionResponse(
            session_id=session_id,
            timestamp=timestamp.strftime("%Y-%m-%d %H:%M:%S")
        )
        
    except Exception as e:
        logger.error(f"Error creating new session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create new session: {str(e)}")
    

from fastapi.responses import JSONResponse
import mimetypes
import base64

@app.get("/api/v1/documents/{file_name}")
async def get_document(
    file_name: str,
    req: Request
):
    try:
        # Get user ID from headers
        user_id = req.headers.get("user-id")
        if not user_id:
            return JSONResponse(
                content={
                    "filename": None,
                    "content_type": None,
                    "base64_data": None
                }
            )

        # Construct the full blob name with user_id prefix
        blob_name = f"{user_id}_{file_name}"

        try:
            # Get blob client
            blob_service_client = get_blob_service_client()
            blob_client = blob_service_client.get_container_client(container_name1).get_blob_client(blob_name)

            # Get blob properties and content
            properties = blob_client.get_blob_properties()
            download_stream = blob_client.download_blob()
            blob_data = download_stream.readall()
            
            # Get content type
            content_type = properties.content_settings.content_type
            if not content_type:
                content_type, _ = mimetypes.guess_type(file_name)
                content_type = content_type or 'application/octet-stream'

            # Encode to base64
            base64_encoded = base64.b64encode(blob_data).decode('utf-8')
            
            return JSONResponse(
                content={
                    "filename": file_name,
                    "content_type": content_type,
                    "base64_data": base64_encoded
                }
            )

        except Exception as e:
            logger.error(f"Error accessing blob: {str(e)}")
            return JSONResponse(
                content={
                    "filename": None,
                    "content_type": None,
                    "base64_data": None
                }
            )

    except Exception as e:
        logger.error(f"Error retrieving document: {str(e)}")
        return JSONResponse(
            content={
                "filename": None,
                "content_type": None,
                "base64_data": None
            }
        )

@app.post("/api/v1/feedback")
def read_item(request: FeedbackRequest):
    if request.id is None or request.id == "":
        raise fastapi.HTTPException(status_code=400, detail="id is required")
    if request.status is None or request.status == "":
        raise fastapi.HTTPException(
            status_code=400, detail="status is required")
    if request.messages is None or len(request.messages) == 0:
        raise fastapi.HTTPException(
            status_code=400, detail="messages is required")

    entity = {}
    entity['PartitionKey'] = 'fb'
    entity['RowKey'] = request.id
    entity['status'] = request.status
    entity['messages'] = ''.join(
        [f"{m.role}: {m.content}\n" for m in request.messages])
    table_manager.upsert_entity(entity)


local_folder = os.path.dirname(os.path.abspath(__file__))
static_foler = os.path.join(local_folder, 'static')
# print(static_foler)
app.mount("/", StaticFiles(directory=static_foler, html=True), name="static")


@app.api_route('/api/v1/status', methods=["GET"])
def read_root():
    return {'status': 'healthy'}


if __name__ == "__main__":
    uvicorn.run(app)
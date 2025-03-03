# Add these imports at the top if not already present
import pandas as pd
import openpyxl
import csv
import docx2txt
from io import BytesIO, StringIO

# Update container variables
container_name = settings.azure_blob_container_name_1  # "docs" - Single container for all file types

# Create a second search index for PDF/Word documents
pdf_word_search_client = SearchClient(
    endpoint=settings.ai_search_endpoint,
    index_name=settings.aiSearchdocIndexName,  # New index for PDF/Word
    credential=DefaultAzureCredential()
)

# Excel/CSV index - use existing search_client or create a new one
excel_csv_search_client = SearchClient(
    endpoint=settings.ai_search_endpoint,
    index_name=settings.aiSearchtableIndexName,  # New index for Excel/CSV
    credential=DefaultAzureCredential()
)

# Function to determine file type and process accordingly
def determine_file_type(file_content_type, file_name):
    """Determine file type based on content type and file extension"""
    file_extension = file_name.split('.')[-1].lower()
    
    if file_content_type in ["application/pdf"] or file_extension == "pdf":
        return "pdf"
    elif file_content_type in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                               "application/msword"] or file_extension in ["doc", "docx"]:
        return "word"
    elif file_content_type in ["application/vnd.ms-excel", 
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"] or file_extension in ["xls", "xlsx"]:
        return "excel"
    elif file_content_type == "text/csv" or file_extension == "csv":
        return "csv"
    elif file_content_type == "text/plain" or file_extension == "txt":
        return "text"
    else:
        return "unknown"

# Function to process Excel files
async def process_excel(file_content, blob_name, debug=False):
    """Process Excel files and index in the Excel/CSV search index"""
    try:
        if debug:
            logging.info(f"Processing Excel file: {blob_name}")
        
        # Read Excel file
        excel_data = BytesIO(file_content)
        df = pd.read_excel(excel_data)
        
        # Generate a parent ID for all chunks from this file
        parent_id = str(uuid4())
        documents = []
        
        # Process each row as a separate document
        for idx, row in df.iterrows():
            # Convert row to JSON string
            row_json = row.to_json()
            
            # Generate embedding for the row
            embedding = generate_embedding(row_json)
            
            # Create document for search index
            document = {
                "chunk_id": str(uuid4()),
                "parent_id": parent_id,
                "chunk": row_json,
                "title": blob_name,
                "metadata_storage_path": container_name,
                "start_char": idx,  # Use row index as position
                "end_char": idx,
                "text_vector": embedding,
                "file_type": "excel"
            }
            documents.append(document)
        
        # Upload documents to search index in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            excel_csv_search_client.upload_documents(documents=batch)
        
        if debug:
            logging.info(f"Successfully indexed {len(documents)} rows from Excel file: {blob_name}")
        
        return len(documents)
    
    except Exception as e:
        logging.error(f"Error processing Excel file {blob_name}: {str(e)}")
        raise

# Function to process CSV files
async def process_csv(file_content, blob_name, debug=False):
    """Process CSV files and index in the Excel/CSV search index"""
    try:
        if debug:
            logging.info(f"Processing CSV file: {blob_name}")
        
        # Read CSV file
        csv_text = file_content.decode('utf-8')
        
        # Use PapaParse for more robust parsing
        import Papa from 'papaparse'  # For client-side, or use Python's csv module
        csv_data = Papa.parse(csv_text, { header: true, dynamicTyping: true })
        
        # Or use Python's CSV module
        csv_data = BytesIO(file_content)
        df = pd.read_csv(csv_data)
        
        # Generate a parent ID for all chunks from this file
        parent_id = str(uuid4())
        documents = []
        
        # Process each row as a separate document
        for idx, row in df.iterrows():
            # Convert row to JSON string
            row_json = row.to_json()
            
            # Generate embedding for the row
            embedding = generate_embedding(row_json)
            
            # Create document for search index
            document = {
                "chunk_id": str(uuid4()),
                "parent_id": parent_id,
                "chunk": row_json,
                "title": blob_name,
                "metadata_storage_path": container_name,
                "start_char": idx,  # Use row index as position
                "end_char": idx,
                "text_vector": embedding,
                "file_type": "csv"
            }
            documents.append(document)
        
        # Upload documents to search index in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            excel_csv_search_client.upload_documents(documents=batch)
        
        if debug:
            logging.info(f"Successfully indexed {len(documents)} rows from CSV file: {blob_name}")
        
        return len(documents)
    
    except Exception as e:
        logging.error(f"Error processing CSV file {blob_name}: {str(e)}")
        raise

# Function to process Word documents
async def process_word(file_content, blob_name, debug=False):
    """Process Word documents and index in the document search index"""
    try:
        if debug:
            logging.info(f"Processing Word document: {blob_name}")
        
        # Extract text from Word document
        text = docx2txt.process(BytesIO(file_content))
        
        # Split text into chunks
        chunks = split_text_into_chunks(text)
        
        if not chunks:
            raise ValueError("No valid chunks generated from document")
        
        # Index chunks with document index
        indexed_count = await index_chunks_in_document_index(chunks, blob_name, debug)
        
        if indexed_count == 0:
            raise ValueError("No chunks were successfully indexed")
        
        if debug:
            logging.info(f"Successfully processed and indexed {indexed_count} chunks from Word document")
        
        return indexed_count
        
    except Exception as e:
        logging.error(f"Error processing Word document {blob_name}: {str(e)}")
        raise

# Function to process text files
async def process_text(file_content, blob_name, debug=False):
    """Process text files and index in the document search index"""
    try:
        if debug:
            logging.info(f"Processing text file: {blob_name}")
        
        # Extract text from file
        text = file_content.decode('utf-8')
        
        # Split text into chunks
        chunks = split_text_into_chunks(text)
        
        if not chunks:
            raise ValueError("No valid chunks generated from document")
        
        # Index chunks with document index
        indexed_count = await index_chunks_in_document_index(chunks, blob_name, debug)
        
        if indexed_count == 0:
            raise ValueError("No chunks were successfully indexed")
        
        if debug:
            logging.info(f"Successfully processed and indexed {indexed_count} chunks from text file")
        
        return indexed_count
        
    except Exception as e:
        logging.error(f"Error processing text file {blob_name}: {str(e)}")
        raise

# New function to index document chunks in pdf_word_search_client
async def index_chunks_in_document_index(chunks, blob_name, debug):
    """Index text chunks in the document search index"""
    try:
        updated_documents = []
        parent_id = str(uuid4())
        
        # Increase batch size and process embeddings in parallel
        batch_size = 20
        
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
                                "text_vector": embedding,
                                "file_type": "document" # Indicates PDF or Word
                            }
                            batch_documents.append(document)
                        except Exception as e:
                            logging.error(f"Error processing chunk: {str(e)}")
                            continue
                
                if batch_documents:
                    # Upload batch to document search index
                    pdf_word_search_client.upload_documents(documents=batch_documents)
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
        
        logging.info(f"Successfully indexed {len(updated_documents)} chunks from {blob_name} in document index")
        return len(updated_documents)
        
    except Exception as e:
        logging.error(f"Error in index_chunks_in_document_index: {str(e)}")
        raise

# Update the upload_file_to_blob function to use the new file processing flow
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
        # Read file content once
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Determine file type
        file_type = determine_file_type(file.content_type, file_name)
        
        if file_type == "unknown":
            raise HTTPException(
                status_code=400, 
                detail="Unsupported file type. Please upload PDF, Word, Excel, CSV, or text files."
            )

        # Prepare blob storage parameters
        modified_file_name = f"{user_id}_{file_name}"
        content_settings = ContentSettings(content_type=file.content_type or "application/octet-stream")

        # Execute upload to the single container
        async def upload_to_blob():
            blob_client = blob_service_client.get_blob_client(container=container_name, blob=modified_file_name)
            await asyncio.to_thread(
                blob_client.upload_blob,
                file_content,
                overwrite=True,
                content_settings=content_settings,
                max_concurrency=4
            )
            return f"https://{blob_service_client.account_name}.blob.core.windows.net/{container_name}/{modified_file_name}"

        # Process file based on type
        async def process_file():
            if file_type == "pdf":
                return await process_blob(file_content, modified_file_name, debug)
            elif file_type == "word":
                return await process_word(file_content, modified_file_name, debug)
            elif file_type == "excel":
                return await process_excel(file_content, modified_file_name, debug)
            elif file_type == "csv":
                return await process_csv(file_content, modified_file_name, debug)
            elif file_type == "text":
                return await process_text(file_content, modified_file_name, debug)
            else:
                return 0

        # Run upload and processing in parallel
        upload_task = asyncio.create_task(upload_to_blob())
        process_task = asyncio.create_task(process_file())

        # Wait for both tasks to complete
        blob_url, indexed_count = await asyncio.gather(upload_task, process_task)

        return {
            "url": blob_url,
            "name": file_name,
            "chunks_indexed": indexed_count,
            "file_type": file_type
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        logging.error(f"Error in upload_file_to_blob: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
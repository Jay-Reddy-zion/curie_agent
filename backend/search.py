async def search_across_indexes(search_text, user_id, file_type=None, limit=10):
    """
    Search across both indexes based on user's query and filter by file type if specified
    
    Args:
        search_text (str): User's search query
        user_id (str): User ID for filtering results
        file_type (str, optional): Filter by file type ('document', 'excel', 'csv')
        limit (int): Maximum number of results to return
    
    Returns:
        list: Combined search results from both indexes
    """
    try:
        # Prepare user filter
        user_filter = f"search.ismatchscoring(title, '{user_id}_*')"
        
        # Add file type filter if specified
        if file_type:
            if file_type in ['pdf', 'word', 'text']:
                # For PDF/Word searches, use the document index with file_type filter
                file_filter = f"file_type eq 'document'"
                user_filter = f"{user_filter} and {file_filter}"
                
                # Search only in document index
                search_request = AISearchRequest(
                    search=search_text.strip(),
                    semanticConfiguration=settings.aiSearchSemanticConfig,
                    filter=user_filter,
                    top=limit,
                    vectorQueries=[
                        VectorQuery(
                            text=search_text.strip(),
                            kind="text",
                            fields="text_vector"
                        )
                    ]
                )
                doc_results, doc_error = await search(settings, search_request, client=pdf_word_search_client)
                return doc_results, doc_error
                
            elif file_type in ['excel', 'csv']:
                # For Excel/CSV searches, use the tabular data index with file_type filter
                file_filter = f"file_type eq '{file_type}'"
                user_filter = f"{user_filter} and {file_filter}"
                
                # Search only in tabular data index
                search_request = AISearchRequest(
                    search=search_text.strip(),
                    semanticConfiguration=settings.aiSearchSemanticConfig,
                    filter=user_filter,
                    top=limit,
                    vectorQueries=[
                        VectorQuery(
                            text=search_text.strip(),
                            kind="text",
                            fields="text_vector"
                        )
                    ]
                )
                tabular_results, tabular_error = await search(settings, search_request, client=excel_csv_search_client)
                return tabular_results, tabular_error
        
        # If no file type specified, search across both indexes
        # Search in document index
        doc_search_request = AISearchRequest(
            search=search_text.strip(),
            semanticConfiguration=settings.aiSearchSemanticConfig,
            filter=user_filter,
            top=limit // 2,  # Split the limit between the two indexes
            vectorQueries=[
                VectorQuery(
                    text=search_text.strip(),
                    kind="text",
                    fields="text_vector"
                )
            ]
        )
        
        # Search in tabular data index
        tabular_search_request = AISearchRequest(
            search=search_text.strip(),
            semanticConfiguration=settings.aiSearchSemanticConfig,
            filter=user_filter,
            top=limit // 2,  # Split the limit between the two indexes
            vectorQueries=[
                VectorQuery(
                    text=search_text.strip(),
                    kind="text",
                    fields="text_vector"
                )
            ]
        )
        
        # Execute searches in parallel
        doc_task = asyncio.create_task(search(settings, doc_search_request, client=pdf_word_search_client))
        tabular_task = asyncio.create_task(search(settings, tabular_search_request, client=excel_csv_search_client))
        
        # Wait for both tasks to complete
        (doc_results, doc_error), (tabular_results, tabular_error) = await asyncio.gather(doc_task, tabular_task)
        
        # Handle errors
        if doc_error and tabular_error:
            return None, "Error searching both indexes"
        
        # Combine results
        combined_results = []
        if doc_results:
            combined_results.extend(doc_results)
        if tabular_results:
            combined_results.extend(tabular_results)
            
        # Sort by relevance score
        combined_results.sort(key=lambda x: x.searchRerankerScore, reverse=True)
        
        # Limit results
        combined_results = combined_results[:limit]
        
        return combined_results, None
        
    except Exception as e:
        logging.error(f"Error searching across indexes: {str(e)}")
        return None, str(e)

# Update the post_aisearch function to use the unified search
@app.post('/api/v1/aisearch', response_model=SearchResponse)
async def post_aisearch(req: Request, request: SearchRequest, session_id: Optional[str] = Query(None, alias="session-id"), file_type: Optional[str] = Query(None)):
    user_id = req.headers.get("user-id")
    logger.info(f"Processing AI search request with input: {request.input}")
    logger.info(f"Chunk IDs received: {request.chunkIds}")
    logger.info(f"User input: {request.input}")
    logger.info(f"File type filter: {file_type}")
    

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
            content=message,
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
        # Use the unified search function
        results, error = await search_across_indexes(
            search_text=request.input.strip(),
            user_id=user_id,
            file_type=file_type,
            limit=request.limit or 10
        )

        logger.info(f"Search results count: {len(results) if results else 0}")
        
        if error:
            error_msg = "I apologize, but I encountered an issue processing your request. Please try asking your question again."
            await update_db_session(request.input, error_msg)
            logger.error(f"Search operation failed: {error}")
            return await create_error_response(error_msg)

        # Rest of the function remains the same as in the original code...
        # Filter results, process them, and generate the response
        
        # Continue with the existing code to process search results...
        # [rest of the post_aisearch function]

    except Exception as e:
        error_msg = "I apologize, but something went wrong on our end. Please try your request again in a moment."
        if session_id and session_id != "null":
            await update_db_session(request.input, error_msg, session_id)
        logger.error(f"Global error in post_aisearch: {str(e)}")
        return await create_error_response(error_msg)a
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex, 
    SimpleField, 
    SearchableField, 
    SearchFieldDataType, 
    VectorSearch, 
    VectorSearchProfile,
    VectorSearchAlgorithmConfiguration,
    SemanticSearch,
    SemanticConfiguration,
    SemanticPrioritizedFields,
    SemanticField,
    VectorSearchAlgorithmKind
)

def create_search_indexes():
    """
    Create or update the search indexes for different file types
    """
    try:
        # Create a search index client
        index_client = SearchIndexClient(
            endpoint=settings.ai_search_endpoint,
            credential=DefaultAzureCredential()
        )
        
        # Create the document index (PDF/Word/Text)
        document_index = SearchIndex(
            name="document-index",
            fields=[
                SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="parent_id", type=SearchFieldDataType.String),
                SearchableField(name="chunk", type=SearchFieldDataType.String),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SimpleField(name="metadata_storage_path", type=SearchFieldDataType.String),
                SimpleField(name="start_char", type=SearchFieldDataType.Int32),
                SimpleField(name="end_char", type=SearchFieldDataType.Int32),
                SimpleField(name="text_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
                         dimensions=1536, vector_search_configuration="vector-config"),
                SimpleField(name="file_type", type=SearchFieldDataType.String)
            ],
            vector_search=VectorSearch(
                algorithms=[
                    VectorSearchAlgorithmConfiguration(
                        name="vector-config",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="vector-config"
                    )
                ]
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name="semantic-config",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[SemanticField(field_name="chunk")]
                        )
                    )
                ]
            )
        )
        
        # Create the tabular data index (Excel/CSV)
        tabular_index = SearchIndex(
            name="tabular-data-index",
            fields=[
                SimpleField(name="chunk_id", type=SearchFieldDataType.String, key=True),
                SimpleField(name="parent_id", type=SearchFieldDataType.String),
                SearchableField(name="chunk", type=SearchFieldDataType.String),
                SearchableField(name="title", type=SearchFieldDataType.String),
                SimpleField(name="metadata_storage_path", type=SearchFieldDataType.String),
                SimpleField(name="start_char", type=SearchFieldDataType.Int32),
                SimpleField(name="end_char", type=SearchFieldDataType.Int32),
                SimpleField(name="text_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), 
                         dimensions=1536, vector_search_configuration="vector-config"),
                SimpleField(name="file_type", type=SearchFieldDataType.String)
            ],
            vector_search=VectorSearch(
                algorithms=[
                    VectorSearchAlgorithmConfiguration(
                        name="vector-config",
                        kind=VectorSearchAlgorithmKind.HNSW,
                        parameters={
                            "m": 4,
                            "efConstruction": 400,
                            "efSearch": 500,
                            "metric": "cosine"
                        }
                    )
                ],
                profiles=[
                    VectorSearchProfile(
                        name="vector-profile",
                        algorithm_configuration_name="vector-config"
                    )
                ]
            ),
            semantic_search=SemanticSearch(
                configurations=[
                    SemanticConfiguration(
                        name="semantic-config",
                        prioritized_fields=SemanticPrioritizedFields(
                            title_field=SemanticField(field_name="title"),
                            content_fields=[SemanticField(field_name="chunk")]
                        )
                    )
                ]
            )
        )
        
        # Create or update the indexes
        try:
            index_client.create_index(document_index)
            logging.info("Document index created successfully")
        except ResourceExistsError:
            index_client.create_or_update_index(document_index)
            logging.info("Document index updated successfully")
            
        try:
            index_client.create_index(tabular_index)
            logging.info("Tabular data index created successfully")
        except ResourceExistsError:
            index_client.create_or_update_index(tabular_index)
            logging.info("Tabular data index updated successfully")
            
        logging.info("Search indexes created/updated successfully")
        return True
        
    except Exception as e:
        logging.error(f"Error creating search indexes: {str(e)}")
        raise

# Add this function to your startup code
@app.on_event("startup")
async def startup_event():
    try:
        # Initialize database handler
        try:
            logger.info("Initializing database handler...")
            db_handler = DataBaseHandler()
            db_handler._ensure_connection()
            app.state.db = db_handler
            logger.info("Database handler successfully initialized")
        except Exception as e:
            logger.critical(f"Failed to initialize database handler: {e}")
            # Continue even if database initialization fails
        
        # Create search indexes
        try:
            logging.info("Creating search indexes...")
            create_search_indexes()
        except Exception as e:
            logging.error(f"Failed to create search indexes: {e}")
            # Continue even if index creation fails
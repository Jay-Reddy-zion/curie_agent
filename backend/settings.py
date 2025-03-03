import os

class Settings:
    def __init__(self):
        # Azure OpenAI configurations
        self.Endpoint = "https://tqzccxbjw7nvu-openai.openai.azure.com/"
        self.model = "gpt-4o"
        self.version = "2024-08-01-preview"

        # AI Search configurations
        self.aiSearchdocIndexName = "document-index"
        self.aiSearchtableIndexName = "tabular-data-index"
        self.aiSearchApiVersion = "2024-05-01-preview"
        self.ai_search_endpoint = "https://datafusionaikt-tqzccxbjw7nvu-aisearch.search.windows.net"
        self.aiSearchEndpoint = self.ai_search_endpoint + "/indexes/" + self.aiSearchdocIndexName + "/docs/search?api-version=" + self.aiSearchApiVersion
        self.aiSearchSemanticConfig = "vector-health-semantic-configuration"

        # Load secrets from environment variables (DO NOT hardcode them)
        self.azure_storage_connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.azure_storage_account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
        self.azure_blob_container_name = os.getenv("AZURE_BLOB_CONTAINER_NAME")
        self.blob_connection_string = os.getenv("BLOB_CONNECTION_STRING")

        self.server = os.getenv("DB_SERVER")
        self.database = os.getenv("DB_NAME")
        self.username = os.getenv("DB_USERNAME")
        self.password = os.getenv("DB_PASSWORD")

settings = None

def get_settings():
    global settings
    if settings is None:
        settings = Settings()
    return settings

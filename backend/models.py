from typing import Any, List, Optional, Dict
from pydantic import BaseModel, Field


class VectorQuery(BaseModel):
    text: str
    kind: str = "text"
    fields: str = "text_vector"


class AISearchRequest(BaseModel):
    search: str
    vectorQueries: List[VectorQuery]
    semanticConfiguration: str
    top: int = 3
    queryType: str = "semantic"
    select: str = "chunk_id,parent_id,chunk,title,metadata_storage_path"
    queryLanguage: str = "en-US"
    


class Message(BaseModel):
    role: str
    content: str


class SearchRequest(BaseModel):
    input: str
    messages: List[Message]
    limit: int = 3
    relevance: float = 0.75
    maxTokens: int = 500
    temperature: float = 0.3
    chunkIds: Optional[List[str]] = None
    chunks: Optional[List[str]] = None
    session_id: Optional[str] = None

# class SearchRequest(BaseModel):
#     input: str
#     messages: List[Message]
#     limit: int = 3
#     relevance: float = 0.75
#     maxTokens: int = 500
#     temperature: float = 0.3
#     chunkIds: Optional[List[str]] = None
#     chunks: Optional[List[str]] = None
    

class Citation(BaseModel):
    label: str
    link: str

class SearchResponse(BaseModel):
    limit: int
    relevance: float
    resultsUsed: int
    content: str
    citations: list[Citation] 
    memories: List[Any]
    lastPrompt: str
    chunkids: Optional[List[str]] = []
    chunk: Optional[List[str]] = []


class ChunkInfo(BaseModel):
    searchScore: float = Field(alias="@search.score")
    searchRerankerScore: float = Field(alias="@search.rerankerScore")
    chunk_id: str
    parent_id: str
    chunk: str
    title: str
    metadata_storage_path: str



class AISearchResult(BaseModel):
    value: List[ChunkInfo]


class MemoryRecordMetadata(BaseModel):
    isReference: bool
    id: str
    text: str
    description: str
    externalSourceName: str
    additionalMetadata: str



class MemoryQueryResult(BaseModel):
    metadata: MemoryRecordMetadata
    relevance: float
    embedding: List[float] | None


class FeedbackRequest(BaseModel):
    id: str
    status: str
    messages: list[Message]

class FileList(BaseModel):
    files: List[str]

class DeleteFileRequest(BaseModel):
    filename: str
    debug: bool

class chunks(BaseModel):
    chunks : list[str]

class ChunkResponse(BaseModel):     
    file_filter: List[str]     
    text_related_to_chunk_id: List[str] 

class FileListResponse(BaseModel):
    files: List[str]
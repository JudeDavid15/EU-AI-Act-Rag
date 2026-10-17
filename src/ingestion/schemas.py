from typing import List,Optional
from pydantic import BaseModel,Field


class ChildChunk(BaseModel):
    chunk_id:str = Field(..., description="Unique ID: e.g. art_12_p1")
    parent_id:str = Field(...,description="ID of parent article: e.g. art_12")
    article_number:str = Field(...,description="Article Number: e.g. Article 12")
    title:str = Field(...,description="Title of the Article")
    text:str = Field(...,description="Text Content of the specific paragraph/clause")

class ParentChunk(BaseModel):
    parent_id:str = Field(...,description="Unique ID: e.g. art_12")
    article_number:str = Field(...,description="e.g. Article 12")
    title:str = Field(...,description="Title of the article")
    full_text:str = Field(...,description="Full combined text of the article")
    children:List[ChildChunk] = Field(default_factory=list)

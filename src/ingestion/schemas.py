from typing import List,Optional
from pydantic import BaseModel,Field


class ChildChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique child chunk ID")
    parent_id: str = Field(..., description="Parent article ID")
    article_number: str = Field(..., description="Article number")
    title: str = Field(..., description="Title of the article")
    text: str = Field(..., description="Text of the clause")

    paragraph_number: Optional[str] = Field(
        default=None,
        description="Number of the legal paragraph, when identifiable",
    )
    citation_label: Optional[str] = Field(
        default=None,
        description="Legal citation label, such as Article 12(2)",
    )
class ParentChunk(BaseModel):
    parent_id:str = Field(...,description="Unique ID: e.g. art_12")
    article_number:str = Field(...,description="e.g. Article 12")
    title:str = Field(...,description="Title of the article")
    full_text:str = Field(...,description="Full combined text of the article")
    children:List[ChildChunk] = Field(default_factory=list)

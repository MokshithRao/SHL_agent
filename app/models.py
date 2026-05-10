from typing import List, Literal
from pydantic import BaseModel, ConfigDict, Field

class ChatMessage(BaseModel):
    """Represents a single message in the chat conversation."""
    model_config = ConfigDict(from_attributes=True)

    role: Literal["user", "assistant", "system"] = Field(
        ...,
        description="The role of the message sender.",
        examples=["user"]
    )
    content: str = Field(
        ...,
        min_length=1,
        description="The text content of the message.",
        examples=["I need an assessment for software engineers."]
    )

class RecommendationItem(BaseModel):
    """Represents a specific assessment recommendation from the catalog."""
    model_config = ConfigDict(from_attributes=True)

    name: str = Field(
        ...,
        min_length=1,
        description="The name of the recommended assessment.",
        examples=["Software Engineering Cognitive Test"]
    )
    url: str = Field(
        ...,
        min_length=1,
        description="The URL to the assessment details or booking page.",
        examples=["https://shl.com/assessments/se-cognitive"]
    )
    test_type: str = Field(
        ...,
        min_length=1,
        description="The type of the assessment (e.g., 'Cognitive', 'Personality', 'Skills').",
        examples=["Cognitive"]
    )

class ChatResponse(BaseModel):
    """Represents the agent's response to the user's message."""
    model_config = ConfigDict(from_attributes=True)

    reply: str = Field(
        ...,
        min_length=1,
        description="The conversational reply from the agent.",
        examples=["Based on your needs, I recommend the Software Engineering Cognitive Test."]
    )
    recommendations: List[RecommendationItem] = Field(
        default_factory=list,
        description="A list of recommended assessments based on the user's input.",
        examples=[
            [
                {
                    "name": "Software Engineering Cognitive Test",
                    "url": "https://shl.com/assessments/se-cognitive",
                    "test_type": "Cognitive"
                }
            ]
        ]
    )
    end_of_conversation: bool = Field(
        default=False,
        description="Flag indicating whether the conversation has reached a natural conclusion.",
        examples=[False]
    )

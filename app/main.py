import logging
from contextlib import asynccontextmanager
from typing import List, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from app.agent import SHLConversationalAgent
from app.models import ChatMessage, ChatResponse

# Set up concise logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global agent instance
agent: SHLConversationalAgent | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager to handle startup and shutdown events.
    Initializes the agent globally once to avoid rebuilding the retriever per request.
    """
    global agent
    logger.info("Starting up: Initializing SHLConversationalAgent...")
    try:
        agent = SHLConversationalAgent()
        logger.info("SHLConversationalAgent initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize agent: {e}")
        raise e
    yield
    logger.info("Shutting down Application...")

# Initialize FastAPI App
app = FastAPI(
    title="SHL Conversational Assessment Recommendation Agent",
    description="A conversational AI agent recommending SHL assessments based on your hiring requirements.",
    version="1.0.0",
    lifespan=lifespan
)

# ----------------------------------------------------
# Exception Handling
# ----------------------------------------------------

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler to ensure we safely handle unhandled errors,
    log them, and return a clean HTTP 500 without exposing stack traces.
    """
    logger.error(f"Unhandled exception processing {request.method} {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected internal server error occurred."}
    )

# ----------------------------------------------------
# Pydantic Request Models
# ----------------------------------------------------

class ChatRequest(BaseModel):
    """
    Payload model for the /chat endpoint.
    """
    messages: List[ChatMessage]

# ----------------------------------------------------
# Endpoints
# ----------------------------------------------------

@app.get("/")
def read_root():
    """
    Optional root endpoint offering a quick sanity check of the service.
    """
    logger.info("Root endpoint requested")
    return {
        "message": "SHL Conversational Assessment Recommendation Agent API",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def handle_chat(request: ChatRequest):
    """
    Main chat endpoint to process user messages and return recommendations.
    """
    logger.info("Processing incoming chat request.")

    if not agent:
        raise RuntimeError("Conversational agent was not initialized properly.")

    # Convert Pydantic models to dicts to pass to the agent
    messages_payload = [msg.model_dump() for msg in request.messages]

    # Delegate the heavy lifting to the global agent
    agent_output = agent.chat(messages_payload)

    # Returning parsed output, validated by FastAPI against ChatResponse
    return ChatResponse(**agent_output)

# ----------------------------------------------------
# Main Execution Block
# ----------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

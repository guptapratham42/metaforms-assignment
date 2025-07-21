import json
import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from openai import OpenAI

from extraction_service import perform_extraction
from models import ComplexityEstimate, ExtractResponse
from utils import estimate_complexity, get_model_for_complexity

# Configure logger
type_ = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=type_, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY not set in environment")
    raise RuntimeError("OPENAI_API_KEY not set in environment")

# Initialize OpenAI client and FastAPI app
client = OpenAI(api_key=OPENAI_API_KEY)
app = FastAPI(
    title="JSON Extraction API",
    description="Extract structured JSON data from unstructured text using AI",
    version="1.0.0",
)


@app.post("/extract", response_model=ExtractResponse)
async def extract(
    schema_file: UploadFile,
    text_file: UploadFile,
) -> ExtractResponse:
    """
    Extract structured JSON data from unstructured text.

    This endpoint has been refactored to reduce cognitive complexity.
    All business logic is delegated to focused, single-purpose functions.
    """
    return await perform_extraction(client, schema_file, text_file)


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    logger.info("Health check invoked")
    return {"status": "ok"}


@app.get("/complexity-estimate", response_model=ComplexityEstimate)
async def estimate_schema_complexity(schema_file: UploadFile) -> ComplexityEstimate:
    """Analyze schema complexity without processing text."""
    try:
        schema_bytes = await schema_file.read()
        schema = json.loads(schema_bytes.decode("utf-8"))
        complexity = estimate_complexity(schema)

        return ComplexityEstimate(
            complexity=complexity,
            recommended_model=get_model_for_complexity(complexity, 10000),
            processing_notes={
                "high_nesting": complexity["max_depth"] > 5,
                "many_objects": complexity["nested_objects"] > 50,
                "many_literals": complexity["literals"] > 100,
            },
        )
    except Exception as e:
        logger.error("Error analyzing schema complexity: %s", e)
        raise HTTPException(
            status_code=400, detail=f"Failed to analyze schema: {e}"
        ) from e

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AssessmentItem(BaseModel):
    """Pydantic model representing an SHL assessment item."""
    model_config = ConfigDict(extra="ignore")  # Ignore unexpected fields

    entity_id: str = Field(..., description="Unique identifier for the assessment")
    name: str = Field(..., description="Name of the assessment")
    link: Optional[str] = Field(default=None, description="URL link to the assessment details")
    description: Optional[str] = Field(default=None, description="Detailed description")
    job_levels: Optional[List[str]] = Field(default_factory=list, description="Target job levels")
    languages: Optional[List[str]] = Field(default_factory=list, description="Available languages")
    duration: Optional[str] = Field(default=None, description="Expected time duration")
    keys: Optional[List[str]] = Field(default_factory=list, description="Key skills or attributes")
    remote: Optional[bool] = Field(default=None, description="Whether it can be taken remotely")
    adaptive: Optional[bool] = Field(default=None, description="Whether the test is adaptive")


def clean_text(text: Optional[str]) -> str:
    """
    Cleans whitespace and newlines from a text safely.

    Args:
        text (Optional[str]): The string to clean.

    Returns:
        str: The cleaned string, with no redundant spaces or newlines.
    """
    if not text:
        return ""
    # Remove newlines, tabs, and collapse multiple spaces into a single space
    cleaned = re.sub(r'\s+', ' ', text)
    return cleaned.strip()


def assessment_to_document(assessment: AssessmentItem) -> str:
    """
    Converts an assessment object into a consolidated, searchable text document.
    Optimized for semantic retrieval using structured labeled sections and name repetition.

    Args:
        assessment (AssessmentItem): The parsed assessment model.

    Returns:
        str: A searchable text document.
    """
    clean_name = clean_text(assessment.name)

    # Core components tailored for semantic density
    components = [
        f"Assessment Name: {clean_name}",
        f"Assessment Name: {clean_name}"  # Repeated for semantic emphasis
    ]

    if assessment.keys:
        components.append(f"Skills: {', '.join(assessment.keys)}")

    if assessment.job_levels:
        components.append(f"Job Levels: {', '.join(assessment.job_levels)}")

    if assessment.languages:
        components.append(f"Languages: {', '.join(assessment.languages)}")

    if assessment.description:
        components.append(f"Description: {clean_text(assessment.description)}")

    # Use newline instead of pipe for cleaner semantic blocks
    return "\n".join(components)


def load_catalog(filepath: Path | str) -> Tuple[List[AssessmentItem], List[str]]:
    """
    Loads the catalog from a JSON file, validates entries, and generates searchable documents.
    Malformed entries are skipped.

    Args:
        filepath (Path | str): Path to the JSON catalog file.

    Returns:
        Tuple[List[AssessmentItem], List[str]]: A tuple containing a list of validated
        AssessmentItem objects and their corresponding searchable document strings.
    """
    filepath = Path(filepath)
    assessments: List[AssessmentItem] = []
    documents: List[str] = []

    if not filepath.exists():
        logger.error(f"Catalog file not found at: {filepath}")
        return assessments, documents

    logger.info(f"Loading catalog from {filepath}...")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON file {filepath}: {e}")
        return assessments, documents
    except Exception as e:
        logger.error(f"Unexpected error reading {filepath}: {e}")
        return assessments, documents

    # Handle different JSON structures (list or wrapped list)
    if not isinstance(data, list):
        if isinstance(data, dict):
            logger.warning("Top-level JSON is a dictionary. Attempting to extract list values...")
            extracted = False
            for val in data.values():
                if isinstance(val, list):
                    data = val
                    extracted = True
                    break
            if not extracted:
                logger.error("JSON structure is not a list and contains no list. Cannot load catalog.")
                return assessments, documents
        else:
            logger.error("JSON structure is not a list. Cannot load catalog.")
            return assessments, documents

    skipped_count = 0
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            skipped_count += 1
            continue

        try:
            assessment = AssessmentItem(**item)
            document = assessment_to_document(assessment)

            assessments.append(assessment)
            documents.append(document)
        except ValidationError as e:
            # Handle validation errors gracefully
            error_msg = e.errors()[0]['msg']
            logger.warning(f"Validation error for item at index {i}. Skipping. Error: {error_msg}")
            skipped_count += 1
        except Exception as e:
            logger.warning(f"Unexpected error validating item at index {i}: {e}")
            skipped_count += 1

    logger.info(f"Successfully loaded {len(assessments)} assessments. Skipped {skipped_count} malformed entries.")
    return assessments, documents


if __name__ == "__main__":
    # Resolve the data directory based on the location of this script
    # This automatically finds data/shl_product_catalog.json universally
    base_dir = Path(__file__).resolve().parent.parent
    catalog_path = base_dir / "data" / "shl_product_catalog.json"

    assessments, documents = load_catalog(catalog_path)

    print("\n--- Loader Summary ---")
    print(f"Total assessments loaded: {len(assessments)}")
    print(f"Total documents generated: {len(documents)}\n")

    if assessments:
        print("--- Sample Assessment Document ---")
        print(documents[0])
        print("\n--- Sample Assessment Object ---")
        print(assessments[0].model_dump_json(indent=2))
    else:
        print("No assessments were loaded. Please check the catalog path and format.")

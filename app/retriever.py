import logging
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.catalog_loader import AssessmentItem, load_catalog

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class AssessmentRetriever:
    """
    Retrieves SHL assessments based on semantic similarity using FAISS and sentence-transformers.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_store_dir: str = "vector_store",
        index_filename: str = "shl_index.faiss",
        metadata_filename: str = "shl_metadata.pkl",
    ):
        """
        Initializes the AssessmentRetriever.

        Args:
            model_name (str): The name/path of the SentenceTransformers model.
            vector_store_dir (str): Directory where the FAISS index and metadata will be saved.
            index_filename (str): Name of the FAISS index file.
            metadata_filename (str): Name of the metadata file.
        """
        self.model_name = model_name
        
        # Resolve paths
        self.base_dir = Path(__file__).resolve().parent.parent
        self.vector_store_path = self.base_dir / vector_store_dir
        self.vector_store_path.mkdir(parents=True, exist_ok=True)
        
        self.index_file = self.vector_store_path / index_filename
        self.metadata_file = self.vector_store_path / metadata_filename

        logger.info(f"Loading embedding model '{self.model_name}'...")
        self.model = SentenceTransformer(self.model_name)
        
        # State placeholders
        self.index: Optional[faiss.Index] = None
        self.assessments: List[AssessmentItem] = []
        self.documents: List[str] = []

    def build_index(self, catalog_path: Path | str) -> None:
        """
        Loads the catalog, generates embeddings, and creates the FAISS index.

        Args:
            catalog_path (Path | str): Path to the JSON catalog file.
        """
        # Load the assessments and textual documents
        self.assessments, self.documents = load_catalog(catalog_path)
        
        if not self.assessments:
            logger.error("No assessments loaded. Index build failed.")
            return

        logger.info(f"Generating embeddings for {len(self.documents)} documents...")
        
        # Generate embeddings with normalization (for cosine similarity)
        embeddings = self.model.encode(self.documents, normalize_embeddings=True, show_progress_bar=True)
        embeddings = np.array(embeddings).astype("float32")

        # Dimensionality of the embeddings
        d = embeddings.shape[1]
        logger.info(f"Embedding dimension: {d}. Building FAISS IndexFlatIP...")

        # Initialize and populate the FAISS index (Inner Product with normalized vectors computes Cosine Similarity)
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        
        logger.info(f"Successfully built FAISS index with {self.index.ntotal} vectors.")

    def save_index(self) -> None:
        """
        Saves the FAISS index and metadata (assessments, documents) to disk.
        """
        if self.index is None:
            logger.error("No index to save. Please build or load the index first.")
            return

        logger.info(f"Saving FAISS index to {self.index_file}...")
        faiss.write_index(self.index, str(self.index_file))

        logger.info(f"Saving metadata to {self.metadata_file}...")
        metadata = {
            "assessments": self.assessments,
            "documents": self.documents,
        }
        with open(self.metadata_file, "wb") as f:
            pickle.dump(metadata, f)

        logger.info("Index and metadata successfully saved.")

    def load_index(self) -> bool:
        """
        Loads the FAISS index and metadata from disk if they exist.

        Returns:
            bool: True if loaded successfully, False otherwise.
        """
        if not self.index_file.exists() or not self.metadata_file.exists():
            logger.warning("Index file or metadata file not found.")
            return False

        logger.info(f"Loading FAISS index from {self.index_file}...")
        try:
            self.index = faiss.read_index(str(self.index_file))
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False

        logger.info(f"Loading metadata from {self.metadata_file}...")
        try:
            with open(self.metadata_file, "rb") as f:
                metadata = pickle.load(f)
            self.assessments = metadata.get("assessments", [])
            self.documents = metadata.get("documents", [])
        except Exception as e:
            logger.error(f"Failed to load metadata: {e}")
            return False

        if self.index.ntotal != len(self.assessments):
            logger.error("Mismatch between number of vectors in index and metadata length.")
            return False

        logger.info("Successfully loaded FAISS index and metadata.")
        return True

    def normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalizes internal FAISS scores (cosine similarity ranges roughly from -1 to 1)
        to a 0-1 range or a percentage for easier interpretation.

        Args:
            scores (np.ndarray): The raw similarity scores.

        Returns:
            np.ndarray: Normalized similarity scores mapping conceptually to [0.0, 1.0].
        """
        # Cosine similarity is [-1, 1]. Map to [0, 1].
        # (score + 1) / 2
        return (scores + 1.0) / 2.0

    def _tokenize_text(self, text: str) -> Set[str]:
        """
        Tokenizes text into lowercase alphanumeric words.
        Useful for lightweight keyword overlap scoring.
        """
        if not text:
            return set()
        # Extract alphanumeric sequences
        tokens = re.findall(r'\b\w+\b', text.lower())
        return set(tokens)

    def _compute_keyword_overlap(self, query: str, assessment_name: str, document: str) -> float:
        """
        Computes a lightweight keyword overlap score between the query and the assessment.
        Gives stronger weight to exact matches in the assessment name, and partial matches in the document.
        """
        query_tokens = self._tokenize_text(query)
        if not query_tokens:
            return 0.0

        name_tokens = self._tokenize_text(assessment_name)
        doc_tokens = self._tokenize_text(document)

        overlap_score = 0.0
        for q_token in query_tokens:
            if q_token in name_tokens:
                overlap_score += 1.0  # Strong match in name
            elif q_token in doc_tokens:
                overlap_score += 0.5  # Partial match in document body

        # Normalize score bounds conceptually to 0.0 - 1.0 based on query length
        normalized_overlap = min(overlap_score / len(query_tokens), 1.0)
        return normalized_overlap

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Embeds the user query, searches the index, and returns structured results.
        Uses a lightweight hybrid retrieval technique:
        final_score = (semantic_score * 0.8) + (keyword_score * 0.2)

        Args:
            query (str): The search query.
            top_k (int, optional): Number of top results to return. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing match details
                (score, assessment, document) sorted by final score.
        """
        if self.index is None:
            logger.error("Index is not loaded or built. Cannot perform search.")
            return []

        # Generate embedding for the query properly normalized for Inner Product (cosine)
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        query_embedding = np.array(query_embedding).astype("float32")

        # Perform the search (fetch more candidate results for re-ranking)
        candidate_k = min(top_k * 4, len(self.assessments))
        scores, indices = self.index.search(query_embedding, candidate_k)
        
        # Normalize semantic scores
        normalized_scores = self.normalize_scores(scores[0])

        candidates = []
        # scores and indices are 2D arrays, fetch the first vector's results
        for idx, sem_score in zip(indices[0], normalized_scores):
            if idx == -1:  # Indicates not enough items in index
                continue
            
            assessment = self.assessments[idx]
            document = self.documents[idx]
            
            # Compute keyword overlap score
            kw_score = self._compute_keyword_overlap(query, assessment.name, document)
            
            # Hybrid scoring
            final_score = (sem_score * 0.8) + (kw_score * 0.2)
            
            result = {
                "score": float(final_score),
                "semantic_score": float(sem_score),
                "keyword_score": float(kw_score),
                "assessment": assessment,
                "document": document
            }
            candidates.append(result)

        # Sort candidates descending by final_score and take top_k
        candidates = sorted(candidates, key=lambda x: x["score"], reverse=True)
        top_results = candidates[:top_k]

        return top_results

    def pretty_print_results(self, results: List[Dict[str, Any]]) -> None:
        """
        Helper method to nicely format and print retrieval results.

        Args:
            results (List[Dict[str, Any]]): The output from the search() method.
        """
        if not results:
            print("\nNo results found.")
            return

        print(f"\n{'='*50}")
        print(f"Top {len(results)} Retrieval Results")
        print(f"{'='*50}")

        for i, res in enumerate(results, 1):
            assessment: AssessmentItem = res["assessment"]
            
            print(f"\nResult #{i} | Match Score: {res['score']:.4f}")
            print(f"Name: {assessment.name}")
            print(f"Job Levels: {', '.join(assessment.job_levels) if assessment.job_levels else 'N/A'}")
            print(f"Keys: {', '.join(assessment.keys) if assessment.keys else 'N/A'}")
            print(f"Link: {assessment.link if assessment.link else 'N/A'}")
            
            # Print a snippet of the description for brevity
            desc = assessment.description or ""
            desc_snippet = desc[:150] + "..." if len(desc) > 150 else desc
            print(f"Description: {desc_snippet}")
            print("-" * 50)


if __name__ == "__main__":
    # Test block
    retriever = AssessmentRetriever()
    
    # Path to catalog
    catalog_filepath = retriever.base_dir / "data" / "shl_product_catalog.json"
    
    # Attempt to load an existing index to avoid rebuilding
    if not retriever.load_index():
        logger.info("Building a new index from scratch...")
        retriever.build_index(catalog_filepath)
        retriever.save_index()
    else:
        logger.info("Skipped building index because a saved copy was successfully loaded.")
        
    sample_query = "Hiring backend Java developers with Spring and AWS experience"
    logger.info(f"Running sample query: '{sample_query}'")
    
    top_results = retriever.search(sample_query, top_k=5)
    retriever.pretty_print_results(top_results)

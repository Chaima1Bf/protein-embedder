import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
import logging
from typing import Dict
import os
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generates protein embeddings using a pre-trained protein model."""
    
    def __init__(self, model_name: str = "facebook/esm2_t33_650M_UR50D"):
        """Initialize the model and tokenizer for the specified protein model."""
        logger.info(f"Loading protein model {model_name}")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.model.eval()  # Set model to evaluation mode
            self.embedding_stats = {"successful": 0, "failed": 0}
            logger.info(f"Model {model_name} loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise ValueError(f"Failed to load model {model_name}: {str(e)}")

    def generate_embedding(self, sequence: str, max_length: int = 1024) -> np.ndarray:
        """Generate embedding for a single protein sequence."""
        try:
            # Truncate sequence if it exceeds max_length
            if len(sequence) > max_length:
                logger.warning(f"Sequence truncated to {max_length} residues")
                sequence = sequence[:max_length]

            # Tokenize sequence
            inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
            inputs = {key: val.to(self.device) for key, val in inputs.items()}

            # Generate embedding
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling over the sequence dimension for flexibility across models
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            
            logger.debug(f"Generated embedding with dimension {embedding.shape[0]}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.array([])

    def batch_generate_embeddings(self, sequences: Dict[str, str], output_dir: str) -> None:
        """Generate embeddings for all sequences and save them to output_dir."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory {output_dir}")

        for protein_id, sequence in tqdm(sequences.items(), desc="Generating embeddings"):
            if not sequence:
                logger.warning(f"Skipping empty sequence for {protein_id}")
                self.embedding_stats["failed"] += 1
                continue
            embedding = self.generate_embedding(sequence)
            if embedding.size > 0:
                output_path = os.path.join(output_dir, f"{protein_id}_embedding.npy")
                np.save(output_path, embedding)
                self.embedding_stats["successful"] += 1
                logger.info(f"Saved embedding for {protein_id} to {output_path}, dimension: {embedding.shape[0]}")
            else:
                self.embedding_stats["failed"] += 1
                logger.warning(f"Failed to generate embedding for {protein_id}")

        logger.info(f"Embedding generation summary: {self.embedding_stats}")
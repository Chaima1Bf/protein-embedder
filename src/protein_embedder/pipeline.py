import logging
from .processor import DataProcessor
from .embedder import EmbeddingGenerator

# Set up logging
logger = logging.getLogger(__name__)

class ProteinEmbeddingPipeline:
    """Orchestrates the processing of FASTA files and embedding generation."""
    
    def __init__(self, dataset_path: str, output_dir: str, model_name: str = "facebook/esm2_t33_650M_UR50D"):
        """Initialize the pipeline with dataset path, output directory, and model name."""
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.data_processor = DataProcessor(dataset_path)
        self.embedding_generator = EmbeddingGenerator(model_name)

    def run(self) -> None:
        """Run the pipeline to process FASTA files and generate embeddings."""
        logger.info("Starting protein embedding pipeline")
        sequences = self.data_processor.process_dataset()
        self.embedding_generator.batch_generate_embeddings(sequences, self.output_dir)
        
        # Generate summary report
        summary = {
            "total_proteins": self.data_processor.processing_stats["total_proteins"],
            "sequences_extracted": self.data_processor.processing_stats["successful"],
            "sequences_failed": self.data_processor.processing_stats["failed"],
            "embeddings_generated": self.embedding_generator.embedding_stats["successful"],
            "embeddings_failed": self.embedding_generator.embedding_stats["failed"]
        }
        logger.info(f"Pipeline summary: {summary}")
        logger.info("Pipeline completed")
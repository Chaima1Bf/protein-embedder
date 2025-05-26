import pytest
import os
from protein_embedder import DataProcessor, ProteinEmbeddingPipeline

def test_data_processor_initialization(tmp_path):
    processor = DataProcessor(str(tmp_path))
    assert processor.dataset_path == str(tmp_path)
    assert processor.protein_sequences == {}
    assert processor.processing_stats == {"total_proteins": 0, "successful": 0, "failed": 0}

def test_process_dataset_nonexistent_path():
    processor = DataProcessor("/nonexistent/path")
    with pytest.raises(FileNotFoundError):
        processor.process_dataset()

def test_pipeline_initialization(tmp_path):
    pipeline = ProteinEmbeddingPipeline(str(tmp_path), str(tmp_path), model_name="facebook/esm2_t33_650M_UR50D")
    assert pipeline.dataset_path == str(tmp_path)
    assert pipeline.output_dir == str(tmp_path)
    assert isinstance(pipeline.data_processor, DataProcessor)
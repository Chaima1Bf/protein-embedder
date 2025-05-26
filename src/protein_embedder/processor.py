import os
from Bio import SeqIO
import logging
from typing import Dict
from tqdm import tqdm

# Set up logging
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles dataset traversal and sequence extraction from FASTA files."""
    
    def __init__(self, dataset_path: str):
        """Initialize with the path to the dataset directory."""
        self.dataset_path = dataset_path
        self.protein_sequences = {}  # Dictionary to store protein ID -> merged sequence
        self.processing_stats = {"total_proteins": 0, "successful": 0, "failed": 0}

    def process_dataset(self) -> Dict[str, str]:
        """Traverse the dataset directory and extract sequences from FASTA files."""
        if not os.path.exists(self.dataset_path):
            logger.error(f"Dataset path {self.dataset_path} does not exist.")
            raise FileNotFoundError(f"Dataset path {self.dataset_path} does not exist.")

        logger.info(f"Processing dataset at {self.dataset_path}")
        protein_folders = [f for f in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, f))]
        self.processing_stats["total_proteins"] = len(protein_folders)

        for protein_folder in tqdm(protein_folders, desc="Processing protein folders"):
            protein_path = os.path.join(self.dataset_path, protein_folder)
            fasta_file = self._find_fasta_file(protein_path)
            if fasta_file:
                try:
                    sequence = self._extract_and_merge_sequence(fasta_file)
                    if self._validate_sequence(sequence):
                        self.protein_sequences[protein_folder] = sequence
                        self.processing_stats["successful"] += 1
                        logger.info(f"Processed FASTA file for protein {protein_folder}, sequence length: {len(sequence)}")
                    else:
                        self.processing_stats["failed"] += 1
                        logger.warning(f"Invalid or empty sequence for {protein_folder}")
                except Exception as e:
                    self.processing_stats["failed"] += 1
                    logger.error(f"Error processing FASTA file for {protein_folder}: {str(e)}")
            else:
                self.processing_stats["failed"] += 1
                logger.warning(f"No FASTA file found in {protein_folder}")
        
        logger.info(f"Dataset processing summary: {self.processing_stats}")
        return self.protein_sequences

    def _find_fasta_file(self, protein_path: str) -> str:
        """Find the .fasta file in the protein folder."""
        for file in os.listdir(protein_path):
            if file.endswith(".fasta"):
                return os.path.join(protein_path, file)
        return ""

    def _extract_and_merge_sequence(self, fasta_file: str) -> str:
        """Extract and merge sequences from all chains in the FASTA file."""
        sequences = []
        try:
            for record in SeqIO.parse(fasta_file, "fasta"):
                seq = str(record.seq)
                if seq:
                    sequences.append(seq)
                    logger.debug(f"Extracted chain sequence from {fasta_file}: {seq[:20]}... (length: {len(seq)})")
            merged_sequence = "".join(sequences)
            if not merged_sequence:
                logger.warning(f"No sequences found in {fasta_file}")
            return merged_sequence
        except Exception as e:
            logger.error(f"Error parsing FASTA file {fasta_file}: {str(e)}")
            return ""

    def _validate_sequence(self, sequence: str) -> bool:
        """Validate that the sequence is non-empty and contains valid amino acid characters."""
        valid_amino_acids = set("ACDEFGHIKLMNPQRSTVWY")
        if not sequence:
            return False
        if not all(c in valid_amino_acids for c in sequence):
            logger.warning(f"Sequence contains invalid characters: {set(sequence) - valid_amino_acids}")
            return False
        return True
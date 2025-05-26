Protein Embedder
A Python library for processing protein FASTA files and generating embeddings using pre-trained models like ESM2.
Installation
pip install protein-embedder

Usage :

* from protein_embedder import ProteinEmbeddingPipeline

* dataset_path = "path/to/your/dataset"
* output_dir = "path/to/output/embeddings"
* model_name = "facebook/esm2_t33_650M_UR50D"

* pipeline = ProteinEmbeddingPipeline(dataset_path, output_dir, model_name)
* pipeline.run()

Requirements:

* Python >= 3.8
* biopython >= 1.79
* torch >= 1.9.0
* transformers >= 4.20.0
* numpy >= 1.21.0
* tqdm >= 4.62.0

License
MIT License

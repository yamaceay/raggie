# Raggie: Latent Space Trainer, Retriever, and Visualizer
![![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyPI](https://img.shields.io/pypi/v/raggie.svg)](https://pypi.org/project/raggie/)
[![Build Status](https://img.shields.io/github/actions/workflow/status/yamaceay/raggie/ci.yml?branch=main)](https://github.com/yamaceay/raggie/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/yamaceay/raggie/main.svg)](https://codecov.io/gh/yamaceay/raggie)
[![Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://yamaceay.github.io/raggie/)
[![Downloads](https://img.shields.io/pypi/dm/raggie.svg)](https://pypistats.org/packages/raggie)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/license/mit/)
![![uv](https://img.shields.io/badge/uv-1.0.0-blue.svg)](https://pypi.org/project/uv/1.0.0/)

![Raggie Logo](https://raw.githubusercontent.com/yamaceay/raggie/refs/heads/master/assets/logo.png)

Raggie is a Python-based project for training, retrieving, and visualizing sentence embeddings. It provides tools to train a model on paired text data, retrieve relevant documents based on queries, evaluate retrieval performance, and visualize embeddings using t-SNE.

## Features

- **Latent Space Trainer**: Train a Sentence Transformer model with positive and negative pairs of text data.
- **Negative Sampling**: Automatically generate negative samples to improve training.
- **Latent Retriever**: Retrieve the most relevant documents for a given query using FAISS for efficient similarity search.
- **Evaluation**: Evaluate retrieval performance using rank-based metrics.
- **t-SNE Visualization**: Visualize embeddings in 2D space with clustering and annotation support.
- **Customized Data Handling**: Extend or customize data handling by using the abstract classes.

Visit the [Raggie API Documentation](https://yamaceay.github.io/raggie/raggie.html) for more detailed information.

## Project Structure

```
├── raggie/
│   ├── data.py              # Abstract and default data handling
│   ├── model.py             # Model training logic
│   ├── main.py              # Retrieval and evaluation logic
│   ├── utils.py             # Plotting capabilities and t-SNE visualization
│   ├── __init__.py          # High-level API
├── examples/
│   ├── user.py              # Example usage of Raggie
├── data/
│   ├── user_train.jsonl     # Training data in JSONL format
│   ├── user_test.jsonl      # Evaluation data in JSONL format
├── output/                  # Directory for saving trained models and configurations
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project metadata
├── uv.lock                  # Dependency lock file
├── README.md                # Project documentation
```

## Installation

### Recommended: Using `uv`

1. Install `uv` package manager:

    ```bash
    # On Unix-like systems (Linux, macOS)
    curl -LsSf https://astral.sh/uv/install.sh | sh

    # On Windows PowerShell
    powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

    For other installation methods, visit [uv documentation](https://docs.astral.sh/uv/getting-started/installation/).

### For end-users

1. Install the package from PyPI:
   ```bash
   uv pip install raggie
   ```

1. Test the installation by running the example script:
   ```python
   import raggie
   print(raggie.__version__)
   ```

### For developers

1. Clone the repository:
   ```bash
   git clone https://github.com/yamaceay/raggie.git
   cd raggie
   ```

2. Test the example script [examples/user.py](https://github.com/yamaceay/raggie/blob/master/examples/user.py):
   ```bash
   uv run examples/user.py
   ```
    
## Basic Usage

Here's a step-by-step guide to using Raggie for training, retrieving, and visualizing embeddings:

1. First, initialize the core components:
    You can use the `RaggieData` class to load your data and the `RaggieModel` class to work with the model.

    ```python
    from raggie import Raggie, RaggieModel, RaggieData

    # Set up data and model paths
    data = RaggieData(data_dir="data/user")
    model = RaggieModel(output_dir="output/user")
    ```

2. Train the model:
    Raggie requires training data in the form of paired topics and texts. You can use the `RaggieData` class to load your data and the `RaggieModel` class to train the model.

    ```python
    model.train(data)
    raggie = Raggie(model, data)
    ```

3. Perform retrievals in different ways:
    Raggie supports various retrieval methods, including using document queries and keys. Here are some examples.

    Given the following arguments:

    ```python
    key = "Dr. Xandor Quill"
    query = "I am looking for a librarian who has specialized in underwater chess mostly played by dolphins"
    ```

    You can also retrieve entities based on a query:

    ```python
        # Retrieve documents based on a query
        results = raggie.retrieve([query], return_all_scores=True)

        # [('Dr. Xandor Quill', np.float32(0.29358667)), ('Coach Zenith Stormweaver', np.float32(1.3610729)), ('Librarian Pixel Stardust', np.float32(1.4311827)), ('Maestro Quasar Dreamweaver', np.float32(1.5779625)), ('Designer Shadow Prism', np.float32(1.6315038))]
    ```

    In a similar way, you can retrieve entities based on one entity key:

    ```python
        # Find similar documents using a key
        similar_keys = raggie.most_similar(keys=[key], return_all_scores=True)

        # [('Dr. Xandor Quill', np.float32(7.7822574e-13)), ('Dr. Echo Starwhisper', np.float32(1.3583667)), ('Artist Zephyr Clockwork', np.float32(1.3616264)), ('Maestro Quasar Dreamweaver', np.float32(1.5282866)), ('Trainer Nebula Sparksmith', np.float32(1.5660846))]
    ```

    You can also retrieve documents based on a query:

    ```python
        # Find similar documents using content
        similar_docs = raggie.most_similar(queries=[query], return_all_scores=True)

        # [('A deep-sea librarian who invented underwater chess which is now played by dolphins', np.float32(0.29358667)), ('Trains butterflies for underwater marathon racing', np.float32(1.3610729)), ('Catalogs books that write themselves when no one is looking', np.float32(1.4311827)), ('Conducts orchestras of wind-up toys and raindrops', np.float32(1.5779625)), ('Creates video games that can only be played while sleeping', np.float32(1.6315038))]
    ```

    All retrieval methods are set `return_all_scores=False` by default, so that they return keys instead of key - similarity score pairs.

4. Evaluate retrieval performance:
    ```python
    # Check retrieval performance
    rank = raggie.evaluate_rank(query, key)
    # Rank of document for key 'Dr. Xandor Quill': 1
    ```

   The rank is the number of guesses it takes to find the correct document for the given query. A rank of 1 means the document was found on the first try. A high rank might indicate that the query embedding needs improvement.

5. Evaluate and visualize:
    Use the `RaggiePlotter` class to visualize embeddings in 2D space using t-SNE. You can also cluster embeddings and annotate centroids with group names.

    ```python
    # Import for visualization
    from raggie.utils import RaggiePlotter

    # Visualize embeddings
    keys = ["Dr. Xandor Quill", "Coach Zenith Stormweaver", "Librarian Pixel Stardust", … more keys]

    plotter = RaggiePlotter(model)
    plotter.plot(keys, n_clusters=5)
    ```

![Raggie Visualization Example](https://raw.githubusercontent.com/yamaceay/raggie/refs/heads/master/assets/tsne.png)

## Extending Data Handling

The current implementation uses abstract classes for allowing custom functionality in data handling, model training and visualization. You can extend the functionality by implementing your own logic by respecting the interfaces provided in each file.

The types can be imported as follows:

```python
from raggie.types import RaggieDataClass, RaggieModelClass, RaggiePlotterClass
```

The API specification is available in [Raggie API Documentation](https://yamaceay.github.io/raggie/raggie.html).

## Data Preparation

The current project includes example training and evaluation data in the `data` directory:
- `data/user_train.jsonl`: Training data with paired topics and texts.
- `data/user_test.jsonl`: Evaluation data for testing retrieval performance.

Here's how you can bring your own training and evaluation data for your use case:

1. Example JSONL format:
   ```json
   {"key": "topic1", "value": "This is the text for topic 1."}
   {"key": "topic2", "value": "This is the text for topic 2."}
   ```

2. Example CSV format:
   ```csv
   key,value
   topic1,This is the text for topic 1.
   topic2,This is the text for topic 2.
   ```

3. Example JSON format:
   ```json
   [
       {"key": "topic1", "value": "This is the text for topic 1."},
       {"key": "topic2", "value": "This is the text for topic 2."}
   ]
   ```

Store the data in a directory like `data/<use_case>` so that you'll be able to pass the directory path to the data loader class later.

The default data loader automatically detects the files with `_train.<ext>`, `_test.<ext>`, and `_val.<ext>` suffixes, where `<ext>` can be `csv`, `jsonl` or `json`.

## Output Logging

The trained model and related configurations are saved per default in the `output` directory. This includes:
- Model weights (`model.safetensors`)
- Tokenizer configuration (`tokenizer.json`, `vocab.txt`)
- Additional metadata files.

Since the data files are expected to be large, the output directory is not included in the source repository.

## Dependencies

The project requires the following Python libraries:
- `faiss_cpu`
- `numpy`
- `sentence_transformers`
- `torch`
- `datasets`
- `accelerate`
- `matplotlib`
- `seaborn`
- `scikit-learn`

All dependencies are managed using `uv` and listed in `requirements.txt` and `uv.lock`.

## Python Version

This project requires Python 3.12. Ensure you have the correct version installed.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project uses the [Sentence Transformers](https://www.sbert.net/) library for training and embedding generation, and [FAISS](https://github.com/facebookresearch/faiss) for efficient similarity search.

## Thank you for using Raggie!
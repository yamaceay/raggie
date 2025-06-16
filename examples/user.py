from raggie import Raggie, RaggieModel, RaggieData
from raggie.utils import RaggiePlotter

# from raggie.types import RaggieDataClass, RaggieModelClass, RaggiePlotterClass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate a latent space model.")
    parser.add_argument("-d", "--data_dir", type=str, default="data/user", help="Path to data directory.")
    parser.add_argument("-o", "--output_dir", type=str, default="output/user", help="Path to output directory.")
    args = parser.parse_args()

    data = RaggieData(data_dir=args.data_dir)
    model = RaggieModel(output_dir=args.output_dir)
    model.train(data)

    raggie = Raggie(model, data)

    key = "Dr. Xandor Quill"
    query = "I am looking for a librarian who has specialized in underwater chess mostly played by dolphins"

    # Retrieve keys based on values
    results = raggie.retrieve([query], return_all_scores=True)
    print(f"Retrieved keys by '{query}':")
    for result in results[0]:
        if isinstance(result, tuple) and len(result) == 2:
            name, score = result
            print(f"Key: {name}, Score: {score}")
        else:
            name = result
            print(f"Key: {name}")
    print()

    # Retrieve similar keys based on keys
    results = raggie.most_similar(keys=[key], return_all_scores=True)
    print(f"Retrieved keys by '{key}':")
    for result in results[0]:
        if isinstance(result, tuple) and len(result) == 2:
            name, score = result
            print(f"Key: {name}, Score: {score}")
        else:
            name = result
            print(f"Key: {name}")
    print()

    # Retrieve similar documents based on values
    results = raggie.most_similar(queries=[query], return_all_scores=True)
    print(f"Retrieved documents by '{query}':")
    for result in results[0]:
        if isinstance(result, tuple) and len(result) == 2:
            doc, score = result
            print(f"Document: {doc}, Score: {score}")
        else:
            doc = result
            print(f"Document: {doc}")
    print()

    # Evaluate rank of a document
    rank = raggie.evaluate_rank(query, key)
    print(f"Rank of document '{query}' for key '{key}': {rank}")

    # Visualize key embeddings
    plotter = RaggiePlotter(model)
    train_data = data.train_data
    keys = [pair[0] for pair in train_data]
    plotter.plot(keys, n_clusters=5)
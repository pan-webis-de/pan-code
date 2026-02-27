#!/usr/bin/env python3
from pathlib import Path

import click
import pandas as pd
import pyterrier as pt
from tira.third_party_integrations import ir_datasets


def get_index(ir_dataset, index_directory):
    # PyTerrier needs an absolute path
    index_directory = index_directory.resolve().absolute()

    if (
        not index_directory.exists()
        or not (index_directory / "data.properties").exists()
    ):
        indexer = pt.IterDictIndexer(str(index_directory), overwrite=True, meta={"docno": 100, "text": 20480})

        # you can do some custom document processing here
        docs = ({"docno": i.doc_id, "text": i.default_text()} for i in ir_dataset.docs_iter())
        indexer.index(docs)

    return pt.IndexFactory.of(str(index_directory))


def process_dataset(ir_dataset, index_directory, output_directory):
    if (output_directory / "run.txt.gz").exists():
        return

    output_directory.mkdir(exist_ok=True, parents=True)
    index_directory.mkdir(exist_ok=True, parents=True)

    index = get_index(ir_dataset, index_directory)
    bm25 = pt.terrier.Retriever(index, wmodel="BM25")

    # potentially do some query processing
    topics = pd.DataFrame(
        [
            {"qid": i.query_id, "query": i.default_text()}
            for i in ir_dataset.queries_iter()
        ]
    )

    # PyTerrier needs to use pre-tokenized queries
    tokeniser = pt.java.autoclass(
        "org.terrier.indexing.tokenisation.Tokeniser"
    ).getTokeniser()

    topics["query"] = topics["query"].apply(
        lambda i: " ".join(tokeniser.getTokens(i))
    )

    run = bm25(topics)

    pt.io.write_results(run, output_directory / "run.txt.gz")


@click.command()
@click.option("--dataset", type=str, required=True, help="The dataset id or a local directory.")
@click.option("--output", type=Path, required=True, help="The output directory.")
@click.option("--index", type=Path, required=True, help="The index directory.")
def main(dataset, output, index):
    ir_dataset = ir_datasets.load(dataset)
    
    process_dataset(ir_dataset, index, Path(output))


if __name__ == "__main__":
    main()

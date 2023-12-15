import argparse
from pathlib import Path

from datasets import Dataset

from data.gigacorpus import fora, books, split_by_empty_lines, twitter, wiki, kamerstukken_extra, dbnl_extra


GIGACORPUS = {
    'subtitles': (split_by_empty_lines, {'n': 4}),
    'wiki': (wiki, {}),
    'twitter': (twitter, {}),
    'recht': (split_by_empty_lines, {'n': 6}),
    'books': (books, {}),
    'articles': (split_by_empty_lines, {'n': 1}),
    'fora': (fora, {}),
    'dbnl-extra': (dbnl_extra, {}),
    'kamerstukken-extra': (kamerstukken_extra, {}),
}


def main(root, username, dataset, private=True):
    root = Path(root)

    if 'all' in dataset:
        dataset = GIGACORPUS.keys()

    for name in dataset:
        processor, kwargs = GIGACORPUS[name]
        print(f'Processing dataset "{name}"…')

        kwargs |= {'path': root / name}
        hf_dataset = Dataset.from_generator(processor,
                                            gen_kwargs=kwargs)

        hf_dataset = hf_dataset.train_test_split(test_size=1e-3, seed=42)

        # rename test split to validation
        hf_dataset['validation'] = hf_dataset['test']
        del hf_dataset['test']

        hf_dataset.push_to_hub(f'{username}/gigacorpus-nl-{name}', private=private)
    print('Done!')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Gigacorpus-nl datasets into Huggingface datasets.")

    # Mandatory Arguments
    parser.add_argument("--root", type=str, required=True,
                        help="Path to a folder where the Gigacorpus files are extracted")
    parser.add_argument("--username", type=str, required=True, help="Huggingface Hub username")

    # Optional Arguments
    parser.add_argument("--dataset", type=str, nargs="+",
                        choices=list(GIGACORPUS.keys()) + ['all'],
                        default=["all"], help="Datasets to run. Default is 'all' which runs all parsable datasets.")
    parser.add_argument("--public", dest="private", action="store_false",
                        help="If set, the dataset on the Huggingface Hub will be public.")

    args = parser.parse_args()
    main(**vars(args))

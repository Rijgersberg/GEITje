from datasets import concatenate_datasets, load_dataset, DatasetDict, Value


def add_column(ds, name, value):
    return ds.add_column(name, [value] * len(ds))

def main():
    """Assemble the GEITje training set from various sources.

    Number of tokens for each dataset are pre-estimated. Some datasets,
    like Wikipedia, are oversampled (in train only)."""
    dataset = {'train': [],
               'validation': []}

    for split in ('validation', 'train'):
        # subtitles
        subtitles = load_dataset('Rijgersberg/gigacorpus-nl-subtitles', split=split)
        N_tokens = 300_000_000
        N_selected = 100_000_000
        N_docs = int(N_selected / N_tokens * len(subtitles))

        subtitles = subtitles.shuffle(seed=42).select(range(N_docs))

        subtitles = add_column(subtitles, 'source', 'gigacorpus-nl-subtitles')
        subtitles = add_column(subtitles, 'subset', None)
        subtitles = add_column(subtitles, 'id', None)

        dataset[split].append(subtitles)

        # wiki
        wiki = load_dataset('Rijgersberg/gigacorpus-nl-wiki', split=split)
        wiki = add_column(wiki, 'source', 'gigacorpus-nl-wiki')
        wiki = add_column(wiki, 'subset', None)
        wiki = wiki.rename_column('title', 'id')

        dataset[split].append(wiki)
        if split == 'train':  # oversample wiki to 3 epochs for training, but not for validation
            dataset[split].append(wiki)
            dataset[split].append(wiki)

        # twitter
        twitter = load_dataset('Rijgersberg/gigacorpus-nl-twitter', split=split)

        twitter = add_column(twitter, 'source', 'gigacorpus-nl-twitter')
        twitter = add_column(twitter, 'subset', None)
        twitter = add_column(twitter, 'id', None)

        dataset[split].append(twitter)

        # recht
        recht = load_dataset('Rijgersberg/gigacorpus-nl-recht', split=split)
        N_tokens = 2_300_000_000
        N_selected = 250_000_000
        N_docs = int(N_selected / N_tokens * len(recht))

        recht = recht.shuffle(seed=42).select(range(N_docs))

        recht = add_column(recht, 'source', 'gigacorpus-nl-recht')
        recht = add_column(recht, 'subset', None)
        recht = add_column(recht, 'id', None)

        dataset[split].append(recht)

        # books
        books = load_dataset('Rijgersberg/gigacorpus-nl-books', split=split)
        N_tokens = 11_100_000_000
        N_selected = 1_800_000_000
        N_docs = int(N_selected / N_tokens * len(books))

        books = books.shuffle(seed=42).select(range(N_docs))

        books = add_column(books, 'source', 'gigacorpus-nl-books')
        books = add_column(books, 'subset', None)

        dataset[split].append(books)

        # articles
        articles = load_dataset('Rijgersberg/gigacorpus-nl-articles', split=split)

        articles = add_column(articles, 'source', 'gigacorpus-nl-articles')
        articles = add_column(articles, 'subset', None)
        articles = add_column(articles, 'id', None)

        dataset[split].append(articles)
        if split == 'train':  # oversample articles to 3 epochs for training, but not for validation
            dataset[split].append(wiki)
            dataset[split].append(wiki)

        # fora
        fora = load_dataset('Rijgersberg/gigacorpus-nl-fora', split=split)
        N_tokens = 42_500_000_000
        N_selected = 1_000_000_000
        N_docs = int(N_selected / N_tokens * len(fora))

        fora = fora.shuffle(seed=42).select(range(N_docs))

        fora = add_column(fora, 'source', 'gigacorpus-nl-fora')
        fora = fora.rename_column('forum', 'subset')
        fora = fora.remove_columns('title')
        fora = fora.cast_column('id', Value('string'))

        dataset[split].append(fora)

        # dbnl-extra
        dbnl = load_dataset('Rijgersberg/gigacorpus-nl-dbnl-extra', split=split)
        N_tokens = 2_000_000_000
        N_selected = 100_000_000
        N_docs = int(N_selected / N_tokens * len(dbnl))

        dbnl = dbnl.shuffle(seed=42).select(range(N_docs))

        dbnl = add_column(dbnl, 'source', 'gigacorpus-nl-dbnl-extra')
        dbnl = add_column(dbnl, 'subset', None)
        dbnl = add_column(dbnl, 'id', None)

        dataset[split].append(dbnl)

        # kamerstukken-extra
        kamerstukken = load_dataset('Rijgersberg/gigacorpus-nl-kamerstukken-extra', split=split)
        N_tokens = 2_900_000_000
        N_selected = 250_000_000
        N_docs = int(N_selected / N_tokens * len(kamerstukken))

        kamerstukken = kamerstukken.shuffle(seed=42).select(range(N_docs))

        kamerstukken = add_column(kamerstukken, 'source', 'gigacorpus-nl-kamerstukken-extra')
        kamerstukken = add_column(kamerstukken, 'subset', None)
        kamerstukken = add_column(kamerstukken, 'id', None)

        dataset[split].append(kamerstukken)

    # MADLAD-400
    madlad = load_dataset("allenai/madlad-400", "nl", split='clean')

    N_tokens = 115_000_000_000
    validation_size = 1e-3
    N_selected = 4_500_000_000 * (1 + validation_size)
    N_docs = int(N_selected / N_tokens * len(madlad))

    madlad = madlad.shuffle(seed=42).select(range(N_docs))

    def fix_madlad_whitespace(batch):
        return {'text': [doc.replace('\\n', '\n').replace('\\t', '\t')
                         for doc in batch['text']]}

    madlad = madlad.map(fix_madlad_whitespace, batched=True)

    madlad = add_column(madlad, 'source', 'allenai/madlad-400')
    madlad = add_column(madlad, 'subset', 'nl,clean')
    madlad = add_column(madlad, 'id', None)

    madlad = madlad.train_test_split(test_size=validation_size, seed=42)

    dataset['train'].append(madlad['train'])
    dataset['validation'].append(madlad['test'])

    # dataset, assemble!
    dataset = DatasetDict({
        'train': concatenate_datasets(dataset['train']),
        'validation': concatenate_datasets(dataset['validation'])
    }).shuffle(seed=42)

    dataset.push_to_hub('Rijgersberg/GEITje-pretrain-10b', private=True)


if __name__ == '__main__':
    main()

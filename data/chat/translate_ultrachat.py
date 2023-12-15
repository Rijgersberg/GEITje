import datasets

from .translate import translate_conversation


def translate(row):
    return {'messages_nl': translate_conversation(row['messages'])}


if __name__ == '__main__':
    ultrachat = datasets.load_dataset('HuggingFaceH4/ultrachat_200k')

    ultrachat_subset = datasets.DatasetDict({
        'test_sft': ultrachat['test_sft'].shuffle(seed=42).select(range(500)),
        'train_sft': ultrachat['train_sft'].shuffle(seed=42).select(range(9500)),
    })

    ultrachat_nl = ultrachat_subset.map(translate, batched=False, num_proc=10)

    # TODO do afterwards
    ultrachat_nl = ultrachat_nl.filter(
        lambda row: all(turn['content'] != '<TRANSLATION FAILED>'
                        for turn in row['messages_nl']))

    ultrachat_nl.push_to_hub('Rijgersberg/ultrachat_10k_nl', private=True)

import datasets

from translate import translate_conversation


def translate(row):
    return {'messages_nl': translate_conversation(row['messages'])}


if __name__ == '__main__':
    no_robots = datasets.load_dataset('HuggingFaceH4/no_robots')

    no_robots_nl = no_robots.map(translate, batched=False, num_proc=10)
    no_robots_nl.push_to_hub('Rijgersberg/no_robots_nl', private=True)

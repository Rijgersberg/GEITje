from pathlib import Path
from data.gigacorpus.fora import Thread


def fora(path, thread_id='4417a475470c48a6bd0a3f1ca97a2442'):
    def row(buffer, n):
        thread = Thread.from_string(''.join(buffer))

        return {
            'text': str(thread),
            'title': thread.title,
            'forum': thread.forum,
            'id': n
        }
    n = 0
    with open(path, 'r', encoding='utf8') as f:
        buffer = []
        for line in f:
            if thread_id in line and buffer:
                yield row(buffer, n)
                n += 1
                buffer = []

            buffer.append(line)
        yield row(buffer, n)


def books(path, marker='d81a52adf1b443e99610457ee489b5ef'):
    def row(buffer, book_id):
        return {
            'text': ''.join(buffer).strip(),
            'id': book_id,
        }
    with open(path, 'r', encoding='utf8') as f:
        next(f)  # first line is a newline

        buffer = []
        for line in f:
            if marker in line:
                if buffer:
                    yield row(buffer, book_id)
                    buffer = []
                _, book_id = line.strip().split('-')
            else:
                buffer.append(line)
        yield row(buffer, book_id)


def split_by_empty_lines(path, n=1):
    def row(buffer):
        return {'text': ''.join(buffer).strip()}

    buffer = []
    count = 0
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            if line == '\n':
                count += 1
            else:
                if count >= n:
                    yield row(buffer)
                    buffer = []
                count = 0
            buffer.append(line)
        yield row(buffer)


def twitter(path):
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            if text := line.strip():
                yield {'text': text}


def wiki(path):
    def row(buffer, title):
        return {
            'text': f'{title}\n\n' + ''.join(buffer).strip(),
            'title': title
        }

    with open(path, 'r', encoding='utf8') as f:
        next(f)  # file starts with newline
        title = next(f).strip()

        buffer = []
        for line in f:
            try:
                if buffer[:-2] and line != '\n' and buffer[-1] == '\n' and buffer[-2] != '\n':
                    yield row(buffer[:-2], title)
                    title = buffer[-2].strip()
                    buffer = []
                buffer.append(line)
            except Exception as e:
                print(e)
        yield row(buffer, title)


def dbnl_extra(path):
    for book_path in (Path(path) / 'books').glob('*.txt'):
        with open(book_path, 'r', encoding='utf-8') as f:
            yield {'text': f.read()}


def kamerstukken_extra(path):
    for kamerstuk_path in Path(path).glob('*.txt'):
        for encoding in ('utf-8', 'latin-1'):
            try:
                with open(kamerstuk_path, 'r', encoding=encoding) as f:
                    yield {'text': f.read()}
                    break
            except UnicodeDecodeError:
                pass
        else:
            raise ValueError(f"Failed to decode {kamerstuk_path} using any of the provided encodings.")


def law(path):
    raise NotImplementedError


def commoncrawl(path):
    raise NotImplementedError


def dcep(path):
    raise NotImplementedError


def openwebtext(path):
    raise NotImplementedError


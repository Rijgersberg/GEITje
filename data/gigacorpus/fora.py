import locale
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Any
from zoneinfo import ZoneInfo

TIMEZONE = ZoneInfo('Europe/Amsterdam')
locale.setlocale(locale.LC_TIME, 'nl_NL.UTF-8')  # for Dutch strftimes

import logging

# Configure the logging settings
logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')


@dataclass
class Post:
    username: int
    dt: datetime
    something: Optional[Any]
    text: str

    def __str__(self):
        dt_str = self.dt.strftime('%A %d %B %Y, %H:%M:%S')
        return f'Op {dt_str} schreef {self.username}:\n{self.text}'

    @staticmethod
    def parse_header(string):
        string = string.removeprefix('d81a52adf1b443e99610457ee489b5ef-')

        if string.count('-') == 1:
            timestamp, username = string.split('-')
            something = None
        else:
            timestamp, _ = string.split('-', maxsplit=1)
            string = string.removeprefix(f'{timestamp}-')

            something, username = string.rsplit('-', maxsplit=1)

        return int(timestamp), something, int(username)

    @classmethod
    def from_lines(cls, lines):
        try:
            timestamp, something, username = cls.parse_header(lines[0].strip())
        except Exception as e:
            logging.error(lines[0])
            timestamp = 1695278556
            something = None
            username = ''

        dt = datetime.fromtimestamp(int(timestamp), tz=TIMEZONE)
        text = '\n'.join(lines[2:]).strip()

        return cls(username=username, dt=dt, something=something, text=text)


@dataclass
class Thread:
    forum: str
    title: str
    posts: list[Post]

    def __str__(self):
        string = self.title
        for post in self.posts:
            string += f'\n\n\n{str(post)}'
        return string

    @classmethod
    def from_string(cls, string):
        lines = string.strip().splitlines()

        _, forum = lines[0].strip().split('--')
        title = lines[2].strip()

        # magic identifier in Gigacorpus fora, see http://gigacorpus.nl
        post_prefix_id = 'd81a52adf1b443e99610457ee489b5ef'

        posts = []
        post_lines = []
        first_post = True

        for line in lines[4:]:
            if post_prefix_id in line:
                if not first_post:
                    posts.append(Post.from_lines(post_lines))
                first_post = False
                post_lines = [line]
            else:
                post_lines.append(line)
        if post_lines:
            posts.append(Post.from_lines(post_lines))

        return cls(forum=forum, title=title, posts=posts)

from data.gigacorpus.fora import Post

import pytest

@pytest.mark.parametrize(
    'string, timestamp, something, username',
    [
        ('d81a52adf1b443e99610457ee489b5ef-955400400-0-6500545793948755741',
         955400400, '0', 6500545793948755741),
        ('d81a52adf1b443e99610457ee489b5ef-1612260000-15527018103110460100',
         1612260000, None, 15527018103110460100),
        ('d81a52adf1b443e99610457ee489b5ef-1612256400-Malapropisme----5818016069547141337',
         1612256400, 'Malapropisme---', 5818016069547141337),
        ('d81a52adf1b443e99610457ee489b5ef-1612252800--joorgeweggist-5818016069547141337',
         1612252800, '-joorgeweggist', 5818016069547141337),
    ])
def test_post_parse_header(string, timestamp, something, username):
    assert Post.parse_header(string) == (timestamp, something, username)

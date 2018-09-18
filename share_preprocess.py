# -*- coding: utf-8 -*-
#

"""preprocess seq2seq, shared vocab
"""

import argparse
import collections
import io
import progressbar
import re

from utils import count_lines
import smart_open

split_pattern = re.compile(r'([.,!?"\':;)(])')
digit_pattern = re.compile(r'\d')


def split_sentence(s, use_lower):
    if use_lower:
        s = s.lower()
    s = s.replace('\u2019', "'")
    s = digit_pattern.sub('0', s)
    words = []
    for word in s.strip().split():
        words.extend(split_pattern.split(word))
    words = [w for w in words if w]
    return words


def read_file(path, use_lower):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    with io.open(path, encoding='utf-8', errors='ignore') as f:
        for line in bar(f, max_value=n_lines):
            words = split_sentence(line, use_lower)
            yield words


def preprocess_dataset(src_path, tgt_path, src_outpath, tgt_outpath, vocab_path=None, vocab_size=None,
                       use_lower=False, ignore_number=False):
    token_count = 0
    counts = collections.Counter()
    for path, outpath in [(src_path, src_outpath), (tgt_path, tgt_outpath)]:
        with io.open(outpath, 'w', encoding='utf-8') as f:
            for words in read_file(path, use_lower):
                line = ' '.join(words)
                f.write(line)
                f.write('\n')
                if vocab_path is not None:
                    for word in words:
                        if ignore_number and word.isnumeric(): continue
                        counts[word] += 1
                        token_count += len(words)
    print('number of tokens: %d' % token_count)

    if vocab_path and vocab_size:
        vocab = [word for (word, _) in counts.most_common(vocab_size)]
        with io.open(vocab_path, 'w', encoding='utf-8') as f:
            for word in vocab:
                f.write(word)
                f.write('\n')


def main(args):
    preprocess_dataset(
        args.SOURCE_INPUT,
        args.TARGET_INPUT,
        args.SOURCE_OUTPUT,
        args.TARGET_OUTPUT,
        vocab_path=args.vocab_file,
        vocab_size=args.vocab_size,
        use_lower=args.lower,
        ignore_number=args.ignore_numbers
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('SOURCE_INPUT', help='path to input')
    parser.add_argument('TARGET_INPUT', help='path to input')
    parser.add_argument('SOURCE_OUTPUT', help='path to input')
    parser.add_argument('TARGET_OUTPUT', help='path to input')
    parser.add_argument('--vocab-file', help='vocabulary file to save')
    parser.add_argument('--vocab-size', type=int, default=30000,
                        help='vocabulary file to save')
    parser.add_argument('--lower', action='store_true',
                        help='use lower case')
    parser.add_argument('--ignore_numbers', action='store_true',
                        help='make numbers as unkown')
    args = parser.parse_args()

    main(args)

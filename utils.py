"""Utility funtions."""
import copy
import progressbar

import numpy

from chainer import cuda
from chainer.dataset import convert

# speical symbols
PAD = -1
UNK = 0
EOS = 1


def get_subsequence_before_eos(seq, eos=EOS):
    index = numpy.argwhere(seq == eos)
    return seq[:index[0, 0] + 1] if len(index) > 0 else seq


def seq2seq_pad_concat_convert(xy_batch, device):
    """

    Args:
        xy_batch: List of tuple of source and target sentences
        device: Device ID to which an array is sent.

    Returns:
        Tuple of Converted array.

    """
    x_seqs, tx_seqs, y_seqs = zip(*xy_batch)

    x_block = convert.concat_examples(x_seqs, device, padding=-1)
    tx_block = convert.concat_examples(tx_seqs, device, padding=-1)
    y_block = convert.concat_examples(y_seqs, device, padding=-1)
    xp = cuda.get_array_module(x_block)

    x_block = xp.pad(x_block, ((0, 0), (0, 1)),
                     'constant', constant_values=PAD)
    for i_batch, seq in enumerate(x_seqs):
        x_block[i_batch, len(seq)] = EOS

    tx_block = xp.pad(tx_block, ((0, 0), (0, 1)),
                      'constant', constant_values=PAD)
    for i_batch, seq in enumerate(tx_seqs):
        tx_block[i_batch, len(seq)] = EOS

    y_out_block = xp.pad(y_block, ((0, 0), (0, 1)),
                         'constant', constant_values=PAD)
    for i_batch, seq in enumerate(y_seqs):
        y_out_block[i_batch, len(seq)] = EOS

    return (x_block, tx_block, y_out_block)


def count_lines(path):
    with open(path) as f:
        return sum([1 for _ in f])


def load_vocabulary(path):
    with open(path) as f:
        # +2 for UNK and EOS
        word_ids = {line.strip(): i + 2 for i, line in enumerate(f)}
    word_ids['<UNK>'] = UNK
    word_ids['<EOS>'] = EOS
    return word_ids


def read_data(paths):
    words = []
    for path in paths:
        with open(path) as f:
            words.extend([w for line in f for w in line.strip().split()])
    return words


def load_data(vocabulary, path):
    n_lines = count_lines(path)
    bar = progressbar.ProgressBar()
    data = []
    print('loading...: %s' % path)
    with open(path) as f:
        for line in bar(f, max_value=n_lines):
            words = line.strip().split()
            array = numpy.array([vocabulary.get(w, UNK) for w in words], 'i')
            data.append(array)
    return data


def make_vocabulary_with_source_side_unks(source_paths, vocabulary):
    vocab_ids_with_unks = copy.deepcopy(vocabulary)

    sources = read_data(source_paths)

    for word in sources:
        if word not in vocab_ids_with_unks:
            vocab_ids_with_unks[word] = len(vocab_ids_with_unks)

    return vocab_ids_with_unks


def calculate_unknown_ratio(data, unk_threshold):
    unknown = sum((s >= unk_threshold).sum() for s in data)
    total = sum(s.size for s in data)
    return unknown / total

def load_source_target(src_path, tgt_path, vocab):
    """make dataset from source/target files and shared vocab

    Args:
        src_path: filename of source sequences
        tgt_path: filename of target sequences
        vocab: shared vocabulary

    Returns:
        src_data: BoW representation of source with extended vocab
        tgt_data_t: BoW representation of target with extended vocab by source
        tgt_data: BoW representation of traget with unknown
        oov_data: array of Out of Vocab words
    """
    with open(src_path, 'r') as sf, open(tgt_path, 'r') as tf:
        src_data = []
        tgt_data = []
        tgt_data_t = []
        oov_data = []
        for src, tgt in zip(sf, tf):
            # source
            words = src.strip().split()
            art_words, oov = article2ids(words, vocab)
            array = numpy.array(art_words)
            src_data.append(array)
            # target
            words = tgt.strip().split()
            abs_words = abstract2ids(words, vocab, oov)
            array = numpy.array(abs_words)
            tgt_data.append(array)
            new_array = copy.deepcopy(array)
            new_array[new_array > len(vocab)] = UNK
            tgt_data_t.append(new_array)
            oov_data.append(oov)
        return src_data, tgt_data_t, tgt_data, oov_data

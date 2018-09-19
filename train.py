"""Train a seq2seq model."""
import argparse

import numpy
import six

import chainer
from chainer import training
from chainer.training import extensions

from net import Seq2seq
from metrics import CalculateBleu
from utils import seq2seq_pad_concat_convert
from utils import load_vocabulary
from utils import load_data
from utils import make_vocabulary_with_source_side_unks
from utils import calculate_unknown_ratio
import utils

def id2words_with_oov(i, vocab, oov):
    if i >= len(vocab):
        oi = i - len(vocab)
        if oi < len(oov):
            w = oov[oi]
        else:
            w = '<UNK:%d>' % oi
            return w
    else:
        return vocab[i]

def main():
    parser = argparse.ArgumentParser(description='Attention-based NMT')
    parser.add_argument('SOURCE', help='source sentence list')
    parser.add_argument('TARGET', help='target sentence list')
    parser.add_argument('VOCAB', help='vocabulary file')
    parser.add_argument('--validation-source',
                        help='source sentence list for validation')
    parser.add_argument('--validation-target',
                        help='target sentence list for validation')
    parser.add_argument('--batchsize', '-b', type=int, default=128,
                        help='number of sentence pairs in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='number of sweeps over the dataset to train')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--resume', '-r', default='',
                        help='resume the training from snapshot')
    parser.add_argument('--encoder-unit', type=int, default=128,
                        help='number of units')
    parser.add_argument('--encoder-layer', type=int, default=3,
                        help='number of layers')
    parser.add_argument('--encoder-dropout', type=int, default=0.1,
                        help='number of layers')
    parser.add_argument('--decoder-unit', type=int, default=128,
                        help='number of units')
    parser.add_argument('--attention-unit', type=int, default=128,
                        help='number of units')
    parser.add_argument('--maxout-unit', type=int, default=128,
                        help='number of units')
    parser.add_argument('--min-source-sentence', type=int, default=1,
                        help='minimium length of source sentence')
    parser.add_argument('--max-source-sentence', type=int, default=50,
                        help='maximum length of source sentence')
    parser.add_argument('--log-interval', type=int, default=200,
                        help='number of iteration to show log')
    parser.add_argument('--validation-interval', type=int, default=4000,
                        help='number of iteration to evlauate the model '
                        'with validation dataset')
    parser.add_argument('--out', '-o', default='result',
                        help='directory to output the result')
    args = parser.parse_args()

    vocab_ids = load_vocabulary(args.VOCAB)

    train_source, train_target, train_target_s, target_oovs \
        = utils.load_source_target(args.SOURCE, args.TARGET, vocab_ids)

    assert len(train_source) == len(train_target)
    train_data = [(s, t, ts, v) for s, t, ts, v in six.moves.zip(
        train_source, train_target, train_target_s, target_oovs)
                  if args.min_source_sentence <= len(s) <= args.max_source_sentence and
                  args.min_source_sentence <= len(t) <= args.max_source_sentence]

    train_source_unk = calculate_unknown_ratio(
        [s for s, _, _, _ in train_data],
        len(vocab_ids)
    )
    train_target_unk = calculate_unknown_ratio(
        [t for _, t, _, _ in train_data],
        len(vocab_ids)
    )

    print('All vocabulary size: {}'.format(len(vocab_ids)))
    print('Train data size: {}'.format(len(train_data)))
    print('Train source unknown: {0:.2f}'.format(train_source_unk))
    print('Train target unknown: {0:.2f}'.format(train_target_unk))

    vocab_words = {i: w for w, i in vocab_ids.items()}

    model = Seq2seq(
        len(vocab_ids),
        args.encoder_layer, args.encoder_unit,
        args.encoder_dropout, args.decoder_unit,
        args.attention_unit, args.maxout_unit
    )
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()
        model.to_gpu(args.gpu)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train_data, args.batchsize)
    updater = training.StandardUpdater(
        train_iter, optimizer, converter=seq2seq_pad_concat_convert,
        device=args.gpu
    )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'))
    trainer.extend(
        extensions.LogReport(trigger=(args.log_interval, 'iteration'))
    )
    trainer.extend(
        extensions.PrintReport(
            ['epoch', 'iteration', 'main/loss', 'validation/main/loss',
             'main/perp', 'validation/main/perp', 'validation/main/bleu',
             'elapsed_time']
        ),
        trigger=(args.log_interval, 'iteration')
    )

    if args.validation_source and args.validation_target:
        test_source, test_target, test_target_s, test_oovs \
            = utils.load_source_target(args.validation_source,
                                       args.validation_target,
                                       vocab_ids)
        assert len(test_source) == len(test_target)
        test_data = list(
            six.moves.zip(test_source, test_target, test_target_s, test_oovs)
        )
        test_data = [(s, t, ts, v) for s, t, ts, v in test_data
                     if 0 < len(s) and 0 < len(t)]
        test_source_unk = calculate_unknown_ratio(
            [s for s, _, _, _ in test_data],
            len(vocab_ids)
        )
        test_target_unk = calculate_unknown_ratio(
            [t for _, _, t, _ in test_data],
            len(vocab_ids)
        )

        print('Validation data: {}'.format(len(test_data)))
        print('Validation source unknown: {0:.2f}'.format(test_source_unk))
        print('Validation target unknown: {0:.2f}'.format(test_target_unk))

        @chainer.training.make_extension()
        def translate(_):
            source, target, target_s, oovs = seq2seq_pad_concat_convert(
                [test_data[numpy.random.choice(len(test_data))]],
                args.gpu
            )
            result = model.translate(source, oovs)[0].reshape(1, -1)

            source, target, result = source[0], target[0], result[0]

            max_art_oovs = 0
            for v in oovs:
                max_art_oovs = len(v) if len(v) > max_art_oovs else max_art_oovs
            def id2word_oov(i, oov):
                if i >= len(vocab_words):
                    oi = i - len(vocab_words)
                    if oi < len(oov):
                        w = oov[oi]
                    else:
                        w = '<UNK>'
                    return w
                else:
                    return vocab_words[i]

            source_sentence = ' '.join([id2words_with_oov(
                int(x), vocab_words, oovs[0]) for x in source])
            target_sentence = ' '.join([id2words_with_oov(
                int(y), vocab_words, oovs[0]) for y in target])
            result_sentence = ' '.join([id2words_with_oov(
                int(y), vocab_words, oovs[0]) for y in result])
            print('# source : ' + source_sentence)
            print('# result : ' + result_sentence)
            print('# expect : ' + target_sentence)

        trainer.extend(
            translate,
            trigger=(args.validation_interval, 'iteration')
        )

        trainer.extend(
            CalculateBleu(
                model, test_data, device=args.gpu,
                key='validation/main/bleu'
            ),
            trigger=(args.validation_interval, 'iteration')
        )

    print('start training')
    trainer.run()


if __name__ == '__main__':
    main()

import argparse
import sys
import codecs
from operator import itemgetter

import torch

from simple_nmt.data_loader import DataLoader
import simple_nmt.data_loader as data_loader

from simple_nmt.model.transformer import Transformer


def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument('--model_fn', required=True, help='Model file name to use')
    p.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use. -1 for CPU. Default=%(default)s')

    p.add_argument('--batch_size', type=int, default=128, help='Mini batch size for parallel inference. Default=%(default)s')
    p.add_argument('--max_length', type=int, default=255, help='Maximum sequence length for inference. Default=%(default)s')
    
    p.add_argument('--n_best', type=int, default=1, help='Number of best inference result per sample. Default=%(default)s')    
    p.add_argument('--beam_size', type=int, default=5, help='Beam size for beam search. Default=%(default)s')
    # beam search 실행 안할거면 beam_size 1
    p.add_argument('--lang', type=str, default=None, help='Source language and target language. Example: enko')
    p.add_argument('--length_penalty', type=float, default=1.2, help='Length penalty parameter that higher value produce shorter results. Default=%(default)s')

    config = p.parse_args()

    return config


def read_text(batch_size=128):
    # This method gets sentences from standard input and tokenize those.
    lines = []

    sys.stdin = codecs.getreader("utf-8")(sys.stdin.detach())

    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip().split(' ')]

        if len(lines) >= batch_size:
            yield lines
            lines = []

    if len(lines) > 0:
        yield lines


def to_text(indice, vocab):
    # This method converts index to word to show the translation result.
    lines = []

    for i in range(len(indice)):
        line = []
        for j in range(len(indice[i])):
            index = indice[i][j]

            if index == data_loader.EOS:
                # line += ['<EOS>']
                break
            else:
                line += [vocab.itos[index]]     # index to text(string)

        line = ' '.join(line)
        lines += [line]

    return lines        # bpe된 형태로 return



def get_vocabs(train_config, config, saved_data):
    
    src_vocab = saved_data['src_vocab']
    tgt_vocab = saved_data['tgt_vocab']

    return src_vocab, tgt_vocab, False          # load_vocab in data_loader.py



def get_model(input_size, output_size, train_config, is_reverse=False):

    if 'use_transformer' in vars(train_config).keys() and train_config.use_transformer:
        model = Transformer(
            input_size,
            train_config.hidden_size,
            output_size,
            n_splits=train_config.n_splits,
            n_enc_blocks=train_config.n_layers,
            n_dec_blocks=train_config.n_layers,
            dropout_p=train_config.dropout,
        )
    else:
        raise Exception('Set model architecture as Transformer.')


    model.load_state_dict(saved_data['model'])  # Load weight parameters from the trained model.
    model.eval()

    return model


if __name__ == '__main__':
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    config = define_argparser()

    # Load saved model.
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    # Load configuration setting in training.
    train_config = saved_data['config']

    src_vocab, tgt_vocab, is_reverse = get_vocabs(train_config, config, saved_data) # load_vocab in data_loader.py

    # Initialize dataloader, but we don't need to read training & test corpus.
    # What we need is just load vocabularies from the previously trained model.
    loader = DataLoader()
    loader.load_vocab(src_vocab, tgt_vocab) # load_vocab in data_loader.py

    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
    model = get_model(input_size, output_size, train_config, is_reverse)


    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)

    with torch.no_grad():
        # Get sentences from standard input.
        for lines in read_text(batch_size=config.batch_size): # lines: raw text with bpe splited by " "
            # Since packed_sequence must be sorted by decreasing order of length,
            # sorting by length in mini-batch should be restored by original order.
            # Therefore, we need to memorize the original index of the sentence.
            lengths         = [len(line) for line in lines]     # #of tokens in each sample for packed sequence
            original_indice = [i for i in range(len(lines))]    # 문장 순서 뒤바뀔 때 기억하고 있을 original index

            sorted_tuples = sorted(
                zip(lines, lengths, original_indice),           # lenght 기준, 길이가 긴것이 앞으로.
                key=itemgetter(1),
                reverse=True,
            )
            sorted_lines    = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]
            lengths         = [sorted_tuples[i][1] for i in range(len(sorted_tuples))]
            original_indice = [sorted_tuples[i][2] for i in range(len(sorted_tuples))]

            # Converts string to list of index.
            x = loader.src.numericalize(    # tensor (batch_size, length), one-hot index 로 만들기
                loader.src.pad(sorted_lines),       # sorted_lines(text): list of list, 모자란 부분 pad채움
                device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu'
            )

            if config.beam_size == 1:
                y_hats, indice = model.search(x)

                output = to_text(indice, loader.tgt.vocab)  # 길이가 긴 순서대로 sorting된 output 이므로               
                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1)) # 원래 순서대로 sorting
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]   # 튜플에서 원하는 출력만 받아옴

                sys.stdout.write('\n'.join(output) + '\n')  # 미니배치단위로 for문, 현재 iteration 결과받고 다음 iteration결과..
            
            else:
                
                batch_indice, _ = model.batch_beam_search(
                    x,
                    beam_size=config.beam_size,
                    max_length=config.max_length,
                    n_best=config.n_best,
                    length_penalty=config.length_penalty,
                )

                # Restore the original_indice.
                output = []
                for i in range(len(batch_indice)):
                    output += [to_text(batch_indice[i], loader.tgt.vocab)]
                sorted_tuples = sorted(zip(output, original_indice), key=itemgetter(1))
                output = [sorted_tuples[i][0] for i in range(len(sorted_tuples))]

                for i in range(len(output)):
                    sys.stdout.write('\n'.join(output[i]) + '\n')

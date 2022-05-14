import argparse
import os, torch
import numpy as np
from utils import predict_essentiality

device = torch.device("cpu")


parser = argparse.ArgumentParser()
parser.add_argument('--seq_type', type=str, default="protein")
parser.add_argument('--cell_line', type=str, default="HCT-116")
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--input', type=str, default='../data/test_seqs.fasta', help='the input sequence file for predicting essentiality')
parser.add_argument('--save', type=str, default=1, help='the results will be saved at the same path with input file if 1')
parser.add_argument('--visual', type=int, default=0)
args, _ = parser.parse_known_args()
model_path = os.path.join(os.path.abspath(os.path.dirname(os.getcwd())), 'saved_models', args.seq_type, '1230', args.cell_line)



def read_fasta_file(fasta_file, param=''):
    seq_file = open(fasta_file, "r")
    all_seq = seq_file.read()
    seq_file.close()
    seq_dict = {}
    each_seq = all_seq.split('>')
    for each in each_seq:
        if each != "":
            name_seq = each.split('\n')
            # 　print(name_seq)
            seq_dict[name_seq[0]] = param.join(name_seq[1:])
    return seq_dict


def predict():
    # fasta -> dict
    input_file = args.input
    seq_dict = read_fasta_file(input_file)
    # save to txt [num, entry, seq length, essentiality, is essential?]
    if args.save==1:
        save_file = os.path.splitext(input_file)[0]+'_result.txt'
        # print(save_file)
        with open(save_file, 'w') as f:
            f.write('Num\tEntry\tLength\tEssentiality\tIs Essential?\n')
            for i, (entry, seq) in enumerate(seq_dict.items()):
                # get essenlity & seq attention:  [n, max_len, max_len]
                pred_ess, seq_attn = predict_essentiality(seq, model_path, device)
                is_ess = 'YES' if pred_ess>=0.5 else 'NO'
                print(i + 1, entry, len(seq), pred_ess, is_ess)
                f.write('%d\t%s\t%d\t%0.3f\t%s\n' % (i+1, entry, len(seq), pred_ess, is_ess))

    if args.visual:
        # attention visualization
        # heatmap plot to pdf
        pass



if __name__ == '__main__':
    if not os.path.exists(args.input):
        print(f'Can not find the input sequence file of {args.input}.')
        exit(0)
    elif not os.path.isdir(model_path):
        print(f'Please train models first and put them in {model_path}.')
        exit(0)
#     elif not os.path.isdir(model_path):
#         print(f'Please train models first and put them in {model_path}.')
#         exit(0)
    else:
        predict()

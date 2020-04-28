import argparse
from solver import Solver
from utils import load_pickle


def parse():
    parser = argparse.ArgumentParser(description='tree transformer')
    parser.add_argument('-model_dir', default='train_model', help='output model weight dir')
    parser.add_argument('-batch_size', type=int, default=256)
    parser.add_argument('-model', default='sbt-transformer', help='[ast-transformer, sbt-transformer, transformer]')
    parser.add_argument('-num_step', type=int, default=250)
    parser.add_argument('-num_layers', type=int, default=2, help='layer num')
    parser.add_argument('-model_dim', type=int, default=256)
    parser.add_argument('-num_heads', type=int, default=8)
    parser.add_argument('-ffn_dim', type=int, default=2048)

    parser.add_argument('-data_dir', default='../dataset')
    parser.add_argument('-code_max_len', type=int, default=200, help='max length of code')
    parser.add_argument('-comment_max_len', type=int, default=30, help='comment max length')
    parser.add_argument('-relative_pos', type=bool, default=True, help='use relative position')
    parser.add_argument('-k', type=int, default=5, help='relative window size')

    parser.add_argument('-dropout', type=float, default=0.2)

    parser.add_argument('-load', action='store_true', help='load pretrained model')
    parser.add_argument('-train', action='store_true')
    parser.add_argument('-test', action='store_true')

    parser.add_argument('-log_dir', default='train_log/')

    parser.add_argument('-g', default='1')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()

    i2code = load_pickle(args.data_dir + '/code_i2w.pkl')
    i2nl = load_pickle(args.data_dir + '/nl_i2w.pkl')
    i2ast = load_pickle(args.data_dir + '/ast_i2w.pkl')

    ast2id = {v: k for k, v in i2ast.items()}
    code2id = {v: k for k, v in i2code.items()}
    nl2id = {v: k for k, v in i2nl.items()}

    solver = Solver(args, ast2id, code2id, nl2id, i2nl)

    if args.train:
        solver.train()
    elif args.test:
        solver.test()

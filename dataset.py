import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from train_utils import Batch
from utils import load_json, traverse_tree, load_pickle, pad_seq, subsequent_mask, make_std_mask


class TreeDataSet(Dataset):
    def __init__(self,
                 file_name,
                 ast_path,
                 ast2id,
                 nl2id,
                 max_ast_size,
                 k,
                 max_comment_size,
                 use_code):
        """
        :param file_name: 数据集名称
        :param ast_path: AST存放路径
        :param max_ast_size: 最大AST节点数
        :param k: 最大相对位置
        :param max_comment_size: 最大评论长度
        """
        super(TreeDataSet, self).__init__()
        print('loading data...')
        self.data_set = load_json(file_name)
        print('loading data finished...')

        self.max_ast_size = max_ast_size
        self.k = k
        self.max_comment_size = max_comment_size
        self.ast_path = ast_path
        self.ast2id = ast2id
        self.nl2id = nl2id

        self.use_code = use_code

        self.len = len(self.data_set)

    def __getitem__(self, index):
        data = self.data_set[index]
        ast_num = data['ast_num']
        nl = data['nl']

        ast = load_pickle(self.ast_path + ast_num)
        seq, rel_par, rel_bro = traverse_tree(ast, self.max_ast_size, self.k)

        seq_id = [self.ast2id[x] if x in self.ast2id else self.ast2id['<UNK>'] for x in seq]
        nl_id = [self.nl2id[x] if x in self.nl2id else self.nl2id['<UNK>'] for x in nl]

        """to tensor"""
        seq_tensor = torch.LongTensor(seq_id)
        nl_tensor = torch.LongTensor(pad_seq(nl_id, self.max_comment_size).long())

        if self.use_code:
            return seq_tensor, nl_tensor, rel_par, rel_bro
        else:
            return seq_tensor, nl_tensor

    def __len__(self):
        return self.len

    @staticmethod
    def make_std_mask(comment, pad):
        comment_mask = (comment != pad).unsqueeze(-2)
        tgt_mask = comment_mask & Variable(
            subsequent_mask(comment.size(-1)).type_as(comment_mask.data))
        return tgt_mask


class SeqDataSet(Dataset):
    def __init__(self,
                 file_name,
                 code2id,
                 nl2id,
                 max_code_size,
                 max_comment_size
                 ):
        super(SeqDataSet, self).__init__()
        print('loading data...')
        self.data_set = load_json(file_name)
        print('loading data finished...')
        self.code2id = code2id
        self.nl2id = nl2id
        self.max_code_size = max_code_size
        self.max_comment_size = max_comment_size

        self.len = len(self.data_set)

    def __getitem__(self, index):
        data = self.data_set[index]
        code = data['code']
        nl = data['nl']

        code_id = [self.code2id[x] if x in self.code2id else self.code2id['<UNK>'] for x in code]
        nl_id = [self.nl2id[x] if x in self.nl2id else self.nl2id['<UNK>'] for x in nl]

        """to tensor"""
        code_tensor = torch.LongTensor(pad_seq(code_id, self.max_code_size).long())
        nl_tensor = torch.LongTensor(pad_seq(nl_id, self.max_comment_size).long())

        return code_tensor, nl_tensor

    def __len__(self):
        return self.len


def collate_fn(inputs):
    codes = []
    nls = []
    rel_pars = []
    rel_bros = []

    for i in range(len(inputs)):
        if len(inputs[i]) == 4:
            code, nl, rel_par, rel_bro = inputs[i]

            codes.append(code)
            nls.append(nl)
            rel_pars.append(rel_par)
            rel_bros.append(rel_bro)
        elif len(inputs[i]) == 2:
            code, nl = inputs[i]

            codes.append(code)
            nls.append(nl)

    batch_code = torch.stack(codes, dim=0)
    batch_nl = torch.stack(nls, dim=0)

    batch_comments = batch_nl[:, :-1]
    batch_predicts = batch_nl[:, 1:]

    comment_mask = make_std_mask(batch_comments, 0)
    code_mask = (code != 0).unsqueeze(-2)

    if len(rel_pars) != 0:
        re_par_ids = torch.stack(rel_pars, dim=0)
        re_bro_ids = torch.stack(rel_bros, dim=0)

        return (batch_code, re_par_ids, re_bro_ids,batch_comments, code_mask, comment_mask), batch_predicts
    else:
        return (batch_code, batch_comments, code_mask, comment_mask), batch_predicts









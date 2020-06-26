import json
import pickle
import torch
import math
import numpy as np
from torch.autograd import Variable


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def make_std_mask(comment, pad):
    comment_mask = (comment != pad).unsqueeze(-2)
    tgt_mask = comment_mask & Variable(
        subsequent_mask(comment.size(-1)).type_as(comment_mask.data))
    return tgt_mask


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def load_pickle(path):
    return pickle.load(open(path, "rb"))


def load_dict(path):
    return pickle.load(open(path, "rb"))


def traverse_tree(root, max_size, k):
    # 先反转根节点的子节点顺序，尽可能多的保留信息
    ls = list(reversed(root.children))
    root.children = ls

    # 按照深度优先遍历取出前200个节点
    sub_tree(root, max_size=max_size)

    # 生成父子关系和兄弟关系矩阵
    root_id = root.num

    seq = [''] * max_size
    relative_parent_ids = torch.zeros((max_size, max_size))
    relative_brother_ids = torch.zeros((max_size, max_size))

    i = 0

    parent_map = {}
    brother_map = {}

    queue = [root]

    while queue:
        current_node = queue.pop()
        node_id = current_node.num
        seq[node_id] = current_node.label

        if node_id == root_id:
            parent_map[node_id] = [node_id]
            brother_map[node_id] = [node_id]

        if len(current_node.children) > 0:
            brother_node_ids = [x.num for x in current_node.children if x.num < max_size]
            for child in reversed(current_node.children):
                if child.num >= max_size:
                    continue
                child_id = child.num
                queue.append(child)

                parent_map[child_id] = parent_map[node_id] + [child_id]
                brother_map[child_id] = brother_node_ids

    for i in range(max_size):
        for j in range(max_size):
            if i not in parent_map or j not in parent_map:
                continue
            if i in parent_map[j] or j in parent_map[i]:
                """存在父子关系"""
                key = i if j in parent_map[i] else j
                rp = parent_map[key].index(j) - parent_map[key].index(i)
                rp = relative_range_map(rp, k)
                relative_parent_ids[i][j] = rp
            if i in brother_map[j]:
                rp = brother_map[j].index(j) - brother_map[j].index(i)
                rp = relative_range_map(rp, k)
                relative_brother_ids[i][j] = rp

    return seq, relative_parent_ids.long(), relative_brother_ids.long()


def relative_range_map(value, k):
    """
    map value from [-k, k] to [1, 2k+1]
    """
    return max(-k, min(k, value)) + k + 1


def pad_seq(data_list, max_len):
    data = torch.zeros(max_len)
    for i in range(min(max_len, len(data_list))):
        data[i] = data_list[i]
    return data


def sub_tree(root, i=0, max_size=200):
    """
    树的最大节点个数不超过200
    """
    root.num = i
    i = i + 1
    if i > max_size:
        return -1
    else:
        for j, child in enumerate(root.children):
            i = sub_tree(child, i, max_size)
            if i == -1:
                root.children = root.children[:j]
                return -2
            if i == -2:
                root.children = root.children[:j + 1]
                return i
        return i


class Node:
    """树节点抽象"""
    def __init__(self, label="", parent=None, is_simple_name=False, simple_name=None, is_leaf_node=False, children=[]):
        self.label = label
        self.parent = parent
        self.children = children
        self.is_simple_name = is_simple_name
        self.simple_name = simple_name
        self.is_leaf_node = is_leaf_node



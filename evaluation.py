import json
import math
import os
import numpy as np

import nltk
from nltk.translate.meteor_score import *

from nltk.translate.meteor_score import _generate_enums, _enum_allign_words, _count_chunks

from rouge import Rouge
from tensorboardX import SummaryWriter
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

from utils import load_json

rouge = Rouge(metrics=['rouge-l'])


def rouge_l_score(hypothesis, reference):
    scores = rouge.get_scores(hypothesis, reference)
    rough_l = scores[0]['rouge-l']
    return rough_l['p'], rough_l['r'], rough_l['f']


def meteor_score(hypothesis, reference):
    return single_meteor_score(reference, hypothesis, alpha=0.85, beta=0.2, gamma=0.6)


def batch_evaluate(comments, predicts, nl_i2w):
    batch_size = comments.size(0)
    references = []
    hypothesises = []
    for i in range(batch_size):
        reference = [nl_i2w[c.item()] for c in comments[i]]
        if '</s>' in reference and reference.index('</s>') < len(reference):
            reference = reference[:reference.index('</s>')]
        hypothesis = [nl_i2w[c.item()] for c in predicts[i]][1:]
        if '</s>' in hypothesis and hypothesis.index('</s>') < len(hypothesis):
            hypothesis = hypothesis[:hypothesis.index('</s>')]
        references.append(reference)
        hypothesises.append(hypothesis)
    return references, hypothesises


def batch_rouge_l(comments, predicts, nl_i2w, i):
    references, hypothesises = batch_evaluate(comments, predicts, nl_i2w)
    if i == 0:
        for j in range(5):
            print("真实:", references[j], "\n猜测:", hypothesises[j])
    scores = []
    for i in range(len(references)):
        if len(hypothesises[i]) == 0:
            scores.append(0)
        else:
            _, _, score = rouge_l_score(' '.join(hypothesises[i]), ' '.join(references[i]))
            scores.append(score)
    return scores


def batch_meteor(comments, predicts, nl_i2w):
    references, hypothesises = batch_evaluate(comments, predicts, nl_i2w)
    scores = []
    for i in range(len(references)):
        if len(hypothesises[i]) == 0:
            scores.append(0)
        else:
            _, _, _, score = meteor_score(' '.join(hypothesises[i]), ' '.join(references[i]))
            scores.append(score)
    return scores


def batch_bleu(comments, predicts, nl_i2w, i):
    references, hypothesises = batch_evaluate(comments, predicts, nl_i2w)
    if i == 0:
        for j in range(3):
            print("真实:", references[j], "\n猜测:", hypothesises[j])
    scores = []
    for i in range(len(references)):
        bleu_score = nltk_sentence_bleu(hypothesises[i], references[i])
        scores.append(bleu_score)
    return scores


def nltk_sentence_bleu(hypothesis, reference, order=4):
    cc = SmoothingFunction()
    if len(reference) < 4:
        return 0
    return nltk.translate.bleu([reference], hypothesis, smoothing_function=cc.method4)


def single_meteor_score(
    reference,
    hypothesis,
    preprocess=str.lower,
    stemmer=PorterStemmer(),
    alpha=0.9,
    beta=3,
    gamma=0.5,
):
    enum_hypothesis, enum_reference = _generate_enums(
        hypothesis, reference, preprocess=preprocess
    )
    translation_length = len(enum_hypothesis)
    reference_length = len(enum_reference)
    matches, _, _ = _enum_allign_words(enum_hypothesis, enum_reference, stemmer=stemmer)
    matches_count = len(matches)
    try:
        precision = float(matches_count) / translation_length
        recall = float(matches_count) / reference_length
        fmean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
        chunk_count = float(_count_chunks(matches))
        frag_frac = chunk_count / matches_count
    except ZeroDivisionError:
        return 0.0, 0.0, 0.0, 0.0
    penalty = gamma * frag_frac ** beta
    return precision, recall, fmean, (1 - penalty) * fmean


def bleu4(true, pred):
    c = len(pred)
    r = len(true)
    bp = 1. if c > r else np.exp(1 - r / (c + 1e-10))
    score = 0
    for i in range(1, 5):
        true_ngram = set(ngram(true, i))
        pred_ngram = ngram(pred, i)
        length = float(len(pred_ngram)) + 1e-10
        count = sum([1. if t in true_ngram else 0. for t in pred_ngram])
        score += math.log(1e-10 + (count / length))
    score = math.exp(score * .25)
    bleu = bp * score
    return bleu


def ngram(words, n):
    return list(zip(*(words[i:] for i in range(n))))


def evaluate_on_file(path):
    files = os.listdir(path)
    for file in files:
        if file.startswith('.D') or file.endswith('.txt'):
            continue
        writer = SummaryWriter(logdir='test_log/'+file.split('.')[0]+'/')
        file_path = path + file
        data = load_json(file_path)

        node_len_bleu = {i: [] for i in range(41)}
        com_len_bleu = {i: [] for i in range(31)}
        bleu = []
        nodes_num = {i: 0 for i in range(41)}

        for i, d in enumerate(data):
            node_len = int(int(d['node_len']) / 5)

            predict = d['predict'].strip().split()
            true = d['true'].split()

            com_len = len(true)
            if com_len > 30:
                com_len = 30

            if len(true) < 4:
                continue
            score = sentence_bleu([true], predict, smoothing_function=SmoothingFunction().method4)

            node_len_bleu[node_len].append(score)
            com_len_bleu[com_len].append(score)
            bleu.append(score)
            nodes_num[node_len] += 1

        if file not in ['transformer_seq.json']:
            for key, value in node_len_bleu.items():
                if len(value) == 0:
                    continue
                score = np.mean(value)
                writer.add_scalar('node_len_bleu', score, (key+1)*5)
        for key, value in com_len_bleu.items():
            if len(value) == 0:
                continue
            score = np.mean(value)
            writer.add_scalar('com_len_bleu', score, key)
        for key, value in nodes_num.items():
            writer.add_scalar('nodes_num', value, key)

        writer.close()
        print(file.split('.')[0] + ': ' + str(np.mean(bleu)))


if __name__ == '__main__':
    # reference = 'This is a test Ok'
    # candidate = 'This is a test unk Yes'
    # print(rouge_l_score(candidate, reference))
    evaluate_on_file('./evaluate_file/')
    #print(nltk_sentence_bleu(['D', 'A', 'object'], ['Frees', 'the', 'object']))

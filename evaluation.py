import nltk
from nltk.translate.meteor_score import *

from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import _generate_enums, _enum_allign_words, _count_chunks

from rouge import Rouge

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
    if len(hypothesis) < order:
        return 0
    else:
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


if __name__ == '__main__':
    reference = 'This is a test Ok'
    candidate = 'This is a test unk Yes'
    print(rouge_l_score(candidate, reference))

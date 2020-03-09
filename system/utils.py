import os
import numpy as np
from nltk import Tree
from PYEVALB.scorer import Scorer
from PYEVALB import parser

def read_file(path_grammar) :
    grammars = []
    with open(path_grammar, mode='r', encoding='utf-8') as file :
        for sentence in file : 
            tokens = sentence.split()
            for i, token in enumerate(tokens) : 
                if token[0] == '(' : tokens[i] = token.split('-')[0]
            grammars.append(' '.join(tokens).strip()[2:-1])
    return grammars


def write_file(file_to_write, corpus_test,
               path_output='evaluation_data.parser_output',
               path_for_sentence='test_sentences.txt') :
    
    print('The prediction on the test will be written here : ', path_output)
    with open(path_output, 'w' ,encoding ='utf-8') as file :
        for (i, text) in enumerate(file_to_write) : 
            file.write(text+'\n')
    print('The corresponding ground sentences will be written :', path_for_sentence)
    with open(path_for_sentence, 'w' ,encoding ='utf-8') as file :
        for (i, text) in enumerate(corpus_test) : 
            file.write(text +'\n')


def compute_precision(prediction_train, 
                      grammars_train) :
    scorer = Scorer()
    tuple_to = []
    for i in range(len(prediction_train)) :
        if prediction_train[i][1] == 1 :
            tuple_to.append((prediction_train[i][0], grammars_train[i]))
    precision = [scorer.score_trees(parser.create_from_bracket_string(pred),
                                    parser.create_from_bracket_string(real)).prec for\
                 (pred, real) in tuple_to]
    return np.sum(precision)/len(grammars_train)

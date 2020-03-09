from itertools import product
import numpy as np
from nltk import Tree
class PCKY():
    def __init__(self, oov, dico_lexicons,
                 dico_cfg, axiomes, lexicons) :
        self.oov = oov
        self.proba_cfg = dico_cfg
        self.proba_lexicons = dico_lexicons
        self.axiomes = axiomes
        self.lexicons = lexicons

    def induce_CYK(self, sentence, show=True):

        binaries = {}
        for lhs in  self.proba_cfg.keys() :
            for rhs in  self.proba_cfg[lhs] :
                if not rhs in binaries.keys() : binaries[rhs] = set()
                binaries[rhs].add(lhs)

        lb = set([B[0] for B in binaries.keys()])
        rb = set([B[1] for B in binaries.keys()])
        bi = set(binaries.keys())            
        axioms = self.axiomes    
        #import pdb; pdb.set_trace()
        init_sentence = sentence
        sentence = self.oov.beam_search_decoder(sentence=init_sentence, show=show).split() 
        n = len(sentence)

        score_table = [[dict() for i in range(n+1)] for k in range(n+1)]
        way_back = [[dict() for i in range(n+1)] for k in range(n+1)]

        right_sets= [[set() for i in range(n+1)] for k in range(n+1)]
        left_sets= [[set() for i in range(n+1)] for k in range(n+1)]

        for i, word in enumerate(sentence):
            #word = word.lower()
            for A, words in self.proba_lexicons.items():
                if word in words.keys():
                    score_table[i][i+1][A] = words[word]
                    if A in lb : left_sets[i][i+1].add(A)
                    if A in rb : right_sets[i][i+1].add(A)

        for window in range(2, n + 1):
            for start in range(n + 1 - window):
                end = start + window
                for split in range(start + 1, end):
                    left, right = score_table[start][split], score_table[split][end]
                    l_interest, r_interest = left_sets[start][split] & lb, right_sets[split][end] & rb
                    final_interest = set(product(l_interest, r_interest)) & bi
                    for (B,C) in final_interest:
                        for A in binaries[(B,C)] :
                            prob = left[B] * right[C] * self.proba_cfg[A][(B,C)]
                            if prob > score_table[start][end].get(A, 0.0) :
                                score_table[start][end][A] = prob
                                way_back[start][end][A] = (split, B, C)
                                if A in lb : left_sets[start][end].add(A)

                                if A in rb : right_sets[start][end].add(A)
        tree = self.get_tree(way_back, score_table, 0, n, init_sentence.split(), n, 'SENT')
        if tree == 'NOT IN GRAMMAR' : 
            ret = self.OOG(init_sentence.split()), 0
        else :
            tree = Tree.fromstring(tree)
            tree.un_chomsky_normal_form()
            ret = ' '.join(tree.pformat().split()), 1
        return ret 
    def OOG(self, sentence):
        ret = '(SENT '
        for i, word in enumerate(sentence[:-1]):
            ret += '(OOG ' + word + ')'
        ret += '(OOG ' + sentence[-1] + '))'
        return ret
    def get_tree(self, way_back, score, start,
                 end, sentence, n,
                 nonterm):
        if n == 1 : 
            nonterm = max(score[start][end], key = score[start][end].get)
            if 'SENT' not in nonterm : return 'NOT IN GRAMMAR'
            return '(' + nonterm + ' ' + sentence[start] + ')'

        if start == end - 1: 
            return '(' + nonterm + ' ' + sentence[start] + ')'

        if end - start == n : 
            cands = [k for k in way_back[start][end].keys() if k in self.axiomes]
            if not cands : return 'NOT IN GRAMMAR'
            best = cands[np.argmax([score[start][end][k] for k in cands])]
            split, lhs, rhs = way_back[start][end][best]

        else : 
            split, lhs, rhs = way_back[start][end][nonterm]
        return '(' + nonterm + ' ' + self.get_tree(way_back, score, start, split, sentence, n, lhs) +\
                ' ' + self.get_tree(way_back, score, split, end, sentence, n, rhs) + ')'

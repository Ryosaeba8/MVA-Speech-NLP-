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

    def induce_CYK(self, sentence, show=True, beam_size=20):

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
        sentence = self.oov.beam_search_decoder(sentence=init_sentence, show=show, beam_size=beam_size).split() 
        n = len(sentence)

        hist_scores = [[dict() for i in range(n+1)] for k in range(n+1)]
        hist_track = [[dict() for i in range(n+1)] for k in range(n+1)]

        r_pos= [[set() for i in range(n+1)] for k in range(n+1)]
        l_pos= [[set() for i in range(n+1)] for k in range(n+1)]

        for i, word in enumerate(sentence):
            #word = word.lower()
            for A, words in self.proba_lexicons.items():
                if word in words.keys():
                    hist_scores[i][i+1][A] = words[word]
                    if A in lb : l_pos[i][i+1].add(A)
                    if A in rb : r_pos[i][i+1].add(A)

        for window in range(2, n + 1):
            for start in range(n + 1 - window):
                end = start + window
                for split in range(start + 1, end):
                    left, right = hist_scores[start][split], hist_scores[split][end]
                    l_int, r_int = l_pos[start][split] & lb, r_pos[split][end] & rb
                    final_int = set(product(l_int, r_int)) & bi
                    for (B,C) in final_int:
                        for A in binaries[(B,C)] :
                            prob = left[B] * right[C] * self.proba_cfg[A][(B,C)]
                            if prob > hist_scores[start][end].get(A, 0.0) :
                                hist_scores[start][end][A] = prob
                                hist_track[start][end][A] = (split, B, C)
                                if A in lb : l_pos[start][end].add(A)

                                if A in rb : r_pos[start][end].add(A)
        tree = self.get_tree(hist_track, hist_scores, 0, n, init_sentence.split(), n, 'SENT')
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
    def get_tree(self, hist_track, score, start,
                 end, sentence, n,
                 nonterm):
        if n == 1 : 
            nonterm = max(score[start][end], key = score[start][end].get)
            if 'SENT' not in nonterm : return 'NOT IN GRAMMAR'
            return '(' + nonterm + ' ' + sentence[start] + ')'

        if start == end - 1: 
            return '(' + nonterm + ' ' + sentence[start] + ')'

        if end - start == n : 
            cands = [k for k in hist_track[start][end].keys() if k in self.axiomes]
            if not cands : return 'NOT IN GRAMMAR'
            best = cands[np.argmax([score[start][end][k] for k in cands])]
            split, lhs, rhs = hist_track[start][end][best]

        else : 
            split, lhs, rhs = hist_track[start][end][nonterm]
        return '(' + nonterm + ' ' + self.get_tree(hist_track, score, start, split, sentence, n, lhs) +\
                ' ' + self.get_tree(hist_track, score, split, end, sentence, n, rhs) + ')'

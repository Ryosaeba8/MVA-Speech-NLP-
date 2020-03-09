from nltk import Tree
import numpy as np
def get_all_trees(sentences):
    tree_cfg_sentences = []
    for sent in sentences :
        cfg = Tree.fromstring(sent)
        cfg.chomsky_normal_form(horzMarkov = 2) ## 2 to reduce size of horizontal siblings
        cfg.collapse_unary(True, True)
        tree_cfg_sentences.append(cfg) ## In order to respect the Chomsky rule 
                                       ## No : (A -> B) with A, B terminals rules.
    return tree_cfg_sentences


def build_pcfg(tree_cfg_sentences) :  
    dico_cfg, count_cfg = {}, {}
    dico_lexicons, count_lexicons = {}, {}
    axiomes = set()
    lexicons = set()
    for tree in tree_cfg_sentences :
        axiome = tree.productions()[0].lhs().symbol()
        axiomes.add(axiome) ## Ajout des axiomes, indispensable pour le CYK
        for elt in tree.productions():
            left, right = elt.lhs().symbol(), list(elt.rhs())

            if len(right) == 2 : # non terminals POS
                right = [str(r) for r in right]
                right = tuple(right)
                if left not in dico_cfg.keys():
                    dico_cfg[left] = {right : 0.}
                    count_cfg[left] = 0.
                if right not in dico_cfg[left].keys() :
                    dico_cfg[left][right] = 0.
                dico_cfg[left][right] = dico_cfg[left][right] + 1.
                count_cfg[left] = count_cfg[left] + 1

            else : # terminals POS aka lexicon
                right = right[0].lower()
                if left not in dico_lexicons.keys():
                    dico_lexicons[left] = {right : 0.}
                    count_lexicons[left] = 0.
                if right not in dico_lexicons[left].keys() :
                    dico_lexicons[left][right] = 0.
                dico_lexicons[left][right] = dico_lexicons[left][right] + 1.
                count_lexicons[left] = count_lexicons[left]  + 1  
                lexicons.add(right)

    for left, dict_left in dico_cfg.items() :
        for right in dict_left.keys() :
            dict_left[right] = dict_left[right]/count_cfg[left]

    for left, dict_left in dico_lexicons.items() :
        for right in dict_left.keys() :
            dict_left[right] = dict_left[right]/count_lexicons[left]
    return axiomes, lexicons, dico_lexicons, dico_cfg
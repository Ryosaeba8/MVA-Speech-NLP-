from proba_contextfg import get_all_trees, build_pcfg
from utils import read_file, write_file, compute_precision
from sklearn.model_selection import train_test_split
from out_of_vocab import OOV_module
from proba_cyk import PCKY
import numpy as np
from nltk import Tree
import warnings
import argparse
warnings.simplefilter('ignore')


path_grammar = 'sequoia-corpus+fct.mrg_strict'
path_embeddings = 'polyglot-fr.pkl'

p = argparse.ArgumentParser( description='Basic run script for the Parser' )
p.add_argument('--mode',  type=str, default= 'test')
args = p.parse_args()
mode = args.mode #mode = 'eval' pour l'évaluation avec de nouvelles phrases
def evaluation(grammars_test, grammars_train,
			   pcky, corpus, mode) :

	if mode == 'test' :
		print("...........Début de l'évaluation........")
		cfg_test = get_all_trees(grammars_test)
		corpus_test = [' '.join(tree.leaves()) for tree in cfg_test] 
		predictions_test = []
		for i, sentence in enumerate(corpus_test) :
			predictions_test.append(pcky.induce_CYK(sentence, show=False))
		status_test, predictions_test_ = [x[1] for x in predictions_test], [x[0] for x in predictions_test]
		print('Precision on test :', compute_precision(predictions_test, grammars_test))
		write_file(predictions_test_, corpus_test)
		print("...........Fin de l'évaluation........")
	elif mode == 'eval' :
		print('....................Début.................')
		print("Pour sortir, entrez : exit")
		while True :
			phrase_to_parse = str(input(">>>>> Veuillez entrer une phrase :"))
			if phrase_to_parse == 'exit' :
				break
			prediction, status = pcky.induce_CYK(phrase_to_parse, show=True)
			if status == 0:
				print("La phrase n'a pas pu être parsée")
			else :
				print(prediction)
			print('....................Fin...................')
if __name__ == '__main__' :
	grammars = read_file(path_grammar)
	grammars_train, grammars_test = train_test_split(grammars, test_size=0.1, shuffle=False) 
	tree_cfg_grammars = get_all_trees(grammars_train)
	corpus = [' '.join(tree.leaves()) for tree in tree_cfg_grammars]
	axiomes, lexicons, dico_lexicons, dico_cfg = build_pcfg(tree_cfg_grammars)
	## Construction d'un modèle CYK
	oov = OOV_module(path_embeddings = path_embeddings,
		lexicons=lexicons, corpus = corpus)
	## Construction des dictionnaires contenant 
	## les probabilités associés au unigrams et bigrams du Corpus
	oov.build_unigram_bigram()
	## Probabilistc CKY
	pcky = PCKY(oov=oov, dico_lexicons=dico_lexicons,
				dico_cfg=dico_cfg, axiomes=axiomes, 
				lexicons=lexicons)

	evaluation(grammars_test, grammars_train,
			   pcky, corpus, mode)

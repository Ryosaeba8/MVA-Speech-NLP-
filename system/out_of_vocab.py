from heapq import nsmallest
import numpy as np
import pickle
import warnings
from pathlib import Path
warnings.simplefilter('ignore')
class OOV_module():
    def __init__(self, lexicons,
                 corpus, path_embeddings = 'polyglot-fr.pkl'):
        
        self.corpus = corpus
        self.load_embeddings(path_embeddings)
        _, words_inds, lexicons_inds = np.intersect1d(self.words_all, 
                                                      list(lexicons), 
                                                      return_indices=True)
        self.words_emb, self.embeddings = np.array(self.words_all)[words_inds], self.embeddings_all[words_inds]
        self.words = list(lexicons)
        self.word2id = {word: idx for idx, word in enumerate(self.words_all)}
        
    def load_embeddings(self, path):
        
        f = open(Path(path), 'rb')
        u = pickle._Unpickler(f)
        u.encoding = 'latin1' 
        self.words_all, self.embeddings_all = u.load()
        f.close()
        
    def encode(self, word):
        ind_word = self.word2id[word] ## finding the index of the word
        return self.embeddings_all[ind_word]
    
    def levenshtein_distance(self, v, w):
        ## v and w are two words
        n, m = len(v), len(w)
        M = np.zeros((n+1, m+1))
        for i in range(n+1):
            M[i, 0] = i
        for j in range(m+1):
            M[0, j] = j
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if v[i-1] == w[j-1]:
                    M[i, j] = min(M[i-1, j] + 1, M[i, j-1] + 1, M[i-1, j-1])
                else:
                    M[i, j] = min(M[i-1, j] + 1, M[i, j-1] + 1, M[i-1, j-1] + 1)
        return M[n, m]

    def score(self, word1, word2):
        # Return the cosine similarity
        if len(word2) == 1 :
            w1 = self.encode(word1)
            w2 = self.encode(word2)[np.newaxis, :] ## for fast computation
        else :
            w1 = word1; w2 = word2
        return np.dot(w2, w1)/(np.linalg.norm(w1)*np.linalg.norm(w2, axis = 1)) ## cosine similarity

    def most_similar(self, word, k):
        # Returns the k most similar words
        if word not in self.words_all :
            return []
        w1 = self.encode(word)
        sim = self.score(w1, self.embeddings)
        inds = np.argsort(sim)[::-1][:k]
        words = [self.words_emb[i] for i in inds] 
        return words
    
    def get_closest_word(self, word, nb_consider = 15) :
        ## First we try to find if there is a spelling 
        ## error by computing the leveinstein distance
        
        lev_dist = {1 :[], 2 :[], 3:[], 4 : []}
        for other_word in self.words :
            lev_distance = self.levenshtein_distance(word, other_word)
            if lev_distance in lev_dist.keys() :
                lev_dist[lev_distance].append(other_word) 
        lev_to_consider = list(lev_dist.values())
        lev_to_consider = sum([i for i in lev_to_consider if i], [])
        lev_to_consider = [i for i in lev_to_consider if i][:nb_consider]
        emb_to_consider = self.most_similar(word, k=nb_consider)
        to_return = lev_to_consider + emb_to_consider
        if (len(to_return) == 0 ) : 
            if (word != word.lower()) :
                word = word.lower()
                for other_word in self.words :
                    lev_distance = self.levenshtein_distance(word, other_word)
                    if lev_distance in lev_dist.keys() :
                        lev_dist[lev_distance].append(other_word) 
                lev_to_consider = list(lev_dist.values())
                lev_to_consider = sum([i for i in lev_to_consider if i], [])
                lev_to_consider = [i for i in lev_to_consider if i][:nb_consider]
                emb_to_consider = self.most_similar(word, k=nb_consider)
                to_return = lev_to_consider + emb_to_consider
                if len(to_return) == 0 :
                    return ['OOV']
                return to_return
            else :
                return ['OOV']
        return to_return
            
    def process_sentence(self, sentence, to_print=False):
        sentence_ = []
        for word in sentence.split():
            word = word.lower()
            if word in self.words :
                sentence_.append([word])
            else :
                to_print = True
                sentence_.append(self.get_closest_word(word))
        return sentence_, to_print      
    def build_unigram_bigram(self) :
        all_words = self.words
        all_words = self.words + ['SOS', 'EOS']
        self.bigrams = {'OOV' : {}}
        self.frequencies = {'OOV' :0}
        self.unigrams = {'OOV' : 0}
        for line in self.corpus:
            arr = ['SOS'] + [w.lower() for w in line.split()] + ['EOS']  
            for i in range(len(arr)-1):  ## for each (word_1, word_2) in the 
                                         ## sentence we update the bigram count and the unigram count
                key = arr[i]
                self.unigrams[key] = self.unigrams.get(key, 0.0) + 1
                self.frequencies[key] = self.frequencies.get(key, 0.0) + 1
                key_ = arr[i+1]
                if key not in self.bigrams.keys() : self.bigrams[key] = {}
                self.bigrams[key][key_] = self.bigrams[key].get(key_, 0.0) + 1
            self.unigrams[key_] = self.unigrams.get(key_, 0.0) + 1
        for key in self.bigrams.keys():  
            if self.frequencies[key] != 0 :
                for key_ in self.bigrams[key].keys():
                    self.bigrams[key][key_] = self.bigrams[key][key_]/self.frequencies[key] 
        for key in self.unigrams.keys():
            self.unigrams[key] = self.unigrams[key]/sum(list(self.unigrams.values()))

    def beam_search_decoder(self, sentence, beam_size = 20, show=True):
        best_scores = {0 : {'SOS': 0}}
        best_edges = {0 : {'SOS': []}}
        active_words = {0 : ['SOS']}
        all_words = self.words + ['SOS', 'EOS']
        sentence_, to_print = self.process_sentence(sentence)
        for i, prediction in enumerate(sentence_ + ['EOS']) : ## we loop over the the predicted probablities 
            my_best = {}
            best_scores[i+1] = {}
            best_edges[i+1] = {}
            for prev_word in active_words[i] : ## At each loop we will focus only 
                                               ## on the best k prediction obtained on the previous iteration 
                if i!= len(sentence_) :
                    for next_ind, next_word in enumerate(prediction) :
                        if next_word not in all_words :
                            #import pdb; pdb.set_trace()
                            #print(sentence[i], 'is not in vocabulary. It will be considered as OOV')
                            next_word = 'OOV'
                        score = best_scores[i][prev_word] -\
                                np.log(self.unigrams[next_word]*\
                                       self.bigrams[prev_word].get(next_word, 0.0))
                        if (next_word not in best_scores[i+1].keys()) or\
                                            (best_scores[i+1][next_word] > score) : ## updating the log probalities
                            best_scores[i+1][next_word] = score
                            best_edges[i+1][next_word] = best_edges[i][prev_word] +\
                                                         [next_word]
                            my_best[next_word] = score
                else : ## dealing with the end of sentence token
                    next_word = 'EOS'
                    score = best_scores[i][prev_word] - np.log(self.bigrams[prev_word].get(next_word, 0.0))
                    if (next_word not in best_scores[i+1].keys()) or\
                                        (best_scores[i+1][next_word] > score) :
                        best_scores[i+1][next_word] = score
                        best_edges[i+1][next_word] = best_edges[i][prev_word] +\
                                                     [next_word]
            active_words[i+1] = nsmallest(beam_size, my_best, key = my_best.get)
        try :
            end_word = nsmallest(1, best_scores[i+1],
                                 best_scores[i+1].get)[0]
        except :
            import pdb; pdb.set_trace()
        pred_sequence = best_edges[i+1][end_word][:-1]
        pred_sequence = ' '.join(pred_sequence)
        if to_print & show :
            print('The sentence : ###', sentence, '### is not recognized.')
            print('We hope you meant :', '### ' + pred_sequence + ' ###' )
        return pred_sequence
        
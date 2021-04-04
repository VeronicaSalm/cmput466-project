import gensim 
from gensim.models import Word2Vec

class Coherence:

    def __init__(self):
        self.__model = None


    def getCoherence(self, words):
        '''
        Gets coherence of a set of words.
        If there are less than 2 words, returns coherence of 1.
        If any word is not contained in the corpus, returns coherence of 0.

        Arguments:
            words (list of strings): The set of words to get the coherence of.
        
        Returns:
            The coherence of the set of words.
        '''
        if len(words) < 2:
            return 1
        out = 0
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                sim = self.getSimilarity(words[i], words[j])
                if sim == None:
                    return 0
                out += sim
        out /= (len(words)*(len(words)-1))//2
        return out


    def getSimilarity(self, word1, word2):
        '''
        Gets cosine similarity of two words.
        If either word is not contained in the corpus, returns None.

        Arguments:
            word1 (string): The first word.
            word2 (string): The second word.
        
        Returns:
            The similarity of the two words.
        '''
        if not self.__model:
            return None
        
        try:
            return self.__model.wv.similarity(word1, word2)
        except:
            return None


    def mapWordsToVecs(self, corpus):
        '''
        Initializes and trains a Word2Vec model based on the given corpus.
        Uses the same data format as DataManager.py.

        Arguments:
            corpus (List of tuples with second element being a string)
        '''
        for i in range(len(corpus)):
            corpus[i] = corpus[i][1].split()
    
        self.__model = gensim.models.Word2Vec(sentences = corpus, min_count = 1, vector_size = 100, window = 5, sg = 1)

import numpy as np
from random import randint, choices

def normalize(vector):
    if np.sum(vector) == 0:
        return vector
    else:
        return vector * (1 / np.sum(vector))

class LDA:

    def __init__(self, num_topics, num_words, corpus, alpha = -1, beta = 0.1):
        # higher alpha -> documents more spread out among topics
        # higher beta -> words more spread out among topics
        self.__K = num_topics
        self.__V = num_words
        self.__corpus = corpus
        self.__alpha = alpha
        if alpha == -1:
            self.__alpha = 50 / self.__K
        self.__beta = beta
        self.__W = np.sum([len(doc) for doc in self.__corpus])
    

    def train(self, num_iterations):
        # K x V word topic matrix
        # wt[i][j] is the number of words j assigned to topic i
        wt = np.zeros([self.__K, self.__V])

        # N x K document topic matrix
        # dt[i][j] is the number of words in document i assigned to topic j
        dt = np.zeros([len(self.__corpus), self.__K])

        # topic assignment matrix
        # ta[i][j] is the assigned topic of the j'th word in the i'th document
        ta = [np.zeros(len(doc)) for doc in self.__corpus]

        # randomly assign topics to all words
        for d in range(len(self.__corpus)):
            for w in range(len(self.__corpus[d])):
                random_topic = randint(0, self.__K - 1)
                ta[d][w] = random_topic
                dt[d][random_topic] += 1
                wt[random_topic][self.__corpus[d][w]] += 1
        
        # apply num_iterations iterations
        for _ in range(num_iterations):
            # calculate values for the new ta
            newta = [np.zeros(len(doc)) for doc in self.__corpus]
            for d in range(len(self.__corpus)):
                for w in range(len(self.__corpus[d])):
                    prob_vector = np.zeros(self.__K)
                    for j in range(self.__K):
                        prob_left = (wt[j][self.__corpus[d][w]] + self.__beta) / (np.sum(wt[j]) + (self.__W * self.__beta))
                        prob_right = (dt[d][j] + self.__alpha) / (np.sum(dt[d]) + self.__K * self.__alpha)
                        prob_vector[j] = (prob_left * prob_right)
                    prob_vector = normalize(prob_vector)
                    new_topic = choices(list(range(self.__K)), prob_vector)
                    newta[d][w] = int(new_topic[0])

            # calculate new values for dt, wt
            dt = np.zeros([len(self.__corpus), self.__K])
            wt = np.zeros([self.__K, self.__V])
            for d in range(len(self.__corpus)):
                for w in range(len(self.__corpus[d])):
                    dt[d][int(newta[d][w])] += 1
                    wt[int(newta[d][w])][self.__corpus[d][w]] += 1
            ta = newta
        print(ta)
        print(dt)
        print(wt)
       
        theta = np.zeros([len(self.__corpus), self.__K])
        for d in range(len(self.__corpus)):
            for j in range(self.__K):
                theta[d][j] = (dt[d][j] + self.__alpha) / (np.sum(dt[d]) + self.__K * self.__alpha)
        return theta



my_lda = LDA(2, 28, [[1, 2, 3, 4, 5], [6, 7, 8, 1, 9, 3, 5], [2, 10, 11, 3, 12, 5], [13, 11, 14, 15], [16, 17, 18, 11], [19, 3, 12], [19, 20, 21, 22, 18, 23, 24, 25, 19], [26, 19, 27]], 1, 1)
theta = my_lda.train(1000)
print(theta)

# coding: utf-8

# In[4]:


import re
from nltk import word_tokenize
from collections import Counter
from string import punctuation
from decimal import *
from nltk.util import ngrams

#ngram word model
class GetNgram:
    def __init__(self, uni_count, bi_count, tri_count, uni_dict, bi_dict, tri_dict, uni_num, bi_num, tri_num):
        self.unigram_count = uni_count
        self.bigram_count = bi_count
        self.trigram_count = tri_count

        self.unigram_dict = uni_dict
        self.bigram_dict = bi_dict
        self.trigram_dict = tri_dict

        self.unigram_number = uni_num
        self.bigram_number = bi_num
        self.trigram_number = tri_num


# In[5]:


#Good_turing Soomthing model
class Good_turing:
    def __init__(self, freq, training):
        self.training = training
        self.freq = freq

    def count_frequency(self, freq, ngram_dict={}):
        number = 0 
        if freq == 0: 
            for value in ngram_dict.values():
                if value == 1:
                    number += 1
        else:
            for value in ngram_dict.values():
                if value == freq:
                    number += 1
        return number
    
    #count probability, if Nc+1 is zero, use add-one smoothing
    def calculate_GT(self, word):
        probability = 0
        words = word.split(" ")
        
        if len(words) == 1:
            freq_1 = Good_turing.count_frequency(self.freq + 1, self.training.unigram_dict)
            if freq_1 == 0:
                probability = (self.training.unigram_dict.setdefault(word, 0) + 1) / (2 * self.training.unigram_number)
            else:
                probability = ((self.freq + 1) * (freq_1) / self.freq) / self.training.unigram_number  
        else:
            if len(words) == 2:
                freq_1 = Good_turing.count_frequency(self.freq + 1, self.training.bigram_dict)
                if freq_1 == 0:
                    probability = (self.freq + 1) / (self.training.unigram_dict.setdefault(words[0], 0) + self.training.unigram_number)
                else:
                    probability = ((self.freq + 1) * (freq_1) / self.freq) / self.training.bigram_number
            else:
                if len(words) == 3:
                    freq_1 = Good_turing.count_frequency(self.freq + 1, self.training.trigram_dict)
                    if freq_1 == 0:
                        probability = (self.freq + 1) / (self.training.bigram_dict.setdefault(" ".join(words[:-1]), 0) + self.training.unigram_number)
                    else:
                        probability = ((self.freq + 1) * (freq_1) / self.freq) / self.training.trigram_number  
        return probability


# In[6]:


#Katz BackOff model
class KatzBackOff:
    def __init__(self, training):
        self.training = training
    
    #count alpha: my method is just ignore it, that is set alpha is 1  
    def count_alpha(self, prefix, ngram):
        return 1
        
    #count probability
    def katz_prob(self, pair):
        words = pair.split(" ")
        probability = 0
        tri_count = KatzBackOff.count_number(self, pair)
        if tri_count == 0:
            bi_word_former = " ".join(w for w in words[:-1])
            bi_count_former = KatzBackOff.count_number(self, bi_word_former)
            if bi_count_former == 0:
                discount_mle_pro = Good_turing(bi_count_former, self.training).calculate_GT(words[2])
                probability = discount_mle_pro
            else:
                alpha_first = KatzBackOff.count_alpha(self, bi_count_former, 3)
                bi_word_later = " ".join(w for w in words[-2:]) 
                bi_count_later = KatzBackOff.count_number(self, bi_word_later)
                if bi_count_later == 0:
                    alpha_second = KatzBackOff.count_alpha(self, words[1], 3) 
                    discount_mle_pro = Good_turing(bi_count_later, self.training).calculate_GT(words[2])
                    probability = alpha_first * alpha_second * discount_mle_pro
                else:
                    discount_mle_pro = Good_turing(bi_count_later, self.training).calculate_GT(" ".join(words[-2:]))#yz
                    probability = alpha_first * discount_mle_pro
        else:
            discount_mle_pro = Good_turing(tri_count, self.training).calculate_GT(pair)
            probability = discount_mle_pro

        return probability
        
    #count frequency in training file
    def count_number(self, pair):
        list = pair.split(" ")
        count = 0
        if len(list) == 2 :
            count = self.training.bigram_dict.setdefault(pair, 0)
        if len(list) == 3 :
            count = self.training.trigram_dict.setdefault(pair, 0)
        return count
        


# In[7]:


#NgramModel
#init: 1. classifyComments[][]: row is kinds; col is comment
class NgramModel:
    def __init__(self, classifyComments):
        self.classifyComments = classifyComments
    
    ##############################################################
    #get Ngram result
    #return a katzBackOff object
    ##############################################################
    def getNgramResult(self):
        katzBackOff = []
        for item in self.classifyComments:
            ngram = NgramModel.setNgramAttributes(self, item)
            katzBackOff.append(KatzBackOff(ngram))

        return katzBackOff
        
    ##############################################################
    #start to calculate test data
    #parameter: test data []
    #return a list
    ##############################################################
    def start(self, testData):
        katzBackOff = NgramModel.getNgramResult(self)
        prob_path = './probability.txt'
        
        #store probability: row: test data; col: line number of kinds
        probability = [[] for i in range(len(testData))]
        
        for j in range(len(testData)):
            #preprocss test data, transfer test data to trigram
            trigram_test = ngrams(NgramModel.pre_process(testData[j]), 3)
                    
            #transfer test data trigram to list format
            trigram_test_sentence = "\n".join('{}'.format(' '.join(key)) for key in trigram_test).split('\n')

            #if trigram doesn't exist, ignore
            if len(trigram_test_sentence[0]) == 0:
                continue
            else:
                for i in range(len(katzBackOff)):
                    #prob is the probability of current line in test data
                    prob = 0
                    for pair in trigram_test_sentence :
                        prob += Decimal.from_float(katzBackOff[i].katz_prob(pair)).log10()
                    
                    probability[j].append(str(prob))
                    
            with open(prob_path, 'w+') as f:
                for line in probability:
                    f.write(' '.join(line))
                    f.write('\n')
                
        #select most likely kind -> []
        resultList = NgramModel.maxProb(self)

    ##############################################################
    # perplexity
    # parameter: test data []
    # return a list
    ##############################################################
    def perplexity(self, testData):
        katzBackOff = NgramModel.getNgramResult(self)
        probability_perplexity_path = './probability_perplexity.txt'

        token = NgramModel.pre_process(''.join(testData))
        token_number = len(token)

        trigram_test = ngrams(NgramModel.pre_process(''.join(testData)), 3)

        trigram_test_sentence = "\n".join('{}'.format(' '.join(key)) for key in trigram_test).split('\n')

        # store probability: row: test data; col: line number of kinds
        probability_perplexity = []


        if len(trigram_test_sentence) != 0:
            for i in range(len(katzBackOff)):
                # prob is the probability of current line in test data
                prob = 0
                for pair in trigram_test_sentence:
                    prob += Decimal.from_float(katzBackOff[i].katz_prob(pair)).log10()

                probability_perplexity.append(str(prob))

        with open(probability_perplexity_path, 'w') as f:
            for line in probability_perplexity:
                f.write(''.join(line))
                f.write('\n')

    
    ##############################################################
    #find the most like kind
    #parameter: probability []
    #return a list
    ##############################################################    
    def maxProb(self):
        result = []
        prob_path = './probability.txt'
        result_path = './result.txt'
        
        with open(prob_path, 'r') as f:
            probability = f.readlines()
            
        for item in probability:
            item = item.split(' ')
            d = {}
            for i in range(len(item)):
                d[i] = item[i]
            sortedDict = sorted(d.items(), key=lambda d:d[1], reverse=True)
            maxType = sortedDict[0][0]
            result.append(maxType)
    
        with open(result_path, 'w') as f:
            for item in result:
                f.write(str(item))
                f.write('\n')

        


    ##############################################################
    #set ngram attributes
    #parameter: item
    #return GetNgarm object
    ############################################################## 
    def setNgramAttributes(self, item):
        unigram_count = Counter(NgramModel.pre_process(' '.join(item)))
        bigram_count = Counter(ngrams(NgramModel.pre_process(' '.join(item)), 2))
        trigram_count = Counter(ngrams(NgramModel.pre_process(' '.join(item)), 3))
        
        unigram_dict = NgramModel.collect_unigram(unigram_count.items())
        bigram_dict = NgramModel.collect_ngram(bigram_count.items())
        trigram_dict = NgramModel.collect_ngram(trigram_count.items())
        
        unigram_number = NgramModel.total_number(unigram_dict)
        bigram_number = NgramModel.total_number(bigram_dict)
        trigram_number = NgramModel.total_number(trigram_dict)
        
        return GetNgram(unigram_count, bigram_count, trigram_count, unigram_dict, bigram_dict, trigram_dict, unigram_number, bigram_number, trigram_number)
    
    
    #preprocess
    #remove blankspace, remove \n, punctuation, numbers etc and replace all capitial letter to lowercase
    def pre_process(sentence):
        #sentence = sentence.lower()
        sentence = re.sub(r'\n', r'', sentence)
        #sentence = re.sub(r'[,.\'\";?\-\!]', r' ',sentence)
        sentence = re.sub(r'[0-9]', r'', sentence)
        word_tok = word_tokenize(sentence)
        return word_tok
    
    #count the number of the same letter
    def collect_unigram(dict={}.items()):
        unigram_dict = {}
        for key,value in dict:
            key = ''.join(key)
            unigram_dict[key] = unigram_dict.setdefault(key, 0) + value
        return unigram_dict
    
    #count the number of the same nigram
    def collect_ngram(dict={}.items()):
        ngram_dict = {}
        for key,value in dict:
            key = ' '.join(key)
            ngram_dict[key] = ngram_dict.setdefault(key, 0) + value
        return ngram_dict
    
    #count total number
    def total_number(dicts={}):
        number = 0
        for value in dicts.values():
            number += value
        return number



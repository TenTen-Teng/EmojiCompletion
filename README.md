# Emoji Completion
Use NGram model to implement auto complete emoji based on user's input

## NGram model
### Use Katz's Back-off Model 
### Dataset: newUniData_5.json
	Source: Twitter
	Code link: https://github.com/TenTen-Teng/ScrapyCrawl-TwitterCommentEmoji.git
	Number: 807,244 comments with emoji
	Format: Json (content is comment, image is emoji image, title is emoji title)

### File Explanation: 
###### HandleRawData.py uses for handle raw data. 
    function getSortedEmojiData uses for category raw data by emoji and sort emoji, save it as count_sort_emoji.txt
    function classifyComment uses for category comments by emoji, save them as folder rawClassifyData
    function sperateRawData uses for sperate data to training data and test data(8:2), save them as folder trainingData and testData, respecti
Ngram.py is Ngram model
	class Good_turing is for handling unknown case by using Good Turing Smoothing model
	class KatzBackOff is for implementing Back-off model
	class NgramModel is for data preprocess(function pre_process), calculate perplexity(function perplexity), start model(function start) and so on.

Main.py is the beginning of model
	change parameters to change how many kinds and how many data as input

AccuracyRate uses for calculate accuracy of each model

## Acknowledgement:
Yassine Benajiba (George Washington University, Intro to Statistical NLP, CSCI 6907)

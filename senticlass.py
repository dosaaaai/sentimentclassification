
# import dependencies
import pandas as pd
from collections import defaultdict
from pathlib import Path
import nltk as nl
from nltk.tokenize import word_tokenize
import re
from nltk.tokenize.casual import TweetTokenizer
import numpy as np
import math
import sys
import operator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#nl.download('punkt')

# pass path as train and test folder

#stdoutorigin = sys.stdout
#sys.stdout = open("D:\\Spring 2020\\assignments\\sentiment_classification\\log.txt", "w")
################ Functions######################
def get_data(path):
    pospath = path + '/positive'
    # create list to store text
    results = defaultdict(list)

    # loop through files and append text to list
    for file in Path(pospath).iterdir():
        with open(file, "r", encoding="utf8") as file_open:
            results["text"].append(file_open.read())

    # read the list in as a dataframe
    df_pos = pd.DataFrame(results)

    #set directory path
    my_dir_path_neg = path + '/negative'

    # create list to store text
    results_neg = defaultdict(list)

    # loop through files and append text to list
    for file in Path(my_dir_path_neg).iterdir():
        with open(file, "r", encoding="utf8") as file_open:
            results_neg["text"].append(file_open.read())
    # read the list in as a dataframe
    df_neg = pd.DataFrame(results_neg)
    df_neg.head()

    #add sentiment to both datasets and then combine them for test data 1 for positive and 0 for negative
    df_pos['Sentiment']=1
    df_neg['Sentiment']=0
    frames = [df_pos, df_neg]
    df = pd.concat(frames)

    # increase column width to see more of the tweets
    pd.set_option('max_colwidth', 140)

    # reshuffle the tweets to see both pos and neg in random order
    df = df.sample(frac=1).reset_index(drop=True)

    # explore top 5 rows
    df.head(5)
    return df

def clean(df):

    # Remove any markup tags (HTML), all the mentions of handles(starts with '@') and '#' character
    def cleantweettext(raw_html):
        pattern = re.compile('<.*?>')
        cleantext = re.sub(pattern, '', raw_html)
        cleantext = " ".join(filter(lambda x:x[0]!='@', cleantext.split()))
        cleantext = cleantext.replace('#', '')
        return cleantext

    def removeat(text):
        atlist=[]
        for word in text:
            pattern = re.compile('^@')
            if re.match(pattern,word):
                #cleantext1 = re.sub(pattern, word[1:], word)
                atlist.append(word[1:])
            else:
                atlist.append(word)
        return atlist

    def tolower(text):
        lowerlist=[]
        for word in text:
            pattern = re.compile('[A-Z][a-z]+')
            if re.match(pattern,word):
                cleantext1 = re.sub(pattern, word.lower(), word)
                lowerlist.append(cleantext1)
            else:
                lowerlist.append(word)
        return lowerlist

    cleantweet= []
    for doc in df.text:
        cleantweet.append(cleantweettext(doc))


    tokentweet=[]
    df.text= cleantweet
    for doc in df.text:
        tokentweet.append(TweetTokenizer().tokenize(doc))
    df.text= tokentweet

    removeattweet=[]
    for doc in df.text:
        removeattweet.append(removeat(doc))
    df.text =removeattweet

    lowertweet=[]
    for doc in df.text:
        lowertweet.append(tolower(doc))
    df.text = lowertweet

    tweets=[]
    for x in df.text:
        tweet = ''
        for word in x:
            tweet += word+' '
        tweets.append(word_tokenize(tweet))
    df.text= tweets

    #stemming
    stemtweets=[]
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english", ignore_stopwords=False)
    #ps= PorterStemmer()
    for x in df.text:
        stemtweet=''
        for word in x:
            stemtweet=stemtweet+stemmer.stem(word)+' '
        stemtweets.append(word_tokenize(stemtweet))
    df['stemmed']=stemtweets

    df_unstemmed = pd.DataFrame()
    df_unstemmed['text'] = df['text']
    df_unstemmed['Sentiment'] = df['Sentiment']
    df_stemmed = pd.DataFrame()
    df_stemmed['text'] = df['stemmed']
    df_stemmed['Sentiment'] = df['Sentiment']
    
    ### Finalize both the stemmed and unstemmed dataframes
    #df_unstemmed = df.drop(['stemmed'], axis=1)
    #df_unstemmed.head()

    # create a df with stemmed text
    #df_stemmed = df.drop(['text'], axis=1)
    
    return df_stemmed,df_unstemmed


# initialize count vectorizer
def dummy_fun(doc):
    return doc

def getrep(df, rep):
    if rep == 'binary':
        vectorizer = CountVectorizer(binary = True, analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)  
    elif rep == 'freq': #for freq
        vectorizer = CountVectorizer(analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)          
    elif rep == 'tfidf':
        vectorizer = TfidfVectorizer(analyzer='word', tokenizer=dummy_fun, preprocessor=dummy_fun, token_pattern=None)
    text = df.text
    vectorizer.fit(text)
    freqVocab = vectorizer.vocabulary_
    train_vector = vectorizer.transform(text)
    #Create bigdoc that contains words in V, their corresponding frequencies for each class

    #1.Transform pos and neg tweets into seprate vectors
    train_pos_vector1 = vectorizer.transform(df[df['Sentiment']==1]['text'])
    train_neg_vector1 = vectorizer.transform(df[df['Sentiment']==0]['text'])

    #2. column sum of vectors(word per column)
    sum_pos = train_pos_vector1.sum(axis = 0)
    sum_neg = train_neg_vector1.sum(axis = 0)

    #3. Initialize bigdoc as a dataframe
    bigdoc = pd.DataFrame(index = list(set(freqVocab.keys())), columns = ['pos', 'neg'])

    #4. get the corresponding frequency from the above matrx and set it to bigdoc
    for word in freqVocab.keys():
        index = freqVocab.get(word)
        bigdoc.at[word, 'pos'] = sum_pos[:, index].item()
        bigdoc.at[word, 'neg'] = sum_neg[:, index].item()
#here
    #print("length of vocab: ")
    #print(len(freqVocab))
    return bigdoc, freqVocab, train_vector, vectorizer

   # https://www.geeksforgeeks.org/g-fact-41-multiple-return-values-in-python/

####### NAIVE BAYES#######
def Naivebayes(data,category,vector,bigvec):
    logprob = bigvec.copy()
    priors = []
    for cat in category:
        ndoc= len(data)
        nc= len(data[data['Sentiment']== cat])
        prior = math.log(nc/ndoc)
        priors.append(prior)
        if cat == 0:
            colname = 'neg'
        else:
            colname = 'pos'
        denominator = bigvec[colname].sum() + len(bigvec) #denominator for likelihood
        logprob[colname] = bigvec[colname].apply(lambda x:math.log((x+1)/denominator)) #likelihood
    return logprob,priors

def TestNaiveBayes(testdf, logprior, loglikelihood, category, V):
    y_pred = []
    y_true= testdf['Sentiment']
    print("testdf len = " + str(len(testdf)))
    for testdoc in testdf.text:
      mle= {0:0, 1:0}
      for cat in category:
        if cat == 0:
            colname = 'neg'
        else:
            colname = 'pos'
        mle[cat] = logprior[cat]
        #print(testdoc)
        for word in testdoc:
            if word in V.keys():
                #print(word)
                mle[cat] = mle[cat] + loglikelihood.loc[word,colname]
                #print(mle)
      y_pred.append(max(mle.items(), key=operator.itemgetter(1))[0])
    
    testdf['pred']= y_pred
    #confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=None)#, sample_weight=None, normalize=None)
    print(cm)
    #accuracy
    accuracy = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    print("accuracy is " + str(accuracy))
    return testdf
     
##### LOGISTIC REGRESSION######
def logisticreg(X,y):
    print("Split the data into training and validation datasets")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    #print(type(X))
    #print(X)
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def compute_cost(X, y, theta):
        m = len(y)
        h = sigmoid((X@theta))
        epsilon = 1e-5
        cost = (1/m)*(((-y).T @ np.log(h + epsilon))-((1-y).T @ np.log(1-h + epsilon)))
        return cost

    def gradient_descent(X, y, params, learning_rate, iterations):
        cost_history = np.zeros((iterations,1))

        for i in range(iterations):
            params = params - (learning_rate/m) * (X.T @ (sigmoid(X @ params) - y))
            # gradient = X.T*(sigmoid(np.multiply(params,X)-y))
            # params= params - learning_rate*gradient
            cost_history[i] = compute_cost(X, y, params)

        return (cost_history, params)
    
    # L2 regularization
    def compute_regularized_cost(X, y, theta,Lambda):
        m=len(y)
        predictions = sigmoid(X @ theta)
        error = (-y * np.log(predictions)) - ((1-y)*np.log(1-predictions))
        cost = 1/m * sum(error)
        regCost= cost + Lambda/(2*m) * sum(theta**2)
      
        # compute gradient
        j_0= 1/m * (X.transpose() @ (predictions - y))[0]
        j_1 = 1/m * (X.transpose() @ (predictions - y))[1:] + (Lambda/m)* theta[1:]
        grad= np.vstack((j_0[:,np.newaxis],j_1))
        return regCost[0], grad
        

    def reggradientDescent(X,y,theta,alpha,num_iters,Lambda):
        """
      Take in numpy array X, y and theta and update theta by taking num_iters gradient steps
      with learning rate of alpha
      
      return theta and the list of the cost of theta during each iteration
      """
        m=len(y)
        J_history =[]
        for i in range(num_iters):
            cost, grad = compute_regularized_cost(X,y,theta, Lambda)
            theta = theta - (alpha * grad)
            J_history.append(cost)      
        return J_history, theta
   
  
    def predict(X, params):
        #return np.round(sigmoid(X @ params))
        pred = sigmoid(X @ params)
        predictions = []
        for p in pred:
            if(p >= 0.5):
                predictions.append(1)
            else:
                predictions.append(0)
        return np.asarray(predictions).reshape(pred.shape)
    
    def model(X, y, params,learning_rate, iterations, reg):
        if (reg == 'Yes'):
            Lambda = 100
            initial_cost = compute_regularized_cost(X, y, params, Lambda)    
            #print("Initial Cost is: {} \n".format(initial_cost))    
            (cost_history, params_gd) = reggradientDescent(X, y, params, learning_rate, iterations, Lambda)           
            
        else:
            initial_cost = compute_cost(X, y, params)    
            #print("Initial Cost is: {} \n".format(initial_cost))         
            (cost_history, params_gd) = gradient_descent(X, y, params, learning_rate, iterations)
               
        return params_gd, cost_history 
   
    m = len(y_train)
    n = np.size(X_train,1)
    params = np.zeros((n,1))
    y_train = y_train[:,np.newaxis]
    y_test = y_test[:,np.newaxis]
    iterations = 50
    learning_rates = np.arange(0.05, 0.11, 0.05)# [0.001, 0.005, 0.01, 0.05, 0.1, 0.03]
    scores = []
    #Train logistic model with different learning rates
    for learning_rate in learning_rates:  
        thetas, ch = model(X_train, y_train, params, learning_rate, iterations, 'No')        
        y_pred = predict(X_test, thetas)
        score = float(sum(y_pred == y_test))/ float(len(y_test))
        print("accuracy at learning rate= ", np.round(learning_rate, 2), " is ", score)
        scores.append(score)
    #Get good estimate of larning rate
    index, value = max(enumerate(scores), key=operator.itemgetter(1))   
    maxlr =  np.round(learning_rates[index],2)
    print("Maximum accuracy is ", value, " at lr = ", maxlr) 
    #Initializations
    m = len(y)
    n = np.size(X,1)
    params = np.zeros((n,1))
    y = y[:,np.newaxis]
    #Train on entire training data with good learning rate 
    print("Now train on the entire training dataset")  
    optimal_thetas, cost_history = model(X, y, params, maxlr, iterations, 'No')
    plt.figure()
    sns.set_style('white')
    plt.plot(range(len(cost_history)), cost_history, 'r')
    plt.title("Convergence Graph of Cost Function")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show() 
    
    #Train regularized logistic model with different learning rates
    print("Training using regularization...")  
    scores = []
    params = np.zeros((n,1))
    for learning_rate in learning_rates:  
        thetas, _ = model(X_train, y_train, params, learning_rate, iterations, 'Yes')        
        y_pred = predict(X_test, thetas)
        score = float(sum(y_pred == y_test))/ float(len(y_test))
        print("accuracy at learning rate= ", np.round(learning_rate, 2), " is ", score)
        scores.append(score)
    #Get good estimate of larning rate    
    index, value = max(enumerate(scores), key=operator.itemgetter(1))   
    print("Maximum accuracy is ", value, " at lr = ", np.round(learning_rates[index],2) ) 
    #Initializations     
    n = np.size(X,1)    
    params = np.zeros((n,1))    
    #Train on entire training data with good learning rate
    print("Now train on the entire training dataset using regularization...")  
    optimal_thetas_reg, cost_history = model(X, y, params, learning_rates[index], iterations, 'Yes')
    plt.figure()
    sns.set_style('white')
    plt.plot(range(len(cost_history)), cost_history, 'r')
    plt.title("Convergence Graph of Cost Function")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Cost")
    plt.show() 
    return optimal_thetas, optimal_thetas_reg

def TestLogistic(X, params, y_true):
    def sigmoidfn(x):
        return 1 / (1 + np.exp(-x))
    pred = sigmoidfn(X @ params)
    predictions = []
    for p in pred:
        if(p >= 0.5):
            predictions.append(1)
        else:
            predictions.append(0)
    y_pred = np.asarray(predictions).reshape(pred.shape)
    y_true = y_true[:,np.newaxis]
#    score = float(sum(y_pred == y_true))/ float(len(y_true))
#    print("accuracy of test data is ", score)    
    cm = confusion_matrix(y_true, y_pred, labels=None)#, sample_weight=None, normalize=None)
    print(cm)
    #accuracy
    accuracy = accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)
    print("accuracy of test data is  " + str(accuracy))
    return y_pred
  
def main():
    # print command line arguments
    train= get_data(sys.argv[1])
    test= get_data(sys.argv[2])
    
    print("cleaning data")
    clean_train_stem,clean_train_nostem= clean(train)
    clean_test_stem, clean_test_nostem= clean(test)
    print("cleaning done")
    print(clean_train_stem.head(5))
    print(clean_train_nostem.head(5))
    
    print("create vectors")
    #Binary and Freq representations for stemmed and nonstemmed data
    traindf_stem_bin, stem_vocab, trainvector_stem_bin, bs_vectorizer= getrep(clean_train_stem, 'binary')
    traindf_stem_freq, stem_vocab, trainvector_stem_freq, fs_vectorizer= getrep(clean_train_stem, 'freq')
    traindf_nostem_bin, nostem_vocab, trainvector_nostem_bin, bn_vectorizer = getrep(clean_train_nostem, 'binary')
    traindf_nostem_freq, nostem_vocab, trainvector_nostem_freq, fn_vectorizer = getrep(clean_train_nostem, 'freq')
    
    #Bonus question
    traindf_stem_tfidf, stem_vocab_tf, trainvector_stem_tfidf, ts_vectorizer= getrep(clean_train_stem, 'tfidf')
    traindf_nostem_tfidf, stem_vocab_tf, trainvector_nostem_tfidf, tn_vectorizer= getrep(clean_train_stem, 'tfidf')
    
       
    #calling naive bayes and logistic regression 4 times each to train on the 4 data vectors above
    #Naive Bayes
    print("**********NAIVE BAYES**********") 
    print("****stem & Binary*******")
    print("Training...")
    likelihood1,priors1 = Naivebayes(clean_train_stem, [0,1], stem_vocab,traindf_stem_bin)
    print("testing...")
    TestNaiveBayes(clean_test_stem, priors1, likelihood1,  [0,1], stem_vocab)
    print("******stem & frequency*******")
    print("Training...")
    likelihood2,priors2 = Naivebayes(clean_train_stem, [0,1], stem_vocab,traindf_stem_freq)
    print("testing...")
    TestNaiveBayes(clean_test_stem, priors2, likelihood2,  [0,1], stem_vocab)
    print("******no-stem & binary*******")
    print("Training...")
    likelihood3,priors3 = Naivebayes(clean_train_nostem, [0,1], nostem_vocab,traindf_nostem_bin)
    print("testing...")
    TestNaiveBayes(clean_test_nostem, priors3, likelihood3,  [0,1], nostem_vocab)
    print("******no-stem & frequency*******")
    print("Training...")
    likelihood4,priors4 = Naivebayes(clean_train_nostem, [0,1], nostem_vocab,traindf_nostem_freq)
    print("testing...")
    TestNaiveBayes(clean_test_nostem, priors4, likelihood4,  [0,1], nostem_vocab)
        
        
    #Logistic Regression
    #betas1,score1 = logisticregression(trainvector_stem_bin,clean_train_stem['Sentiment'])
    
    output = pd.DataFrame()
    output['actual'] = clean_test_stem['Sentiment']
    print("**********LOGISTIC REGRESSION**********")
    print("******stem & frequency*******")
    print("Training...")
    betas_stem, betas_stem_reg = logisticreg(trainvector_stem_freq.toarray(), clean_train_stem['Sentiment'])
    test_stem_vector_freq = fs_vectorizer.transform(clean_test_stem.text)    
    print("Testing...")
    output['stemFreq'] = TestLogistic(test_stem_vector_freq.toarray(), betas_stem, clean_test_stem['Sentiment'])
    print("Testing with regularization...")
    output['stemFreqreg'] = TestLogistic(test_stem_vector_freq.toarray(), betas_stem_reg, clean_test_stem['Sentiment'])
    
    print("******no-stem & frequency*******")
    print("Training...")
    betas_nostem, betas_nostem_reg = logisticreg(trainvector_nostem_freq.toarray(), clean_train_nostem['Sentiment'])
    test_nostem_vector_freq = fn_vectorizer.transform(clean_test_nostem.text)    
    print("Testing...")
    output['nostemFreq'] = TestLogistic(test_nostem_vector_freq.toarray(), betas_nostem, clean_test_nostem['Sentiment'])
    print("Testing with regularization...")
    output['nostemFreqreg'] = TestLogistic(test_nostem_vector_freq.toarray(), betas_nostem_reg, clean_test_nostem['Sentiment'])

    print("******stem & TfIDF*******")
    print("Training...")
    betas_stem, betas_stem_reg = logisticreg(trainvector_stem_tfidf.toarray(), clean_train_stem['Sentiment'])
    test_stem_vector_tfidf = ts_vectorizer.transform(clean_test_stem.text)    
    print("Testing...")
    output['stemtfidf'] = TestLogistic(test_stem_vector_tfidf.toarray(), betas_stem, clean_test_stem['Sentiment'])
    print("Testing with regularization...")
    output['stemtfidfreg'] = TestLogistic(test_stem_vector_tfidf.toarray(), betas_stem_reg, clean_test_stem['Sentiment'])
    
    print("******no-stem & TfIDF*******")
    print("Training...")
    betas_nostem, betas_nostem_reg = logisticreg(trainvector_nostem_tfidf.toarray(), clean_train_nostem['Sentiment'])
    test_nostem_vector_tfidf = tn_vectorizer.transform(clean_test_nostem.text)    
    print("Testing...")
    output['nostemtfidf'] = TestLogistic(test_nostem_vector_tfidf.toarray(), betas_nostem, clean_test_nostem['Sentiment'])
    print("Testing with regularization...")
    output['nostemtfidfreg'] = TestLogistic(test_nostem_vector_tfidf.toarray(), betas_nostem_reg, clean_test_nostem['Sentiment'])
    
    #output.to_csv("D:\\Spring 2020\\assignments\\sentiment_classification\\predictions.csv")
    
    print("Bonus questions:		")			
    print("1. How would the results change if you used term frequency instead of binary representation for both logistic regression and Naïve Bayes (1 point)? ")					
    					
    print("Ans: Binary representations are often expected to perform better in text classification tasks compared to frequency representations. Consider the following test tweet: \[A good, good food and great name, but poor travel.] \n	\n Let us use the two model representations and the two models to classify the sentiment:")					
    print("NB Binary:	Negative"	)			
    print("\n NB Frequency:	Positive")				
    					
    print("\n LR Binary:	Negative")				
    print("\n LR Frequency:	Positive")				
    					
    print("We see that the Binary representations classify tweets like this to negative due to the presence of the word 'bad' irrespective of the presence of the word good multiple times")					
    					
    print("2.What about term frequency x inverse document frequency (1 point)\n" )					
    print("\n Ans: IDF reduces the weight given to common words, and highlights the uncommon words in a document.Theoretically the TF-IDF representation should perform better than just the TF model.")					
    					
    print("\n 3. How do your results change if you regularize your logistic regression (1 point)?\n")					
    					
    print("\n Ans: We will compare the logistic regression method that has the cross entropy loss to the loss with L2 regularization term. We will use penalty value (lambda) of 100 on the euclidean norm of the coefficients. Adding the regularization term ensures the parameters in the logistic regression don't overfit and that a few words are not given high weights.")					

   
if __name__ == "__main__":
    main()
 

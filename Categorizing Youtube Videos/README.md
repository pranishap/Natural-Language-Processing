____________________________________________________________________________

   Categorization of YouTube Videos based on an Analysis of their Comments
____________________________________________________________________________

•	The goal was to classify videos on YouTube which come up in search results as deceptive or not based on their comments.
•	Collected real time data from YouTube using YouTube API (30 search queries, 15 top results for each query, 20 comment threads on each video i.e. 8000 comments in total).
•	Manually tagged the videos as Deceptive and Not Deceptive.
•	Built the model using this data and fine-tuned the same based on cross-validation results.


For Executing the code please follow the below steps:

1). <b>Run Collect.py</b>: (This needs manual tagging and has already been done.)
    This file will collect all the data from youtube. Then it takes the tagged file and creates a new pickle file called actualData.pkl. Running Collect.py with command line arguement 0 (python Collect.py 0) will collect the data from YouTube, search queries should be specified within Collect.py. Running with command line arguement 1(python Collect.py 1), will read the manually tagged excel file and create the pickle file of the data. 

2). <b>DataPreprocessing.py</b>:
    Here preprocessing of the data is done and have implemented baseline methods like predict_random, predict_dominant_class and MultinomialNB and used evalution parameters like F1, accuracy and precision.

3). <b>NBwithNgram.py</b>:
    Implemented Naive Bayes classifier with tri-gram.

4). <b>Model_Latest.py</b>:
    This is the main file where additional pickle files are created as needed by different classifiers. Also have implemented SVM, Neural Network and Multinomial Naive Bayes.

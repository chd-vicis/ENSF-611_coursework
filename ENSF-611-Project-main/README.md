# ENSF 611 - Final Project 

## Introduction
This repository is for the ENSF 611 final project.  The main goal of the project was to compete in a Kaggle competition and test my ML skills.  The competition uses generated data for a supposed spaceship that was on it's way to another planet, but during the trip a "spacetime anomaly" transported roughly half of the passengers to another dimension.  The goal is to try and create a model that can predict if a passenger was transported to another dimension or not based on passenger data.

The models I used included:
1. Random Forest Classifier
2. Logistic Regression
3. Support Vector Machines
4. Gradient Boosting Classifier
5. K-Nearest Neighbours
6. Kmeans Clustering

All the classification models can be found in the two notebooks titled "ML_models....", while the clustering model can be found in the notebook titled "clustering_model".  Both ML notebooks are the same, except with different preprocessing (removing nan values vs. replacing nan values).

The Kaggle competitoin can be found here:
https://www.kaggle.com/competitions/spaceship-titanic/overview

The competition provides you with two sets of data: a test set and training set.  The goal is to use the training set which is comprised of passenger data (which room they stayed it, where they came from, how much they spent on food, etc) to try and predict if they were transported or not.  The test set does not contain a prediction column and it's the job of my model to make those predictions and then submit them to Kaggle to determine how well my models performed.  Each submission receives an accuracy score which is a reflection of how successful a model is. 

The data has categorical data and numerical data.  In the repository you can find the full test and training data along with a sample submission.  All my model predictions can be found in the results folder.

See snapshot of data below.
![alt text](https://github.com/chd-vicis/ENSF-611-Project/blob/main/photos/training_data_snapshot.png?raw=true)

## Goal 

The goal of this project was to see how well the ML techniques that I learned in this course can compete with data scientists and other ML enthusiast.  To test my skills I entered the Kaggle competition mentioned above with the goal of scoring as high as possible on the leaderboard.  More broadly speaking, one of the main challenges of ML is that there are so many techniques and tools available that without experience it is difficult to judge which tools should be used and which should be discarded, so the goal is two-fold: get the highest possible score and obtain real experience that requires on the fly adaptation to maximize my accuracy.  This is a deviation from most procedural assignments and projects which limit the freedom to choose models and parameters.

I also wanted to get a better idea of how difficult it is to obtain higher and higher accuracy score which becomes increasingly more difficult as accuracy reaches 100% and which techniques become more and less important as accuracy rises (is model selection, hyper parameter tuning or pre-processing more important as accuracy becomes more difficult to increase?).  I also want to get a sense of the tradeoff between more powerful techniques and the time/effort required to implement them vs their potential benefit. 

  

## Results 

I was able to obtain a fairly high rank of 386/2,427 which is in the top 20th percentile (at the time of the submission.  The leaderboard updates constantly). 

The best model was Gradient Boosting Classifier with the NaN values replaced with mean/mode value.  As expected, unsupervised learning did not provide nearly as accurate results as supervised due to the lack of information, although more powerful clustering methods might have helped.  It should be noted that using PCA to help graph the data showed that the data did not form neat clusters, so even with more powerful clustering techniques it would still have been challenging to obtain good results.   

It was also interesting to note that most models had roughly the same performance with no model in particular performing extremely bad or good.  Due to lack of time I did not execute a neural net, but it would have been interesting to compare a more complex model with these simpler models.

It was also a trend that the models with nan values replaced with the mean and mode of their respective column depending on if the data was nurmerical or categorical performed slightly better.  This indicates that at least for this test set it is better to keep the rows with missing data and replace values even though this introduced information leakage between samples.


See the full results below. 

Model Performance | Nans Removed | Nans Replaced with mean/mode |
--- | --- | --- |
GBC | 0.802 | 0.804 | 
Logistic Regression | 0.787 | 0.789 | 
SVM | 0.790 | 0.794 | 
Random Forest | 0.789 | 0.788 | 
K-NN | 0.780 | 0.784 | 
K-mean Clustering | --- | 0.485 | 


Note that my username is David which is my middle name.  I tend not to use my full name for online public user displays. 
![alt text](https://github.com/chd-vicis/ENSF-611-Project/blob/main/photos/rank_screenshot.png?raw=true)


See the cluster created below via PCA.  Notice there is no distinct boundary (at least intuitively), so next steps would likely be trying to draw different boundaries based on the training data and then to try to see if there is a clear boundary or not. 

![alt text](https://github.com/chd-vicis/ENSF-611-Project/blob/main/photos/cluster_graph.png?raw=true)

Below is the training data with true predictions.  As you can see there is a distinct cluster, but it is not intuitive so without the training data you could not reasonably cluster the data.  This boundary also isn't suitable for most clustering algorithms because they work on differences in distance between points or other methods to identify distinct clusters.  In theory you could draw this boundary manually and then reapply it to the test data but this would be difficult, time consuming and likely inaccurate in comparision to supervised models.

github.com/chd-vicis/ENSF-611-Project

![alt text](https://github.com/chd-vicis/ENSF-611-Project/blob/main/photos/cluster_graph_true_values_training.png?raw=true)


To address model selection directly, I decided to apply most of the models that I learned in this course instead of just 1 or 2.  This explanation for this was to gain a better understanding of which model is the most powerful, takes the longest to run, is the most sensitive to hyper parameter tuning and which is most sensitive to preprocessing.  As it turns out they are all relatively similar in terms of predictive power but they differ drastically in run time with SVM taking far longer than the rest.  It was also evident that many models were not sensitive to hyper parameter tuning while others were.  For instance random forest was quite insensitive while SVM with a poly kernel seemed to be more sensitive, but not overly sensitive.

## Interpretation
The question of "can I successfully compete in a Kaggle competition" was answered with a yes.  While it took much fine tuning of the models I was able to get a fairly high score.  That said it was obvious that better/more advanced ML techniques would have helped immensely in obtaining better results.  I found that my knowledge of preprocessing probably held me back because I couldn't think of proper ways to use the "passenger group id" in a predictive way so I ended up ignoring this data completely.  I also suspect that had I used a neural net I may have gotten better results.

Another large challenge was that I didn't know how to combine categorical data (were they asleep or not during transit) well with continuous numerical data (how much money was spent on food).  I suspect that my models did not perform well because of this (specifically Naive Bayes). 
  

I also could have used more interesting techniques to replace Nan values.  For instance if there was a NaN value I could have replaced it with the most common value based on their group ID instead of the entire data set.  I also could have done more to understand/test if the decks were related or not (ordinal data vs unordered).  Other than that the project was a success. 


## Reflection
1. I selected this problem because it seemed difficult because of the different types of data used.  It also seemed interesting because the data appeared to have many ways to process and interpret it.  For instance passenger ID have a group component which if I had more time would have been useful to analyze.  I also selected this problem because it seemed within the scope of the course. 
2. The only deviation was that I was not able to quantify the amount of  diminishing returns in comparison to model complexity and time spent.  While I did gain a great intuition for how much time to spend on what problem (preprocessing vs hyper parameter tuning for instance) and how that directly translated to results I was not able to quantify it numerically. 
3. I found the most difficult part to be to try and find the optimal models.  For instance there were many kernels for SVM and it took me quite some time to figure out that some are far more powerful than others.  I also found that many models took extended periods of time to run or never ran at all and it was difficult to understand if it was because of the hyper parameters I was choosing or if it was related to data size/types (based on what I've read online its a combination).  I found the easiest part to be troubleshooting the code.  There was little difficulty in dealing with bugs or error messages because of how simple the python language is and how extension sklearn is.  The most important thing I learned was how  to optimize my time with respect to choosing models.  I found that you could spend hours trying to optimize a model that has mediocre performance while a simpler model generalizes far better (SVM vs random forest).  I also learned that clustering seems to be less powerful but at the same time I think that combining the PCA graph with the test data that is labeled may reveal how the data is clustered. 



# CodingAssessISW

CANDIDATE: ISAIAH S. WILLIAMS
INSTITUTE: UNIVERSITY OF GEORGIA


Coded using Python
Packages used:

pandas
numpy
sklearn
seaborn
statsmodels
matplotlib
decimal




#----------------------------------------------------------------------

1) Unsupervised + supervised learning.

After processing the data I chose to use K-Means Clustering after visualizing through the pairplots, and noticing sepreate clusters of data. I plotted the inertia for each cluster and the difference in distance diminishes past 4 clusters, so the initial model was fitted and labeled for 4 clusters.

I fitted a multinomial logistic regression for the initial model, and model accuracy was 100%. Because the clusters were so distinct, the model accuarcy could easily account for what seperates each label from other labels. Next I fitted a SVM with a linear kernal and it also received a 100% accuracy.

The next part of the code tests what would happen to model accuracy if there were more than 4 clusters. The K-Means works well because it can isolate features from linear relationships to give a class, and linear machine learning models like logistic regression can create functions that map class back to features. However if the K-Means is told to create more clusters, model accuracy for logistic regression and Random Forest plummets.

SVM's model accuracy does not plummet because it can model in multidimensional space.

#----------------------------------------------------------------------

2)Prediction + filtering

I fit 70% of the training data to a polynomial regression of the cleanSine wave. The other 30% I used to see if the model actually captured the sine function and correctly mapped it. I used MSE and R-squared as metrics to compare fits. The MSE and R-Squared had the largest improvement when the degree was 3 compared to 1, but the most "overfitted" when the degree was 5.

The noisySine seemed to be a similar to the cleanSine's function, but cluttered with noise. Before filtering, I attempted to smooth, and use the MSE as an accuracy metric. The fit still had some noise, but very close to the cleanSine.

Next a Hodrick Prescott Filter was used to filter the data, and more closely resembled the cleanSine curve with a bit of noise. The MSE was much smaller than the exponetial smoothing.

#----------------------------------------------------------------------

3)Time Series with Pi

Data Preprocessing to fit the Pi numbers into dataframes.

First, I removed the cycle from the series in order to get the trend line, but it removed the digits and created and interval between 2 and 6 because the mean floats around 4.5 for the sum of the series.

Forecasting with a linear model shows that this is true (mean ~ 4.5), and R-squared below 1%. Specifiying a higher order polynomial barely improves the model because the sequence is so noisy.

The most interesting observation is that each digit almost occurs equally, and not skewed towards any other digit. From a time series perspective, the ability to forecast is a mighty task. I cannot conclude it is irrational based on the methods I used here. However the seemingly random sequence of numbers do not help forecast future numbers.

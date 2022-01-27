import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.api import SimpleExpSmoothing as SES

#Data Prep
X = pd.read_csv("xvalsSine.csv",header=None)
X = X.rename(columns = {0:"X"})
clean = pd.read_csv("cleanSine.csv",header=None)
clean = clean.rename(columns = {0:"y"})
clean_t = pd.concat((X,clean),axis = 1)

#Initial Fig of Clean Sine
fig, ax = plt.subplots(figsize=(10,7.5))
ax.scatter(X,clean, color = 'blue')
ax.set_title('Clean Wave',size= 15)
plt.show()

#Higher Order Regression to fit Sine Wave

X_train, X_test, y_train, y_test = train_test_split(
                                                    clean_t['X'],
                                                    clean_t['y'],
                                                    test_size=0.30)

train = pd.concat((X_train,y_train), axis = 1)

#Specifying 10 Degrees for Polynomial Fit
poly = np.polyfit(train['X'], train['y'],5)
poly1d = np.poly1d(poly)
res = smf.ols(formula='y ~ poly1d(X)', data=train).fit()
y_pred = res.predict(X_test)
    



fig, ax = plt.subplots(figsize=(10,7.5))
ax.scatter(X_train,y_train, color = 'red',label = 'Clean Wave')
ax.scatter(X_test,y_pred, color = 'blue',linestyle='--',label = 'Fitted Values')
ax.legend(frameon = False)
ax.set_xlabel('X Values',size=14)
ax.set_ylabel('Y Values',size=14)
ax.set_title('Fitted Sine Wave',size=20)
plt.show()

#Loop to save MSE and R-Squared to capture optimal polynomial degree
mse = []
r_s = []
for i in range(1,10):
    poly = np.polyfit(train['X'], train['y'],i)
    poly1d = np.poly1d(poly)
    res = smf.ols(formula='y ~ poly1d(X)', data=train).fit()
    y_pred = res.predict(X_test)
    
    mse.append((mean_squared_error(y_test,y_pred)))
    r_s.append(r2_score(y_test,y_pred))
    
fig1, ax1 = plt.subplots(2,figsize=(10,7.5))
ax1[0].plot(range(1,10),mse, color = 'red')
ax1[0].set_ylabel('MSE',size=14)
ax1[0].set_title("Mean Squared Error")

ax1[1].plot(range(1,10),r_s, color = 'blue')
ax1[1].set_xlabel('Poly Degree',size=14)
ax1[1].set_ylabel('R2',size=14)
ax1[1].set_title("R-Squared")

plt.show()

#Noisy Sine Wave Set Up and Visual


noisy = pd.read_csv("noisySine.csv",header=None)
noisy = noisy.rename(columns = {0:"y"})
noisy_t = pd.concat((X,noisy),axis = 1)

fig, ax = plt.subplots(figsize=(10,7.5))
ax.scatter(X,noisy, color = 'red')
ax.set_title('Noisy Sine',size= 15)
plt.show()

#Visualizing Smoothing before Filtering
count = []
smooth = []
for i in np.arange(.01, 1, 0.001):

    fit1 = SES(noisy_t['y'], initialization_method="heuristic").fit(
        smoothing_level=i, optimized=False
    )
    noisy_smooth = pd.DataFrame(fit1.fittedvalues)
    noisy_smooth = noisy_smooth.rename(columns={0:'y'})
    meansq = mean_squared_error(clean['y'], noisy_smooth['y'])
    
    count.append(i)
    smooth.append(meansq)
    
#Storing and plotting MSE

fig, ax = plt.subplots(figsize=(15,10))
ax.scatter(count,smooth, color="black")
ax.set_xlabel('Alpha',size=14)
ax.set_ylabel('MSE',size=14)
ax.set_title('Alpha and MSE Curve',size=20)
plt.show()
mapped = dict(zip(count, smooth))
summary = pd.DataFrame({'alpha': list(mapped.keys()),'mse': list(mapped.values())})

#Plotting Smoothed, Clean, and Noisy Lines
fit_min = SES(noisy_t['y'], initialization_method="heuristic").fit(
    smoothing_level=.1, optimized=False)
fig2, ax2 = plt.subplots(figsize=(15,10))
ax2.plot(clean, color="blue",label = 'Clean')
ax2.plot(noisy, color="red",label = 'Noisy')
ax2.plot(fit_min.fittedvalues, color="black", label='Smoothed')
ax2.set_xlabel('T',size=14)
ax2.set_ylabel('Y',size=14)
ax2.set_title('Smoothed, Clean, Noisy',size=20)
ax2.legend(frameon = False)
plt.show()

meansq_smooth = mean_squared_error(clean['y'], fit_min.fittedvalues)
print(meansq_smooth)
#Smoothing doesn't quite do the job

#Hodrick Prescott Filter, cycle removed

cycle, trend = sm.tsa.filters.hpfilter(noisy_t['y'])

fig, ax = plt.subplots(figsize=(20,10))
ax.plot(noisy_t['y'], color="red",label = 'Noisy')
ax.plot(cycle, color="blue", label = 'Cycle')
ax.plot(trend, color="black", label = 'Trend')
ax.legend(frameon = False)
ax.set_xlabel('T',size=14)
ax.set_ylabel('Y',size=14)
ax.set_title('HP - Filter',size=20)
plt.show()
#Compare MSE with clean Sine
meansq_filter = mean_squared_error(clean['y'], trend)
print(meansq_filter)

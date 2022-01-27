import pandas as pd
import numpy as np
from decimal import getcontext, Decimal
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import SimpleExpSmoothing as SES

def genPiAppxDigits(numdigits,appxAcc):
    getcontext().prec = numdigits
    mypi = (Decimal(4) * sum(-Decimal(k%4 - 2) / k for k in range(1, 2*appxAcc+1, 2)))
    return mypi

pi = genPiAppxDigits(1000,100000)

#Formatting Digits to DataFrame

pi_str = str(pi).replace(".","")
pi_list=list(pi_str)
pi_num = []
for i in pi_list:
    pi_num.append(int(i))

count = [num for num in range(0,1000)]
mapped = dict(zip(count, pi_num))
pi_df = pd.DataFrame({'nth': list(mapped.keys()),'Digit': list(mapped.values())})

pi_train = pi_df[0:950]
pi_test = pi_df[950::]

plt.hist(pi_df['Digit'])
plt.show()
#Visualizations for Smoothing and HP Filters

cycle, trend = sm.tsa.filters.hpfilter(pi_train['Digit'])
fit = SES(pi_train['Digit'], initialization_method="heuristic").fit(
    smoothing_level=.02, optimized=False)

pi_smooth = pd.DataFrame(fit.fittedvalues)
pi_smooth = pi_smooth.rename(columns={0:'Digit'})


fig, ax = plt.subplots(figsize = (20,7.5))
ax.plot(pi_train['Digit'], color = 'blue', label = 'Pi Approx')
ax.plot(cycle, color="red", label = 'Cycle (HPF)')
ax.plot(trend, color="black", label = 'Trend')
ax.plot(pi_smooth, color = 'green', label = 'Smooth')
ax.set_xlabel('nth',size=14)
ax.set_ylabel('Digit',size=14)
ax.set_title('Insights on Detrending',size=20)
ax.legend()

fig2, ax2 = plt.subplots(figsize=(20,8))
ax2.scatter(pi_train['nth'],pi_train['Digit'],color = 'blue')
ax2.set_xlabel('nth',size=14)
ax2.set_ylabel('Digit',size=14)
ax2.set_title('Scatter of Original Series',size=20)


#checking mean for insights
#
print(pi_smooth['Digit'].mean(),pi_train['Digit'].mean())

pi_range = [num for num in range(0,10)]
sum(pi_range)/len(pi_range)

#Poly Regression for prediction, using nth digit as time stamp


poly = np.polyfit(pi_train['nth'], pi_train['Digit'],3)
poly1d = np.poly1d(poly)
res = smf.ols(formula='Digit ~ poly1d(nth)', data=pi_train).fit()
Yhat = res.predict()
pi_train['Yhat'] = Yhat

#creating DataFrame for prediction
forecast_count=[num for num in range(950,1000)]
forecast = pd.DataFrame()
forecast['nth'] = forecast_count
y_pred = res.predict(forecast)
forecast['Digit'] = y_pred

#Visualizing 800 - 1000 digits

fig, ax = plt.subplots(figsize=(20,8))
ax.plot(pi_df['nth'][800::],pi_df['Digit'][800::],color = 'black',label = 'Actual Aprox Pi')
ax.plot(pi_train['nth'][800::],pi_train['Yhat'][800::],color = 'blue',label = 'Fitted Pi')
ax.plot(forecast['nth'],forecast['Digit'],color = 'red',label = '50 Predicted Pi Digits')
ax.set_xlabel('nth',size=14)
ax.set_ylabel('Digit',size=14)
ax.set_title('Prediction',size=20)
ax.legend()
plt.show()



#--------------------------------------------------
pi_1 = genPiAppxDigits(1000,1000)
pi_5 = genPiAppxDigits(1000,5000)
pi_10 = genPiAppxDigits(1000,10000)
pi_50 = genPiAppxDigits(1000,50000)
appx = [pi_1,pi_5,pi_10,pi_50]

pies = []

for i in appx:
    pi_str = str(i).replace(".","")
    pi_list=list(pi_str)
    pi_num = []
    for j in pi_list:
        pi_num.append(int(j))
    pies.append(pi_num)
pi_df['Digit_1K'] = pies[0]
pi_df['Digit_5K'] = pies[1]
pi_df['Digit_10K'] = pies[2]
pi_df['Digit_50K'] = pies[3]

pi_df.corr()
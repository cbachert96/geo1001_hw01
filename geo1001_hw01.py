#-- GEO1001.2020--hw01
#-- Carolin Bachert
#-- 5382998

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.stats import sem, t


S1 = pd.read_excel(r"C:\Users\carol\Documents\Geomatics\Q1 - Sensing technologies and mathematics\hw01/HEAT - A_final.xls", header=3,skiprows=range(4, 5))
S2 = pd.read_excel(r"C:\Users\carol\Documents\Geomatics\Q1 - Sensing technologies and mathematics\hw01/HEAT - B_final.xls", header=3,skiprows=range(4, 5))
S3 = pd.read_excel(r"C:\Users\carol\Documents\Geomatics\Q1 - Sensing technologies and mathematics\hw01/HEAT - C_final.xls", header=3,skiprows=range(4, 5))
S4 = pd.read_excel(r"C:\Users\carol\Documents\Geomatics\Q1 - Sensing technologies and mathematics\hw01/HEAT - D_final.xls", header=3,skiprows=range(4, 5))
S5 = pd.read_excel(r"C:\Users\carol\Documents\Geomatics\Q1 - Sensing technologies and mathematics\hw01/HEAT - E_final.xls", header=3,skiprows=range(4, 5))

############################################# A1 ################################################################################

## S1 ##
print("mean statistics for SA")
print(" ")
print(S1.mean(numeric_only=True))
print(S1.var(numeric_only=True))
print(S1.std(numeric_only=True))

## S2 ##
print("mean statistics for SB")
print(" ")
print(S2.mean(numeric_only=True))
print(S2.var(numeric_only=True))
print(S2.std(numeric_only=True))

## S3 ##
print("mean statistics for SC")
print(" ")
print(S3.mean(numeric_only=True))
print(S3.var(numeric_only=True))
print(S3.std(numeric_only=True))

## S4 ##
print("mean statistics for SD")
print(" ")
print(S4.mean(numeric_only=True))
print(S4.var(numeric_only=True))
print(S4.std(numeric_only=True))

## S5 ##
print("mean statistics for SE")
print(" ")
print(S5.mean(numeric_only=True))
print(S5.var(numeric_only=True))
print(S5.std(numeric_only=True))

### Histograms Temperature ###

fig = plt.figure(figsize=(10,8))
plt.subplots_adjust(wspace = 0.2, hspace = 0.6)
fig.suptitle("Histogram of Tempearture Values for each Sensor - 5 Bins", fontsize = 14)

S1h = fig.add_subplot(321)
S2h = fig.add_subplot(322)
S3h = fig.add_subplot(323)
S4h = fig.add_subplot(324)
S5h = fig.add_subplot(325)

fs = 9 # character height
b = 5 # setting number of bins

S1h.hist(x=S1["Temperature"], bins = b, color='b',alpha=0.7, rwidth=0.85)
S1h.set_title("Sensor A")
S1h.set_ylabel('Frequency',fontsize=fs)
S1h.set_xlabel('Temperature in °C',fontsize=fs)
#S1h.tick_params(labelsize=fs)

S2h.hist(x=S2["Temperature"], bins = b, color='b',alpha=0.7, rwidth=0.85)
S2h.set_title("Sensor B")
S2h.set_ylabel('Frequency',fontsize=fs)
S2h.set_xlabel('Temperature in °C',fontsize=fs)
S2h.tick_params(labelsize=fs)

S3h.hist(x=S3["Temperature"], bins = b, color='b',alpha=0.7, rwidth=0.85)
S3h.set_title("Sensor C")
S3h.set_ylabel('Frequency',fontsize=fs)
S3h.set_xlabel('Temperature in °C',fontsize=fs)
S3h.tick_params(labelsize=fs)

S4h.hist(x=S4["Temperature"], bins = b, color='b',alpha=0.7, rwidth=0.85)
S4h.set_title("Sensor D")
S4h.set_ylabel('Frequency',fontsize=fs)
S4h.set_xlabel('Temperature in °C',fontsize=fs)
S4h.tick_params(labelsize=fs)

S5h.hist(x=S5["Temperature"], bins = b, color='b',alpha=0.7, rwidth=0.85)
S5h.set_title("Sensor E")
S5h.set_ylabel('Frequency',fontsize=fs)
S5h.set_xlabel('Temperature in °C',fontsize=fs)
S5h.tick_params(labelsize=fs)

plt.show()


## Frequency polygon ##

fs = 9

[frequency1,bins]=np.histogram(S1["Temperature"], bins=20)
[frequency2,bins]=np.histogram(S2["Temperature"], bins=20)
[frequency3,bins]=np.histogram(S3["Temperature"], bins=20)
[frequency4,bins]=np.histogram(S4["Temperature"], bins=20)
[frequency5,bins]=np.histogram(S5["Temperature"], bins=20)
cdf_S1 = np.cumsum(frequency1)
cdf_S2 = np.cumsum(frequency2)
cdf_S3 = np.cumsum(frequency3)
cdf_S4 = np.cumsum(frequency4)
cdf_S5 = np.cumsum(frequency5)

fig = plt.figure()
fig.suptitle("Frequency Polygon Temperature for all Sensors", fontsize = 14)
ax=plt.axes()
x=np.linspace(0, 10, 1000)
ax.plot(bins[:-1], cdf_S1, label= "Sensor A")
ax.plot(bins[:-1], cdf_S2, label = "Sensor B")
ax.plot(bins[:-1], cdf_S3, label = "Sensor C")
ax.plot(bins[:-1], cdf_S4, label = "Sensor D")
ax.plot(bins[:-1], cdf_S5, label = "Sensor E")

plt.ylabel('Cumulative number of samples',fontsize=fs)
plt.xlabel('Temperature in °C',fontsize=fs)
plt.legend(prop={'size': 10})
plt.tick_params(labelsize=fs)

plt.show()

## Boxplots ##

### Temperature ###

dat1 = S1["Temperature"]
dat2 = S2["Temperature"]
dat3 = S3["Temperature"]
dat4 = S4["Temperature"]
dat5 = S5["Temperature"]

data_to_plot = [dat1, dat2, dat3, dat4, dat5]

fig = plt.figure(1, figsize=(9, 6))

# Create an axes instance
ax = fig.add_subplot(111)

# Create the boxplot
bp = ax.boxplot(data_to_plot, showmeans = True)
ax.set_title("Boxplots for the measured Temperature for all Sensors", fontsize = 14)
ax.set_ylabel("Temperature in °C", fontsize = 10)
ax.set_xticklabels(['Sensor A', 'Sensor B', 'Sensor C', 'Sensor D', 'Sensor E'], fontsize = 10)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
plt.show()


### Windspeed ###

dat1_ws = S1["Wind Speed"]
dat2_ws = S2["Wind Speed"]
dat3_ws = S3["Wind Speed"]
dat4_ws = S4["Wind Speed"]
dat5_ws = S5["Wind Speed"]

data_to_plot_ws = [dat1_ws, dat2_ws, dat3_ws, dat4_ws, dat5_ws]

fig = plt.figure(1, figsize=(9, 6))
axws = fig.add_subplot(111)

bp_ws = axws.boxplot(data_to_plot_ws, showmeans = True)
axws.set_title("Boxplots for the measured Wind Speed for all Sensors", fontsize = 14)
axws.set_ylabel("Wind Speed in m/s", fontsize = 10)
axws.set_xticklabels(['Sensor A', 'Sensor B', 'Sensor C', 'Sensor D', 'Sensor E'], fontsize = 10)
axws.get_xaxis().tick_bottom()
axws.get_yaxis().tick_left()
plt.show()

### Wind direction ###

dat1_wd = S1["Direction ‚ True"]
dat2_wd = S2["Direction ‚ True"]
dat3_wd = S3["Direction ‚ True"]
dat4_wd = S4["Direction ‚ True"]
dat5_wd = S5["Direction ‚ True"]

print(dat1_wd)

data_to_plot_wd = [dat1_wd, dat2_wd, dat3_wd, dat4_wd, dat5_wd]

fig = plt.figure(1, figsize=(9, 6))
axwd = fig.add_subplot(111)

bp_wd = axwd.boxplot(data_to_plot_wd, showmeans = True)
axwd.set_title("Boxplots for the measured Wind Direction for all Sensors", fontsize = 14)
axwd.set_ylabel("Wind Direction in degrees", fontsize = 10)
axwd.set_xticklabels(['Sensor A', 'Sensor B', 'Sensor C', 'Sensor D', 'Sensor E'], fontsize = 10)
axwd.get_xaxis().tick_bottom()
axwd.get_yaxis().tick_left()
plt.show()

######################################## A2 #################################################################################

tS1 = S1["Temperature"].dropna()
tS2 = S2["Temperature"].dropna()
tS3 = S3["Temperature"].dropna()
tS4 = S4["Temperature"].dropna()
tS5 = S5["Temperature"].dropna()

nb = 20
fs = 12

## PDF ##

fig = plt.figure(figsize=(10,8))
plt.subplots_adjust(wspace = 0.2, hspace = 0.6)
fig.suptitle("Probability Density Function (PDF) - Temperature", fontsize = 14)

ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)

a1 = ax1.hist(x=tS1.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
sns.distplot(tS1.astype(float), color='k',ax=ax1, hist = False)
ax1.set_title("Sensor A")
ax1.set_xlabel('Temperature in °C')
ax1.set_ylabel('Density')
ax1.set_xlim(0, 40)
ax1.set_ylim(0.0, 0.20)

a2 = ax2.hist(x=tS2.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
sns.distplot(tS2.astype(float), color='k',ax=ax2, hist = False)
ax2.set_title("Sensor B")
ax2.set_xlabel('Temperature in °C')
ax2.set_ylabel('Density')
ax2.set_xlim(0, 40)
ax2.set_ylim(0.0, 0.20)

a3 = ax3.hist(x=tS3.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
sns.distplot(tS3.astype(float), color='k',ax=ax3, hist = False)
ax3.set_title("Sensor C")
ax3.set_xlabel('Temperature in °C')
ax3.set_ylabel('Density')
ax3.set_xlim(0, 40)
ax3.set_ylim(0.0, 0.20)

a4 = ax4.hist(x=tS4.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
sns.distplot(tS4.astype(float), color='k',ax=ax4, hist = False)
ax4.set_title("Sensor D")
ax4.set_xlabel('Temperature in °C')
ax4.set_ylabel('Density')
ax4.set_xlim(0, 40)
ax4.set_ylim(0.0, 0.20)

a5 = ax5.hist(x=tS5.astype(float),bins=nb, density=True, color='b',alpha=0.7, rwidth=0.85)
sns.distplot(tS5.astype(float), color='k',ax=ax5, hist = False)
ax5.set_title("Sensor E")
ax5.set_xlabel('Temperature in °C')
ax5.set_ylabel('Density')
ax5.set_xlim(0, 40)
ax5.set_ylim(0.0, 0.20)

plt.show()


## CDF ##

fig = plt.figure(figsize=(10,8))
plt.subplots_adjust(wspace = 0.2, hspace = 0.6)
fig.suptitle("Cumulative Distribution Function (CDF) - Temperature", fontsize = 14)

ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)

a1=ax1.hist(x=tS1.astype(float),bins=nb, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85, label = " Empirical")
ax1.plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='dimgrey', label = "Theoretical")
ax1.set_title("Sensor A")
ax1.set_xlabel('Temperature in °C')
ax1.set_ylabel('CDF')
ax1.set_xlim(0, 35)

a2=ax2.hist(x=tS2.astype(float),bins=nb, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85, label = " Empirical")
ax2.plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2,a2[0], color='dimgrey', label = "Theoretical")
ax2.set_title("Sensor B")
ax2.set_xlabel('Temperature in °C')
ax2.set_ylabel('CDF')
ax2.set_xlim(0, 35)

a3=ax3.hist(x=tS3.astype(float),bins=nb, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85, label = " Empirical")
ax3.plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2,a3[0], color='dimgrey', label = "Theoretical")
ax3.set_title("Sensor C")
ax3.set_xlabel('Temperature in °C')
ax3.set_ylabel('CDF')
ax3.set_xlim(0, 35)

a4=ax4.hist(x=tS4.astype(float),bins=nb, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85, label = " Empirical")
ax4.plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2,a4[0], color='dimgrey', label = "Theoretical")
ax4.set_title("Sensor D")
ax4.set_xlabel('Temperature in °C')
ax4.set_ylabel('CDF')
ax4.set_xlim(0, 35)

a5=ax5.hist(x=tS5.astype(float),bins=nb, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85, label = " Empirical")
ax5.plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2,a5[0], color='dimgrey', label = "Theoretical")
ax5.set_title("Sensor E")
ax5.set_xlabel('Temperature in °C')
ax5.set_ylabel('CDF')
ax5.set_xlim(0, 35)

plt.show()

## PMF ##

def pmf(sample):
    c = sample.value_counts()
    p = c/len(sample)
    return p
df1 = pmf(tS1)
df2 = pmf(tS2)
df3 = pmf(tS3)
df4 = pmf(tS4)
df5 = pmf(tS5)

fig = plt.figure(figsize=(10,8))
plt.subplots_adjust(wspace = 0.3, hspace = 0.6)
fig.suptitle("Probability Mass Function (PMF) - Temperature", fontsize = 14)

c1 = df1.sort_index()
ax1 = fig.add_subplot(321)
ax1.bar(c1.index,c1)
ax1.set_title("Sensor A")
ax1.set_xlabel('Temperature in °C')
ax1.set_ylabel('Probability')
ax1.set_ylim(0.00, 0.025)
ax1.set_xlim(0, 35)

c2 = df2.sort_index()
ax2 = fig.add_subplot(322)
ax2.bar(c2.index,c2)
ax2.set_title("Sensor B")
ax2.set_xlabel('Temperature in °C')
ax2.set_ylabel('Probability')
ax2.set_ylim(0.00, 0.025)
ax2.set_xlim(0, 35)

c3 = df3.sort_index()
ax3 = fig.add_subplot(323)
ax3.bar(c3.index,c3)
ax3.set_title("Sensor C")
ax3.set_xlabel('Temperature in °C')
ax3.set_ylabel('Probability')
ax3.set_ylim(0.00, 0.025)
ax3.set_xlim(0, 35)

c4 = df4.sort_index()
ax4 = fig.add_subplot(324)
ax4.bar(c4.index,c4)
ax4.set_title("Sensor D")
ax4.set_xlabel('Temperature in °C')
ax4.set_ylabel('Probability')
ax4.set_ylim(0.00, 0.025)
ax4.set_xlim(0, 35)

c5 = df5.sort_index()
ax5 = fig.add_subplot(325)
ax5.bar(c5.index,c5)
ax5.set_title("Sensor E")
ax5.set_xlabel('Temperature in °C')
ax5.set_ylabel('Probability')
ax5.set_ylim(0.00, 0.025)
ax5.set_xlim(0, 36)

plt.show()

## PDF and KD for Windspeed ##

wS1 = S1["Wind Speed"].dropna()
wS2 = S2["Wind Speed"].dropna()
wS3 = S3["Wind Speed"].dropna()
wS4 = S4["Wind Speed"].dropna()
wS5 = S5["Wind Speed"].dropna()


fig, axes = plt.subplots (3,2, figsize=(10,8))
fig.suptitle("PDF and Kernel Density for Wind Speed", fontsize = 14)
plt.subplots_adjust(wspace = 0.3, hspace = 0.6)


axes[0,0].set_title("Sensor A")
axes[0,0].set_xlabel('Windspeed in m/s')
axes[0,0].set_ylabel('Density')
sns.distplot(wS1, hist=True, kde=True, bins=27, color = 'darkblue', hist_kws={'edgecolor':'black'}, kde_kws={'linewidth': 2.5}, ax=axes[0,0])

axes[0,1].set_title("Sensor B")
axes[0,1].set_ylabel('Density')
sns.distplot(wS2, hist=True, kde=True, bins=27, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2.5}, ax=axes[0,1])

axes[1,0].set_title("Sensor C")
axes[1,0].set_ylabel('Density')
sns.distplot(wS3, hist=True, kde=True, bins=27, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2.5}, ax=axes[1,0])

axes[1,1].set_title("Sensor D")
axes[1,1].set_ylabel('Density')
sns.distplot(wS4, hist=True, kde=True, bins=27, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2.5}, ax=axes[1,1])

axes[2,0].set_title("Sensor E")
axes[2,0].set_ylabel('Density')
sns.distplot(wS5, hist=True, kde=True, bins=27, color = 'darkblue', 
             hist_kws={'edgecolor':'black'},
             kde_kws={'linewidth': 2.5}, ax=axes[2,0])


plt.show()

############################################### A3 ##################################################################

### Get columns Temperature ###
tS1 = S1["Temperature"]
tS2 = S2["Temperature"]
tS3 = S3["Temperature"]
tS4 = S4["Temperature"]
tS5 = S5["Temperature"]

# Remove na values
tS1 = tS1[~np.isnan(tS1)]
tS2 = tS2[~np.isnan(tS2)]
tS3 = tS3[~np.isnan(tS3)]
tS4 = tS4[~np.isnan(tS4)]
tS5 = tS5[~np.isnan(tS5)]

# Make them same lenght
tS1_i = np.interp(np.linspace(0,len(tS3),len(tS3)),np.linspace(0,len(tS1),len(tS1)),tS1)
tS2_i = np.interp(np.linspace(0,len(tS3),len(tS3)),np.linspace(0,len(tS2),len(tS2)),tS2)
tS3_i = np.interp(np.linspace(0,len(tS3),len(tS3)),np.linspace(0,len(tS3),len(tS3)),tS3)
tS4_i = np.interp(np.linspace(0,len(tS3),len(tS3)),np.linspace(0,len(tS4),len(tS4)),tS4)
tS5_i = np.interp(np.linspace(0,len(tS3),len(tS3)),np.linspace(0,len(tS5),len(tS5)),tS5)

# Normalize values
tS1_n = (tS1_i - tS1_i.mean())/tS1_i.std()
tS2_n = (tS2_i - tS2_i.mean())/tS2_i.std()
tS3_n = (tS3_i - tS3_i.mean())/tS3_i.std()
tS4_n = (tS4_i - tS4_i.mean())/tS4_i.std()
tS5_n = (tS5_i - tS5_i.mean())/tS5_i.std()

# Calculate Coefficients

tab_p, _= stats.pearsonr(tS1_n,tS2_n)
tab_s, _ = stats.spearmanr(tS1_n, tS2_n)

tac_p, _ = stats.pearsonr(tS1_n,tS3_n)
tac_s, _ = stats.spearmanr(tS1_n, tS3_n)

tad_p, _ = stats.pearsonr(tS1_n,tS4_n)
tad_s, _ = stats.spearmanr(tS1_n, tS4_n)

tae_p, _ = stats.pearsonr(tS1_n,tS5_n)
tae_s, _ = stats.spearmanr(tS1_n, tS5_n)

tbc_p, _ = stats.pearsonr(tS2_n,tS3_n)
tbc_s, _ = stats.spearmanr(tS2_n, tS3_n)

tbd_p, _ = stats.pearsonr(tS2_n,tS4_n)
tbd_s, _ = stats.spearmanr(tS2_n, tS4_n)

tbe_p, _ = stats.pearsonr(tS2_n,tS5_n)
tbe_s, _ = stats.spearmanr(tS2_n, tS5_n)

tcd_p, _ = stats.pearsonr(tS3_n,tS4_n)
tcd_s, _ = stats.spearmanr(tS3_n, tS4_n)

tce_p, _ = stats.pearsonr(tS3_n,tS5_n)
tce_s, _ = stats.spearmanr(tS3_n, tS5_n)

tde_p, _ = stats.pearsonr(tS4_n,tS5_n)
tde_s, _ = stats.spearmanr(tS4_n, tS5_n)

xlabel = ["AB", "AC", "AD", "AE", "BC", "BD", "BE", "CD", "CE", "DE"]
pearson_t =[tab_p, tac_p, tad_p, tae_p, tbc_p, tbd_p, tbe_p, tcd_p, tce_p, tde_p]
spearman_t = [tab_s, tac_s, tad_s, tae_s, tbc_s, tbd_s, tbe_s, tcd_s, tce_s, tde_s]



### Get columns Wet Bulb Globe Temperature ###
wS1 = S1["WBGT"]
wS2 = S2["WBGT"]
wS3 = S3["WBGT"]
wS4 = S4["WBGT"]
wS5 = S5["WBGT"]

#Remove na values
wS1 = wS1[~np.isnan(wS1)]
wS2 = wS2[~np.isnan(wS2)]
wS3 = wS3[~np.isnan(wS3)]
wS4 = wS4[~np.isnan(wS4)]
wS5 = wS5[~np.isnan(wS5)]

wS1_i = np.interp(np.linspace(0,len(wS3),len(wS3)),np.linspace(0,len(wS1),len(wS1)),wS1)
wS2_i = np.interp(np.linspace(0,len(wS3),len(wS3)),np.linspace(0,len(wS2),len(wS2)),wS2)
wS3_i = np.interp(np.linspace(0,len(wS3),len(wS3)),np.linspace(0,len(wS3),len(wS3)),wS3)
wS4_i = np.interp(np.linspace(0,len(wS3),len(wS3)),np.linspace(0,len(wS4),len(wS4)),wS4)
wS5_i = np.interp(np.linspace(0,len(wS3),len(wS3)),np.linspace(0,len(wS5),len(wS5)),wS5)

wS1_n = (wS1_i - wS1_i.mean())/wS1_i.std()
wS2_n = (wS2_i - wS2_i.mean())/wS2_i.std()
wS3_n = (wS3_i - wS3_i.mean())/wS3_i.std()
wS4_n = (wS4_i - wS4_i.mean())/wS4_i.std()
wS5_n = (wS5_i - wS5_i.mean())/wS5_i.std()

wab_p, _= stats.pearsonr(wS1_n,wS2_n)
wab_s, _ = stats.spearmanr(wS1_n, wS2_n)

wac_p, _ = stats.pearsonr(wS1_n,wS3_n)
wac_s, _ = stats.spearmanr(wS1_n, wS3_n)

wad_p, _ = stats.pearsonr(wS1_n,wS4_n)
wad_s, _ = stats.spearmanr(wS1_n, wS4_n)

wae_p, _ = stats.pearsonr(wS1_n,wS5_n)
wae_s, _ = stats.spearmanr(wS1_n, wS5_n)

wbc_p, _ = stats.pearsonr(wS2_n,wS3_n)
wbc_s, _ = stats.spearmanr(wS2_n, wS3_n)

wbd_p, _ = stats.pearsonr(wS2_n,wS4_n)
wbd_s, _ = stats.spearmanr(wS2_n, wS4_n)

wbe_p, _ = stats.pearsonr(wS2_n,wS5_n)
wbe_s, _ = stats.spearmanr(wS2_n, wS5_n)

wcd_p, _ = stats.pearsonr(wS3_n,wS4_n)
wcd_s, _ = stats.spearmanr(wS3_n, wS4_n)

wce_p, _ = stats.pearsonr(wS3_n,wS5_n)
wce_s, _ = stats.spearmanr(wS3_n, wS5_n)

wde_p, _ = stats.pearsonr(wS4_n,wS5_n)
wde_s, _ = stats.spearmanr(wS4_n, wS5_n)

pearson_w =[wab_p, wac_p, wad_p, wae_p, wbc_p, wbd_p, wbe_p, wcd_p, wce_p, wde_p]
spearman_w = [wab_s, wac_s, wad_s, wae_s, wbc_s, wbd_s, wbe_s, wcd_s, wce_s, wde_s]

## Get columns Crosswind Speed ##
sS1 = S1["Crosswind Speed"]
sS2 = S2["Crosswind Speed"]
sS3 = S3["Crosswind Speed"]
sS4 = S4["Crosswind Speed"]
sS5 = S5["Crosswind Speed"]

sS1 = sS1[~np.isnan(sS1)]
sS2 = sS2[~np.isnan(sS2)]
sS3 = sS3[~np.isnan(sS3)]
sS4 = sS4[~np.isnan(sS4)]
sS5 = sS5[~np.isnan(sS5)]

sS1_i = np.interp(np.linspace(0,len(sS3),len(sS3)),np.linspace(0,len(sS1),len(sS1)),sS1)
sS2_i = np.interp(np.linspace(0,len(sS3),len(sS3)),np.linspace(0,len(sS2),len(sS2)),sS2)
sS3_i = np.interp(np.linspace(0,len(sS3),len(sS3)),np.linspace(0,len(sS3),len(sS3)),sS3)
sS4_i = np.interp(np.linspace(0,len(sS3),len(sS3)),np.linspace(0,len(sS4),len(sS4)),sS4)
sS5_i = np.interp(np.linspace(0,len(sS3),len(sS3)),np.linspace(0,len(sS5),len(sS5)),sS5)

sS1_n = (sS1_i - sS1_i.mean())/sS1_i.std()
sS2_n = (sS2_i - sS2_i.mean())/sS2_i.std()
sS3_n = (sS3_i - sS3_i.mean())/sS3_i.std()
sS4_n = (sS4_i - sS4_i.mean())/sS4_i.std()
sS5_n = (sS5_i - sS5_i.mean())/sS5_i.std()

sab_p, _= stats.pearsonr(sS1_n,sS2_n)
sab_s, _ = stats.spearmanr(sS1_n, sS2_n)

sac_p, _ = stats.pearsonr(sS1_n,sS3_n)
sac_s, _ = stats.spearmanr(sS1_n, sS3_n)

sad_p, _ = stats.pearsonr(sS1_n,sS4_n)
sad_s, _ = stats.spearmanr(sS1_n, sS4_n)

sae_p, _ = stats.pearsonr(sS1_n,sS5_n)
sae_s, _ = stats.spearmanr(sS1_n, sS5_n)

sbc_p, _ = stats.pearsonr(sS2_n,sS3_n)
sbc_s, _ = stats.spearmanr(sS2_n, sS3_n)

sbd_p, _ = stats.pearsonr(sS2_n,sS4_n)
sbd_s, _ = stats.spearmanr(sS2_n, sS4_n)

sbe_p, _ = stats.pearsonr(sS2_n,sS5_n)
sbe_s, _ = stats.spearmanr(sS2_n, sS5_n)
print(sbe_p)
print(sbe_s)

scd_p, _ = stats.pearsonr(sS3_n,sS4_n)
scd_s, _ = stats.spearmanr(sS3_n, sS4_n)
print(scd_p)
print(scd_s)

sce_p, _ = stats.pearsonr(sS3_n,sS5_n)
sce_s, _ = stats.spearmanr(sS3_n, sS5_n)

sde_p, _ = stats.pearsonr(sS4_n,sS5_n)
sde_s, _ = stats.spearmanr(sS4_n, sS5_n)

pearson_s =[sab_p, sac_p, sad_p, sae_p, sbc_p, sbd_p, sbe_p, scd_p, sce_p, sde_p]
spearman_s = [sab_s, sac_s, sad_s, sae_s, sbc_s, sbd_s, sbe_s, scd_s, sce_s, sde_s]


fig = plt.figure(figsize=(10,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
fig.suptitle("Correlation in between the Sensors for selected Variables", fontsize = 14)

ax1.set_xlabel('Correlation between Sensors')
ax1.set_ylabel("Pearson rank coefficient")
ax1.scatter(xlabel, pearson_t)
ax1.scatter(xlabel, pearson_w)
ax1.scatter(xlabel, pearson_s)

ax2.set_xlabel('Correlation between Sensors')
ax2.set_ylabel('Spearman rank coefficient')
ax2.scatter(xlabel, spearman_t, label = "Temperature")
ax2.scatter(xlabel, spearman_w, label = "Wet Bulb Globe Temperature")
ax2.scatter(xlabel, spearman_s, label = "Crosswind Speed")

#ax1.set_xlabel('Heights [cm]')
plt.legend(prop={'size': 10})
plt.legend(bbox_to_anchor=(0.05, -0.4), loc= 8)
plt.tight_layout()
plt.show()


################################ A4 ################################################################

tS1 = S1["Temperature"].dropna()
tS2 = S2["Temperature"].dropna()
tS3 = S3["Temperature"].dropna()
tS4 = S4["Temperature"].dropna()
tS5 = S5["Temperature"].dropna()

wS1 = S1["Wind Speed"].dropna()
wS2 = S2["Wind Speed"].dropna()
wS3 = S3["Wind Speed"].dropna()
wS4 = S4["Wind Speed"].dropna()
wS5 = S5["Wind Speed"].dropna()

fig = plt.figure(figsize=(10,8))
plt.subplots_adjust(wspace = 0.2, hspace = 0.6)
fig.suptitle("Cumulative Distribution Function (CDF) - Temperature", fontsize = 14)

ax1 = fig.add_subplot(321)
ax2 = fig.add_subplot(322)
ax3 = fig.add_subplot(323)
ax4 = fig.add_subplot(324)
ax5 = fig.add_subplot(325)

a1=ax1.hist(x=tS1.astype(float),bins=nb, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85, label = " Empirical")
ax1.plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='dimgrey', label = "Theoretical")
ax1.set_title("Sensor A")
ax1.set_xlabel('Temperature in °C')
ax1.set_ylabel('CDF')
ax1.set_xlim(0, 35)

a2=ax2.hist(x=tS2.astype(float),bins=nb, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85, label = " Empirical")
ax2.plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2,a2[0], color='dimgrey', label = "Theoretical")
ax2.set_title("Sensor B")
ax2.set_xlabel('Temperature in °C')
ax2.set_ylabel('CDF')
ax2.set_xlim(0, 35)

a3=ax3.hist(x=tS3.astype(float),bins=nb, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85, label = " Empirical")
ax3.plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2,a3[0], color='dimgrey', label = "Theoretical")
ax3.set_title("Sensor C")
ax3.set_xlabel('Temperature in °C')
ax3.set_ylabel('CDF')
ax3.set_xlim(0, 35)

a4=ax4.hist(x=tS4.astype(float),bins=nb, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85, label = " Empirical")
ax4.plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2,a4[0], color='dimgrey', label = "Theoretical")
ax4.set_title("Sensor D")
ax4.set_xlabel('Temperature in °C')
ax4.set_ylabel('CDF')
ax4.set_xlim(0, 35)

a5=ax5.hist(x=tS5.astype(float),bins=nb, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85, label = " Empirical")
ax5.plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2,a5[0], color='dimgrey', label = "Theoretical")
ax5.set_title("Sensor E")
ax5.set_xlabel('Temperature in °C')
ax5.set_ylabel('CDF')
ax5.set_xlim(0, 35)

plt.show()

def cdf_ws(sample1, sample2, sample3,sample4,sample5):
    fig = plt.figure(figsize=(10,8))
    plt.subplots_adjust(wspace = 0.2, hspace = 0.6)
    fig.suptitle("Cumulative Distribution Function (CDF) - Wind Speed", fontsize = 14)

    ax1 = fig.add_subplot(321)
    ax2= fig.add_subplot(322)
    ax3= fig.add_subplot(323)
    ax4= fig.add_subplot(324)
    ax5= fig.add_subplot(325)

    a1=ax1.hist(x=sample1, bins=27, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85)
    ax1.plot(a1[1][1:]-(a1[1][1:]-a1[1][:-1])/2,a1[0], color='dimgrey')
    ax1.set_title("CDF of Sensor A")
    ax1.set_xlabel('Wind Speed in m/s')
    ax1.set_ylabel('Probability')
    #ax1.set_xlim(0, 10)

    a2=ax2.hist(x=sample2, bins=27, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85)
    ax2.plot(a2[1][1:]-(a2[1][1:]-a2[1][:-1])/2,a2[0], color='dimgrey')
    ax2.set_title("CDF of Sensor B")
    ax2.set_xlabel('Wind Speed in m/s')
    ax2.set_ylabel('Probability')
    #ax2.set_xlim(0, 10)

    a3=ax3.hist(x=sample3, bins=27, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85)
    ax3.plot(a3[1][1:]-(a3[1][1:]-a3[1][:-1])/2,a3[0], color='dimgrey')
    ax3.set_title("CDF of Sensor C")
    ax3.set_xlabel('Wind Speed in m/s')
    ax3.set_ylabel('Probability')
    #ax3.set_xlim(0, 10)

    a4=ax4.hist(x=sample4, bins=27, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85)
    ax4.plot(a4[1][1:]-(a4[1][1:]-a4[1][:-1])/2,a4[0], color='dimgrey')
    ax4.set_title("CDF of Sensor D")
    ax4.set_xlabel('Wind Speed in m/s')
    ax4.set_ylabel('Probability')
    #ax4.set_xlim(0, 10)

    a5=ax5.hist(x=sample5, bins=27, density = True, cumulative=True, histtype = "step", color='blue',alpha=0.7, rwidth=0.85)
    ax5.plot(a5[1][1:]-(a5[1][1:]-a5[1][:-1])/2,a5[0], color='dimgrey')
    ax5.set_title("CDF of Sensor E")
    ax5.set_xlabel('Wind Speed in m/s')
    ax5.set_ylabel('Probability')
    #ax5.set_xlim(0, 10)

    plt.show ()
cdf_ws(wS1, wS2, wS3, wS4, wS5)

### Calculate Confidence Intervals and put it in a txt ###

def confidence (sample, c, s):
    confidence = c
    n = len(sample)
    m = np.mean(sample)
    std_err = stats.sem(sample)
    h = std_err * t.ppf((1 + confidence) / 2, n - 1)
    start = m - h
    end = m + h
    start = round(start, 3)
    end = round(end, 3)
    file = open('95% Confidence Intervals.txt','a')    
    file.write(s+ " , " + str(start)+ " , " +str(end) +"\n")
    file.close()
confidence(tS1, 0.95, "Temperature Sensor A")
confidence(tS2, 0.95, "Temperature Sensor B")
confidence(tS3, 0.95, "Temperature Sensor C")
confidence(tS4, 0.95, "Temperature Sensor D")
confidence(tS5, 0.95, "Temperature Sensor E")

confidence(wS1, 0.95, "Wind Speed Sensor A")
confidence(wS2, 0.95, "Wind Speed Sensor B")
confidence(wS3, 0.95, "Wind Speed Sensor C")
confidence(wS4, 0.95, "Wind Speed Sensor D")
confidence(wS5, 0.95, "Wind Speed Sensor E")


def student_t(arr1,arr2):

    data = arr1, arr2
    t, p=stats.ttest_ind(data[0],data[1])
    print(t, p)

student_t(tS5.astype(float).values, tS4.astype(float).values)
student_t(tS4.astype(float).values, tS3.astype(float).values)
student_t(tS3.astype(float).values, tS2.astype(float).values)
student_t(tS2.astype(float).values, tS1.astype(float).values)

student_t(wS5.astype(float).values, wS4.astype(float).values)
student_t(wS4.astype(float).values, wS3.astype(float).values)
student_t(wS3.astype(float).values, wS2.astype(float).values)
student_t(wS2.astype(float).values, wS1.astype(float).values)

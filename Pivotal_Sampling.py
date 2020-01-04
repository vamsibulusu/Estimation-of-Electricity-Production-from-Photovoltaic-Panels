import numpy as np
import random
import pandas as pd
from scipy import stats

from joblib import Parallel, delayed
from multiprocessing import Pool
#A date parser is created to parse the date from the csv file
dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S')

#input is read from the csv file containing the unsampled data
df = pd.read_csv("PATH_TO_INPUT_FILE")
daymax = df.groupby('date').max().reset_index()

# mean inclusion probability gives an idea about the sampling frequency at any point of time.
# The pivotal sampling algorithms sampling frequency varies with varying deviation from the last measured value and charge of the device
# A larger change gives a higher inclusion probability and in turn a more chance of being recorded.
# A lower charge decreases the probabilities to increase the device up time.

mean_pi = []
res = []
# parallel implementation of the pivotal sampling algorithm
def Parallel_pivotal(o):
    print(o)
    for i in range(0,daymax.shape[0]):
        temp = df[df.date == daymax.loc[i].date].copy()
        temp = temp.reset_index()
        temp.columns = ['indexx', 'TimeStamp', 'Ppv', 'date', 'Power', 'pi']
        last = temp.loc[0]
        df.loc[last.indexx, 'pi'] = 1
        for j in range(1,temp.shape[0]):
            x = temp.loc[j].Power
            power_factor = 1.013 - (0.9531*(pow(2.71828182846,-0.0428*x))) #A curve can be fit to meet the required conditions of decrease in sampling rate as power (%charge) decreases.
            #power_factor = 1
            foo = (abs(temp.loc[j].Ppv - last.Ppv) / ((o/100)*temp.Ppv.mean())) * (power_factor)
            if foo > 1:
                foo = 1
            df.loc[temp.loc[j].indexx, 'pi'] = foo
            last = temp.loc[j]
    result = pd.DataFrame()
    a = df.loc[0].pi #pi1. Pivotal sampling algorithm is implement from here as proposed in Sampling Algorithms by Yves Tille
    b = df.loc[1].pi #pi2
    i = 1
    j = 2
    N = df.shape[0]
    s = [0] * N
    k = 2
    while k < N :
        u = random.uniform(0,1)
        if a + b > 1:
            if u < ((1-b)/(2-a-b)):
                b = a + b - 1
                a = 1
            else:
                a = a + b - 1
                b = 1
        else :
            if u < (b / (a + b)):
                b = a + b
                a = 0
            else:
                a = a + b
                b = 0
        if a % 1 == 0 and k <= N:
            s[i] = a
            a = df.loc[k].pi
            i = k
            k = k + 1
        if b % 1 == 0 and k <= N:
            s[j] = b
            b = df.loc[k].pi
            j = k
            k = k + 1
    bb = df.Ppv.values
    samples = []
    for i in range(0,df.shape[0]):
        if s[i] == 1:
            samples.append(bb[i])
    all_hist, all_bins = np.histogram(df.Ppv.values, bins = 100, density = False)
    sample_hist, sample_bins = np.histogram(samples, bins = all_bins, density=False)
    weights = []
    for i in range(0,len(all_bins)-1):
        weights.append((all_bins[i]+all_bins[i+1])/2)
    print(df.pi.mean())
    print(scipy.stats.wasserstein_distance(all_hist, sample_hist,weights,weights))# The wasserstein distance is used to find the divergence between the original and the sampled data.
    return(o,df.pi.mean(),scipy.stats.wasserstein_distance(all_hist, sample_hist,weights,weights))

    result.to_csv("./piv/pivotal_" + str(pii) + ".csv")
    return 0

args = list(range(100,150,1))
p = Pool(9)
results = p.map(Parallel_pivotal, args)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib
from matplotlib import cm
import math
import os
import SS as ss
import TPI as tpi



df_main = pd.read_stata('p13i6.dta')
df_summ = pd.read_stata('rscfp2013.dta')
df = pd.concat([df_main, df_summ], axis=1)
ID = np.arange(1, len(df)+1)
df ['ID'] = ID
df = df[['ID','X5804', 'X5805','X5809', 'X5810', 'X5814', 'X5815', 'X8022', 'networth', 'wgt']]
df.rename(columns={'X5804': 'b1', 'X5805': 't1','X5809':'b2', 'X5810':'t2', 'X5814':'b3', 'X5815':'t3', 'X8022':'age'}, inplace=True)
# print(df.shape)
# print (df['b2'].sum())

#data cleaning
df.loc[df.age<0, 'age'] = 2013+df.loc[df.age<0, 'age']

df.loc[df.t1<2011, 'b1'] = 0
df.loc[df.t2<2011, 'b2'] = 0
df.loc[df.t3<2011, 'b3'] = 0
df['age_b1']=df['age']-(2013-df['t1'])
df['age_b2']=df['age']-(2013-df['t2'])
df['age_b3']=df['age']-(2013-df['t3'])
df = df[((df[['age']].T >=21) & (df[['age']].T <=100) ).any()]
df = df[((df[['age_b1']].T >=21) & (df[['age_b1']].T <=100) ).any()]


#adjust for inflation
df[df.t1 == 2011].b1 *= 0.9652
df[df.t2 == 2011].b2 *= 0.9652
df[df.t3 == 2011].b3 *= 0.9652
df[df.t1 == 2012].b1 *= 0.9854
df[df.t2 == 2012].b2 *= 0.9854
df[df.t3 == 2012].b3 *= 0.9854

total= np.sum(df.b1 * df.wgt) / np.sum(df[df.b1!=0].wgt) * len(df[df.b1!=0]) + \
    np.sum(df.b2 * df.wgt) / np.sum(df[df.b2!=0].wgt) * len(df[df.b2!=0]) + \
    np.sum(df.b3 * df.wgt) / np.sum(df[df.b3!=0].wgt) * len(df[df.b3!=0])

print (total)


def get_bq_tot (df, start_age, end_age, graph=True):
    bq_tot = dict(zip(list(range(start_age, end_age + 1)), [0] * (end_age - start_age + 1)))
    bq_pct = dict(zip(list(range(start_age, end_age + 1)), [0] * (end_age - start_age + 1)))
    for age in range (start_age, end_age + 1):

        sum_age_b1 = np.sum(df[df.age_b1 == age].b1 * df[df.age_b1 == age].wgt)/ np.sum(df[df.age_b1 == age].wgt) * \
                     len(df[df.age_b1 == age])
        # print (f'sum_age_b1:{sum_age_b1}')
        sum_age_b2 = np.sum(df[df.age_b2 == age].b2 * df[df.age_b2 == age].wgt) / np.sum(df[df.age_b2 == age].wgt) * \
                     len(df[df.age_b2 == age])
        # print(f'sum_age_b2:{sum_age_b2}')
        sum_age_b3 = np.sum(df[df.age_b3 == age].b3 * df[df.age_b3 == age].wgt) / np.sum(df[df.age_b3 == age].wgt) * \
                     len(df[df.age_b3 == age])
        # print(f'sum_age_b3:{sum_age_b3}')
        if math.isnan(sum_age_b1): sum_age_b1 = 0
        if math.isnan(sum_age_b2): sum_age_b2 = 0
        if math.isnan(sum_age_b3): sum_age_b3 = 0
        pct_age = (sum_age_b1 + sum_age_b2 + sum_age_b3) / total

        # bq_tot[age] = pct_age
        bq_tot[age] = sum_age_b1 + sum_age_b2 + sum_age_b3
        # print (f'bq_tot[{age}]: {bq_tot[age]}')
        bq_pct[age] = pct_age
        # print(f'bq_pct[{age}]: {bq_pct[age]}')
    if graph:
        cur_path = os.path.split(os.path.abspath(__file__))[0]
        output_fldr = "images"
        output_dir = os.path.join(cur_path, output_fldr)
        if not os.access(output_dir, os.F_OK):
            os.makedirs(output_dir)
        plt.plot(bq_pct.keys(), bq_pct.values(), 'go--', label='bequest distribution across ages')
        plt.grid(b=True, which='major', color='0.65', linestyle='-')
        plt.title('Distribution of Bequest across Ages', fontsize=20)
        plt.xlabel('age')
        plt.ylabel('pct')
        # plt.legend()
        # plt.show()
        output_path = os.path.join(output_dir, 'distribution')
        plt.savefig(output_path)
        plt.close()

    # print(bq_tot.values())
    # print(list(bq_tot.values()))
    return list(bq_pct.values())


bq_tot_all= get_bq_tot(df, 21, 100, graph=True)

# print (df.networth.quantile(.25)[1])
cut_25, cut_50, cut_75, cut_100=df.networth.quantile([.25,0.5,0.75,1]).values
# print (cutoffs.loc[0.25])
df_25 = df[df.networth < cut_25]
df_50 = df[((df[['networth']].T >=cut_25) & (df[['networth']].T <cut_50) ).any()]
df_75 = df[((df[['networth']].T >=cut_50) & (df[['networth']].T <cut_75) ).any()]
df_100 = df[((df[['networth']].T >=cut_75) & (df[['networth']].T <cut_100) ).any()]

# j=np.array([1, 2, 3, 4])

bq_tot_25 = get_bq_tot(df_25, 21, 100, graph=False)
bq_tot_50 = get_bq_tot(df_50, 21, 100, graph=False)
bq_tot_75 = get_bq_tot(df_75, 21, 100, graph=False)
bq_tot_100 = get_bq_tot(df_100, 21, 100, graph=False)

bq_tot_mat=[bq_tot_25/sum(bq_tot_25), bq_tot_50/sum(bq_tot_50), bq_tot_75/sum(bq_tot_75), bq_tot_100/sum(bq_tot_100)]

graph2=True
if graph2:
    cur_path = os.path.split(os.path.abspath(__file__))[0]
    output_fldr = "images"
    output_dir = os.path.join(cur_path, output_fldr)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    nbins = 80
    for c, z, to_plot in zip(['r', 'g', 'b', 'y'], [0, 0.25, 0.5, 0.75], bq_tot_mat ):
        ys = to_plot
        # hist, bins = np.histogram(ys, bins=nbins)
        # xs = (bins[:-1] + bins[1:]) / 2
        xs=list(range(21,101))
        ax.bar(xs, ys, zs=z, zdir='y', color=c, ec=c, alpha=0.8)

    ax.set_xlabel('age')
    ax.set_ylabel('percentile')
    ax.set_zlabel('bequest percentage')
    ax.set_title ('Distribution of Total Bequest Across Ages and Income Quantiles')

    # plt.show()

    output_path = os.path.join(output_dir, 'bq_pct_quantile')
    plt.savefig(output_path)
    plt.close()

print (bq_tot_all)

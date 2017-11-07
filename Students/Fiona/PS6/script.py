import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict



df_main = pd.read_stata('p13i6.dta')
df_summ = pd.read_stata('rscfp2013.dta')
df = pd.concat([df_main, df_summ], axis=1)
ID = np.arange(1, len(df)+1)
df ['ID'] = ID
df = df[['ID','X5804', 'X5805','X5809', 'X5810', 'X5814', 'X5815', 'X8022', 'networth', 'wgt']]
df.rename(columns={'X5804': 'b1', 'X5805': 't1','X5809':'b2', 'X5810':'t2', 'X5814':'b3', 'X5815':'t3', 'X8022':'age'}, inplace=True)
print(df.shape)
print (df['b2'].sum())
bq_pct=dict(zip(list(range(21, 101)), [0]*80))


# df = df[((df[['age']].T >21) & (df[['age']].T <100) ).any()]
print(df.shape)
print (df['b2'].sum())
df.loc[df.t1<2011, 'b1'] = 0
df.loc[df.t2<2011, 'b2'] = 0
df.loc[df.t3<2011, 'b3'] = 0
df['age_b1']=df['age']-(2013-df['t1'])
df['age_b2']=df['age']-(2013-df['t2'])
df['age_b3']=df['age']-(2013-df['t3'])
df = df[((df[['age']].T >=21) & (df[['age']].T <=100) ).any()]
df = df[((df[['age_b1']].T >=21) & (df[['age_b1']].T <=100) ).any()]
print(df.shape)
print (df['b1'].sum())

#adjust for inflation
df[df.t1 == 2011].b1 *= 0.9652
df[df.t2 == 2011].b2 *= 0.9652
df[df.t3 == 2011].b3 *= 0.9652
df[df.t1 == 2012].b1 *= 0.9854
df[df.t2 == 2012].b2 *= 0.9854
df[df.t3 == 2012].b3 *= 0.9854

bq_tot= np.sum(df.b1 * df.wgt) / (np.sum(df[df.b1!=0].wgt))
#+ df.b2 * df.wgt + df.b3 * df.wgt
         #+ np.sum(df[(df[['b2']].T != 0).any()].wgt) +\
         #np.sum(df[(df[['b3']].T != 0).any()].wgt))
print (bq_tot)

for age in range (21, 101):
    pct_age = (np.sum(df[df.age_b1  == age].b1 * df[df.age_b1 == age].wgt) +
               np.sum(df[df.age_b2  == age].b2 * df[df.age_b2 == age].wgt) +
               np.sum(df[df.age_b3 == age].b3 * df[df.age_b3 == age].wgt)) / \
              (np.sum(df[df.age_b1 == age].wgt) +
               np.sum(df[df.age_b2 == age].wgt) +
               np.sum(df[df.age_b3 == age].wgt))
    bq_pct[age] = pct_age

print (sum(bq_pct.values()))
# plt.plot
plt.plot (bq_pct.keys(), bq_pct.values(), 'go--', label = 'bequest distribution across ages')
plt.grid(b=True, which='major', color='0.65', linestyle='-')
plt.title('Distribution of Bequest across Ages', fontsize=20)
plt.xlabel('age')
plt.ylabel('pct')
plt.legend()
# plt.show()
#
# print(df.shape)
# print (df['b2'].sum())
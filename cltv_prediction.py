##############
## TASK 1 ###
#############
#Making a 6-month CLTV prediction for 2010-2011 UK customers.

!pip install Lifetimes
import datetime as dt
import pandas as pd
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

df_ = pd.read_excel("Datasets/online_retail_II.xlsx", sheet_name="Year 2010-2011")
df = df_.copy()
df.head()

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

#Data Pre-Processing

df = df[df["Country"] == "United Kingdom"]
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[df["Quantity"] > 0]

df.describe().T

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]
df["TotalPrice"].head()
df.head()
df.describe().T
today_date = dt.datetime(2011, 12, 11)

#Lifetime Data Structure Preparation

# recency: The elapsed time since the last purchase. Weekly. (according to analysis day in rfm, user specific here)
# T: The age of the customer. Weekly. (how long before the analysis date the first purchase was made)
# frequency: total number of repeat purchases (frequency>1)
# monetary_value: average earnings per purchase

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate" : [lambda date: (date.max() - date.min()).days,
                                                          lambda date: (today_date - date.min()).days],
                                         "Invoice" : lambda num : num.nunique(),
                                         "TotalPrice" : lambda totalprice : totalprice.sum()})

cltv_df.head()

#for hierarchical index problem
cltv_df.columns = cltv_df.columns.droplevel(0)

#naming columns
cltv_df.columns = ['recency', 'T', 'frequency', 'monetary']
cltv_df.head()

# Expressing "monetary value" as average earnings per purchase
cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

#frequency must be greater than 1.
cltv_df = cltv_df[(cltv_df['frequency'] > 1)]

cltv_df.head()

# Choosing monetaries that greater than zero
cltv_df = cltv_df[cltv_df["monetary"] > 0]
cltv_df.head()

# Converting recency and T to weekly for BGNBD
cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7

cltv_df.head(12)


#######################
# Establishing BG/NBD Model
#######################


bgf = BetaGeoFitter(penalizer_coef=0.001)

bgf.fit(cltv_df['frequency'],
        cltv_df['recency'],
        cltv_df['T'])

#Top 10 customers we expect to purchase the most in a week?
bgf.predict(1,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T']).sort_values(ascending=False).head(10)

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])


cltv_df.head(10)

#Top 10 customers we expect to purchase the most in 1 month
bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T']).sort_values(ascending=False).head(10)


cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])

cltv_df.head(10)
#Expected sales of the entire company in 3 months
bgf.predict(4 * 3, cltv_df['frequency'], cltv_df['recency'], cltv_df['T']).sum()

plot_period_transactions(bgf)
plt.show()


#######################
#Establishing the GAMMA-GAMMA Model
#######################



ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])
#en karlı 10 müsteri
ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                        cltv_df['monetary']).sort_values(ascending=False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])

cltv_df.head()

#######################
# Calculation of CLTV with BG-NBD and GG model.
#######################

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head()
cltv = cltv.reset_index()

cltv.sort_values(by="clv", ascending=False).head(50)
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(20)

#Transform between 1-100
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(1, 100))
scaler.fit(cltv_final[["clv"]])
cltv_final["SCALED_CLV"] = scaler.transform(cltv_final[["clv"]])
print("6 months CLTV Prediction for 2010-2011 UK customers")
print("-------------------------------------------------------")
cltv_final.sort_values(by="clv", ascending=False).head()


##############################################################
# TASK 2
##############################################################
#1. Calculate 1-month and 12-month CLTV for 2010-2011 UK customers.
# 2. Analyze the 10 highest individuals at 1 month CLTV and the 10 highest at 12 months. Is there a difference?
# If there is, why do you think it might be?

##################
#1 month calculation

cltv1_month = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  #1 month
                                   freq="W",  # T frequency info
                                   discount_rate=0.01)

cltv1_month.head()

cltv1_month = cltv1_month.reset_index()



cltv1_month = cltv_df.merge(cltv1_month, on ='Customer ID', how ='left')

cltv1_month.head(10)



scaler = MinMaxScaler(feature_range=(0,100))
scaler.fit(cltv1_month[["clv"]])

cltv1_month['scaled_clv'] = scaler.transform(cltv1_month[["clv"]])

#1 month CLTV Prediction of 2010-2011 UK Customers

cltv1_month.sort_values(by="scaled_clv", ascending= False).head(10)
"""
      Customer ID  recency        T  frequency   monetary  expected_purc_1_week  expected_purc_1_month  expected_average_profit        clv  scaled_clv
1485  16000.00000  0.00000  0.42857          3  685.26222               0.41601                1.64094                713.64093 1257.59476   100.00000
2009  17084.00000  0.00000  5.14286          2  737.43750               0.21612                0.85533                784.02677  720.40148    57.28407
1604  16240.00000  7.57143 11.14286          2  872.98000               0.17244                0.68423                927.78213  682.13666    54.24137
589   14096.00000 13.85714 14.57143         17  185.82806               0.72305                2.87354                187.29838  578.42891    45.99486
1265  15531.00000  3.14286  4.42857          2  501.86000               0.24903                0.98516                534.17504  565.30434    44.95123
2499  18139.00000  0.00000  2.71429          6  234.39833               0.52343                2.06949                239.55947  532.52973    42.34510
1954  16984.00000  5.85714 18.71429          2 1120.33750               0.10331                0.41069               1190.12763  525.32023    41.77182
1082  15113.00000  6.28571  7.85714          3  401.03667               0.25595                1.01452                418.16139  455.81120    36.24468
986   14893.00000  0.28571  1.71429          2  309.46250               0.29754                1.17475                330.11970  416.50796    33.11941
1384  15786.00000  9.71429 16.42857          3  548.02222               0.16713                0.66418                570.96688  407.56458    32.40826
"""

####################
#12 Months calculation

cltv12 = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,
                                   freq="W",
                                   discount_rate=0.01)

cltv12.head()
cltv12 = cltv12.reset_index()


cltv12 = cltv_df.merge(cltv12, on ='Customer ID', how ='left')
cltv12.head(10)


scaler = MinMaxScaler(feature_range=(0,100))
scaler.fit(cltv12[["clv"]])

cltv12['scaled_clv'] = scaler.transform(cltv12[["clv"]])

#12 Months CLTV Prediction of 2010-2011 UK Customers

cltv12.sort_values(by="scaled_clv", ascending= False).head(10)

"""
 Customer ID  recency        T  frequency   monetary  expected_purc_1_week  expected_purc_1_month  expected_average_profit         clv  scaled_clv
1485  16000.00000  0.00000  0.42857          3  685.26222               0.41601                1.64094                713.64093 12837.16973   100.00000
2009  17084.00000  0.00000  5.14286          2  737.43750               0.21612                0.85533                784.02677  7472.85377    58.21263
1604  16240.00000  7.57143 11.14286          2  872.98000               0.17244                0.68423                927.78213  7182.78928    55.95306
589   14096.00000 13.85714 14.57143         17  185.82806               0.72305                2.87354                187.29838  6156.47198    47.95817
1265  15531.00000  3.14286  4.42857          2  501.86000               0.24903                0.98516                534.17504  5850.94704    45.57817
1954  16984.00000  5.85714 18.71429          2 1120.33750               0.10331                0.41069               1190.12763  5602.89414    43.64587
2499  18139.00000  0.00000  2.71429          6  234.39833               0.52343                2.06949                239.55947  5497.07151    42.82152
1082  15113.00000  6.28571  7.85714          3  401.03667               0.25595                1.01452                418.16139  4769.26655    37.15201
1384  15786.00000  9.71429 16.42857          3  548.02222               0.16713                0.66418                570.96688  4336.62781    33.78181
986   14893.00000  0.28571  1.71429          2  309.46250               0.29754                1.17475                330.11970  4269.25821    33.25701
"""


# 2. Analyze the 10 highest individuals at 1 month CLTV and the 10 highest at 12 months. Is there a difference?
# If there is, why do you think it might be?
"""
There is no difference between two analysis because top 10 customers are same.
"""

################
# TASK 3
################

#Splitting the scaled values into 4 segments and adding the group names to the dataset
cltv_final["segment"] = pd.qcut(cltv_final["SCALED_CLV"],4,labels=["D","C","B","A"])

cltv_final.head(20)


"""
Customers in Segment A are the best customers and discounts, special offers, etc. can be made to keep them connected to the company. 
Customers in Segment C can be offered promotions to remind ourselves
"""

##############################################################
# TASK 4
##############################################################

cltv_final.head()

cltv_final["Customer ID"] = cltv_final["Customer ID"].astype(int)
cltv_final.to_sql(name='cltv_prediction', con=conn, if_exists='replace', index=False)







# HEDEF BU DOSYADAKI 39302 değerinden daha düşük hataya sahip model kurmak.
# Bunun için ne yapılabilir?
# Yeni değişkenler türebilebilir.
# Zaman değişkeni üzerinde değişiklikler yapılabilir.
# Farklı değişken dönüştürme yöntemleri denenebilir.
# Genel olarak en üst seviyeden ele alınan projeyi değişkenleri daha detaylı inceleyerek ve hatta isterseniz
# notebook'ta ele alarak, analiz ederek inceleyerek ilerleyebilirsiniz.

#1.deneme
#numeric gibi görünen ama kategorik olan 15 değişkeni kategoriğe çevirdim
#'YearBuilt', 'YearRemodAdd', 'GarageYrBlt', değişkenlerini 'YrSold'  dan çıkararak fark değerlerini buluyorum. Kaç sene önce yapıldıklarını anlıyorum
#'YrSold'  değişkeninin max değerine 1 ekleyerek tüm 'YrSold'  değerlerini max+1 değerden çıkarıyorum ve kaç sene önce satıldıklarını buluyorum.
#Yr sold hariç diğer year değişkenlerini 14 kategoride sınıflandırarak kaç yıl önce satıldıklarını zaman aralıklarına dağıtıyorum. Örneğin 6-10 yıl önce gibi
#1000 değerden az değeri olan değişkenleri sildim.
###Sonuç 39.200

#2.deneme
#bir önceki denemdeki numeric kategorik değişken çevirme işi , year işlemleri kaldı
#Yeni bir değişken açarak yeni ve eski evleri sınıflandırdım ###( YENİ )###
#1000 değişken silme işlemi de kaldı.
####Sonuç39.800 :)


# 1. GEREKLILIKLER

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV


import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)



# train ve test setlerinin bir araya getirilmesi.
train = pd.read_csv("5.Hafta/Dataset/train.csv")
test = pd.read_csv("5.Hafta/Dataset/test.csv")
df = train.append(test).reset_index()
df.head()



def data_house_price():
    dataframe = pd.read_csv("5.Hafta/Dataset/train.csv")
    return dataframe

# 2. EDA


#Histogram heatmap ile genel bakış
corela=df.corr()
cols = corela.nlargest(15, 'SalePrice').index
corr = df.corr()
cordf=df[cols].corr()
sns.set(font_scale=1.4)
f,ax=plt.subplots(figsize=(11,9))
sns.heatmap(cordf,annot=True,annot_kws={'size': 13},  linewidths=1.5,cmap="YlOrBr", fmt='.1f',)
plt.show()

corela=df.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corela,vmin = 0,vmax=1,square=True,cmap="YlOrBr",ax=ax)
plt.show()


# KATEGORIK DEGISKEN ANALIZI
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']
print('Kategorik Değişken Sayısı: ', len(cat_cols))



def cat_summary(data, categorical_cols, target, number_of_classes=10):
    var_count = 0
    vars_more_classes = []
    for var in categorical_cols:
        if len(df[var].value_counts()) <= number_of_classes:  # sınıf sayısına göre seç
            print(pd.DataFrame({var: data[var].value_counts(),
                                "Ratio": 100 * data[var].value_counts() / len(data),
                                "TARGET_MEDIAN": data.groupby(var)[target].median()}), end="\n\n\n")
            var_count += 1
        else:
            vars_more_classes.append(data[var].name)
    print('%d categorical variables have been described' % var_count, end="\n\n")
    print('There are', len(vars_more_classes), "variables have more than", number_of_classes, "classes", end="\n\n")
    print('Variable names have more than %d classes:' % number_of_classes, end="\n\n")
    print(vars_more_classes)

cat_summary(df, cat_cols, "SalePrice")


# 10'dan fazla sınıfı olan değişkenler:
for col in ['Neighborhood', 'Exterior1st', 'Exterior2nd']:
    print(df[col].value_counts())



# SAYISAL DEGISKEN ANALIZI
num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in "Id"]
print('Sayısal değişken sayısı: ', len(num_cols))



def hist_for_nums(data, numeric_cols):
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")


hist_for_nums(df, num_cols)

#Aslında numeric olarak geçen ama kategorik değişken sayılabilecekleri ayırıyorum.
def cat_count(data,num,number_of_classes=17):
    num_but_cat= 0
    col_counter_name = []
    for var in num:
        if len(df[var].value_counts())<=number_of_classes:
            num_but_cat += 1
            col_counter_name.append(df[var].name)
    print("%d adet değişken aslında kategorik" % num_but_cat,end="\n\n")
    print("Kategorik değişkenler: " , num_but_cat,end="\n\n")
    print(col_counter_name)


cat_count(df,num_cols)

#Elde ettiğimiz değişkenleri bir listeye topluyorum ( Listeye toplamadan işlem yapamadım! ) (Return kullanmadığım için mi ? )
cat_count1 = [col for col in num_cols if len(df[col].value_counts()) <= 17]
print("Numeric gibi görünen kategorik değişkenler:", cat_count1)

#Değişkenlerin tiplerini tekrar kontrol ediyorum
num= 0
for i in cat_count1:
    if df[i].dtype !="O":
        num +=1

len(cat_count1)


### for döngüsü ile cat_list içerisindeki değişkenlerin tipini kategorik yapıyorum.
for i in cat_count1:
    df[i]=df[i].astype(str)


#tek bir değişkeni manuel kontrol ediyorum
df["MSSubClass"].dtype

#num_cols döngüsünü tekrar çalıştırarak içerisinden kategoriklerin çıkmasını sağlıyorum
num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in "Id"]

#cat_count1 'i tekrar çalıştırarak 17 değerden az değişken olup olmadığını kontrol ettim.
cat_count1 = [col for col in num_cols if len(df[col].value_counts()) <= 17]

#cat_count1 listemiz 15'di. Şu an 0
len(cat_count1)

#cat_cols listesini de tekrar çalıştırarak kategorik değişkenlerin eklenmesini sağlıyoruz
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']

#43'tü 58 oldu
len(cat_cols)


# TARGET ANALIZI
df["SalePrice"].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99])


# target ile bagımsız degiskenlerin korelasyonları
def find_correlation(dataframe, corr_limit=0.60):
    high_correlations = []
    low_correlations = []
    for col in num_cols:
        if col == "SalePrice":
            pass

        else:
            correlation = dataframe[[col, "SalePrice"]].corr().loc[col, "SalePrice"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


low_corrs, high_corrs = find_correlation(df)

#Sale Price ile yüksek oranda ilişkisi olan değişkenler
high_corrs
#TotalBsmtSF , 1stFlrSF, GrLivArea , GarageArea

# yüksek ilişkili olanların grafiğini inceledim
sns.jointplot(df["SalePrice"],df["GrLivArea"],df)
plt.show()

#Log Bölümünü Deniyorum
# LOG TRANSFORMATION
"""When checking distribution of the dependent variable (SalePrice),  this is not a normal distribution. 
It shows a positive skewness and kurtosis. Log transformation is a good choose."""

print("Skewness: %f" % df['SalePrice'].skew())
print("Kurtosis: %f" % df['SalePrice'].kurt())

# Applying the log1p function log(1+x) to “SalePrice

df["SalePrice"] = np.log1p(df["SalePrice"])




# 3. DATA PREPROCESSING & FEATURE ENGINEERING

#İçerisinde Year ve Yr geçen değişkenleri çıkartıyoruz
year_name = [col for col in df.columns if "Year" in col or "Yr" in col]
year_name
#['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
df["YearBuilt"].sort_values(ascending=False)

names=['YearBuilt', 'YearRemodAdd']
names
#işlemlere başlamadan önce YrSold değerinin yapım tarihlerinden küçük olduğu indexleri bulmaya çalıştım. ama olmadı.
#def big_year_find(data,years):
    #big_year=[]
    #for var in years:
        #if var == "YrSold":
         #   pass
        #else:
         #   if ((data["YrSold"].astype(int))-(data[var].astype(int)) < 0).sum():
          #      big_year.append(df[var].values)
        #print(df[var].name)
         #   return big_year

#big_year_find(df,names)


#YearBuilt  ve diğer değişkenleri satış yılından çıkararak kaç yıllıkken satıldığını buluyorum
df["YearBuilt2"]=df["YrSold"].astype(int)-df["YearBuilt"].astype(int)
df["YearRemodAdd2"]=df["YrSold"].astype(int)-df["YearRemodAdd"].astype(int)
df["GarageYrBlt2"]=df["YrSold"].astype(int)-df["GarageYrBlt"]

names2=["YearBuilt2","YearRemodAdd2","GarageYrBlt2"]

# Eksi değer olanları tek tek kontrol ettim
df["YearBuilt2"].values < 0
df["YearRemodAdd2"].sort_values(ascending=True)
df["YearBuilt2"].sort_values(ascending=True)
df["GarageYrBlt2"].sort_values(ascending=True)

#ilgili indexleri sildim
#df.drop([2592,2295,523,2549],inplace=True)

#En son satılan evin tarihini buluyorum. Üzerine 1 ekliyorum.
a=int(df["YrSold"].max())+1

#2011 yılında olduğumuzu varsayarak kaç sene önce satıldıklarını çıkarıyorum.
df["YrSold2"]=a-(df["YrSold"].astype(int))

df.GarageYrBlt.dtype


df.drop(["YearBuilt","YearRemodAdd","GarageYrBlt"],axis=1,inplace=True)


#num_cols'tan yearbuilt ve remodadd değişkenlerini çıkarıyoruz.
num_cols.remove("YearBuilt")
num_cols.remove("YearRemodAdd")
#Yrsold'u df ten silip sonra cat cols tan çıkarıyorum
df.drop(["YrSold"], axis=1, inplace=True)
cat_cols.remove("YrSold")

df["YrSold2"]=df["YrSold2"].astype(int)
df["YrSold2"].dtype
num_cols.append("YrSold2")



#Yapılan yıl işlemlerinden sonra kategorik olup değişkene dönen YrSold değişkenini tekrar düzenliyorum
#for i in cat_count1:
 #   df[i]=df[i].astype(str)

#num_cols döngüsünü tekrar çalıştırarak içerisinden kategoriklerin çıkmasını sağlıyorum
num_cols = [col for col in df.columns if df[col].dtypes != 'O' and col not in "Id"]

"GarageYrBlt" in num_cols

#cat_count1 'i tekrar çalıştırarak 17 değerden az değişken olup olmadığını kontrol ettim.
cat_count1 = [col for col in num_cols if len(df[col].value_counts()) <= 17]

#cat_count1 listemiz tekrar 0 oldu
len(cat_count1)
cat_count1

#cat_cols listesini de tekrar çalıştırarak kategorik değişkenlerin eklenmesini sağlıyoruz
cat_cols = [col for col in df.columns if df[col].dtypes == 'O']

#yapılan değişikliklerden sonra 58 olarak gerçekleşti.
len(cat_cols)

#df["YearBuilt"].between(5,31).sum()
#(df["YearBuilt"].values == 0).sum()
#df["YearBuilt"].head()
#df["YearBuilt"].value_counts()

#forgap=["YearBuilt","YearRemodAdd","GarageYrBlt"]
#forgap


#gap - aralıkları ayarlama / Yapım yıllarına göre sınıflandırma yaptım
def gap(dataframe,gap_name):
    for variable in gap_name:
        if variable == "YrSold":
            pass
        else:
            dataframe.loc[(dataframe[variable].values == 0), variable] = 0
            dataframe.loc[(dataframe[variable].values == 1), variable] = 1
            dataframe.loc[(dataframe[variable].values == 2), variable] = 2
            dataframe.loc[(dataframe[variable].values == 3), variable] = 3
            dataframe.loc[(dataframe[variable].values == 4), variable] = 4
            dataframe.loc[(dataframe[variable].between(5,10)), variable] = 5
            dataframe.loc[(dataframe[variable].between(11,15)), variable] = 6
            dataframe.loc[(dataframe[variable].between(16,20)), variable] = 7
            dataframe.loc[(dataframe[variable].between(21,25)), variable] = 8
            dataframe.loc[(dataframe[variable].between(26,30)), variable] = 9
            dataframe.loc[(dataframe[variable].between(31,40)), variable] = 10
            dataframe.loc[(dataframe[variable].between(41,50)), variable] = 11
            dataframe.loc[(dataframe[variable].between(51,70)), variable] = 12
            dataframe.loc[(dataframe[variable].between(71,90)), variable] = 13
            dataframe.loc[(dataframe[variable].values >= 91), variable] = 14


gap(df,names2)


#Index ve ID'nin veri steinden silinmesi
#df.drop(["index","Id"],axis=1, inplace=True)
#num_cols içerisinden indexi çıkarıyoruz.
#num_cols.remove("index")
num_cols

#ev tiplerini belirten MsSubClass değişkenini ele alıyoruz
df["MSSubClass"].describe().T

df["MSSubClass"].value_counts()

df.groupby("MSSubClass").agg({"SalePrice":["max", "min", "mean"]}).T

#Yeni bir değişken türetiyoruz
df["new_old"]=""

#new_old değişkenine MSSubClass değerlerini atadım.
df["new_old"]=[col for col in df["MSSubClass"].values ]

#yeni, eski, tüm yaşları içeren ve diğerlerini 1-4 arası sıralandırıyorum.
#20,60,120,160, newer = 1
#30,70,older = 2
#40,45,50,75,90,150,190 all ages = 3
#80,85,180 other = 4

df.loc[df.new_old.isin(["60", "20","160", "120"]),"new_old"] = 1
df.loc[df.new_old.isin(["30", "70"]),"new_old"] = 2
df.loc[df.new_old.isin(["40","45","50","75","90","150","190"]),"new_old"] = 3
df.loc[df.new_old.isin(["80","85","180"]),"new_old"] = 4
df["new_old"].value_counts()

#yeni oluşturduğumuz new_old değişkeninin tipini konrol edip ilgili listeye ekliyorum
df["new_old"].dtype
cat_cols.append("new_old")
cat_cols

#Genel imar sınıflandırmasını inceliyorum
df["MSZoning"].value_counts()

#eksik değer olup olmadığını kontrol ediyorum
df["MSZoning"].isnull().sum()

#Groupby ile imar sınıfına göre satış fiyatı ve feet kare alanlarını inceliyorum.
df.groupby("MSZoning").agg({"SalePrice":["max","min","mean"],"GrLivArea":["max","min","mean"]})

#Groupby ile ms zoningdeki eksik değerlerin feet kare değerlerini inceliyorum
df.groupby(df["MSZoning"].isnull()).agg({"GrLivArea":["max","min","mean"]})

#Eksik değerlerin index numaralarını topluyorum
df[df['MSZoning'].isnull()].index.tolist()

#MSzoningteki eksik değerlerini ev tipine, büyüklüğüne, yapım yılına vs göre değerlendiriyorum
df.loc[df["MSZoning"].isnull(),("MSSubClass","GrLivArea","YearBuilt2","new_old","YrSold2")]

#Mszoningte ev tipi 30 olan kaç adet değer var onu kontrol ediyorum
df.loc[df["MSSubClass"] == "30",("MSZoning")].value_counts()

#Mszoningte 14 yıl önce yapılan kaç değer var onu kontrol ediyorum
df.loc[df["YearBuilt2"] == 14,("MSZoning")].value_counts()

#Mszoningte ev büyüklüğü 730-1836 feet kare olan kaç ev var onu inceliyorum
df.loc[df["GrLivArea"].between(730,1836),("MSZoning")].value_counts()

#yaptığım incelemelere göre eksik değerleri RL olarak doldurmaya karar verdim.
df.loc[df["MSZoning"].isnull(),"MSZoning"]="RL"
#kontrol ettiğimde eksik değer yok
df["MSZoning"].isnull().sum()

#SalePrice ile yüksek ilişkisi olan overallqual değişkenini inceliyorum
df["OverallQual"].value_counts().sort_values(ascending=False)
#eksik değer olmadığı için şu an herhangi bir işlem yapmıyorum.
df["OverallQual"].isna().sum()
#Overall condition değerleri de tam
df["OverallCond"].isna().sum()

#1stFlrSF ( 1.katın feet kare değeri) inceliyorum
df["1stFlrSF"].isnull().sum()

#Zaten GrLivArea'da toplam feet kare verildiği için 1. ve 2.kat değerlerini siliyorum
df.drop(["1stFlrSF","2ndFlrSF"],axis=1,inplace=True)
num_cols.remove("1stFlrSF")
num_cols.remove("2ndFlrSF")

#Veranda alanlarının toplayıp tek bir değişken oluşturuyoruz.
df["TotalPorch"] =df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]

#tek bir değişken üretip diğer 4 değişkeni siliyorum
df.drop(["OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch"],axis=1,inplace=True)
num_cols.remove("OpenPorchSF")
num_cols.remove("EnclosedPorch")
num_cols.remove("3SsnPorch")
num_cols.remove("ScreenPorch")

#Genel kalite ve şartların değerlendirildiği değişkenleri tek bir değişken olarak oluşturuyorum.
df["OverallQual"].isnull().sum()
df["OverallCond"].isnull().sum()

df["TotalQual"]=(df["OverallQual"].astype(int))+(df["OverallCond"].astype(int))
df.drop(["OverallQual","OverallCond"], axis=1, inplace=True)

cat_cols.remove("OverallQual")
cat_cols.remove("OverallCond")

#Banyo sayılarını gösteren tek bir değişken türetiyoruz
#BsmtFullBath hata verdiği için yapamadım.
#df["TotalBath"]=(df["BsmtFullBath"].astype(int))+(df["BsmtHalfBath"].astype(int)+df["FullBath"].astype(int))+(df["HalfBath"].astype(int))

#Zaten FullBath olarak toplam değişkeni varmış. Onun için diğerlerini siliyorum.
df.drop(["BsmtFullBath","HalfBath","BsmtHalfBath"], axis=1, inplace=True)
#df["FullBath"]=df["FullBath"].astype("O")
#df["FullBath"].dtype
cat_cols.remove("BsmtFullBath")
cat_cols.remove("HalfBath")
cat_cols.remove("BsmtHalfBath")


#Kalite değerlendirmesinde 5 değer olan değişkenleri numaralandırıp toplayacağım
df["ExterCond"].dtype

qual=["ExterCond","HeatingQC","KitchenQual","PoolQC","ExterQual"]

def gap_qual(dataframe,gap_name):
    for variable in gap_name:
        if dataframe[variable].dtype == "O":
            dataframe.loc[(dataframe[variable].values == "Ex"), variable] = 1
            dataframe.loc[(dataframe[variable].values == "Gd"), variable] = 2
            dataframe.loc[(dataframe[variable].values == "TA"), variable] = 3
            dataframe.loc[(dataframe[variable].values == "Fa"), variable] = 4
            dataframe.loc[(dataframe[variable].values == "Po"), variable] = 5

gap_qual(df,qual)

#KitchenQuall ve PoolQc değişkenlerinde eksik değerleri 0'a eşitledim. Çünkü 0 o özelliğin o evde olmadığını gösteriyor
df["KitchenQual"] = df.loc[(df["KitchenQual"].isnull().sum() ), "KitchenQual"] = 0
df["PoolQC"] = df.loc[(df["PoolQC"].isnull().sum() ), "PoolQC"] = 0

# Kategorik olanları numeriğe çevirip cat cols içerisinde çıkarmak istedim ama return edemedim.
def convert_int(dataframe,liste):
    for var in liste:
        if dataframe[var].dtype == "O":
            dataframe[var].astype(int)
            if var in cat_cols:
                cat_cols.remove(var)

convert_int(df,qual)

#Yeni bir değişken üretiyorum
df["TotalQual2"]=(df["ExterCond"].astype(int))+(df["HeatingQC"].astype(int))+(df["KitchenQual"].astype(int))+(df["PoolQC"].astype(int))+(df["ExterQual"].astype(int))

#DF'den ve ilgili kategori listelerinden siliyorum
df.drop(["ExterCond","HeatingQC","KitchenQual","ExterQual","PoolQC"], axis=1, inplace=True)
#aşağıdaki değişkenler ilgili listelerde değilmiş.
#cat_cols.remove("ExterCond")
#cat_cols.remove("HeatingQC")
cat_cols.remove("KitchenQual")
#cat_cols.remove("ExterQual")
cat_cols.remove("PoolQC")


df["Street"].value_counts()
df["Street"].isnull().sum()

df["Alley"].value_counts()
df["Alley"].isnull().sum()

df.groupby("Alley").agg({"GrLivArea":"mean"})

#Alley değişkeninde pave değerinde new oldu değişkenine baktığımda çoğunluğu 1(yani yeni ev )
(df.loc[(df["Alley"].values=="Pave"),"new_old"]).value_counts()

#Alley değişkeninde Grvl değerinde new oldu değişkenine baktığımda çoğunluğu 2(yani eski ev )
(df.loc[(df["Alley"].values=="Grvl"),"new_old"]).value_counts()

#Alley değişkeninde eksik olan değerlerin new old değişkenindeki karşılıklarına bakıyorum.
(df.loc[(df["Alley"].isnull()),"new_old"]).value_counts()

#newold değişkeninde 1 olanlar yeni evler. yeni olduğu için de pave yani asfalt ile dolduruyorum.
##Yapamadım!!!
df.loc[(df["Alley"].isnull()),"new_old"].values==1

#Değerleri inceliyorum
df["BsmtQual"].isnull().sum()
df["BsmtQual"].values=="nan"
df["GarageCond"].dtype
#df.loc[(df["FireplaceQu"].isnull()), "FireplaceQu"]= 0
df["BsmtQual"].isnull().any()

#eksik değerleri 0 a eşitliyoruz. bu o dairede bu özelliğin olmadığını gösteriyor
df.loc[(df["BsmtQual"].isnull()), "BsmtQual"] = 0
df.loc[(df["BsmtCond"].isnull()), "BsmtCond"] = 0
df.loc[(df["FireplaceQu"].isnull()), "FireplaceQu"] = 0
df.loc[(df["GarageQual"].isnull()), "GarageQual"] = 0
df.loc[(df["GarageCond"].isnull()), "GarageCond"] = 0


qual2=["BsmtQual","BsmtCond","FireplaceQu","GarageQual","GarageCond"]

def gap_qual2(dataframe,gap_name):
    for variable in gap_name:
        if dataframe[variable].dtype == "O":
            dataframe.loc[(dataframe[variable].values == "Ex"), variable] = 1
            dataframe.loc[(dataframe[variable].values == "Gd"), variable] = 2
            dataframe.loc[(dataframe[variable].values == "TA"), variable] = 3
            dataframe.loc[(dataframe[variable].values == "Fa"), variable] = 4
            dataframe.loc[(dataframe[variable].values == "Po"), variable] = 5
            dataframe.loc[(dataframe[variable].values == "NA"), variable] = 0
            dataframe.loc[(dataframe[variable].values == "nan"), variable] = 0
            dataframe.loc[(dataframe[variable].isnull().sum()), variable] = 0


gap_qual(df,qual2)

#Yeni bir değişken oluşturarak tüm koşulları topluyorum
df["TotalCond2"]= (df["BsmtQual"].astype(int))+(df["BsmtCond"].astype(int))+(df["FireplaceQu"].astype(int))+(df["GarageQual"].astype(int))+(df["GarageCond"].astype(int))

#DF'den ve ilgili kategori listelerinden siliyorum
df.drop(["BsmtQual","BsmtCond","FireplaceQu","GarageQual","GarageCond"], axis=1, inplace=True)
#aşağıdaki değişkenler ilgili listelerde değilmiş.
cat_cols.remove("BsmtQual")
cat_cols.remove("BsmtCond")
cat_cols.remove("FireplaceQu")
cat_cols.remove("GarageQual")
cat_cols.remove("GarageCond")

#df["Alley"].dtype

#"Street" in num_cols
#"Alley" in cat_cols

#cat_cols.remove("Alley")
#df.drop(["Alley"], axis=1, inplace=True)


# RARE ANALYZER
def rare_analyser(dataframe, target, rare_perc):
    rare_columns = [col for col in df.columns if len(df[col].value_counts()) <= 20
                    and (df[col].value_counts() / len(df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        print(var, ":", len(dataframe[var].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[var].value_counts(),
                            "RATIO": dataframe[var].value_counts() / len(dataframe),
                            "TARGET_MEDIAN": dataframe.groupby(var)[target].median()}), end="\n\n\n")


rare_analyser(df, "SalePrice", 0.01)



def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]
    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])
    return temp_df

df = rare_encoder(df, 0.01)
rare_analyser(df, "SalePrice", 0.01)

### 1000 değerden daha az olan değişkenleri sildim.
def delete_low_classes(dataframe,low_number=1000):
    for col in dataframe.columns:
        if (dataframe[col].count() <=low_number):
            dataframe.drop(col, axis=1, inplace=True)
    return dataframe

delete_low_classes(df).shape
df.shape

### Kategorik değişken listesinden sildiğimiz değerleri çıkartıyoruz.
#cat_cols.remove('PoolQC')
cat_cols.remove('MiscFeature')
cat_cols.remove('Alley')
cat_cols.remove('Fence')

# LABEL ENCODING & ONE-HOT ENCODING
def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=nan_as_category, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns

#rare içeren değişken kaldığı için hata var. o çıkartıldı
df, new_cols_ohe = one_hot_encoder(df, cat_cols)
cat_summary(df, new_cols_ohe, "SalePrice")


#df.info()
#liste=[]

#for i in df.columns:
    #if df[i].dtype == "O":
        #liste.append(i)

#df[liste].head()

#df.drop("BsmtFinSF1", axis=1, inplace=True)

# MISSING_VALUES
def missing_values_table(dataframe):
    variables_with_na = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[variables_with_na].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[variables_with_na].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df)
    return variables_with_na

num_cols.remove("SalePrice")

missing_values_table(df)
df[num_cols] = df[num_cols].apply(lambda x: x.fillna(x.median()), axis=0)
missing_values_table(df)




# OUTLIERS
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def has_outliers(dataframe, num_col_names, plot=False):
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, ":", number_of_outliers)
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    return variable_names


has_outliers(df, num_cols)


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in num_cols:
    replace_with_thresholds(df, col)

has_outliers(df, num_cols)


# STANDARTLASTIRMA

df.info()
liste=[]




df.head()
like_num = [col for col in df.columns if df[col].dtypes != 'O' and len(df[col].value_counts()) < 20]
cols_need_scale = [col for col in df.columns if col not in new_cols_ohe
                   and col not in "Id"
                   and col not in "SalePrice"
                   and col not in like_num]

df[cols_need_scale].head()
df[cols_need_scale].describe([0.05, 0.10, 0.25, 0.50, 0.75, 0.80, 0.90, 0.95, 0.99]).T
hist_for_nums(df, cols_need_scale)

#df.drop(["Alley","Street"], axis=1, inplace=True)

def robust_scaler(variable):
    var_median = variable.median()
    quartile1 = variable.quantile(0.25)
    quartile3 = variable.quantile(0.75)
    interquantile_range = quartile3 - quartile1
    if int(interquantile_range) == 0:
        quartile1 = variable.quantile(0.05)
        quartile3 = variable.quantile(0.95)
        interquantile_range = quartile3 - quartile1
        z = (variable - var_median) / interquantile_range
        return round(z, 3)
    else:
        z = (variable - var_median) / interquantile_range
    return round(z, 3)


for col in cols_need_scale:
    df[col] = robust_scaler(df[col])



df[cols_need_scale].head()
df[cols_need_scale].describe().T
hist_for_nums(df, cols_need_scale)


# son kontrol
missing_values_table(df)
has_outliers(df, num_cols)



# 4. MODELLEME

train_df = df[df['SalePrice'].notnull()]
test_df = df[df['SalePrice'].isnull()]

test_df.drop("SalePrice",axis=1,inplace=True)
train_df.shape, test_df.shape

train_df.to_pickle("5.Hafta/Dataset/prepared_data/train_df_.pkl")
test_df.to_pickle("5.Hafta/Dataset/prepared_data/test_df_.pkl")
#
# train_df = pd.read_pickle("5.Hafta/Dataset/prepared_data/train_df_.pkl")
# test_df = pd.read_pickle("5.Hafta/Dataset/prepared_data/test_df_.pkl")


# train & test ayrımını yapalım:
# X = train_df.drop('SalePrice', axis=1)
# y = np.ravel(train_df[["SalePrice"]])
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)
# y_train = np.ravel(y_train)  # boyut ayarlaması

# train_df tüm veri setimiz gibi davranarak derste ele aldığımız şekilde modelelme işlemini gerçekleştiriniz.
X = train_df.drop('SalePrice', axis=1)
y = train_df[["SalePrice"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=46)

from scipy.stats import norm
# #We use the numpy fuction log1p which  applies log(1+x) to all elements of the column
# y_train = np.log1p(y_train)
# sns.distplot(y_train , fit=norm);
# plt.show()

#train_df.to_pickle("5.Hafta/Dataset/prepared_data/train_df_.pkl")
#test_df.to_pickle("5.Hafta/Dataset/prepared_data/test_df_.pkl")



# TODO scaler'i burada çalıştırıp deneyebilirsiniz.

models = [('LinearRegression', LinearRegression()),
          ('Ridge', Ridge()),
          ('Lasso', Lasso()),
          ('ElasticNet', ElasticNet())]

# evaluate each model in turn
results = []
names = []

for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    result = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append(result)
    names.append(name)
    msg = "%s: %f" % (name, result)
    print(msg)

np.expm1(df["SalePrice"].mean())

#LinearRegression: 113475.620133
#Ridge: 9.424712
#Lasso: 9.423908
#ElasticNet: 9.423908



#train_df.to_pickle("5.Hafta/Dataset/prepared_data/train_df_.pkl")
#test_df.to_pickle("5.Hafta/Dataset/prepared_data/test_df_.pkl")

X_train.shape, test_df.shape

#test_df.drop("SalePrice",axis=1,inplace=True)


# Kaggle için


model = Ridge()
model.fit(X_train, y_train)
y_pred = model.predict(test_df)
ridge_pred = np.expm1(model.predict(test_df))

test2 = pd.read_csv("5.Hafta/Dataset/test.csv")
sub = pd.DataFrame()
sub['Id'] = test2.Id
sub['SalePrice'] = ridge_pred
sub.to_csv('submission42.csv',index=False)

test2.shape
test_df.shape






#All the imports and data import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sklearn.preprocessing as skl_pre
import sklearn.linear_model as skl_lm
import sklearn.discriminant_analysis as skl_da
import sklearn.neighbors as skl_nb
import sklearn.model_selection as skl_ms
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import seaborn as sns

import os, sys

cwd = os.getcwd()
URI = (cwd+"\\train.csv")
film_data = pd.read_csv(URI, dtype={"Lead":str}).dropna().reset_index(drop=True)

#Creating dummy values of the category "Lead"
film_data_dummies = pd.get_dummies(film_data).copy()

#Adding two columns that summerizes the number of words M/F with the lead's words if the same gender
film_data_dummies = film_data_dummies.assign(total_words_male=lambda row: (row['Number words male'] + row['Number of words lead'] * row['Lead_Male']))
film_data_dummies = film_data_dummies.assign(total_words_female=lambda row: (row['Number words female'] + row['Number of words lead'] * row['Lead_Female']))


#Only saving "Lead_Male" as category and renaming it to "Lead"
film_data_dummies.drop("Lead_Female",1, inplace=True)
y = film_data_dummies['Lead_Male'].rename("Lead")
x = film_data_dummies.drop(columns=['Lead_Male'])




"""
#Heatmap with new dataset correlation
plt.figure(figsize=(12,12))
cor = film_data_new.corr()
sns.heatmap(cor, annot=True, cmap="coolwarm",center=0.0)
plt.show()

"""

"""
#average nr spoken words (M/F) per film per year

df_avg_male_words = df.groupby("Year")["Number words male"].mean()
df_avg_female_words = df.groupby("Year")["Number words female"].mean()
df_avg_total_words= df.groupby("Year")["Total words"].mean()

df_avg_total_words_plot = df_avg_total_words.plot(kind="line", y="Total words", x="Year", color="g", label="Avg number of words per film")
df_avg_male_words_plot = df_avg_male_words.plot(kind="line", y="Number words male", x="Year", color="b", label="Avg number of Male words per film", ax=df_avg_total_words_plot)
df_avg_female_words_plot = df_avg_female_words.plot(kind="line", y="Number words female", x="Year", color ="r",label="Avg number of Female words per film", xlabel="Year", ylabel="Proportion spoken words", title="Avg number of spoken of words per year", ax=df_avg_male_words_plot)
plt.legend()
plt.show()



#Proportion of spoken words (M/F) per film per year
def proportion(x, y):
    x_new = x/(x+y)
    return x_new


df['Proportion female words'] = df.apply(lambda row : proportion(row['Number words female'],row['Total words']), axis = 1)
df['Proportion male words'] = df.apply(lambda row : proportion(row['Number words male'], row['Total words']), axis = 1)

df_male_words_proportion = df.groupby("Year")["Proportion male words"].median()
df_female_words_proportion = df.groupby("Year")["Proportion female words" ].median()

df_male_words_proportion_plot = df_male_words_proportion.plot(kind="line", y="Proportion male words", x="Year", color="b", label="Mean proportion of Male words per year")
df_female_words_proportion_plot = df_female_words_proportion.plot(kind="line", y="Proportion female words", x="Year", color ="r",label="Mean proportion of Female words per year", xlabel="Year", ylabel="Proportion spoken words", title="Proportion of words spoken (M/F) per year", ax=df_male_words_proportion_plot)
plt.legend()
plt.show()



#NUMBER OF FILMS WITH M/F LEAD
df_male = film_data.loc[film_data['Lead'] == "Male",]
df_female = film_data.loc[film_data['Lead'] == "Female"]

df_male_leads = df_male.groupby("Year", as_index=True)["Lead"].count()
df_female_leads = df_female.groupby("Year", as_index=True)["Lead"].count()
df_number_films = film_data.groupby("Year", as_index=True)["Lead"].count()


df_male_leads.columns = ["Year", "Films with male lead"]
df_female_leads.columns = ["Year", "Films with female lead"]

df1 = df_number_films.plot(kind="line", y="Lead",x="Year", color="g", label="Total number of analyzed films")
df2 = df_male_leads.plot(kind="line", y="Films with male lead",x="Year", color="b", label="Nr films with male lead", ax=df1)
df3 = df_female_leads.plot(kind="line", y="Films with female lead",x="Year", color="r", label="Nr films with female lead", ax=df2, xlabel="Year", title="Number of films with leads m/f per year")

plt.legend()
plt.show()
"""



###ALL PLOTTING
##
#

#Spoken word to movie gross
df = film_data_dummies.copy()
#df= df.groupby("Gross")["total_words_male"].mean()

plt.subplot(2,2,1)
ax_male_word_gross = plt.scatter(df["Gross"], df["total_words_male"], color="b", label="Number of Male words to movie gross")
ax_female_word_gross = plt.scatter(df["Gross"], df["total_words_female"], color="r", label="Number of Female words to movie gross")
plt.xlabel("Gross")
plt.ylabel("Words")
plt.title("M/F number of words to gross")
plt.legend()

df = film_data_dummies.copy()

#average nr spoken words (M/F) per film per year
df_avg_male_words = df.groupby("Year")["total_words_male"].mean()
df_avg_female_words = df.groupby("Year")["total_words_female"].mean()
df_avg_total_words= df.groupby("Year")["Total words"].mean()
plt.legend()

plt.subplot(2,2,2)
df_avg_total_words_plot = df_avg_total_words.plot(kind="line", y="Total words", x="Year", color="g", label="Avg number of words per film")
df_avg_male_words_plot = df_avg_male_words.plot(kind="line", y="total_words_male", x="Year", color="b", label="Avg number of Male words per film", ax=df_avg_total_words_plot)
df_avg_female_words_plot = df_avg_female_words.plot(kind="line", y="total_words_female", x="Year", color ="r",label="Avg number of Female words per film", xlabel="Year", ylabel="Avg spoken words", title="Avg number of spoken of words per year", ax=df_avg_male_words_plot)
plt.legend()


#Proportion of spoken words (M/F) per film per year
def proportion(x, y):
    x_new = x/y
    return x_new

df['Proportion female words'] = df.apply(lambda row : proportion(row['total_words_female'],row['Total words']), axis = 1)
df['Proportion male words'] = df.apply(lambda row : proportion(row['total_words_male'], row['Total words']), axis = 1)

df_male_words_proportion = df.groupby("Year")["Proportion male words"].mean()
df_female_words_proportion = df.groupby("Year")["Proportion female words" ].mean()

plt.subplot(2,2,3)
df_male_words_proportion_plot = df_male_words_proportion.plot(kind="line", y="Proportion male words", x="Year", color="b", label="Mean proportion of Male words per year")
df_female_words_proportion_plot = df_female_words_proportion.plot(kind="line", y="Proportion female words", x="Year", color ="r", \
    label="Mean proportion of Female words per year", xlabel="Year", ylabel="Proportion spoken words", title="Proportion of words spoken (M/F) per year", \
        ax=df_male_words_proportion_plot)
plt.legend()


#NUMBER OF M/F actors avg per film
df_male_actors = df.groupby("Year", as_index=True)["Number of male actors"].mean()
df_female_actors = df.groupby("Year", as_index=True)["Number of female actors"].mean()

df_male_actors.columns = ["Year", "Number of male actors"]
df_female_actors.columns = ["Year", "Number of female actors"]

plt.subplot(2,2,4)
df2 = df_male_actors.plot(kind="line", y="Number of male actors",x="Year", color="b", label="Number of Male actors")
df3 = df_female_actors.plot(kind="line", y="Number of female actors",x="Year", color="r", label="Number of Male actors", ax=df2, xlabel="Year", ylabel="Actors", title="Mean of M/F actors per film (yearly avg)",figsize=(15,10))


plt.legend()
plt.show()
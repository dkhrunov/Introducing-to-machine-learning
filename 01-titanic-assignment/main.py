import numpy as np
import pandas as pd
import re


data = pd.read_csv("titanic.csv", index_col="PassengerId")

data.head(10)


#How many men and women were traveling by ship?
len_male = len([m for m in data['Sex'] if m == 'male'])
len_female = len([f for f in data['Sex'] if f == 'female'])
print("{} {}".format(len_male,len_female))

# Pandas Alternative option
sex_counts = data['Sex'].value_counts()
print(sex_counts)


# What part of the passengers managed to survive?
allPass = len(data)
sur = np.count_nonzero(data["Survived"])
print("{:.2f}".format(sur*100/allPass))

# Pandas Alternative option
surv_counts = data['Survived'].value_counts()
surv_percent = 100.0 * surv_counts[1] / surv_counts.sum()
print('{:.2f}'.format(surv_percent))


# What percentage of the first class passengers were among all passengers?
fClass = len(data["Pclass"].where(data["Pclass"] == 1).dropna())
print("{:.2f}".format(fClass*100/allPass))

# Pandas Alternative option
fclass_counts = data["Pclass"].value_counts()
fclass_percent = 100 * fclass_counts[1] / fclass_counts.sum()
print('{:.2f}'.format(fclass_percent))


# How old were the passengers?
ages = data['Age'].dropna()
print("{0:.2f} {1:.2f}".format(ages.mean(), ages.median()))


# Do the number of brothers / sisters / spouses correlate with the number of parents / children?
print("R-Pirson: {:.2f}".format(data['SibSp'].corr(data['Parch'])))


# What is the most popular female name on the ship?
def clean_name(name):
    # The first word before the comma is the surname
    s = re.search('^[^,]+, (.*)', name)
    if s:
        name = s.group(1)

    # If there are braces, that is the name of the passenger in them
    s = re.search('\(([^)]+)\)', name)
    if s:
        name = s.group(1)

    # Deleting appeals
    name = re.sub('(Miss\. |Mrs\. |Ms\. )', '', name)

    # We take the first remaining word and remove the quotes
    name = name.split(' ')[0].replace('"', '')

    return name


names = data[data['Sex'] == 'female']['Name'].map(clean_name)
name_counts = names.value_counts()
print(name_counts.head(1).index.values[0])






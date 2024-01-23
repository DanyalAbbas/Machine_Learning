import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# DUMMY VARIABLE METHOD

df = pd.read_csv("homeprices.csv")
# print(df)

dummies = pd.get_dummies(df.town)
merged = pd.concat([df,dummies], axis= "columns")

# print(merged)

final = merged.drop(["town", "west windsor"], axis= "columns")
# print(final)

model = LinearRegression()

x_ = final.drop("price", axis= "columns").values
y_ = final.price.values

model.fit(x_,y_)

# print(x_)
# print(model.predict([[2800, 0,1]]))

print(model.score(x_,y_))

# ONE HOT ENCODING METHOD

# le =LabelEncoder()

# dfle = df

# dfle.town = le.fit_transform(dfle.town)

# x = dfle[["town", "area"]].values
# y = dfle.price

# ohe = OneHotEncoder(categories= [0])
# x = ohe.fit_transform(x)

# print(x)

# print(dfle)


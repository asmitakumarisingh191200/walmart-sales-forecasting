import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\asmit\Downloads\Python\Walmart.csv")

df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)

print(df.head())
print(df.info())
print(df.isnull().sum())
print(df.duplicated().sum())


plt.figure()
sns.lineplot(x='Date', y='Weekly_Sales', data=df)
plt.title("Weekly Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Weekly Sales")
plt.show()


plt.figure()
sns.barplot(x='Holiday_Flag', y='Weekly_Sales', data=df)
plt.title("Holiday Flag vs Weekly Sales")
plt.xlabel("Holiday Flag")
plt.ylabel("Weekly Sales")
plt.show()


plt.figure()
sns.scatterplot(x='Temperature', y='Weekly_Sales', data=df)
plt.title("Temperature vs Weekly Sales")
plt.xlabel("Temperature")
plt.ylabel("Weekly Sales")
plt.show()


X = df[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']]
y = df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

print(mean_absolute_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred))
print(np.sqrt(mean_squared_error(y_test, y_pred)))
print(r2_score(y_test, y_pred))


df['Sales_Class'] = np.where(
    df['Weekly_Sales'] > df['Weekly_Sales'].median(), 1, 0)

Xc = df[['Temperature', 'Fuel_Price', 'CPI', 'Unemployment', 'Holiday_Flag']]
yc = df['Sales_Class']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    Xc, yc, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=1000)
clf.fit(Xc_train, yc_train)
yc_pred = clf.predict(Xc_test)

print(accuracy_score(yc_test, yc_pred))
print(confusion_matrix(yc_test, yc_pred))
print(classification_report(yc_test, yc_pred))



cm = confusion_matrix(yc_test, yc_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()


scaler = StandardScaler()
scaled_data = scaler.fit_transform(
    df[['Weekly_Sales', 'Temperature', 'Fuel_Price']])

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)

plt.figure()
sns.scatterplot(
    x='Temperature',
    y='Weekly_Sales',
    hue='Cluster',
    data=df)
plt.title("K-Means Clustering of Weekly Sales")
plt.xlabel("Temperature")
plt.ylabel("Weekly Sales")
plt.show()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = {
    "StudyHours":[2,3,4,5,6,7,8,9,10],
    "Attendance":[70,75,80,85,88,90,92,95,98],
    "Assignments":[5,6,7,8,9,9,10,10,10],
    "Marks":[45,52,60,68,75,82,88,94,98]
}

df = pd.DataFrame(data)

X = df[["StudyHours","Attendance","Assignments"]]
y = df["Marks"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

prediction = model.predict([[6,90,9]])

print("Predicted Marks:", round(prediction[0],2))

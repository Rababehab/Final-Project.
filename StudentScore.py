# FINAL PROJECT: Student Exam Score Prediction
# -------------------------------------------------

# 1. Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split   #دالة تقسيم البيانات الي مجموعة تدريب و اختبار
from sklearn.linear_model import LinearRegression  #استدعاء نمو\ج الانحدار الخطي
from sklearn.metrics import mean_squared_error, r2_score # لحساب مقاييس التقييم R2 , MSE

# 2. Load dataset
data = pd.read_csv("StudentScores.csv")   
print(data.head())

# Show column names
print(data.head())
print(data.columns)

# drop rows with missing values (or we can fill them)
data = data.dropna()

# student_id مش مهم في التنبؤ → هنشيله
data = data.drop("student_id", axis=1)

# 4. Visualization
plt.figure(figsize=(6,4)) #صندوق رسم جديد
plt.scatter(data["hours_studied"], data["exam_score"], color="blue") # لرسم النقاط بين قيم العمودين
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Hours Studied vs Exam Score")
plt.show()
plt.figure(figsize=(6,4))
plt.scatter(data["attendance_percent"], data["exam_score"], color="green")
plt.xlabel("Attendance Percent")
plt.ylabel("Exam Score")
plt.title("Attendance vs Exam Score")
plt.show()

# 5. Split data
X = data.drop("exam_score", axis=1)   # features
y = data["exam_score"]                # target

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42) # 20% اختبار و 80% تدريب

# 6. perform Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)

# 7. Evaluate model (prediction)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test , y_pred)
print(" Mean Squared Error:", mse)
print(" R-squared:", r2, "\n")

# 8. Plot Actual vs Predicted
plt.figure(figsize=(6,4))
plt.scatter(y_test, y_pred, color="purple")
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Actual vs Predicted Exam Scores")
plt.show()

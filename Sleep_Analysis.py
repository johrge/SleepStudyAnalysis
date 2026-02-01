# Sleep and Lifestyle Health Analysis
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import ttest_ind

# Load dataset
# =============================================================================
# DATASET ATTRIBUTION
# Title: Sleep and Lifestyle Health Dataset
# Author/Creator: shraddha4ever20
# Repository: Kaggle
# URL: https://www.kaggle.com/datasets/shraddha4ever20/sleep-and-lifestyle-health-dataset
# Access Date: January 18, 2026
# =============================================================================

df = pd.read_csv(r"C:\Users\johrg\OneDrive\Documents\Job Resources\Data Set\sleep_health.csv")
df = df.rename(columns={
    "REMSleepPercentage": "REM_Sleep_Percent",
    "DeepSleepPercentage": "Deep_Sleep_Percent",
    "LightSleepPercentage": "Light_Sleep_Percent"
})
print(df.columns)

# If file loads as one column, split it
if df.shape[1] == 1:
    df = df.iloc[:, 0].str.split(",", expand=True)
    
    df.columns = [
        "ID","Age","Gender","Bedtime","WakeupTime","SleepDuration",
        "SleepEfficiency","REM_Sleep_Percent","Deep_Sleep_Percent",
        "Light_Sleep_Percent","Awakenings","CaffeineConsumption",
        "AlcoholConsumption","SmokingStatus","ExerciseFrequency"
    ]

# ================================
# Data Type Cleaning
# ================================

numeric_cols = [
    "Age","SleepDuration","SleepEfficiency","REM_Sleep_Percent",
    "Deep_Sleep_Percent","Light_Sleep_Percent","Awakenings",
    "CaffeineConsumption","AlcoholConsumption","ExerciseFrequency"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["Bedtime"] = pd.to_datetime(df["Bedtime"], errors="coerce")
df["WakeupTime"] = pd.to_datetime(df["WakeupTime"], errors="coerce")

df = df.dropna()

print(df.head())
print(df.info())

# ================================
# NumPy statistics on sleep duration
# ================================

mean_sleep = np.mean(df["SleepDuration"])
median_sleep = np.median(df["SleepDuration"])
std_sleep = np.std(df["SleepDuration"])

print("Mean Sleep Duration:", mean_sleep)
print("Median Sleep Duration:", median_sleep)
print("Sleep Duration Std Dev:", std_sleep)

# ================================
# Feature Engineering
# ================================

# Create sleep category feature
df.loc[:, "SleepCategory"] = np.where(
    df["SleepDuration"] < 7, "Short Sleep",
    np.where(df["SleepDuration"] <= 9, "Normal Sleep", "Long Sleep")
)

print(df["SleepCategory"].value_counts())

# Create high caffeine consumption flag
df.loc[:, "HighCaffeine"] = np.where(
    df["CaffeineConsumption"] >= 200, 1, 0)

print(df["HighCaffeine"].value_counts())

# ================================
# EDA
# ================================

sleep_by_gender = df.groupby("Gender")["SleepDuration"].mean()
sleep_by_exercise = df.groupby("ExerciseFrequency")["SleepDuration"].mean()

print(sleep_by_gender)
print(sleep_by_exercise)

correlation = df["SleepDuration"].corr(df["ExerciseFrequency"])
print("Correlation between Sleep Duration and Exercise Frequency:", correlation)

# ================================
# Visualizations
# ================================

plt.figure(figsize=(7,4))
plt.hist(df["SleepDuration"], bins=15)
plt.title("Distribution of Sleep Duration")
plt.xlabel("Hours of Sleep")
plt.ylabel("Frequency")
plt.show()

plt.figure(figsize=(6,4))
sleep_by_gender.plot(kind="bar")
plt.title("Average Sleep Duration by Gender")
plt.xlabel("Gender")
plt.ylabel("Hours of Sleep")
plt.show()

plt.figure(figsize=(7,4))
sleep_by_exercise.plot(kind="line")
plt.title("Sleep Duration vs Exercise Frequency")
plt.xlabel("Exercise Frequency")
plt.ylabel("Hours of Sleep")
plt.show()

print("Analysis Complete")

# ================================
# Hypothesis Testing
# ================================

# Separate sleep duration by gender
male_sleep = df[df["Gender"] == "Male"]["SleepDuration"]
female_sleep = df[df["Gender"] == "Female"]["SleepDuration"]

# Perform independent t-test
t_stat, p_value = ttest_ind(male_sleep, female_sleep, equal_var=False)

print("T-Statistic:", t_stat)
print("P-Value:", p_value)

# Interpretation
if p_value < 0.05:
    print("Statistically significant difference in sleep duration by gender")
else:
    print("No statistically significant difference in sleep duration by gender")

plt.show()


print("Analysis Complete")

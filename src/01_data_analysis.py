import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

df = pd.read_csv("../data/student_performance.csv")

print("shape:", df.shape)
print("\ncolumns:", list(df.columns))
print("\ninfo:")
print(df.info())
print("\ndescribe:")
print(df.describe())
print("\nnull values:", df.isnull().sum().sum())
print("duplicated rows:", df.duplicated().sum())

# target distribution
sns.histplot(df["Performance Index"], kde=True, color="steelblue")
plt.title("Performance Index Distribution")
plt.tight_layout()
plt.show()

# hours studied vs performance
sns.scatterplot(x=df["Hours Studied"], y=df["Performance Index"],
                hue=df["Hours Studied"], palette="viridis")
plt.title("Hours Studied vs Performance Index")
plt.tight_layout()
plt.show()

# previous scores vs performance
sns.scatterplot(x=df["Previous Scores"], y=df["Performance Index"],
                hue=df["Previous Scores"], palette="coolwarm")
plt.title("Previous Scores vs Performance Index")
plt.tight_layout()
plt.show()

# extracurricular activities vs performance
sns.barplot(x=df["Extracurricular Activities"],
            y=df["Performance Index"],
            hue=df["Extracurricular Activities"])
plt.title("Extracurricular Activities vs Performance Index")
plt.tight_layout()
plt.show()

# sleep hours vs performance
sns.scatterplot(x=df["Sleep Hours"], y=df["Performance Index"],
                hue=df["Sleep Hours"])
plt.title("Sleep Hours vs Performance Index")
plt.tight_layout()
plt.show()

# correlation heatmap
numeric_df = df.copy()
numeric_df["Extracurricular Activities"] = \
    numeric_df["Extracurricular Activities"].map({"Yes": 1, "No": 0})
plt.figure(figsize=(8, 5))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

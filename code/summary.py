import pandas as pd

SECONDS_IN_A_DAY = 3600 * 24
df = pd.read_csv("./data/project-8400-GitHub-20240509T181710.csv")

df["created_at"] = pd.to_datetime(df["created_at"])
df["closed_at"] = pd.to_datetime(df["closed_at"])

df["delta_days"] = (
    (df["closed_at"] - df["created_at"]).dt.total_seconds() / SECONDS_IN_A_DAY
)

print("Delta days: ")
print(df["delta_days"].describe())

print("Delta days per repository: ")
print(df.groupby(["repository"])["delta_days"].describe())

print("Delta days per author: ")
print(df.groupby(["author_uuid"])["delta_days"].describe())

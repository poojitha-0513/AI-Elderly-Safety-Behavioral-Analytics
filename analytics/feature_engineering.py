import pandas as pd
import os

activity_file = "data/activity_logs.csv"
anomaly_file = "data/anomaly_logs.csv"

if not os.path.exists(activity_file):
    print("activity_logs.csv not found")
    exit()

if not os.path.exists(anomaly_file):
    print("anomaly_logs.csv not found")
    exit()

activity_df = pd.read_csv(activity_file)
anomaly_df = pd.read_csv(anomaly_file)

standing_time = (
    activity_df["activity"]
    .eq("Standing")
    .sum()
)

walking_time = (
    activity_df["activity"]
    .eq("Walking")
    .sum()
)

sitting_time = (
    activity_df["activity"]
    .eq("Sitting")
    .sum()
)

lying_time = (
    activity_df["activity"]
    .eq("Lying")
    .sum()
)

fall_count = (
    activity_df["fall_detected"]
    .eq("Yes")
    .sum()
)

inactivity_alerts = (
    anomaly_df["alert_type"]
    .eq("Prolonged Inactivity")
    .sum()
)

lying_alerts = (
    anomaly_df["alert_type"]
    .eq("Extended Lying Alert")
    .sum()
)

movement_score = (
    walking_time * 5
    +
    standing_time * 2
)

feature_row = {

    "standing_time":
        standing_time,

    "walking_time":
        walking_time,

    "sitting_time":
        sitting_time,

    "lying_time":
        lying_time,

    "fall_count":
        fall_count,

    "inactivity_alerts":
        inactivity_alerts,

    "lying_alerts":
        lying_alerts,

    "movement_score":
        movement_score
}

features_df = pd.DataFrame(
    [feature_row]
)

features_df.to_csv(
    "data/features.csv",
    index=False
)

print("\nFeatures Generated\n")
print(features_df)
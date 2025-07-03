import pandas as pd

# === Load datasets ===
rainfall_df = pd.read_csv("ssd-rainfall-adm2-full.csv")
coords_df = pd.read_csv("coordinates.csv")  # Ensure you saved this with Place, Latitude, Longitude

# === Merge based on district/region name ===
merged_df = pd.merge(
    rainfall_df,
    coords_df,
    how='left',
    left_on="ADM2_NAME",
    right_on="Place"
)

# === Drop the duplicate 'Place' column if you want ===
merged_df.drop(columns=["Place"], inplace=True)

# === Save the merged file ===
merged_df.to_csv("ssd-rainfall-with-coordinates.csv", index=False)

print("✅ Coordinates successfully merged into rainfall dataset.")
print("📁 Saved as 'ssd-rainfall-with-coordinates.csv'")

import pandas as pd

# Step 1: Load the dataset
df = pd.read_csv("P:\MY PROJECTS\Area crime prediction\crime_by_state.csv")

# Step 2: Check for missing values (optional display)
missing_values = df.isnull().sum()
print("Missing values per column:\n", missing_values)

# Step 3: Strip column names (in case of leading/trailing spaces)
df.columns = df.columns.str.strip()

# Step 4: Ensure column types are correct
df['Year'] = df['Year'].astype(int)

# Step 5: Create a new column for the most frequent crime in each row
# This excludes STATE/UT and Year columns (index 0 and 1)
crime_columns = df.columns[2:]
df["Major_Crime"] = df[crime_columns].idxmax(axis=1)

# Step 6: Save the cleaned dataset
df.to_csv("cleaned_crime_data.csv", index=False)
print("✅ Cleaned dataset saved as 'cleaned_crime_data.csv'")

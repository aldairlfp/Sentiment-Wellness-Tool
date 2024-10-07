import pandas as pd

# Sample DataFrame with 194 columns
data = {
    f'col_{i}': range(1, 6) for i in range(194)  # Creating 194 columns with sample data
}
df = pd.DataFrame(data)

print(df)
# Convert the entire DataFrame to a list of lists
list_of_rows = df.values.tolist()  # Using values and converting to list

# Alternatively, using a list comprehension
# list_of_rows = [row.to_list() for _, row in df.iterrows()]

# Print the result
for row in list_of_rows:
    print(row)
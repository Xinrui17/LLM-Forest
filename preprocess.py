import pandas as pd
import json
import os

def generate_description_from_row(row):
    description = []
    missing_features = []

    for col in row.index:
        value = row[col]
        if pd.isna(value):
            missing_features.append(col)  # Track missing features
        else:
            description.append(f"{col}: {value}")

    # Add information about missing features if any
    if missing_features:
        description.append(f"the house has missing features: {', '.join(missing_features)}")
    
    return ",".join(description)

def preprocess(dataset, data_path):
    df = pd.read_csv(os.path.join(data_path,f"{dataset}", ".csv"))
    with open(os.path.join(data_path,f'{dataset}_descriptions.json'), 'w') as outfile:
        outfile.write('[') 
        
        first_entry = True
        for index, row in df.iterrows():
            try:
                description = generate_description_from_row(row)
                description_entry = {"user": index, "description": description}
                
                if not first_entry:
                    outfile.write(',\n')  # Add comma and newline before the next entry, except for the first
                else:
                    first_entry = False

                json.dump(description_entry, outfile)
                print("Generated Description for index", index, ":\n", description, "\n---\n")

            except Exception as e:
                print(f"An error occurred at index {index}: {e}")
        
        outfile.write('\n]')  # Close the list with a newline

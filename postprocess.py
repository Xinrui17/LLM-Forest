import pandas as pd
import json
import re
import os

def extract_json_blocks(record_str):
    json_matches = re.findall(r'```json\s*(\{.*?\})\s*```', record_str, re.DOTALL)
    inferred_values = {}
    confidence_levels = {}

    if json_matches:
        try:
            # Assume the first JSON block contains inferred values
            inferred_values = json.loads(json_matches[0])
            # Assume the second JSON block contains confidence levels, if present
            if len(json_matches) > 1:
                confidence_levels = json.loads(json_matches[1])
        except json.JSONDecodeError as e:
            print(f"JSON decoding error: {e}")

    return inferred_values, confidence_levels

def parse_free_text_records(record_str):
    inferred_values = {}
    confidence_levels = {}

    try:
        inferred_matches = re.findall(r'"([^"]+)":\s*([\d.]+)', record_str)
        for key, value in inferred_matches:
            inferred_values[key.strip()] = float(value)

        confidence_matches = re.findall(r'"([^"]+)":\s*"([^"]+)"', record_str)
        for key, value in confidence_matches:
            confidence_levels[key.strip()] = value.strip()

    except Exception as e:
        print(f"Error parsing free-text record: {e}")

    return inferred_values, confidence_levels

def post_process(dataset, data_path, num_round):
    imputed_data_list = []
    confidence_data_list = []
    confidence_rank = {"high": 1, "medium": 0.4, "low": 0.3}
    for round_id in range(num_round):
        data = pd.read_csv(os.path.join(data_path, f"{dataset}", ".csv"))
        original_columns = data.columns.tolist()

        column_mapping = {col[:9].lower(): col for col in original_columns}  # Map shortened column names to original names

        json_path = os.path.join(data_path, f'{dataset}_imputation_{round_id}.json')
        with open(json_path, 'r') as file:
            imputed_data = json.load(file)

        inferred_data = data.copy()
        confidence_data = data.copy()

        for entry in imputed_data:
            patient_id = entry['patient']
            updated_records_str = entry['description']

            inferred_values, confidence_levels = extract_json_blocks(updated_records_str)

            # If JSON extraction fails, fall back to free-text parsing
            if not inferred_values and not confidence_levels:
                inferred_values, confidence_levels = parse_free_text_records(updated_records_str)

            # Skip the record if both methods fail
            if not inferred_values:
                print(f"No valid data found for patient {patient_id}. Skipping.")
                continue

            for key, value in inferred_values.items():
                normalized_key = key[:9].lower()  
                if normalized_key in column_mapping:
                    original_key = column_mapping[normalized_key]  
                    inferred_data.at[patient_id, original_key] = value
                else:
                    print(f"Unknown key in inferred values for patient {patient_id}: {key}")

            for key, value in confidence_levels.items():
                normalized_key = key[:9].lower()  
                if normalized_key in column_mapping:
                    original_key = column_mapping[normalized_key]
                    confidence_data.at[patient_id, original_key] = value
                else:
                    print(f"Unknown key in confidence levels for patient {patient_id}: {key}")

        inferred_csv_path = os.path.join(data_path, f'{dataset}_{round_id}_values.csv')
        confidence_csv_path = os.path.join(data_path, f'{dataset}_{round_id}_confidence.csv')
        imputed_data_list.append(inferred_csv_path)
        confidence_data_list.append(confidence_data)
        inferred_data.to_csv(inferred_csv_path, index=False)
        confidence_data.to_csv(confidence_csv_path, index=False)

        print(f"Inferred values saved to {inferred_csv_path}")
        print(f"Confidence levels saved to {confidence_csv_path}")
    
    for patient_id in data.index:
        for column in data.columns:
            normalized_key = column.lower()

            if normalized_key not in column_mapping:
                print("unknown normalized_key")
                continue

            if pd.isna(data.at[patient_id, column]):
                imputed_values = []
                confidences = []
                
                for i in range(len(imputed_data_list)): 
                    imputed_data = imputed_data_list[i]
                    confidence_data = confidence_data_list[i]
                    
                    if patient_id < len(imputed_data):
                        imputed_value = imputed_data.at[patient_id, column_mapping[normalized_key]]
                        confidence = confidence_data.at[patient_id, column_mapping[normalized_key]]

                        if pd.notna(imputed_value) and pd.notna(confidence):
                            confidence_score = confidence_rank.get(str(confidence).strip().lower(), 0)  # Default to 0 if unrecognized
                            imputed_values.append(imputed_value)
                            confidences.append(confidence_score)

                if len(imputed_values) > 0:
                    max_confidence = max(confidences)
                    max_confidence_indices = [i for i, conf in enumerate(confidences) if conf == max_confidence]

                    if len(max_confidence_indices) > 1:  
                        final_value = sum(imputed_values[i] for i in max_confidence_indices) / len(max_confidence_indices)  # Use average
                    else:  
                        final_value = imputed_values[max_confidence_indices[0]]

                  
                    data.at[patient_id, column] = final_value

    updated_csv_path = os.path.join(data_path, f'{dataset}_result.csv')
    data.to_csv(updated_csv_path, index=False)

    print(f"Updated data saved to {updated_csv_path}")


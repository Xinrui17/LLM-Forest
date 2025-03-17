import json
import os
from openai import OpenAI
from mistralai import Mistral
import time


descriptions = """
## Below are the detailed information about the house's feature description. Its follws the format of 
Name - Description

# crim - per capita crime rate by town
# zn - proportion of residential land zoned for lots over 25,000 sq.ft.
# indus - proportion of non-retail business acres per town.
# chas - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# nox - nitric oxides concentration (parts per 10 million)
# rm - average number of rooms per dwelling
# age - proportion of owner-occupied units built prior to 1940
# dis - weighted distances to five Boston employment centres
# rad - index of accessibility to radial highways
# tax - full-value property-tax rate per $10,000
# ptratio - pupil-teacher ratio by town
# b - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# lstat - %% lower status of the population
# medvgpt - Median value of owner-occupied homes in $1000's
"""

def read_json_file(file_path):
    """Reads a JSON file and returns the data."""
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def append_to_json_file(data, file_path):
    """Appends data to a JSON file."""
    try:
        with open(file_path, 'r+') as file:
            file_data = json.load(file)
            file_data.append(data)  
            file.seek(0)  
            json.dump(file_data, file, indent=4)
    except json.JSONDecodeError:  
        with open(file_path, 'w') as file:
            json.dump([data], file, indent=4) 

def imputation_missing_value(client, model_name, patinet_id, records):
    # print(records)
    """Generates a concise description for patient records using OpenAI's Chat Completion."""
    try:
        if model_name == 'gpt':
            # print(records)
            response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0.001,
            messages=[
             {"role": "system","content": f"You are a helpful assistant tasked with filling the missing values for hosue {patinet_id} in the Boston Housing Dataset. You must give the results for all the missing value. You have the flexibility to determine the best approach for each missing feature by analyzing both the house's own records and the data of other houses. Possible methods may include in no particular sequence (1) using the most common values of the similar houses (2) using the most similar houses' values (3) uncovering patterns and relationships between the features (4) applying your knowledge in the related domains (5)... After performing the imputation, ensure your inferred results are consistent with domain knowledge and validate them using common sense and reasonable assumptions within the related domains."},
             {"role":"user", "content": "Here are some patterns observed based on the correlations and feature distributions: crim and rad have a strong positive correlation of 0.814, indicating that higher crime rates (crim) are associated with higher accessibility to radial highways (rad). indus and nox have strong positive correlation of 0.780, showing that areas with higher proportions of non-retail business acres (indus) also tend to have higher nitric oxide concentrations (nox). rad and tax have a positive correlation (0.69) showing that towns with better highway accessibility (rad) tend to have higher property tax rates (tax). This could reflect higher infrastructure investment in these areas. indus and rad have moderate positive correlation of 0.564, suggesting that industrial areas are more likely to have better access to radial highways (rad). indus and zn have a moderate negative correlation of -0.537, implying that regions with higher industrial activity (indus) are less likely to have a higher proportion of residential land zoned for lots over 25,000 sq. ft. (zn). For crim, most values cluster around 0.63. The majority of zn values are concentrated at 0, reflecting that many towns have minimal or no large residential zoning. Few towns show much higher values, indicating skewness. For rm, the mean is around 6.24, with a moderate spread, indicating a relatively consistent distribution of housing room sizes. For ptratio, the majority of values are centered around 18.5, showing minimal variability and suggesting uniformity in educational resources. For b, most values cluster around 356.0, with a relatively narrow spread and a few extreme outliers showing significant deviation. For chas,the majority of values are concentrated at 0, indicating that most towns do not bound the Charles River. A small portion of the towns has a value of 1."},
             {"role":"user", "content": f'''The data is from Boston Housing dataset, containing information collected by the U.S Census Service concerning housing in the area of Boston Mass and aiming to predict the price, in which the median value of a home is to be predicted. Some values are missing in the orginal records due to various reasons. Given feature descriptionn {descriptions} and house's records {records}. 
                Imagine that your are working as an analyst to complete this dataset for further research purpose. Provide your understanding of the missing feature and your explanations for choosing the appropriate value before the final inference. Give the imputation results in a succinct JSON format strictly starting with ```json with the following structure: \"feature name\": \"inferred value\". Output your confidence level for the imputation results in a JSON format starting with ```json with the following structure: \"feature name\": \" confidence level\" at the end without any explanations. In the results, use the exact feature names as those in the feature description.'''}    
               ]
            )
    
            print(response.choices[0].message.content)
            return response.choices[0].message.content
        elif model_name == 'mixtral':
            retry_count = 0
            max_retries = 50
            backoff_factor = 2  # Exponential backoff factor
            while retry_count < max_retries:
                try:
                    chat_response = client.chat.complete(
                        model="open-mixtral-8x22b",
                        temperature=0.001,
                        messages=[
             {"role": "system","content": f"You are a helpful assistant tasked with filling the missing values for hosue {patinet_id} in the Boston Housing Dataset. You must give the results for all the missing value. You have the flexibility to determine the best approach for each missing feature by analyzing both the house's own records and the data of other houses. Possible methods may include in no particular sequence (1) using the most common values of the similar houses (2) using the most similar houses' values (3) uncovering patterns and relationships between the features (4) applying your knowledge in the related domains (5)... After performing the imputation, ensure your inferred results are consistent with domain knowledge and validate them using common sense and reasonable assumptions within the related domains."},
             {"role":"user", "content": "Here are some patterns observed based on the correlations and feature distributions: crim and rad have a strong positive correlation of 0.814, indicating that higher crime rates (crim) are associated with higher accessibility to radial highways (rad). indus and nox have strong positive correlation of 0.780, showing that areas with higher proportions of non-retail business acres (indus) also tend to have higher nitric oxide concentrations (nox). rad and tax have a positive correlation (0.69) showing that towns with better highway accessibility (rad) tend to have higher property tax rates (tax). This could reflect higher infrastructure investment in these areas. indus and rad have moderate positive correlation of 0.564, suggesting that industrial areas are more likely to have better access to radial highways (rad). indus and zn have a moderate negative correlation of -0.537, implying that regions with higher industrial activity (indus) are less likely to have a higher proportion of residential land zoned for lots over 25,000 sq. ft. (zn). For crim, most values cluster around 0.63. The majority of zn values are concentrated at 0, reflecting that many towns have minimal or no large residential zoning. Few towns show much higher values, indicating skewness. For rm, the mean is around 6.24, with a moderate spread, indicating a relatively consistent distribution of housing room sizes. For ptratio, the majority of values are centered around 18.5, showing minimal variability and suggesting uniformity in educational resources. For b, most values cluster around 356.0, with a relatively narrow spread and a few extreme outliers showing significant deviation. For chas,the majority of values are concentrated at 0, indicating that most towns do not bound the Charles River. A small portion of the towns has a value of 1."},
             {"role":"user", "content": f'''The data is from Boston Housing dataset, containing information collected by the U.S Census Service concerning housing in the area of Boston Mass and aiming to predict the price, in which the median value of a home is to be predicted. Some values are missing in the orginal records due to various reasons. Given feature descriptionn {descriptions} and house's records {records}. 
                Imagine that your are working as an analyst to complete this dataset for further research purpose. Provide your understanding of the missing feature and your explanations for choosing the appropriate value before the final inference. Give the imputation results in a succinct JSON format strictly starting with ```json with the following structure: \"feature name\": \"inferred value\". Output your confidence level for the imputation results in a JSON format starting with ```json with the following structure: \"feature name\": \" confidence level\" at the end without any explanations. In the results, use the exact feature names as those in the feature description.'''}    
                        ]
            )

                    print(chat_response.choices[0].message.content)
                    return chat_response.choices[0].message.content

                except Exception as e:
                    if "Status 429" in str(e) or "Status 403" in str(e):
                        retry_count += 1
                        wait_time = backoff_factor ** retry_count
                        print(f"Rate limit exceeded. Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                    else:
                        print(f"An error occurred: {e}")
                        return "Failed to generate description due to an error."

    except Exception as e:
        print(f"An error occurred: {e}")
        return "Failed to generate description due to an error."
    print("Maximum retries reached. Skipping this patient.")
    return "Failed to generate description after retries."
        
    
def process_and_write_patient_record(client, model_name, round_id, patient, output_file_path):
    """Generates concise description for a single patient and appends to JSON file."""
    imputation_results = imputation_missing_value(client, model_name, patient["user"], patient[f"UpdatedRecords_Round_{round_id}"])
    
    updated_patient_data = {
        "patient": patient["user"],
        "UpdatedRecords": imputation_results
    }
    append_to_json_file(updated_patient_data, output_file_path)

def llm_imputation(model_name, data_path, dataset, round_id):
    if model_name == "gpt":
        client = OpenAI(api_key = 'put_your_key_here')
    elif model_name == "mixtral":
        client = Mistral(api_key="put_your_key_here")
   
    input_file_path = os.path.join(data_path,f'{dataset}_neighbors_{round_id}.json')  
    output_file_path = os.path.join(data_path, f'{dataset}_imputation_{round_id}.json')  
    open(output_file_path, 'w').close() 
    patient_data = read_json_file(input_file_path)

    for patient in patient_data:
        process_and_write_patient_record(client, model_name, patient, output_file_path)

    print(f"Updated User data has been written to {output_file_path}")

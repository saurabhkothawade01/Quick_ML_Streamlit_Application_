import streamlit as st
import logging
import math
import pandas as pd
from dateutil import parser
import os
import create_db

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

##################################################################
#                                                                #
#                >>>>> MODULE 1 - Data Cleaning <<<<<            #
#                                                                #
##################################################################

# Categorize DataFrame columns by their data types.
def analyze_columns(df):
    if not isinstance(df, pd.DataFrame):
        st.error("The provided input is not a valid DataFrame.")
        logging.error("The provided input is not a valid DataFrame.")
        return None

    if df.empty:
        st.warning("The provided DataFrame is empty.")
        logging.warning("The provided DataFrame is empty.")
        return None
    
    columns_by_type = {
        'nominal': [],
        'ordinal': [],
        'string': [],
        'float': [],
        'int': [],
        'date': []
    }
    
    for col in df.columns:
        try:
            non_null_values = df[col].dropna()
            unique_values = non_null_values.unique()
            num_unique_values = len(unique_values)
            
            max_unique = min(
                math.ceil(0.05 * len(df)),
                math.ceil(10 * math.log10(len(df)))
            )
            
            logging.info(f"Processing column '{col}' with {num_unique_values} unique values.")
            
            if num_unique_values <= max_unique:
                if pd.api.types.is_numeric_dtype(non_null_values):
                    unique_values_sorted = sorted(unique_values)
                    if all(unique_values_sorted[i] < unique_values_sorted[i+1] for i in range(num_unique_values - 1)):
                        columns_by_type['ordinal'].append(col)
                    else:
                        columns_by_type['nominal'].append(col)
                        
                elif pd.api.types.is_string_dtype(non_null_values):
                    columns_by_type['nominal'].append(col)
                    
                elif pd.api.types.is_object_dtype(non_null_values):
                    if is_numeric(non_null_values.iloc[0]):
                        non_null_values = pd.to_numeric(non_null_values, errors='coerce').astype(float)
                        non_null_values = non_null_values.dropna()
                        unique_values = non_null_values.unique()
                        num_unique_values = len(unique_values)
                        unique_values_sorted = sorted(unique_values)
                        if all(unique_values_sorted[i] < unique_values_sorted[i+1] for i in range(num_unique_values - 1)):
                            columns_by_type['ordinal'].append(col)
                            
                        else:
                            columns_by_type['nominal'].append(col)
                    else:
                        columns_by_type['nominal'].append(col)
                    
            elif all(is_date(val) for val in non_null_values):
                columns_by_type['date'].append(col)
                
            elif pd.api.types.is_string_dtype(df[col]):
                columns_by_type['string'].append(col)
                
            elif pd.api.types.is_float_dtype(df[col]):
                columns_by_type['float'].append(col)
                
            elif pd.api.types.is_integer_dtype(df[col]):
                columns_by_type['int'].append(col)
                
            elif pd.api.types.is_object_dtype(df[col]):
                if is_numeric(non_null_values.iloc[0]):
                    contains_float = any('.' in str(val) for val in non_null_values)
                    if contains_float:
                        columns_by_type['float'].append(col)
                    else:
                        columns_by_type['int'].append(col)
                        
                elif is_date(non_null_values.iloc[0]):
                    columns_by_type['date'].append(col)
                    
                else:
                    columns_by_type['string'].append(col)
                    
            else:
                raise ValueError(f"Unknown data type for column {col}.")
            
            logging.info(f"Column '{col}' categorized as: {columns_by_type}")

        except Exception as e:
            logging.error(f"Error processing column '{col}': {e}")
            st.error(f"Error processing column '{col}': {e}")
            columns_by_type['string'].append(col)
 
    return columns_by_type

# Check if a string can be converted to a float.
def is_numeric(value):
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

# Check if a string can be parsed into a date.    
def is_date(value):
    try:
        parser.parse(value)
        return True
    except (ValueError, TypeError):
        return False

# Display a summary of DataFrame columns categorized by their data types.    
def display_analysis(df, target_column):
    if not isinstance(df, pd.DataFrame):
        st.error("The provided input is not a valid DataFrame.")
        logging.error("The provided input is not a valid DataFrame.")
        return None, None
    
    if df.empty:
        st.warning("The provided DataFrame is empty.")
        logging.warning("The provided DataFrame is empty.")
        return None, None
    
    if target_column not in df.columns:
        st.error(f"The target column '{target_column}' does not exist in the DataFrame.")
        logging.error(f"The target column '{target_column}' does not exist in the DataFrame.")
        return None, None
    
    try:
        columns_by_type = analyze_columns(df)
        logging.info(f"Column analysis result: {columns_by_type}")
        
        data = {
            'Data Type': [],
            'Count': [],
            'Columns': []
        }
        
        for data_type, columns in columns_by_type.items():
            data['Data Type'].append(data_type.capitalize())
            data['Count'].append(len(columns))
            data['Columns'].append(', '.join(columns) if columns else 'None')
    
        analysis_df = pd.DataFrame(data)
        
        st.subheader("Column Analysis")
        st.write(f"Total {len(df.columns)} columns are in the provided dataset.")
        st.write("")
        st.dataframe(analysis_df, width=1000) 
        
        removed_info = []
        for key in columns_by_type:
            if target_column in columns_by_type[key]:
                removed_info.append((key, columns_by_type[key].copy()))
                columns_by_type[key].remove(target_column)
            
        return columns_by_type, removed_info
    
    except Exception as e:
        logging.error(f"An error occurred while analyzing columns: {e}")
        st.error("An unexpected error occurred. Please try again later.")
        return None, None
    
# Main function to drive the Streamlit machine learning application
def main(user_id):    
    st.title("Building a Data Cleansing and Transformation solution using Machine Learning")

    st.markdown("""
        Welcome to the **Machine Learning Tool**! This application is designed to streamline your data processing tasks, including:
        - Data Cleaning
        - Data Transformation
        - Model Building
        
        Upload your CSV file to get started with data analysis and visualization, and then move on to building your machine learning model.
    """, unsafe_allow_html=True)
    
    st.header("Instructions")
    st.markdown("""
    1. **Upload your CSV file** using the file uploader below.
    2. Ensure the file is in **CSV format** with headers in the first row.
    """, unsafe_allow_html=True)

    if 'analysis_done' not in st.session_state:
        st.session_state.analysis_done = False

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            file_path = save_uploaded_file(uploaded_file)

            if 'df' not in st.session_state:
                st.session_state.df = load_data(uploaded_file)

                dataset_name = uploaded_file.name

                add_dataset_to_db(user_id, dataset_name, file_path)

                st.success(f"Dataset '{dataset_name}' uploaded and saved successfully.")
                
            if st.session_state.df is not None:
                df = st.session_state.df

                if not st.session_state.analysis_done:
                    try:
                        action_irrelevant = handle_irrelevant_columns()
                        action_duplicate = handle_duplicates()
                        action_missing = handle_50_missing_values()
                        df_without_y, y, target_column = select_target_variable(df)

                        if st.button("Analyze Columns"):
                            with st.spinner("Analyzing columns..."):
                                columns_by_type, removed_info = display_analysis(df, target_column)
                    except Exception as e:
                        st.error(f"An error occurred during data analysis and transformation: {str(e)}")
                        logging.error(f"An error occurred during data analysis and transformation: {e}")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"An error occurred: {e}")

def save_uploaded_file(uploaded_file):
    save_path = os.path.join('datasets', uploaded_file.name)
    
    if not os.path.exists('datasets'):
        os.makedirs('datasets')

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return save_path

def add_dataset_to_db(user_id, dataset_name, dataset_file):
    create_db.c.execute('''
        INSERT INTO datasets (user_id, dataset_name, dataset_file)
        VALUES (?, ?, ?)
    ''', (user_id, dataset_name, dataset_file))
    create_db.conn.commit()

# Load data from a CSV file with multiple encoding attempts.
def load_data(file):
    encodings = ['utf-8', 'cp1252', 'iso-8859-1', 'latin1']
    
    for encoding in encodings:
        try:
            df = pd.read_csv(file, encoding=encoding)
            st.success(f"File successfully uploaded and processed with encoding '{encoding}'")
            logging.info(f"File successfully loaded with encoding '{encoding}'")
            return df
        except UnicodeDecodeError:
            logging.warning(f"Failed to decode file with encoding '{encoding}'. Trying next encoding...")
            continue
        except Exception as e:
            st.error(f"An unexpected error occurred while reading the file: {str(e)}")
            logging.error(f"An unexpected error occurred while reading the file: {e}")
            return None
    st.warning("Failed to read the file with any of the specified encodings.")
    logging.error("Failed to read the file with any of the specified encodings.")
    return None

# Keep or drop irrelevant columns based on user input.
def handle_irrelevant_columns():
    st.subheader("Irrelevant Columns")
    st.write("You can choose to keep or drop columns considered irrelevant.")
    action = st.radio("What would you like to do with the irrelevant columns?",
                      ("Keep all columns", "Drop irrelevant columns"))
    
    if action == "Drop irrelevant columns":
            action_irrelevant = "yes"
            st.write("Irrelevant columns will be dropped.")
    else:
            action_irrelevant = "no"
            st.write("All columns will be kept.")        

    return action_irrelevant

# Keep or drop duplicate records based on user input.
def handle_duplicates():
    st.subheader("Duplicate Data")
    st.write("You can choose to keep or drop duplicate records.")
    action = st.radio("What would you like to do with the duplicate records?",
                      ("Keep all records", "Drop duplicate records"))
           
    if action == "Drop duplicate records":
        action_duplicate = "yes"
        st.write("Duplicate records will be dropped.")
    else:
        action_duplicate = "no"
        st.write("All records will be kept.")

    return action_duplicate

# Keep or drop columns with more than 50% missing values based on user input.
def handle_50_missing_values():
    st.subheader("More than 50% Missing Values")
    st.write("You can choose to keep or drop columns with more than 50% missing values.")
    action = st.radio("What would you like to do with columns having more than 50% missing values?",
                      ("Keep all values", "Drop values"))
            
    if action == "Drop values":
        action_missing = "yes"
        st.write("Columns with more than 50% missing values will be dropped.")
    else:
        action_missing = "no"
        st.write("All columns will be kept.")

    return action_missing

# Allow the user to select a target variable from the DataFrame.
def select_target_variable(df):
    st.subheader("Target Variable")
    options = ["None"] + list(df.columns)
    target_column = st.selectbox("Select target column:", options)
    
    if target_column != "None":
        st.write(f"Selected target column: {target_column}")
        y = df[target_column]
        df = df.drop(columns=[target_column])
        return df, y, target_column
    else:
        st.write("No column selected")
        return df, None, None

if __name__ == "__main__":
        main()
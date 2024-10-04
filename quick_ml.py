import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import math
from sklearn.impute import SimpleImputer
from scipy.stats import shapiro
import scipy.stats as stats
from dateutil import parser
import re
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import VarianceThreshold, RFECV
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, KFold, train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import io
import plotly.graph_objects as go
import time
import logging
import os
import create_db
import sqlite3
from io import BytesIO

# Configures the settings of the page
st.set_page_config(
    page_title="Quick ML",
    page_icon=":robot:",
)

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

# Detect outliers in a DataFrame for categorical, numerical
def detect_outlier(df, columns_by_type, n_neighbors=5, contamination='auto'):
    outliers_percentage = {}
    
    # Detect outliers in categorical columns using Local Outlier Factor.
    def detect_categorical_outliers(df, cat_columns):
        try:
            cat_data = df[cat_columns].copy()
            logging.info(f"Initial categorical data:\n{cat_data.head()}")

            for col in cat_data.columns:
                if col in columns_by_type["ordinal"]:
                    cat_data[col] = pd.to_numeric(cat_data[col], errors='coerce').astype(float)
            
            non_empty_cols = cat_data.dropna(how='all', axis=1)
            empty_cols = set(cat_data.columns) - set(non_empty_cols.columns)

            logging.info(f"Non-empty columns:\n{non_empty_cols.head()}")
            logging.info(f"Empty columns: {empty_cols}")

            for col_type in columns_by_type:
                columns_by_type[col_type] = [col for col in columns_by_type[col_type] if col not in empty_cols]
            
            cat_data_cleaned = non_empty_cols
            imputer = SimpleImputer(strategy='most_frequent')
            cat_data_imputed = pd.DataFrame(imputer.fit_transform(cat_data_cleaned), columns=cat_data_cleaned.columns)
            
            logging.info(f"Imputed categorical data:\n{cat_data_imputed.head()}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ('nom', OneHotEncoder(sparse_output=False, handle_unknown='ignore'), columns_by_type.get("nominal", [])),
                    ('ord', OrdinalEncoder(), columns_by_type.get("ordinal", []))
                ], remainder='drop') 

            encoded_data = preprocessor.fit_transform(cat_data_imputed)
            logging.info(f"Encoded data shape: {encoded_data.shape}")

            nominal_feature_names = preprocessor.named_transformers_['nom'].get_feature_names_out(columns_by_type.get("nominal", []))
            ordinal_feature_names = columns_by_type.get("ordinal", [])
            feature_names = list(nominal_feature_names) + ordinal_feature_names

            if len(feature_names) != encoded_data.shape[1]:
                raise ValueError(f"Feature names length ({len(feature_names)}) does not match encoded data columns ({encoded_data.shape[1]})")

            encoded_df = pd.DataFrame(encoded_data, columns=feature_names)
            logging.info(f"Encoded DataFrame:\n{encoded_df.head()}")

            lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
            outlier_labels = lof.fit_predict(encoded_df)

            total_rows = len(outlier_labels)
            outliers_count = np.sum(outlier_labels == -1)
            cat_outlier_percentage = (outliers_count / total_rows) * 100

            logging.info(f"Categorical outlier percentage: {cat_outlier_percentage}")
            return cat_outlier_percentage, columns_by_type

        except Exception as e:
            logging.error(f"Error occurred in categorical outlier detection: {e}")
            st.error(f"Error occurred in categorical outlier detection: {e}")
            return None
        # Detect outliers in numerical columns using Z-score.
    def detect_numerical_outliers(numerical_columns, threshold=3):
        outliers_percentage = {}
        
        for col in numerical_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                mean = df[col].mean()
                std = df[col].std()
                z_scores = (df[col] - mean) / std
                outlier_count = np.sum(np.abs(z_scores) > threshold)
                
                percentage_outliers = (outlier_count / len(df)) * 100
                outliers_percentage[col] = percentage_outliers

                logging.info(f"Numerical column '{col}' outlier percentage: {percentage_outliers}%")
            
            except Exception as e:
                logging.error(f"Error occurred in numerical outlier detection for column '{col}': {e}")
                st.error(f"Error occurred in numerical outlier detection for column '{col}': {e}")
                outliers_percentage[col] = None
        
        return outliers_percentage
    
    if not isinstance(df, pd.DataFrame):
        st.error("The provided input is not a valid DataFrame.")
        logging.error("The provided input is not a valid DataFrame.")
        return None, None
    
    if not isinstance(columns_by_type, dict):
        st.error("The columns_by_type argument must be a dictionary.")
        logging.error("The columns_by_type argument must be a dictionary.")
        return None, None
    
    if df.empty:
        st.warning("The provided DataFrame is empty.")
        logging.warning("The provided DataFrame is empty.")
        return None, None
    
    cat_columns = columns_by_type.get("nominal", []) + columns_by_type.get("ordinal", [])
    cat_outlier_percentage, columns_by_type = detect_categorical_outliers(df, cat_columns)
        
    if cat_outlier_percentage is not None:
        outliers_percentage["categorical"] = cat_outlier_percentage
    
    numerical_columns = columns_by_type.get('float', []) + columns_by_type.get('int', [])
    numerical_outlier_percentages = detect_numerical_outliers(numerical_columns)
    
    if numerical_outlier_percentages:
        outliers_percentage.update(numerical_outlier_percentages)
    
    st.subheader("Outlier Analysis Results")
    st.write(outliers_percentage)
    
    return outliers_percentage, columns_by_type

# Check the normality of numerical columns and visualize data distributions.
def check_normality_with_graphs(df, columns_by_type, outlier):
    if not isinstance(df, pd.DataFrame):
        st.error("The provided input is not a valid DataFrame.")
        logging.error("The provided input is not a valid DataFrame.")
        return None, None, None
    
    if not isinstance(columns_by_type, dict):
        st.error("The columns_by_type argument must be a dictionary.")
        logging.error("The columns_by_type argument must be a dictionary.")
        return None, None, None
    
    if df.empty:
        st.warning("The provided DataFrame is empty.")
        logging.warning("The provided DataFrame is empty.")
        return None, None, None

    if not isinstance(outlier, dict):
        st.error("The input 'outlier' must be a dictionary.")
        logging.error("The input 'outlier' is not a dictionary.")
        return None, None, None
    
    alpha = 0.05
    normal_col = []
    not_normal_col = []

    fi_columns = columns_by_type["float"] + columns_by_type["int"]

    st.subheader("Data Distribution Analysis")

    for col in fi_columns:
        if col not in df.columns:
            st.warning(f"Column '{col}' is not present in the DataFrame.")
            logging.warning(f"Column '{col}' is not present in the DataFrame.")
            continue
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
            stats.probplot(df[col].dropna(), dist="norm", plot=ax1)
            ax1.set_title(f'Q-Q Plot of {col}', fontsize=12)
            ax1.tick_params(labelsize=8)
            ax1.set_xlabel('Theoretical Quantiles', fontsize=10)
            ax1.set_ylabel('Sample Quantiles', fontsize=10)
            
            sns.histplot(df[col].dropna(), kde=True, ax=ax2)
            ax2.set_title(f'Histogram of {col}', fontsize=12)
            ax2.tick_params(labelsize=8)
            ax2.set_xlabel(col, fontsize=10)
            ax2.set_ylabel('Frequency', fontsize=10)

            plt.tight_layout()
            st.pyplot(fig)

            col1, col2 = st.columns(2)

            with col1:
                stats_text = (
                    f"Statistics for {col}:\n"
                    f"• Mean: {df[col].mean():.2f}\n"
                    f"• Median: {df[col].median():.2f}\n"
                    f"• Mode: {df[col].mode().iloc[0]:.2f}\n"
                    f"• Std Dev: {df[col].std():.2f}\n"
                    f"• Missing Values: {df[col].isna().sum()}\n"
                    f"• Outlier Detection: {outlier.get(col, 'N/A'):.2f}%\n"
                )
                st.text(stats_text)
            
            with col2:
                try:
                    stat, p = stats.shapiro(df[col].dropna())
                    if p > alpha:
                        normality_text = (
                            f"Normality Test Results:\n"
                            f"• P value: {p:.4f}\n"
                            f"• Distribution: Normal\n"
                            f"• Missing values treatment: Mean\n"
                        )
                        normal_col.append(col)
                    else:
                        p_value = f"{p:.4e}" if p < 0.0001 else f"{p:.4f}"
                        normality_text = (
                            f"Normality Test Results:\n"
                            f"• P value: {p_value}\n"
                            f"• Distribution: Not normal\n"
                            f"• Missing values treatment: Median\n"
                        )
                        not_normal_col.append(col)
                except Exception as e:
                    normality_text = f"Normality test failed: {e}"
                    st.warning(f"Normality test failed for column '{col}'. Check logs for details.")
                    logging.error(f"Error occurred in Shapiro-Wilk test for column '{col}': {e}")
                st.text(normality_text)

        except Exception as e:
            st.warning(f"Error occurred while plotting distribution for column '{col}'. Check logs for details.")
            logging.error(f"Error occurred while plotting distribution for column '{col}': {e}")

    def plot_pie_chart(column, value_counts, total):
        try:
            if value_counts.empty:
                fig, ax = plt.subplots(figsize=(6, 5))
                ax.text(0.5, 0.5, "No data available", ha='center', va='center')
                ax.axis('off')
                return fig

            fig, ax = plt.subplots(figsize=(8, 6))
            fig.suptitle(f"Distribution of {column}", fontsize=10, y=0.95)

            def get_display_categories(value_counts, total):
                if len(value_counts) <= 4:
                    return value_counts.index.tolist(), []

                cumulative_sum = 0
                display_categories = []
                for category, count in value_counts.items():
                    cumulative_sum += count
                    display_categories.append(category)
                    if cumulative_sum / total >= 0.8:
                        break

                other_categories = [cat for cat in value_counts.index if cat not in display_categories]
                return display_categories, other_categories

            display_categories, other_categories = get_display_categories(value_counts, total)

            chart_data = [value_counts[cat] for cat in display_categories]
            if other_categories:
                chart_data.append(sum(value_counts[cat] for cat in other_categories))
                categories = display_categories + ['Other']
            else:
                categories = display_categories

            colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

            wedges, texts, autotexts = ax.pie(chart_data, colors=colors, autopct='%1.1f%%',
                                            pctdistance=0.85, wedgeprops=dict(width=0.5, edgecolor='white'),
                                            startangle=90)

            ax.set_aspect('equal')

            for wedge, category in zip(wedges, categories):
                angle = (wedge.theta2 + wedge.theta1) / 2
                x = np.cos(np.deg2rad(angle))
                y = np.sin(np.deg2rad(angle))

                label_x = 1.2 * x
                label_y = 1.2 * y

                ax.plot([x, label_x], [y, label_y], color='black', linestyle='-', linewidth=1)

                ax.text(label_x, label_y, f'{category}: {chart_data[categories.index(category)]}',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'),
                        fontsize=8, ha='left', va='center')

            plt.tight_layout()
            return fig
        except Exception as e:
            st.warning(f"Error occurred while plotting pie chart for column '{column}'. Check logs for details.")
            logging.error(f"Error occurred while plotting pie chart for column '{column}': {e}")
            return plt.Figure()

    no_columns = columns_by_type["nominal"] + columns_by_type["ordinal"]
    pie_charts = []
    stats_texts = []

    for column in no_columns:
        try:
            value_counts = df[column].value_counts()
            total = sum(value_counts.values)

            pie_charts.append(plot_pie_chart(column, value_counts, total))
            
            if value_counts.empty:
                most_common = "N/A"
                most_common_count = 0
            else:
                most_common = df[column].value_counts().index[0] if not df[column].value_counts().empty else "N/A"
                most_common_count = df[column].value_counts().iloc[0] if not df[column].value_counts().empty else 0

            stats_text = (
                f"Statistics for {column}:\n"
                f"• Number of categories: {len(value_counts)}\n"
                f"• Most common: {most_common}\n"
                f"  ({most_common_count} occurrences)\n"
                f"• Missing values: {df[column].isnull().sum()}\n"
                f"• Total count: {len(df)}\n"
                f"• Missing values treatment: Mode\n"
                f"• Outlier Detection: {outlier.get('categorical', 'N/A')}%\n"
            )
            stats_texts.append(stats_text)
        except Exception as e:
            st.warning(f"Error occurred while processing nominal/ordinal column '{column}'. Check logs for details.")
            logging.error(f"Error occurred while processing nominal/ordinal column '{column}': {e}")

    num_cols = len(pie_charts)
    rows = (num_cols + 1) // 2 

    for row in range(rows):
        col1, col2 = st.columns(2)

        with col1:
            if row * 2 < num_cols:
                st.pyplot(pie_charts[row * 2])
                st.text(stats_texts[row * 2])

        with col2:
            if row * 2 + 1 < num_cols:
                st.pyplot(pie_charts[row * 2 + 1])
                st.text(stats_texts[row * 2 + 1])

        st.write("---")
            
    return normal_col, not_normal_col

# Identifies and displays the percentage of duplicate rows in the DataFrame.
def check_duplicity(df):
    if not isinstance(df, pd.DataFrame):
        st.error("The provided input is not a valid DataFrame.")
        logging.error("The provided input is not a valid DataFrame.")
        return None, None
    
    if df.empty:
        st.warning("The provided DataFrame is empty.")
        logging.warning("The provided DataFrame is empty.")
        return None, None
    
    try:
        total_rows = len(df)
        logging.info(f"Total rows in DataFrame: {total_rows}")

        duplicated_rows = df[df.duplicated(keep='first')]
        num_duplicates = len(duplicated_rows)
        logging.info(f"Number of duplicated rows found: {num_duplicates}")
        
        if total_rows > 0:
            percentage_duplicates = (num_duplicates / total_rows) * 100
        else:
            percentage_duplicates = 0.0
        
        st.subheader("Duplicate Rows")
        with st.expander("Duplicate Rows", expanded=True):
            st.info(f"Total Rows: {total_rows}")
            st.info(f"Number of Duplicate Rows: {num_duplicates}")
            st.info(f"Percentage of Duplicate Rows: {percentage_duplicates:.2f}%")
        
        return duplicated_rows, percentage_duplicates

    except Exception as e:
        st.error("An error occurred while checking for duplicate rows. Please check the logs for details.")
        logging.error(f"Error occurred while checking for duplicate rows: {e}")
        return None, None

# Check if a value type is an integer type.
def is_integer_type(val_type):
    return np.issubdtype(val_type, np.integer)

# Check if a value type is a float type.
def is_float_type(val_type):
    return np.issubdtype(val_type, np.floating)

# Check if a value type is either an integer or float.
def is_numeric_type(val_type):
    return is_integer_type(val_type) or is_float_type(val_type)

# Convert a string value to numeric type (int or float) if possible.
def convert_to_numeric(val):
    try:
        if '.' in str(val):
            return float(val)
        else:
            return int(val)
    except (ValueError, TypeError):
        return val

# Detect columns with inconsistent data types and display the percentage and samples of irrelevant data.
def detect_irrelevant(df):
    if not isinstance(df, pd.DataFrame):
        st.error("The provided input is not a valid DataFrame.")
        logging.error("The provided input is not a valid DataFrame.")
        return None, None
    
    if df.empty:
        st.warning("The provided DataFrame is empty.")
        logging.warning("The provided DataFrame is empty.")
        return None, None
    
    try:        
        total_rows = len(df)
        irrelevant_columns = {}
        
        logging.info(f"Total rows in DataFrame: {total_rows}")
        
        # Convert values to numeric where possible
        for col in df.columns:
            df[col] = df[col].apply(lambda x: convert_to_numeric(x) if pd.notnull(x) else x)
        
        for col in df.columns:
            initial_type = None
            
            for val in df[col]:
                if pd.notnull(val):
                    initial_type = type(val)
                    break
            
            if initial_type is None:
                continue
            
            mixed_type_indices = []
            for i in range(len(df[col])):
                if pd.notnull(df[col].iloc[i]):
                    current_type = type(df[col].iloc[i])
                    if current_type != initial_type:
                        if is_numeric_type(initial_type) and is_numeric_type(current_type):
                            continue
                        mixed_type_indices.append(i)
            
            if mixed_type_indices:
                logging.info(f"Column '{col}' has inconsistent data types.")
                irrelevant_columns[col] = mixed_type_indices
        
        column_irrelevancy = {}
        for col, indices in irrelevant_columns.items():
            col_irrelevant_percentage = (len(indices) / total_rows) * 100
            column_irrelevancy[col] = col_irrelevant_percentage

        st.subheader("Data Irrelevancy Analysis")
        text = ""
        for col, percentage in column_irrelevancy.items():
            text += f"• {col}: {percentage:.2f}%\n"
            if percentage > 5:
                indices = irrelevant_columns[col]
                text += f"\n  -> Number of irrelevant entries: {len(indices)}\n"
                text += "\n  -> Sample of irrelevant data:"
                sample_size = min(2, len(indices))
                for i in range(sample_size):
                    index = indices[i]
                    text += f"\n          Row {index}: {df[col].iloc[index]}"
            text += "\n"
        
        if not text:
            text += "• No irrelevancy in provided dataset."

        with st.expander("Column Irrelevancy Analysis", expanded=True):
            st.info(text)
        
        return irrelevant_columns, column_irrelevancy

    except Exception as e:
        st.error("An error occurred during data relevancy analysis. Please check the logs for details.")
        logging.error(f"Error occurred during data relevancy analysis: {e}")
        return {}, {}

# Extract a numeric value from a string if present.
def extract_number(text):
    try:
        number = re.search(r'\d+\.\d+', text)
        if number:
            return float(number.group())
        else:
            return None
    except Exception as e:
        logging.error(f"Error in extracting number: {e}")
        st.error(f"Error in extracting number: {e}")
        return None

# Analyze and handle string and date columns, detecting mixed values and missing data.
def handle_date_text(df, columns_by_type):
    if not isinstance(df, pd.DataFrame):
        st.error("The provided input is not a valid DataFrame.")
        logging.error("The provided input is not a valid DataFrame.")
        return
    
    if not isinstance(columns_by_type, dict):
        st.error("The columns_by_type argument must be a dictionary.")
        logging.error("The columns_by_type argument must be a dictionary.")
        return 
    
    if df.empty:
        st.warning("The provided DataFrame is empty.")
        logging.warning("The provided DataFrame is empty.")
        return

    annotation_text = ""
    
    for col in df.columns:
        try:
            if col in columns_by_type["string"]:
                annotation_text += f"• String Type - Column: {col}\n\n"
                count_checked = 0
                has_number = False
                for item in df[col].iloc[:15]:
                    if isinstance(item, str):
                        number = extract_number(item)
                        if number is not None:
                            has_number = True
                            count_checked += 1
                    elif isinstance(item, (int, float)):
                        continue
                    if count_checked == 10:
                        break
                if count_checked >= 10 and count_checked < 15:
                    annotation_text += f"  Column {col} contains mixed values.\n"
                    logging.info(f"Column {col} contains mixed values based on checked items.")
                
                df[col] = df[col].replace('nan', np.nan)
                
                missing_values = df[col].isna().sum()
                perc_missing = (missing_values / len(df[col])) * 100
                annotation_text += f"  Missing Values: {missing_values} ({perc_missing:.2f}%)\n\n"
                logging.info(f"Column {col} - Missing values: {missing_values} ({perc_missing:.2f}%)")
            
            elif col in columns_by_type["date"]:
                annotation_text += f"• Date Type - Column: {col}\n\n"
                
                df[col] = pd.to_datetime(df[col], errors='coerce')  # Convert to datetime
                df[col] = df[col].replace(pd.NaT, np.nan)  # Handle missing dates
                
                missing_values = df[col].isna().sum()
                perc_missing = (missing_values / len(df[col])) * 100
                annotation_text += f"  Missing Values: {missing_values} ({perc_missing:.2f}%)\n\n"
                logging.info(f"Column {col} - Missing values after conversion: {missing_values} ({perc_missing:.2f}%)")
        
        except Exception as e:
            logging.error(f"Error processing column {col}: {e}")
            st.warning(f"Error processing column {col}. Check logs for details.")
    
    if not annotation_text:
        annotation_text += "• No values for handling free text."

    st.subheader("Free Text Analysis")
    
    with st.expander("String & Date Analysis", expanded=True):
        st.info(annotation_text)
        
# Handle missing values in the DataFrame based on column type and imputation strategy.
def handle_missing_values(df, columns_by_type, normal_col, not_normal_col, removed_info):
    if not isinstance(df, pd.DataFrame):
        st.error("The provided input is not a valid DataFrame.")
        logging.error("The provided input is not a valid DataFrame.")
        return None, None, None
    
    if not isinstance(columns_by_type, dict):
        st.error("The columns_by_type argument must be a dictionary.")
        logging.error("The columns_by_type argument must be a dictionary.")
        return None, None, None
    
    if df.empty:
        st.warning("The provided DataFrame is empty.")
        logging.warning("The provided DataFrame is empty.")
        return None, None, None
    
    if not isinstance(normal_col, list) or not isinstance(not_normal_col, list):
        st.error("The inputs 'normal_col' and 'not_normal_col' must be lists.")
        logging.error("The inputs 'normal_col' or 'not_normal_col' are not lists.")
        return None, None, None
    
    if not isinstance(removed_info, list):
        st.error("The input 'removed_info' must be a list of tuples with two elements each.")
        logging.error("The input 'removed_info' is not in the expected format.")
        return None, None, None

    null_counts = df.isnull().sum()
    
    st.subheader("Handling Missing Values")
    
    annotation_text = ""
    more_missing = []
    
    for key, old_list in removed_info:
        columns_by_type[key] = old_list
    
    if df.isnull().values.any():
        for col in df.columns:
            if null_counts[col] > 0:
                missing_ratio = null_counts[col] / len(df[col])
                if missing_ratio > 0.5:
                    annotation_text += f"• Column {col} has more than 50% missing values.\n\n"
                    more_missing.append(col)
                    logging.info(f"Column {col} has more than 50% missing values.")
                else:
                    try:
                        if col in columns_by_type['nominal'] or col in columns_by_type["ordinal"]:
                            imputer = SimpleImputer(strategy='most_frequent')
                            df[col] = imputer.fit_transform(df[[col]]).ravel()
                            annotation_text += f"• Imputed missing values in categorical column {col} with the most frequent value.\n\n"
                            logging.info(f"Imputed missing values in categorical column {col} with the most frequent value.")
                        elif col in columns_by_type['int']:
                            imputer = SimpleImputer(strategy='median')
                            df[col] = imputer.fit_transform(df[[col]])
                            annotation_text += f"• Imputed missing values in integer column {col} with median.\n\n"
                            logging.info(f"Imputed missing values in integer column {col} with median.")
                        elif col in columns_by_type['float']:
                            if col in normal_col:
                                imputer = SimpleImputer(strategy='mean')
                                df[col] = imputer.fit_transform(df[[col]])
                                annotation_text += f"• Imputed missing values in float column {col} with mean.\n\n"
                                logging.info(f"Imputed missing values in float column {col} with mean.")
                            elif col in not_normal_col:
                                imputer = SimpleImputer(strategy='median')
                                df[col] = imputer.fit_transform(df[[col]])
                                annotation_text += f"• Imputed missing values in float column {col} with median.\n\n"
                                logging.info(f"Imputed missing values in float column {col} with median.")
                        elif col in columns_by_type['string']:
                            imputer = SimpleImputer(strategy='most_frequent')
                            df[col] = imputer.fit_transform(df[[col]]).ravel()
                            annotation_text += f"• Imputed missing values in string column {col} with the most frequent value.\n\n"
                            logging.info(f"Imputed missing values in string column {col} with the most frequent value.")
                        elif col in columns_by_type['date']:
                            imputer = SimpleImputer(strategy='most_frequent')
                            df[col] = imputer.fit_transform(df[[col]]).ravel()
                            annotation_text += f"• Imputed missing values in date column {col} with the most frequent value.\n\n"
                            logging.info(f"Imputed missing values in date column {col} with the most frequent value.")
                        else:
                            imputer = SimpleImputer(strategy='most_frequent')
                            df[col] = imputer.fit_transform(df[[col]]).ravel()
                            annotation_text += f"• Imputed missing values in column {col} with the most frequent value.\n\n"
                            logging.info(f"Imputed missing values in column {col} with the most frequent value.")
                    except ValueError as ve:
                        st.error(f"ValueError handling missing values in column {col}: {ve}")
                        annotation_text += f"• Error handling missing values in column {col}: {ve}\n\n"
                        logging.error(f"ValueError handling missing values in column {col}: {ve}")
                    except TypeError as te:
                        st.error(f"TypeError handling missing values in column {col}: {te}")
                        annotation_text += f"• Error handling missing values in column {col}: {te}\n\n"
                        logging.error(f"TypeError handling missing values in column {col}: {te}")
                    except Exception as e:
                        st.error(f"Unexpected error handling missing values in column {col}: {e}")
                        annotation_text += f"• Unexpected error handling missing values in column {col}: {e}\n\n"
                        logging.error(f"Unexpected error handling missing values in column {col}: {e}")
    else:
        annotation_text += "• No missing values found in the DataFrame.\n\n"
    
    with st.expander("Handling Missing Values", expanded=True):
        st.info(annotation_text)
        
    return df, more_missing, columns_by_type

def get_cleaned_data():
    return df_cleaned_download

##################################################################
#                                                                #
#          >>>>> MODULE 2 - Data Transformation <<<<<            #
#                                                                #
##################################################################

# Identifies and removes ID columns based on uniqueness.
def find_ids_col(df_cleaned, columns_by_type, not_normal):
    if df_cleaned.empty:
        st.warning("The DataFrame is empty.")
        logging.warning("The DataFrame is empty.")
        return df_cleaned, columns_by_type, not_normal
    
    total_rows = len(df_cleaned)
    potential_id_columns = []

    logging.info("Starting ID column identification process.")
    
    for column in df_cleaned.columns:
        if df_cleaned[column].empty:
            potential_id_columns.append(column)
            logging.info(f"Column {column} is empty and thus considered a potential ID column.")
        
        try:
            unique_ratio = df_cleaned[column].nunique() / total_rows
        except Exception as e:
            st.error(f"Error processing column {column}: {e}")
            logging.error(f"Error processing column {column}: {e}")
            continue

        if unique_ratio >= 1:
            if 'id' in column.lower():
                potential_id_columns.append(column)
                if column in not_normal:
                    not_normal.remove(column)
                logging.info(f"Identified potential ID column: {column}")

    if potential_id_columns:
        df_cleaned.drop(columns=potential_id_columns, axis=1, inplace=True)
        logging.info(f"Removed potential ID columns: {', '.join(potential_id_columns)}")

    for col_type, cols in columns_by_type.items():
        if not isinstance(cols, list):
            st.error(f"Expected a list for columns of type {col_type} but got {type(cols).__name__}.")
            logging.error(f"Expected a list for columns of type {col_type} but got {type(cols).__name__}.")
            continue
        columns_by_type[col_type] = [col for col in cols if col not in potential_id_columns]

    return df_cleaned, columns_by_type, not_normal

# Applies user choices to handle duplicates, irrelevant columns, and missing values in the DataFrame.
def operations_on_data(df_trans, columns_by_type, percentage_duplicates, irrelevant_columns, more_missing, action_duplicate, action_irrelevant, action_missing):
    removed_columns = []

    logging.info("Starting data operations.")
    
    if percentage_duplicates > 0:
        if action_duplicate == "yes":
            try:
                df_trans = df_trans.drop_duplicates(keep='first')
                logging.info("Duplicate rows removed from DataFrame.")
            except Exception as e:
                st.error(f"Error removing duplicates: {e}")
                logging.error(f"Error removing duplicates: {e}")
    
    if irrelevant_columns:
        if action_irrelevant == 'yes':
            for col in irrelevant_columns.keys():
                try:
                    df_trans = df_trans.drop(columns=[col])
                    removed_columns.append(col)
                    logging.info(f"Removed irrelevant column: {col}")
                except Exception as e:
                    st.error(f"Error removing irrelevant column {col}: {e}")
                    logging.error(f"Error removing irrelevant column {col}: {e}")
        elif action_irrelevant == 'no':
            for col, indices in irrelevant_columns.items():
                if not df_trans.index.isin(indices).all():
                    st.warning(f"Warning: Some indices in {col} are not valid.")
                    logging.warning(f"Warning: Some indices in {col} are not valid.")

                df_trans.loc[indices, col] = np.nan
                null_counts = df_trans.isnull().sum()
                
                if null_counts[col] > 0:    
                    try:
                        if col in columns_by_type.get('nominal', []) or col in columns_by_type.get('ordinal', []):
                            imputer = SimpleImputer(strategy='most_frequent')
                        elif col in columns_by_type.get('int', []):
                            imputer = SimpleImputer(strategy='median')
                        elif col in columns_by_type.get('float', []):
                            imputer = SimpleImputer(strategy='mean')
                        elif col in columns_by_type.get('string', []):
                            imputer = SimpleImputer(strategy='most_frequent')
                        else:
                            imputer = SimpleImputer(strategy='most_frequent')
                        
                        df_trans[[col]] = imputer.fit_transform(df_trans[[col]])
                        logging.info(f"In irrelevant columns - imputed missing values in column {col}.")
                    except Exception as e:
                        st.error(f"Error imputing values in irrelevant column {col}: {e}")
                        logging.error(f"Error imputing values in irrelevant column {col}: {e}")
    
    if more_missing:
        if action_missing == 'yes':
            for col in more_missing:
                if col in df_trans.columns:
                    try:
                        df_trans = df_trans.drop(columns=[col])
                        removed_columns.append(col)
                        logging.info(f"Removed column with high percentage of missing values: {col}")
                    except Exception as e:
                        st.error(f"Error removing column {col} with high missing values: {e}")
                        logging.error(f"Error removing column {col} with high missing values: {e}")
        elif action_missing == 'no':
            imputer = SimpleImputer(strategy='most_frequent')
            for col in more_missing:
                if col in df_trans.columns:
                    try:
                        if pd.api.types.is_datetime64_any_dtype(df_trans[col]):
                            mode_value = df_trans[col].mode().iloc[0]
                            df_trans[col] = df_trans[col].fillna(mode_value)
                            logging.info(f"Filled missing values in datetime column {col} with mode.")
                        else:
                            if df_trans[[col]].notnull().any().any():
                                df_trans[col] = imputer.fit_transform(df_trans[[col]]).ravel()
                                logging.info(f"Imputed missing values in column {col} with most frequent value.")
                            else:
                                df_trans = df_trans.drop(columns=[col])
                                removed_columns.append(col)
                                logging.info(f"Dropped column {col} as it was completely empty.")
                    except Exception as e:
                        st.error(f"Error processing column {col} with high missing values: {e}")
                        logging.error(f"Error processing column {col} with high missing values: {e}")

    for col in removed_columns:
        for key in columns_by_type:
            if col in columns_by_type[key]:
                columns_by_type[key].remove(col)
                logging.info(f"Updated columns_by_type to remove {col} from {key} list.")
    
    return df_trans, columns_by_type

# Apply transformations to correct skewness in non-normal columns.
def treat_non_normal_columns(df_trans, not_normal):
    df_trans_original = df_trans.copy()
    
    for col in not_normal:
        try:
            if col not in df_trans.columns:
                st.warning(f"Column {col} not found in DataFrame.")
                logging.warning(f"Column {col} not found in DataFrame.")
                continue

            original_skewness = stats.skew(df_trans[col].dropna())
            is_non_negative = np.all(df_trans[col].dropna() >= 0)
            
            logging.info(f"Processing column {col}: original skewness = {original_skewness:.2f}")

            if abs(original_skewness) > 1:
                if is_non_negative:
                    if np.any(df_trans[col].dropna() == 0):
                        df_trans[col].replace(0, 1e-10, inplace=True)
                        df_trans[col] = np.log1p(df_trans[col])
                        logging.info(f"Applied log1p transformation to column {col}.")
                    else:
                        df_trans[col] = np.log10(df_trans[col].replace(0, np.nan))  # Avoid log(0)
                        logging.info(f"Applied log10 transformation to column {col}.")
                else:
                    pt = PowerTransformer(method='yeo-johnson', standardize=False)
                    df_trans[col] = pt.fit_transform(df_trans[col].dropna().values.reshape(-1, 1)).flatten()
                    logging.info(f"Applied Yeo-Johnson transformation to column {col}.")
            
            elif abs(original_skewness) > 0.5:
                if is_non_negative:
                    df_trans[col].replace(0, 1e-10, inplace=True)
                    df_trans[col] = np.sqrt(df_trans[col])
                    logging.info(f"Applied square root transformation to column {col}.")
                else:
                    df_trans[col] = np.cbrt(df_trans[col])
                    logging.info(f"Applied cube root transformation to column {col}.")
            
            new_skewness = stats.skew(df_trans[col].dropna())
            logging.info(f"Column {col}: new skewness = {new_skewness:.2f}")
        
        except Exception as e:
            st.error(f"Error processing column {col}: {e}")
            logging.error(f"Error processing column {col}: {e}")
    
    return df_trans

# Normalize numeric columns in the DataFrame using StandardScaler.
def normalization(df_trans, columns_by_type):
    scaler = StandardScaler()
    
    numeric_columns = columns_by_type['float'] + columns_by_type['int']
    
    for col in numeric_columns:
        try:
            if col not in df_trans.columns:
                st.warning(f"Column {col} not found in DataFrame. Skipping normalization.")
                logging.warning(f"Column {col} not found in DataFrame. Skipping normalization.")
                continue
            
            df_trans[col] = scaler.fit_transform(df_trans[[col]])
            logging.info(f"Normalized column {col}.")
        
        except Exception as e:
            st.error(f"Error normalizing column {col}: {e}")
            logging.error(f"Error normalizing column {col}: {e}")
                
    return df_trans

# Encode categorical columns, update and return the columns_by_type dictionary along with the transformed DataFrame.
def encoding_categoricals(df_trans, columns_by_type):
    new_columns_by_type = {
        'nominal': [],
        'ordinal': columns_by_type.get('ordinal', [])
    }
    nominal_cols_mapping = {}

    for col in columns_by_type.get('nominal', []):
        if col in df_trans.columns:
            try:
                original_cols = [col]
                df_trans = pd.get_dummies(df_trans, columns=[col], prefix=col, drop_first=True)
                new_cols = [c for c in df_trans.columns if c.startswith(col + '')]
                nominal_cols_mapping[col] = new_cols
                new_columns_by_type['nominal'].extend(new_cols)
                logging.info(f"Encoded nominal column '{col}' into new columns: {new_cols}.")
            except Exception as e:
                logging.error(f"Error encoding nominal column '{col}': {e}")
                st.error(f"Error encoding nominal column '{col}': {e}")
        else:
            logging.warning(f"Nominal column '{col}' not found in DataFrame.")
            st.warning(f"Nominal column '{col}' not found in DataFrame.")

    ordinal_encoder = OrdinalEncoder()
    for col in columns_by_type.get('ordinal', []):
        if col in df_trans.columns:
            try:
                df_trans[col] = ordinal_encoder.fit_transform(df_trans[[col]])
                logging.info(f"Encoded ordinal column '{col}'.")
            except Exception as e:
                logging.error(f"Error encoding ordinal column '{col}': {e}")
                st.error(f"Error encoding ordinal column '{col}': {e}")
        else:
            logging.warning(f"Ordinal column '{col}' not found in DataFrame.")
            st.warning(f"Ordinal column '{col}' not found in DataFrame.")

    columns_by_type['nominal'] = new_columns_by_type['nominal']
    columns_by_type['ordinal'] = [col for col in columns_by_type['ordinal'] if col not in nominal_cols_mapping]

    return df_trans, columns_by_type

# Process string columns: apply frequency or TF-IDF encoding as needed.
def string_process(df_trans, columns_by_type):
    try:
        columns_to_remove = [col for col in columns_by_type['string']]
        logging.info(f"Columns to remove (string): {columns_to_remove}")

        df_trans = df_trans.drop(columns=columns_to_remove)
        columns_by_type['string'] = [col for col in columns_by_type['string'] if col not in columns_to_remove]

        logging.info("String columns successfully removed.")
        return df_trans, columns_by_type

    except KeyError as e:
        logging.error(f"KeyError: {e}")
        st.error(f"KeyError: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        st.error(f"Unexpected error: {e}")

# Convert date columns to new features, apply one-hot encoding, and update columns_by_type dictionary.
def handling_date(df_trans, columns_by_type):
    try:
        columns_to_remove = [col for col in columns_by_type['date']]
        logging.info(f"Columns to remove (date): {columns_to_remove}")

        df_trans = df_trans.drop(columns=columns_to_remove)
        columns_by_type['date'] = [col for col in columns_by_type['date'] if col not in columns_to_remove]

        logging.info("Date columns successfully removed.")
        return df_trans, columns_by_type

    except KeyError as e:
        logging.error(f"KeyError: {e}")
        st.error(f"KeyError: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        st.error(f"Unexpected error: {e}")

##################################################################
#                                                                #
#            >>>>> MODULE 3 - ML Model Building <<<<<            #
#                                                                #
##################################################################

# Generate and display a correlation matrix heatmap.
def plot_correlation_matrix(df_trans, figsize=(1000, 800)):
    try:
        logging.info("Generating correlation matrix heatmap.")

        st.subheader("Feature Selection")

        corr = df_trans.corr()
        logging.info("Correlation matrix computed.")

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.index,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            text=np.round(corr.values, decimals=2),
            texttemplate="%{text}",
            hoverongaps=False
        ))

        fig.update_layout(
            title='Correlation Matrix',
            width=figsize[0],
            height=figsize[1],
            xaxis_title='Features',
            yaxis_title='Features',
            xaxis_showgrid=False,
            yaxis_showgrid=False,
            yaxis_autorange='reversed',
            xaxis=dict(
                tickangle=270,
                tickmode='array',
                tickvals=list(range(len(corr.columns))),
                ticktext=corr.columns
            )
        )

        st.plotly_chart(fig, use_container_width=True)
        logging.info("Heatmap displayed successfully.")

        return corr

    except Exception as e:
        logging.error(f"An error occurred while generating the correlation matrix heatmap: {e}")
        st.error(f"An error occurred while generating the correlation matrix heatmap: {e}")
        return None

# drop columns that are highly correlated
def select_features_correlation(df_trans, threshold=0.4):
    try:
        corr_matrix = df_trans.corr().abs()
        
        logging.info("Correlation matrix computed.")
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        logging.info(f"Features to drop based on correlation: {to_drop}")
        
        df_trans_reduced = df_trans.drop(to_drop, axis=1)
        
        return df_trans_reduced
    except Exception as e:
        logging.error(f"An error occurred while selecting features based on correlation: {e}")
        st.error(f"An error occurred while selecting features based on correlation: {e}")
        return df_trans

# drop columns that are low variance
def remove_low_variance_features(df_final, threshold=0.01):
    try:
        if df_final.empty:
            st.warning("The DataFrame is empty. No features to process.")
            logging.warning("The DataFrame is empty. No features to process.")
            return df_final

        selector = VarianceThreshold(threshold=threshold)
        df_final_selected = selector.fit_transform(df_final)
        logging.info("VarianceThreshold applied to remove low variance features.")

        selected_features = selector.get_support()
        selected_columns = df_final.columns[selected_features]
        dropped_columns = df_final.columns[~selected_features]

        logging.info(f"Features with variance below {threshold} that will be dropped: {list(dropped_columns)}")

        df_final_reduced = df_final[selected_columns]

        return df_final_reduced

    except Exception as e:
        logging.error(f"An error occurred while removing low variance features: {e}")
        st.error(f"An error occurred while removing low variance features: {e}")
        return df_final

# detect problem type: if classification return True else false
def detect_problem_type(y):
    try:
        if y is None or len(y) == 0:
            st.warning("The target variable is empty or None. Cannot detect problem type.")
            logging.warning("The target variable is empty or None. Cannot detect problem type.")
            return False

        if not isinstance(y, pd.Series):
            st.warning("The target variable is not a pandas Series. Please provide a pandas Series.")
            logging.warning("The target variable is not a pandas Series. Please provide a pandas Series.")
            return False

        unique_values = y.nunique()
        total_values = len(y)

        classification_threshold = 0.1 * total_values
        is_classification = unique_values <= classification_threshold

        logging.info(f"Detected problem type. Unique values: {unique_values}, Classification threshold: {classification_threshold}, Classification: {is_classification}")

        return is_classification

    except Exception as e:
        logging.error(f"An error occurred while detecting problem type: {e}")
        st.error(f"An error occurred while detecting problem type: {e}")
        return False

# select features wrapper
def select_features_wrapper(df_final, y, is_classification):
    try:
        if len(df_final) != len(y):
            raise ValueError(f"Mismatch between number of rows in df_final ({len(df_final)}) and length of y ({len(y)})")

        df_final = df_final.reset_index(drop=True)
        y = y.reset_index(drop=True)

        model = RandomForestClassifier() if is_classification else RandomForestRegressor()
        cv = StratifiedKFold(5) if is_classification else KFold(5)
        scoring = 'accuracy' if is_classification else 'neg_mean_squared_error'

        rfecv = RFECV(estimator=model, step=1, cv=cv, scoring=scoring)
        rfecv.fit(df_final, y)
        rfe_features = df_final.columns[rfecv.support_].tolist()
        logging.info(f"Features selected by RFECV: {rfe_features}")

        model.fit(df_final, y)
        importances = model.feature_importances_
        rf_features = df_final.columns[importances > importances.mean()].tolist()
        logging.info(f"Features selected by RandomForest importance: {rf_features}")

        l1_model = LogisticRegression(penalty='l1', solver='liblinear') if is_classification else Lasso()
        l1_model.fit(df_final, y)
        coef_ = l1_model.coef_[0] if is_classification else l1_model.coef_
        l1_features = pd.Series(abs(coef_), index=df_final.columns)[abs(coef_) > 0].index.tolist()
        logging.info(f"Features selected by L1 penalty: {l1_features}")

        selected_features = list(set(rfe_features + rf_features + l1_features))
        logging.info(f"Combined selected features: {selected_features}")

        df_final = df_final[selected_features]

        return df_final

    except ValueError as ve:
        logging.error(f"ValueError: {ve}")
        st.error(f"ValueError: {ve}")
        return df_final
    except NotFittedError as nfe:
        logging.error(f"Model not fitted error: {nfe}")
        st.error(f"Model not fitted error: {nfe}")
        return df_final
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        st.error(f"An unexpected error occurred: {e}")
        return df_final

# test 4 models for classification problem
def compare_classification_models(X, y):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVC': SVC(),
        'Gradient Boosting': GradientBoostingClassifier()
    }

    results = {}

    for name, model in models.items():
        try:
            predictions = cross_val_predict(model, X, y, cv=5, method='predict')

            conf_matrix = confusion_matrix(y, predictions)

            results[name] = {
                'conf_matrix': conf_matrix,
            }
            logging.info(f"{name}: Classification metrics computed successfully.")

        except Exception as e:
            logging.error(f"{name}: Error occurred - {e}")
            st.error(f"{name}: Error occurred - {e}")
            results[name] = {'error': str(e)}

    return results

# test 4 models for regression problem
def compare_regression_models(X, y):
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'SVR': SVR(),
        'Gradient Boosting': GradientBoostingRegressor()
    }

    results = {}

    for name, model in models.items():
        try:
            scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
            rmse_scores = np.sqrt(-scores)
            results[name] = {
                'mean_rmse': rmse_scores.mean(),
                'std_rmse': rmse_scores.std()
            }
            logging.info(f"{name}: Regression metrics computed successfully.")

        except Exception as e:
            logging.error(f"{name}: Error occurred - {e}")
            st.error(f"{name}: Error occurred - {e}")
            results[name] = {'error': str(e)}

    return results

# display the result of all 4 models for classification problem
def display_classification_results(results):
    try:
        cols = st.columns(len(results))

        for i, (name, metrics) in enumerate(results.items()):
            with cols[i]:
                st.write(name)
                st.write("")

                if 'conf_matrix' in metrics:
                    try:
                        fig, ax = plt.subplots()
                        sns.heatmap(metrics['conf_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 16})
                        ax.set_xlabel('Predicted', fontsize=16)
                        ax.set_ylabel('Actual', fontsize=16)
                        ax.set_title('Confusion Matrix', fontsize=18)
                        st.pyplot(fig)
                        logging.info(f"{name}: Confusion matrix displayed successfully.")
                    except Exception as e:
                        logging.error(f"{name}: Error displaying confusion matrix - {e}")
                        st.error(f"Error displaying Confusion Matrix for {name}: {e}")
                else:
                    st.write("Confusion Matrix: Not available")

                st.write("\n")

    except Exception as e:
        logging.error(f"An error occurred while displaying classification results: {e}")
        st.error(f"Error displaying classification results: {e}")

# display the result of all 4 models for regression problem
def display_regression_results(results):
    try:
        cols = st.columns(len(results))

        for i, (name, metrics) in enumerate(results.items()):
            with cols[i]:
                st.write(name)
                st.write("")

                mean_rmse = metrics.get('mean_rmse', 'N/A')
                std_rmse = metrics.get('std_rmse', 'N/A')
                st.write(f"Mean RMSE: {mean_rmse:.2f}" if mean_rmse != 'N/A' else "Mean RMSE: Not available")
                st.write(f"RMSE Standard Deviation: {std_rmse:.2f}" if std_rmse != 'N/A' else "RMSE Standard Deviation: Not available")

                st.write("\n")

    except Exception as e:
        logging.error(f"An error occurred while displaying regression results: {e}")
        st.error(f"Error displaying regression results: {e}")

# Function to save model to the database
def save_model_to_db(model, model_name, dataset_id):
    # Serialize the model using joblib
    model_binary = BytesIO()
    joblib.dump(model, model_binary)
    model_binary.seek(0)  # Move the cursor to the beginning of the stream

    # Connect to your SQLite database (or modify this for your database setup)
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()

    # Insert the model into the table
    cursor.execute('''
        INSERT INTO models (dataset_id, model_name, ml_model)
        VALUES (?, ?, ?)
    ''', (dataset_id, model_name, model_binary.read()))

    conn.commit()
    conn.close()

# saved model by choosing model name from select box
def saved_model(X_train, y_train, model_name, is_classification):
    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'SVC': SVC(),
        'Gradient Boosting': GradientBoostingClassifier()
    } if is_classification else {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'SVR': SVR(),
        'Gradient Boosting': GradientBoostingRegressor()
    }

    if model_name not in models:
        st.error(f"Model {model_name} is not available. Please select a valid model.")
        logging.error(f"Model {model_name} is not available.")
        raise ValueError(f"Model {model_name} is not in the available models.")

    try:    
        model = models[model_name]
        model.fit(X_train, y_train)
        st.success(f"Model {model_name} saved successfully!")
        logging.info(f"Model {model_name} trained and saved successfully.")

        save_model_to_db(model, model_name, dataset_id)

        return model

    except Exception as e:
        st.error(f"Error occurred while saving the model {model_name}: {e}")
        logging.error(f"Error occurred while saving the model {model_name}: {e}")

def make_predictions(model, test_data):
    if model is None:
        st.error("No model provided")
        return None
    try:
        predictions = model.predict(test_data)
        return predictions
    except Exception as e:
        st.error(str(e))
        return None
    
def load_real_pred_to_csv(y_test, predictions):
    try:
        df = pd.DataFrame({
            'True Values': y_test,
            'Predictions': predictions
        })
        return df
    except Exception as e:
        st.error(str(e))

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
    if 'model_saved' not in st.session_state:
        st.session_state.model_saved = False

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            file_path = save_uploaded_file(uploaded_file)

            if 'df' not in st.session_state:
                st.session_state.df = load_data(uploaded_file)

                dataset_name = uploaded_file.name

                global dataset_id
                dataset_id = add_dataset_to_db(user_id, dataset_name, file_path)

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
                                outlier, columns_by_type = detect_outlier(df_without_y, columns_by_type)
                                normal_col, not_normal_col = check_normality_with_graphs(df_without_y, columns_by_type, outlier)
                                duplicated_rows, percentage_duplicates = check_duplicity(df_without_y)
                                irrelevant_columns, column_irrelevancy = detect_irrelevant(df_without_y)
                                handle_date_text(df_without_y, columns_by_type)
                                global df_cleaned_download
                                df_cleaned, more_missing, columns_by_type = handle_missing_values(df, columns_by_type, normal_col, not_normal_col, removed_info)
                                df_cleaned_download = df_cleaned.copy()
                                st.session_state.df_cleaned = df_cleaned.copy() 

                                df_trans, columns_by_type, not_normal_col = find_ids_col(df_cleaned, columns_by_type, not_normal_col)
                                df_trans, columns_by_type = operations_on_data(df_trans, columns_by_type, percentage_duplicates, irrelevant_columns, more_missing, action_duplicate, action_irrelevant, action_missing)
                                y = df_trans[target_column]
                                df_trans = df_trans.drop(columns=[target_column])
                                for key in columns_by_type:
                                    if target_column in columns_by_type[key]:
                                        columns_by_type[key].remove(target_column)
                                df_trans = treat_non_normal_columns(df_trans, not_normal_col)
                                df_trans = normalization(df_trans, columns_by_type)
                                df_trans, columns_by_type = encoding_categoricals(df_trans, columns_by_type)
                                df_trans, columns_by_type = handling_date(df_trans, columns_by_type)
                                df_trans, columns_by_type = string_process(df_trans, columns_by_type)

                                df_final = select_features_correlation(df_trans)
                                df_final = remove_low_variance_features(df_final)
                                plot_correlation_matrix(df_final)
                                problem_type = detect_problem_type(y)
                                df_final = select_features_wrapper(df_final, y, problem_type)
                                
                                X_train, X_test, y_train, y_test = train_test_split(df_final, y, test_size=0.2, random_state=42)

                                if problem_type:
                                    st.subheader("Comparing classification models...")
                                    results = compare_classification_models(X_train, y_train)
                                    display_classification_results(results)
                                else:
                                    st.write("Comparing regression models...")
                                    results = compare_regression_models(X_train, y_train)
                                    display_regression_results(results)

                                st.session_state.results = results
                                st.session_state.X_train = X_train
                                st.session_state.y_train = y_train
                                st.session_state.X_test = X_test
                                st.session_state.y_test = y_test
                                st.session_state.problem_type = problem_type
                                st.session_state.df_final = df_final
                                st.session_state.analysis_done = True      

                    except Exception as e:
                        st.error(f"An error occurred during data analysis and transformation: {str(e)}")
                        logging.error(f"An error occurred during data analysis and transformation: {e}")

                if st.session_state.analysis_done:
                    st.success("Analysis completed!")
                    
                    
                    model_names = list(st.session_state.results.keys())
                    selected_model_name = st.selectbox("Select a model to save", model_names)
                    
                    if st.button('Save Model'):
                        with st.spinner("Saving model..."):
                            try:
                                model = saved_model(st.session_state.X_train, st.session_state.y_train, selected_model_name, st.session_state.problem_type)
                                st.session_state.model = model
                                print(model)
                                predictions = make_predictions(model, st.session_state.X_test)
                                df_pred = load_real_pred_to_csv(st.session_state.y_test, predictions)
                                st.session_state.df_pred = df_pred
                                st.session_state.model_saved = True
                            except Exception as e:
                                st.error(f"An error occurred while saving the model: {str(e)}")
                                logging.error(f"An error occurred while saving the model: {e}")

                    if 'model_saved' in st.session_state and st.session_state.model_saved:
                        st.success("Model saved successfully!")
                        st.subheader("Download Options")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                                if 'df_cleaned' in st.session_state:
                                    if st.download_button(
                                        label="Download Cleaned Data",
                                        data=st.session_state.df_cleaned.to_csv(index=False).encode('utf-8'),
                                        file_name="data_cleaned.csv",
                                        mime="text/csv"
                                    ):
                                        st.success("Cleaned data file is ready for download!")
                                
                        with col2:
                            if 'df_pred' in st.session_state:
                                if st.download_button(
                                    label="Download Prediction Data",
                                    data=st.session_state.df_pred.to_csv(index=False).encode('utf-8'),
                                    file_name="predictions.csv",
                                    mime="text/csv"
                                ):
                                    st.success("Prediction data file is ready for download!")

                        with col3:
                            if 'model' in st.session_state:
                                buffer = io.BytesIO()
                                joblib.dump(st.session_state.model, buffer)
                                buffer.seek(0)
                                if st.download_button(
                                    label="Download ML Model",
                                    data=buffer,
                                    file_name=f"{selected_model_name}.pkl",
                                    mime="application/octet-stream"
                                ):
                                    st.success("ML model file is ready for download!")        
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

    dataset_id = create_db.c.lastrowid
    return dataset_id

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
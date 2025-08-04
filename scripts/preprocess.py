import pandas as pd
import numpy as np
import torch
import xarray as xr
import glob
from sklearn.preprocessing import StandardScaler
import warnings
import cdsapi
import os
from retrying import retry

@retry(stop_max_attempt_number=3, wait_fixed=10000)
def download_month(c, year, month, variables, area, filename):
    """A robust function to download a single month of data."""
    print(f"Requesting data from the CDS for {year}-{month}...")
    c.retrieve(
        # --- THE CRITICAL FIX ---
        'reanalysis-era5-single-levels', # Use the main dataset
        {
            'product_type': 'reanalysis',
            'variable': variables,
            'year': year,
            'month': month,
            'day': [f'{d:02d}' for d in range(1, 32)],
            'time': [f'{h:02d}:00' for h in range(24)],
            'area': area,
            'format': 'netcdf',
        },
        filename
    )

def process_weather_data(base_data_folder='data/germany_weather_2015_2020/',
                         opsd_file_path = 'data/Open_Power_System_Data/time_series_60min_singleindex.csv',
                         output_csv_path = 'germany_energy_and_weather_hourly.csv'
                         ):
    """ Process weather data from ERA5 and OPSD, merging them into a single DataFrame. """
    # --- Step 1: Find All Weather Data Files ---
    print("--- Searching for weather data files... ---")

    # Create two separate patterns, one for each file type
    instant_pattern = os.path.join(base_data_folder, '**', '*stepType-instant.nc')
    accum_pattern = os.path.join(base_data_folder, '**', '*stepType-accum.nc')

    # Find all files of each type
    instant_files = sorted(glob.glob(instant_pattern, recursive=True))
    accum_files = sorted(glob.glob(accum_pattern, recursive=True))

    if not instant_files or not accum_files:
        print("Error: Could not find both 'instant' and 'accum' files. Please check the download folder.")
        exit()

    print(f"Found {len(instant_files)} instantaneous and {len(accum_files)} accumulated files.")

    # --- Step 2: Load and Process Each File Type Separately ---
    print("--- Loading and processing file types... ---")

    # Open all instantaneous files together
    ds_instant = xr.open_mfdataset(instant_files, combine="by_coords", engine="netcdf4")
    df_instant = ds_instant.mean(dim=['latitude', 'longitude']).to_dataframe()

    # Open all accumulated files together
    ds_accum = xr.open_mfdataset(accum_files, combine="by_coords", engine="netcdf4")
    df_accum = ds_accum.mean(dim=['latitude', 'longitude']).to_dataframe()

    # --- Step 3: Merge the Two Weather DataFrames ---
    print("--- Merging weather data types... ---")
    # Merge the two dataframes on their shared 'time' index
    df_weather = pd.merge(df_instant, df_accum, left_index=True, right_index=True)


    # --- Step 4: Final Processing of the Combined Weather DataFrame ---
    print("--- Finalizing weather data... ---")

    # Drop metadata columns that came from the NetCDF files
    df_weather.drop(columns=['number', 'expver', 'step'], inplace=True, errors='ignore')

    # Rename columns to be more descriptive
    df_weather.rename(columns={
        't2m': 'temperature_celsius',
        'ssrd': 'solar_radiation',
        'tcc': 'total_cloud_cover',
        'u100': 'wind_u_100m',
        'v100': 'wind_v_100m'
    }, inplace=True)

    # Process the data
    df_weather['temperature_celsius'] = df_weather['temperature_celsius'] - 273.15
    df_weather['wind_speed_100m'] = np.sqrt(df_weather['wind_u_100m']**2 + df_weather['wind_v_100m']**2)
    df_weather.drop(columns=['wind_u_100m', 'wind_v_100m'], inplace=True)

    df_weather.index.name = 'utc_timestamp'
    df_weather = df_weather.tz_localize('UTC')

    print("Weather data processed successfully.")


    # --- Step 5: Load Your OPSD Energy Data ---
    print("\n--- Loading OPSD energy data... ---")
    try:
        df_energy = pd.read_csv(
            opsd_file_path,
            index_col='utc_timestamp',
            parse_dates=True
        )
    except FileNotFoundError:
        print(f"Error: Could not find the OPSD hourly data file at '{opsd_file_path}'.")
        exit()


    # --- Step 6: Merge the Energy and Weather Datasets ---
    print("\n--- Merging energy and weather data... ---")
    df_final = pd.merge(df_energy, df_weather, left_index=True, right_index=True, how='inner')


    # --- Step 7: Final Check and Save ---
    print("\n--- Final Merged DataFrame ---")
    print(f"Final data shape: {df_final.shape}")
    print("Checking for any remaining missing values (top 5)...")
    print(df_final.isnull().sum().sort_values(ascending=False).head())

    df_final.to_csv(output_csv_path)
    print(f"\n✅ Success! Final merged data saved to '{output_csv_path}'")

def create_sequences(data, input_len, output_len, target_col_indices):
    """
    Creates input sequences (X) and output sequences (y) from time series data.

    Args:
        data (pd.DataFrame or np.array): The input data.
        input_len (int): The length of the input sequences (lookback window).
        output_len (int): The length of the output sequences (forecast horizon).
        target_col_indices (list of int): List of column indices for the target variable(s).

    Returns:
        tuple: A tuple containing two NumPy arrays (X, y).
    """
    X_list, y_list = [], []
    data_as_array = data.to_numpy() # Convert dataframe to numpy array for efficiency

    for i in range(len(data_as_array) - input_len - output_len + 1):
        # The input window (all features)
        input_window = data_as_array[i : i + input_len, :]
        X_list.append(input_window)

        # The output window (only the target column(s))
        output_window = data_as_array[i + input_len : i + input_len + output_len, target_col_indices]
        y_list.append(output_window)

    return np.array(X_list), np.array(y_list)

def preprocess_data(file_name='data/Open_Power_System_Data/germany_energy_and_weather_hourly.csv',
                    predict_target='combined', input_length=168, output_length=1, debug=False):
    """
    Loads, cleans, and prepares the German energy dataset for time series forecasting.

    Args:
        file_name (str): Path to the input CSV file.
        predict_target (str): The target to forecast. Must be one of 
                              ['combined', 'solar', 'wind', 'wind_onshore', 'wind_offshore'].
        input_length (int): The number of past time steps to use as input (lookback window).
        output_length (int): The number of future time steps to predict (forecast horizon).
        debug (bool): If True, prints detailed status messages during execution.

    Returns:
        tuple: A tuple containing the prepared PyTorch tensors and the training DataFrame to see how the data looks like:
               (X_train, y_train, X_val, y_val, X_test, y_test, train_df)
    """
    warnings.filterwarnings('ignore', category=UserWarning, module='pandas')

    # --- Step 1: Load Data ---
    
    if debug:
        print(f"--- Starting Preprocessing for Target: '{predict_target}' ---")
    try:
        df = pd.read_csv(file_name, index_col='utc_timestamp', parse_dates=True)
        if debug:
            print(f"--- Step 1: Data Loaded Successfully (Initial Shape: {df.shape}) ---\n")
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        return

    de_cols = [
        # --- Target Components (to be summed up into a single target variable) ---
        'DE_solar_generation_actual',
        'DE_wind_onshore_generation_actual',
        'DE_wind_offshore_generation_actual',
        # --- Core Predictors: Load (a key driver) ---
        'DE_load_actual_entsoe_transparency',      # Actual total electricity demand
        'DE_load_forecast_entsoe_transparency',    # Professional day-ahead forecast for demand
        # --- Core Predictors: Capacity & Profile (for normalization and efficiency) ---
        'DE_solar_capacity',
        'DE_wind_onshore_capacity',
        'DE_wind_offshore_capacity',
        'DE_solar_profile',
        'DE_wind_onshore_profile',
        'DE_wind_offshore_profile',
        # --- Advanced Predictors: TSO-level data for more granular signals ---
        'DE_50hertz_solar_generation_actual',
        'DE_50hertz_wind_onshore_generation_actual',
        'DE_amprion_solar_generation_actual',
        'DE_amprion_wind_onshore_generation_actual',
        'DE_tennet_solar_generation_actual',
        'DE_tennet_wind_onshore_generation_actual',
        'DE_tennet_wind_offshore_generation_actual',
        'DE_transnetbw_solar_generation_actual',
        # --- Weather Features (to capture environmental impacts) ---
        'temperature_celsius',
        'solar_radiation',
        'total_cloud_cover',
        'wind_speed_100m'
    ]

    # Create a new dataframe focused on German data
    df_de = df[de_cols].copy()

    # --- Step 2: Filter, Clean, and Interpolate ---

    # Handle missing values using linear interpolation
    if debug:
        print("--- Step 2: Cleaning and Interpolating ---")
        print(f"Missing values before interpolation:\n{df_de.isnull().sum()}")

    df_de.interpolate(method='linear', inplace=True)

    if debug:
        print(f"\nMissing values after interpolation:\n{df_de.isnull().sum()}")
        print(f"Data shape after filtering: {df_de.shape}")
        print("\n")


    # --- Step 3: Feature Engineering ---
    if debug:
        print("--- Step 3: Engineering Time-Based Features ---")

    # Create time-based features from the datetime index
    df_de['hour'] = df_de.index.hour
    df_de['day_of_week'] = df_de.index.dayofweek # Monday=0, Sunday=6
    df_de['month'] = df_de.index.month
    df_de['day_of_year'] = df_de.index.dayofyear

    # Create cyclical features for 'hour' and 'month' to help the model understand their cyclical nature
    df_de['hour_sin'] = np.sin(2 * np.pi * df_de['hour']/24.0)
    df_de['hour_cos'] = np.cos(2 * np.pi * df_de['hour']/24.0)
    df_de['month_sin'] = np.sin(2 * np.pi * (df_de['month']-1)/12.0)
    df_de['month_cos'] = np.cos(2 * np.pi * (df_de['month']-1)/12.0)

    # Drop the original time features that are now encoded
    df_de.drop(['hour', 'month'], axis=1, inplace=True)

    if debug:
        print("New features created. Here's a look at the data:")
        print(df_de.head())
        print("\n")
        print("--- Creating Final Target Variable and Dropping Edge NaNs ---")

    # Create the combined renewable energy target variable
    df_de['DE_VRE_generation_actual'] = df_de['DE_solar_generation_actual'] + \
                                    df_de['DE_wind_onshore_generation_actual'] + \
                                    df_de['DE_wind_offshore_generation_actual']

    # Drop any rows that still contain NaN values (at the start/end of the dataset)
    df_de.dropna(inplace=True)

    if debug:
        print(f"Final data shape after dropping remaining NaNs: {df_de.shape}")
        print(f"\nMissing values after dropping remaining NaNs:\n{df_de.isnull().sum()}")
        print("Data is now 100% clean.")
        print("\n")

    # --- Step 4: Engineering Advanced Features ---
    if debug:
        print("--- Step 4: Engineering Advanced Features ---")
                                    
    # If we use the 15min data, we need to adjust the offsets accordingly.(*4)
    # For 1-hour data, we use 24 for daily and 168 for weekly
    DAY_OFFSET = 24   
    WEEK_OFFSET = DAY_OFFSET * 7

    # Create Lag Features
    df_de['target_lag_1_day'] = df_de['DE_VRE_generation_actual'].shift(DAY_OFFSET)
    df_de['target_lag_1_week'] = df_de['DE_VRE_generation_actual'].shift(WEEK_OFFSET)

    # Create Rolling Window Features
    df_de['target_rolling_mean_24h'] = df_de['DE_VRE_generation_actual'].rolling(window=DAY_OFFSET).mean()
    df_de['target_rolling_std_24h'] = df_de['DE_VRE_generation_actual'].rolling(window=DAY_OFFSET).std()

    # To Prevent data leakage, we choose the columns to drop based on the target variable
    if predict_target == 'combined':
        columns_to_drop =[
                            'DE_solar_generation_actual',
                            'DE_wind_onshore_generation_actual',
                            'DE_wind_offshore_generation_actual',
                            'DE_50hertz_solar_generation_actual',
                            'DE_50hertz_wind_onshore_generation_actual',
                            'DE_amprion_solar_generation_actual',
                            'DE_amprion_wind_onshore_generation_actual',
                            'DE_tennet_solar_generation_actual',
                            'DE_tennet_wind_onshore_generation_actual',
                            'DE_tennet_wind_offshore_generation_actual',
                            'DE_transnetbw_solar_generation_actual',
                            'DE_solar_profile',
                            'DE_wind_onshore_profile',
                            'DE_wind_offshore_profile',
                            ]
        # Reorder columns to have the target first for easier indexing
        feature_order = ['DE_VRE_generation_actual'] + [col for col in df_de.columns if col != 'DE_VRE_generation_actual']
        df_de = df_de[feature_order] 
        
    elif predict_target == 'solar':
        columns_to_drop =[
                    'DE_wind_onshore_generation_actual',
                    'DE_wind_offshore_generation_actual',
                    'DE_50hertz_solar_generation_actual',
                    'DE_50hertz_wind_onshore_generation_actual',
                    'DE_amprion_solar_generation_actual',
                    'DE_amprion_wind_onshore_generation_actual',
                    'DE_tennet_solar_generation_actual',
                    'DE_tennet_wind_onshore_generation_actual',
                    'DE_tennet_wind_offshore_generation_actual',
                    'DE_transnetbw_solar_generation_actual',
                    'DE_solar_profile',
                    'wind_speed_100m', # we remove this for solar because of irrelevance to solar generation
                    'DE_VRE_generation_actual' 
                    ]
        # Reorder columns to have the target first for easier indexing
        feature_order = ['DE_solar_generation_actual'] + [col for col in df_de.columns if col != 'DE_solar_generation_actual']
        df_de = df_de[feature_order] 
        
    elif predict_target == 'wind':
        columns_to_drop =[
                    'DE_solar_generation_actual',
                    'DE_wind_onshore_generation_actual',
                    'DE_wind_offshore_generation_actual',
                    'DE_50hertz_solar_generation_actual',
                    'DE_50hertz_wind_onshore_generation_actual',
                    'DE_amprion_solar_generation_actual',
                    'DE_amprion_wind_onshore_generation_actual',
                    'DE_tennet_solar_generation_actual',
                    'DE_tennet_wind_onshore_generation_actual',
                    'DE_tennet_wind_offshore_generation_actual',
                    'DE_transnetbw_solar_generation_actual',
                    'DE_solar_profile',
                    'DE_VRE_generation_actual',
                    'DE_wind_onshore_profile',
                    'DE_wind_offshore_profile', 
                    ]
         # Combine onshore and offshore wind generation into a single feature
        df_de['DE_wind_combined_generation_actual'] = df_de['DE_wind_onshore_generation_actual'] + \
                                                    df_de['DE_wind_offshore_generation_actual']
        
        feature_order = ['DE_wind_combined_generation_actual'] + [col for col in df_de.columns if col != 'DE_wind_combined_generation_actual']
        df_de = df_de[feature_order] 
        
        
    elif predict_target == 'wind_offshore':
        columns_to_drop =[
                    'DE_solar_generation_actual',
                    'DE_wind_onshore_generation_actual',
                    'DE_50hertz_solar_generation_actual',
                    'DE_50hertz_wind_onshore_generation_actual',
                    'DE_amprion_solar_generation_actual',
                    'DE_amprion_wind_onshore_generation_actual',
                    'DE_tennet_solar_generation_actual',
                    'DE_tennet_wind_onshore_generation_actual',
                    'DE_tennet_wind_offshore_generation_actual',
                    'DE_transnetbw_solar_generation_actual',
                    'DE_solar_profile',
                    'DE_VRE_generation_actual',
                    'DE_wind_onshore_profile',
                    'DE_wind_offshore_profile', 
                    ]
        
        
        feature_order = ['DE_wind_offshore_generation_actual'] + [col for col in df_de.columns if col != 'DE_wind_offshore_generation_actual']
        df_de = df_de[feature_order] 
        
        
    elif predict_target == 'wind_onshore':
        columns_to_drop =[
                    'DE_solar_generation_actual',
                    'DE_50hertz_solar_generation_actual',
                    'DE_50hertz_wind_onshore_generation_actual',
                    'DE_amprion_solar_generation_actual',
                    'DE_amprion_wind_onshore_generation_actual',
                    'DE_tennet_solar_generation_actual',
                    'DE_tennet_wind_onshore_generation_actual',
                    'DE_tennet_wind_offshore_generation_actual',
                    'DE_transnetbw_solar_generation_actual',
                    'DE_solar_profile',
                    'DE_VRE_generation_actual',
                    'DE_wind_onshore_profile',
                    'DE_wind_offshore_profile', 
                    ]
        
        
        feature_order = ['DE_wind_onshore_generation_actual'] + [col for col in df_de.columns if col != 'DE_wind_onshore_generation_actual']
        df_de = df_de[feature_order] 
        
    df_de.drop(columns=columns_to_drop, axis="columns", inplace=True)

    # IMPORTANT: Clean up NaNs created by shift/rolling operations
    # The first week of data will now be NaN, so we must drop it.
    df_de.dropna(inplace=True)

    if debug:
        print("New features created. Here's the updated data head:")
        print(df_de.head())
        print(f"\nNew data shape after adding features and dropping NaNs: {df_de.shape}")


    # --- Step 5: Data Splitting ---
    if debug:
        print("--- Step 5: Splitting Data Chronologically ---")

    # Training set is everything before 2018.
    train_df = df_de.loc[df_de.index < '2018-01-01'].copy()
    # Validation set is the full year of 2018.
    val_df = df_de.loc[(df_de.index >= '2018-01-01') & (df_de.index < '2019-01-01')].copy()
    # Test set is the full year of 2019 for robust evaluation.
    test_df = df_de.loc[(df_de.index >= '2019-01-01') & (df_de.index < '2020-01-01')].copy()

    if debug:
        print(f"Training set shape:   {train_df.shape}")
        print(f"Validation set shape: {val_df.shape}")
        print(f"Test set shape:       {test_df.shape}")


    # --- Step 6: Scaling the Data ---
    if debug:
        print("--- Step 6: Scaling Features ---")

    # Define the features to be scaled (all columns in our case)
    features_to_scale = train_df.columns

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # CRITICAL: Fit the scaler ONLY on the training data
    scaler.fit(train_df[features_to_scale])

    # Transform the training, validation, and test sets using the fitted scaler
    train_df[features_to_scale] = scaler.transform(train_df[features_to_scale])
    val_df[features_to_scale] = scaler.transform(val_df[features_to_scale])
    test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])

    if debug:
        print("Data scaling complete. Note: Scaler was fit ONLY on training data to prevent data leakage.")
        print("\n")
    # --- Final Confirmation ---
        print("--- Preprocessing Complete ---")
        print("The data is now fully preprocessed and split into train_df, val_df, and test_df.")
        print("You are ready to use these dataframes to create sequences for your Transformer model.")
    
    
    # --- Define Parameters ---
    INPUT_LENGTH = input_length   # in hours 
    OUTPUT_LENGTH = output_length  # in hours 

    # The target column 'DE_VRE_generation_actual' is the first column (index 0)
    TARGET_COLUMN_INDICES = [0]

    # --- Apply the function to your datasets ---
    if debug:
        print("Creating sequences for training, validation, and test sets...")

    X_train, y_train = create_sequences(train_df, INPUT_LENGTH, OUTPUT_LENGTH, TARGET_COLUMN_INDICES)
    X_val, y_val = create_sequences(val_df, INPUT_LENGTH, OUTPUT_LENGTH, TARGET_COLUMN_INDICES)
    X_test, y_test = create_sequences(test_df, INPUT_LENGTH, OUTPUT_LENGTH, TARGET_COLUMN_INDICES)
    
    if debug:
        print("Sequence creation complete.")
        print("\n--- Data Shapes ---")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print("-" * 20)
        print(f"X_val shape: {X_val.shape}")
        print(f"y_val shape: {y_val.shape}")
        print("-" * 20)
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")

    # --- Convert NumPy arrays to PyTorch Tensors ---
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    # Squeeze the target's last dimension for the loss function
    # The reason we do this is to make the shape of our true labels
    # (y_train) perfectly match the shape of our model's predictions.
    y_train, y_val, y_test = y_train.squeeze(), y_val.squeeze(), y_test.squeeze()
    
    if debug:
        print(f"X_train tensor shape: {X_train.shape}")
        print(f"y_train tensor shape: {y_train.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test , train_df

if __name__ == "__main__":
    # =================================================================================================================
    #                  UNCOMMENT THIS SECTION TO DOWNLOAD WEATHER DATA AND TO PROCESS THE WEATHER DATA
    # =================================================================================================================
    # --- Initialize the CDS API client to download the weather data ---
    # c = cdsapi.Client()
    # # --- Parameters (no changes here) ---
    # years_to_download = [str(year) for year in range(2015, 2020)]
    # months_to_download = [f'{m:02d}' for m in range(1, 13)]
    # data_folder = 'data/germany_weather_2015_2020/'
    # germany_area = [55.5, 5.5, 47.0, 15.5] 
    # variables_to_download = [
    #     '2m_temperature',
    #     'total_cloud_cover',
    #     'surface_solar_radiation_downwards',
    #     '100m_u_component_of_wind',
    #     '100m_v_component_of_wind',
    # ]

    # os.makedirs(data_folder, exist_ok=True)

    # # Loop through and download
    # for year in years_to_download:
    #     for month in months_to_download:
    #         output_filename = os.path.join(data_folder, f"germany_weather_{year}-{month}.nc")
            
    #         if os.path.exists(output_filename):
    #             print(f"File '{output_filename}' already exists. Skipping.")
    #             continue

    #         try:
    #             download_month(c, year, month, variables_to_download, germany_area, output_filename)
    #             print(f"✅ Successfully downloaded {output_filename}")
    #         except Exception as e:
    #             print(f"❌ Failed to download for {year}-{month}. Error: {e}")

    # print("\nAll download tasks are complete.")
    
    # print("\n--- Processing Weather Data ---")
    # process_weather_data()
    # =================================================================================================================
    #                  
    # =================================================================================================================
    
    # Example usage of the preprocess_data function
    X_train, y_train, X_val, y_val, X_test, y_test, train_df = preprocess_data(predict_target='combined')
    print("Preprocessing completed successfully.")
    print(train_df.head())  # Display the first few rows of the training DataFrame

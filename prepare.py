from bikes_data import TARGET,make_rentals
from sklearn.model_selection import train_test_split

SEED = 42


#The split
def split_frame(df, seed = SEED):
    # splitting df into 60% train,40% rest
    train,rest = train_test_split(df,test_size = 0.4,random_state= seed)  

    # splitting remaining df equally into validation and test,20% each
    val,test = train_test_split(rest,test_size = 0.5,random_state= seed)  
    return train, val, test


#Scaler and encoder (fitted on train only)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["hour", "temperature_c", "rain_mm", "wind_kph"]
BINARY = ["is_weekend"]
CATEGORY = ["station_type"]

def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC), # transforming NUMERIC columns into  standard scale 
        ("bin", "passthrough", BINARY),     # keeping BINARY column as it is

        # converting CATEGORY column into three numeric columns using oneHotEncoding
        ("cat", OneHotEncoder(sparse_output=False), CATEGORY),  
    ])
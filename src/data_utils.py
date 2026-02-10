def load_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    # Assuming 'dx' is the target column and the rest are features
    X = data.drop(columns=['dx'], errors='ignore')
    y = data['dx']
    return X, y
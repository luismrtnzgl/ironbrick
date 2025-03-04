import pandas as pd
import numpy as np

def preprocess_data(df):
    """Función de preprocesamiento aplicada antes de entrenar el modelo"""
    df = df[df['USRetailPrice'] > 0].copy()
    
    exclusivity_mapping = {'Regular': 0, 'Exclusive': 1}
    df.loc[:, 'Exclusivity'] = df['Exclusivity'].map(exclusivity_mapping)
    
    size_category_mapping = {'Small': 0, 'Medium': 1, 'Large': 2}
    df.loc[:, 'SizeCategory'] = df['SizeCategory'].map(size_category_mapping)
    
    df.loc[:, "PricePerPiece"] = df["USRetailPrice"] / df["Pieces"]
    df.loc[:, "PricePerMinifig"] = np.where(df["Minifigs"] > 0, df["USRetailPrice"] / df["Minifigs"], 0)
    df.loc[:, "YearsOnMarket"] = df["ExitYear"] - df["LaunchYear"]
    df.loc[:, "InteractionFeature"] = df["PricePerPiece"] * df["YearsOnMarket"]
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.infer_objects(copy=False)  
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df.loc[:, numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    return df

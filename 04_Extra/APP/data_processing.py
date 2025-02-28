import pandas as pd

def load_and_clean_data(file_path):
    """
    Cargamos y limpiamos los datos del archivo CSV extraído de la API.
    """
    df = pd.read_csv(file_path)
    
    # Ordenar por número de set y fecha de precio
    df_sorted = df.sort_values(by=['Number', 'PriceDate'], ascending=[True, True])
    
    # Asignar un índice de precios correcto
    df_sorted['PriceIndex'] = df_sorted.groupby('Number').cumcount()
    
    # Pivotear los datos para consolidar precios
    df_transformed = df_sorted.pivot(index=['Number', 'SetName', 'Theme', 'Year', 'Pieces', 'RetailPriceUSD', 'CurrentValueNew', 'ForecastValueNew2Y', 'ForecastValueNew5Y'],
                                     columns='PriceIndex', values='PriceValue').reset_index()
    
    # Renombrar columnas de precios
    df_transformed.columns = [f'Price_{col+1}' if isinstance(col, int) else col for col in df_transformed.columns]
    
    # Identificar columnas de precios
    price_columns = [col for col in df_transformed.columns if col.startswith("Price_")]
    
    # Reordenar los precios para eliminar valores nulos en la secuencia
    def reorder_prices(row):
        prices = row[price_columns].dropna().values
        new_row = [None] * len(price_columns)
        new_row[:len(prices)] = prices
        return pd.Series(new_row, index=price_columns)
    
    df_transformed[price_columns] = df_transformed.apply(reorder_prices, axis=1)
    
    # Eliminar columnas irrelevantes
    columns_to_drop = ['Currency', 'PriceType']
    df_clean = df_transformed.drop(columns=columns_to_drop, errors='ignore')
    
    return df_clean

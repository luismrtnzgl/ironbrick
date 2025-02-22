import pandas as pd
import numpy as np
from datetime import datetime

def limpiar_df(df):
    """
    Realiza la limpieza del dataset de sets de LEGO:
    - Rellena valores nulos en columnas categóricas.
    - Sustituye valores nulos por 0 en columnas numéricas clave.
    - Convierte las fechas de lanzamiento y retiro a solo el año.
    - Asegura la consistencia de los tipos de datos.
    
    Parámetros:
        df (pd.DataFrame): DataFrame de LEGO sin limpiar.
        
    Retorna:
        pd.DataFrame: DataFrame limpio.
    """
    # Rellenamos valores nulos en columnas categóricas
    df['Subtheme'] = df['Subtheme'].fillna('Unknown')
    df['ImageFilename'] = df['ImageFilename'].fillna('Unknown')

    # En las columnas numéricas a reemplazamos valores nulos por 0
    columnas_a_cero = [
        'Pieces', 'BrickLinkSoldPriceNew', 'USRetailPrice', 
        'BrickLinkSoldPriceUsed', 'Depth', 'Height', 'Width', 'Weight', 
        'Minifigs', 'AgeMin', 'AgeMax'
    ]

    for col in columnas_a_cero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Convertimos LaunchDate y ExitDate a solo el año en formato numérico
    if 'LaunchDate' in df.columns:
        df['LaunchDate'] = pd.to_datetime(df['LaunchDate'], errors='coerce').dt.year
        df['LaunchDate'] = df['LaunchDate'].fillna(0).astype(int).replace(0, None)
    
    if 'ExitDate' in df.columns:
        df['ExitDate'] = pd.to_datetime(df['ExitDate'], errors='coerce').dt.year
        df['ExitDate'] = df['ExitDate'].fillna(0).astype(int).replace(0, None)

    return df

def normalizar_df(df):
    """
    Normaliza el dataset de sets de LEGO:
    - Convierte 'LaunchDate' y 'ExitDate' a enteros manteniendo valores nulos.
    - Elimina la columna 'Released' por no aportar información útil.
    - Normaliza 'PackagingType' agrupando valores similares.
    - Normaliza 'Availability' simplificando categorías.
    
    Parámetros:
        df (pd.DataFrame): DataFrame de LEGO sin normalizar.
        
    Retorna:
        pd.DataFrame: DataFrame normalizado.
    """
    # Convertir LaunchDate y ExitDate a enteros manteniendo los valores nulos
    df['LaunchDate'] = pd.to_numeric(df['LaunchDate'], errors='coerce').astype('Int64')
    df['ExitDate'] = pd.to_numeric(df['ExitDate'], errors='coerce').astype('Int64')
    
    # Eliminar la columna 'Released' si existe
    if 'Released' in df.columns:
        df.drop(columns=['Released'], inplace=True)
    
    # Normalizar 'PackagingType'
    df['PackagingType'] = df['PackagingType'].replace({
        '{Not specified}': 'Unknown',
        'Plastic canister': 'Canister',
        'Plastic box': 'Box',
        'Metal canister': 'Canister',
        'Box with handle': 'Box',
        'Box with backing card': 'Box',
        'None (loose parts)': 'None'
    })
    
    # Normalizar 'Availability'
    df['Availability'] = df['Availability'].replace({
        '{Not specified}': 'Unknown',
        'Promotional (Airline)': 'Promotional'
    })
    
    return df


def process_lego_data(df):
    """
    Script para procesar datos de sets de LEGO y calcular métricas de inversión.
    
    Parámetros:
        df (pd.DataFrame): DataFrame de LEGO procesadoy sin métricas.
        
    Retorna:
        pd.DataFrame: DataFrame completo.
    """    
    # Convertir LaunchDate y ExitDate a enteros manteniendo los valores nulos
    df['LaunchDate'] = pd.to_numeric(df['LaunchDate'], errors='coerce').astype('Int64')
    df['ExitDate'] = pd.to_numeric(df['ExitDate'], errors='coerce').astype('Int64')
    
    # Calcular la duración de los sets en venta
    df['Lifespan'] = df['ExitDate'] - df['YearFrom']
    
    # Calcular la mediana de Lifespan por Theme, excluyendo valores nulos
    median_lifespan_by_theme = df.groupby('Theme')['Lifespan'].median()
    
    # Rellenar valores nulos en ExitDate con YearFrom + mediana del Theme correspondiente
    df['ExitDate'] = df.apply(
        lambda row: row['YearFrom'] + median_lifespan_by_theme[row['Theme']]
        if pd.isnull(row['ExitDate']) and row['Theme'] in median_lifespan_by_theme
        else row['ExitDate'],
        axis=1
    )
    
    # Redondear valores con decimales en ExitDate y convertir a enteros
    df['ExitDate'] = df['ExitDate'].apply(lambda x: round(x) if pd.notnull(x) else x)
    df['ExitDate'] = pd.to_numeric(df['ExitDate'], errors='coerce').astype('Int64')
    
    # Obtener el año actual y calcular los años desde la retirada del set
    current_year = datetime.now().year
    df['YearsSinceExit'] = current_year - df['ExitDate']
    df['YearsSinceExit'] = df['YearsSinceExit'].apply(lambda x: max(x, 0))
    df['YearsSinceExit'] = df['YearsSinceExit'].fillna(0).astype(int)
    
    # Calcular el porcentaje de cambio de precio entre el precio de venta en BrickLink y en la web de LEGO
    df['PriceChange'] = ((df['BrickLinkSoldPriceNew'] - df['USRetailPrice']) / df['USRetailPrice']) * 100
    df['PriceChange'] = df['PriceChange'].fillna(0)
    
    # Calcular la demanda de reventa
    df['ResaleDemand'] = df.apply(lambda row: row['BrickLinkSoldPriceNew'] / row['BrickLinkSoldPriceUsed'] 
                                             if row['BrickLinkSoldPriceUsed'] > 0 else 0, axis=1)
    
    # Calcular la tendencia de apreciación
    df['AppreciationTrend'] = df.apply(lambda row: row['PriceChange'] / row['YearsSinceExit']
                                                 if row['YearsSinceExit'] > 0 else 0, axis=1)
    
    # Clasificar los sets por tamaño
    size_labels = ['Small', 'Medium', 'Large']
    df['SizeCategory'] = pd.cut(df['Pieces'], bins=[0, 249, 1000, float('inf')], labels=size_labels, include_lowest=True)
    
    # Definir sets exclusivos
    exclusive_themes = ['Star Wars', 'Modular Buildings', 'Ideas', 'Creator Expert', 'Harry Potter', 
                        'Marvel Super Heroes', 'Ghostbusters', 'Icons', 'The Lord of the Rings',
                        'Pirates of the Caribbean', 'Pirates', 'Trains', 'Architecture']
    df['Exclusivity'] = df['Theme'].apply(lambda x: 'Exclusive' if x in exclusive_themes else 'Regular')
    
    # Calcular popularidad del tema
    theme_popularity = df.groupby('Theme')['PriceChange'].mean().replace([np.inf, -np.inf], np.nan)
    df['ThemePopularity'] = df['Theme'].map(theme_popularity).fillna(0)
    
    # Calcular InvestmentScore
    df['InvestmentScore'] = df.apply(lambda row: (row['PriceChange'] * 0.4) +
                                                         (row['AppreciationTrend'] * 0.3) +
                                                         (row['ThemePopularity'] * 0.2) +
                                                         (10 if row['Exclusivity'] == 'Exclusive' else 0), axis=1)
    
    return df

import pandas as pd
import numpy as np
from datetime import datetime

def clean_lego_data(df_lego):
    # Reemplazo los valores nulos en 'Subtheme' por 'Unknown'
    df_lego['Subtheme'] = df_lego['Subtheme'].fillna('Unknown')
    
    # Reemplazo nulos por 0 en las columnas numéricas seleccionadas
    columns_zero = [
        'Pieces', 'BrickLinkSoldPriceNew', 'BrickLinkSoldPriceNewUS', 'USRetailPrice', 
        'BrickLinkSoldPriceUsed', 'Depth', 'Height', 'Width', 'Weight', 'Minifigs', 'AgeMin', 'AgeMax'
    ]
    
    for col in columns_zero:
        if col in df_lego.columns:
            df_lego[col] = df_lego[col].fillna(0)
    
    # Reemplazo los valores nulos en 'ImageFilename' por 'Unknown'
    df_lego['ImageFilename'] = df_lego['ImageFilename'].fillna('Unknown')
    
    # Convierto a formato de fecha para manejar valores nulos
    df_lego['LaunchDate'] = pd.to_datetime(df_lego['LaunchDate'], errors='coerce')
    df_lego['ExitDate'] = pd.to_datetime(df_lego['ExitDate'], errors='coerce')
    
    # Calcula la duración en años de los sets con datos disponibles
    df_lego['Duration'] = (df_lego['ExitDate'] - df_lego['LaunchDate']).dt.days / 365.25
    
    # Calcular la mediana de duración por Theme
    theme_median_duration = df_lego.groupby('Theme')['Duration'].median()
    
    # Relleno ExitDate usando la mediana de duración por Theme
    for theme, median_duration in theme_median_duration.items():
        mask = (df_lego['Theme'] == theme) & df_lego['ExitDate'].isna() & df_lego['LaunchDate'].notna()
        df_lego.loc[mask, 'ExitDate'] = df_lego.loc[mask, 'LaunchDate'] + pd.to_timedelta(median_duration * 365.25, unit='D')
    
    # Relleno LaunchDate usando YearFrom para los valores NaN
    mask_launch = df_lego['LaunchDate'].isna() & df_lego['YearFrom'].notna()
    df_lego.loc[mask_launch, 'LaunchDate'] = pd.to_datetime(df_lego.loc[mask_launch, 'YearFrom'].astype(int).astype(str) + '-01-01')
    
    # Extraigo año y mes en nuevas columnas
    df_lego['LaunchYear'] = df_lego['LaunchDate'].dt.year
    df_lego['LaunchMonth'] = df_lego['LaunchDate'].dt.month
    df_lego['ExitYear'] = df_lego['ExitDate'].dt.year
    df_lego['ExitMonth'] = df_lego['ExitDate'].dt.month
    
    # Elimino las columnas originales y la auxiliar
    df_lego.drop(columns=['LaunchDate', 'ExitDate', 'Duration'], inplace=True)
    
    # Calculamos la duración en años de los sets con datos disponibles
    df_lego['Duration'] = df_lego['ExitYear'] - df_lego['LaunchYear']
    
    # Calculo la duración media por tema, ignorando NaN
    theme_avg_duration = df_lego.groupby('Theme')['Duration'].mean()
    
    # Calculo la duración media por año de lanzamiento, ignorando NaN
    year_avg_duration = df_lego.groupby('LaunchYear')['Duration'].mean()
    
    # Relleno los valores nulos de ExitYear y ExitMonth usando valores calculados
    for index, row in df_lego.iterrows():
        if pd.isna(row['ExitYear']) and not pd.isna(row['LaunchYear']):
            theme_duration = theme_avg_duration.get(row['Theme'], None)
            year_duration = year_avg_duration.get(row['LaunchYear'], None)
    
            # Usar la duración del tema si está disponible, si no, la del año de lanzamiento
            estimated_duration = theme_duration if pd.notna(theme_duration) else year_duration
    
            if pd.notna(estimated_duration):  # Solo asignar si hay un valor válido
                df_lego.at[index, 'ExitYear'] = int(row['LaunchYear'] + round(estimated_duration))
                df_lego.at[index, 'ExitMonth'] = 12  # Usar diciembre como mes estimado de retiro
    
    # Elimino de nuevo la columna auxiliar de duración
    df_lego.drop(columns=['Duration'], inplace=True)
    
    # Normalizo 'PackagingType'
    df_lego['PackagingType'] = df_lego['PackagingType'].replace({
        '{Not specified}': 'Unknown',
        'Plastic canister': 'Canister',
        'Plastic box': 'Box',
        'Metal canister': 'Canister',
        'Box with handle': 'Box',
        'Box with backing card': 'Box',
        'None (loose parts)': 'None'
    })
    
    # Normalizo 'Availability'
    df_lego['Availability'] = df_lego['Availability'].replace({
        '{Not specified}': 'Unknown',
        'Promotional (Airline)': 'Promotional'
    })
    # Reasigno los sets de "Creator Expert" al tema "Icons". Es una serie que ha cambiado de nombre últimamente y puede hacer que los datos no sean consistentes.
    df_lego.loc[df_lego['Theme'] == 'Creator Expert', 'Theme'] = 'Icons'

    return df_lego





def process_lego_data(df_lego):
    """
    Script para procesar datos de sets de LEGO y calcular métricas de inversión.
    
    Parámetros:
        df (pd.DataFrame): DataFrame de LEGO procesadoy sin métricas.
        
    Retorna:
        pd.DataFrame: DataFrame completo.
    """    
    
    # Obtener el año actual y calcular los años desde la retirada del set
    current_year = datetime.now().year
    df_lego['YearsSinceExit'] = current_year - df_lego['ExitYear']
    df_lego['YearsSinceExit'] = df_lego['YearsSinceExit'].fillna(0).astype(int)
    df_lego['YearsSinceExit'] = df_lego['YearsSinceExit'].fillna(0).astype(int)

    
    # Calcular la mediana de Lifespan por Theme, excluyendo valores nulos
    median_lifespan_by_theme = df_lego.groupby('Theme')['Lifespan'].median().replace(0, np.nan)
    
    # Calcular el porcentaje de cambio de precio entre el precio de venta en BrickLink y en la web de LEGO
    df_lego['PriceChange'] = ((df_lego['BrickLinkSoldPriceNew'] - df_lego['USRetailPrice']) / df_lego['USRetailPrice']) * 100
    df_lego['PriceChange'] = df_lego['PriceChange'].fillna(0)
    
    # Calcular la demanda de reventa
    df_lego['ResaleDemand'] = df_lego.apply(lambda row: row['BrickLinkSoldPriceNew'] / row['BrickLinkSoldPriceUsed'] 
                                             if row['BrickLinkSoldPriceUsed'] > 0 else 0, axis=1)
    
    # Calcular la tendencia de apreciación
    df_lego['AppreciationTrend'] = df_lego.apply(lambda row: row['PriceChange'] / row['YearsSinceExit']
                                                 if row['YearsSinceExit'] > 0 else 0, axis=1)
    
    # Clasificar los sets por tamaño
    size_labels = ['Small', 'Medium', 'Large']
    df_lego['SizeCategory'] = pd.cut(df['Pieces'], bins=[0, 249, 1000, float('inf')], labels=size_labels, include_lowest=True)
    
    # Definir sets exclusivos
    exclusive_themes = ['Star Wars', 'Modular Buildings', 'Ideas', 'Creator Expert', 'Harry Potter', 
                        'Marvel Super Heroes', 'Ghostbusters', 'Icons', 'The Lord of the Rings',
                        'Pirates of the Caribbean', 'Pirates', 'Trains', 'Architecture']
    df_lego['Exclusivity'] = df_lego['Theme'].apply(lambda x: 'Exclusive' if x in exclusive_themes else 'Regular')
    
    # Calcular popularidad del tema
    theme_popularity = df_lego.groupby('Theme')['PriceChange'].mean().replace([np.inf, -np.inf], np.nan)
    df_lego['ThemePopularity'] = df_lego['Theme'].map(theme_popularity).fillna(0)
    
    # Calcular InvestmentScore
    df_lego['InvestmentScore'] = df_lego.apply(lambda row: (row['PriceChange'] * 0.4) +
                                                         (row['AppreciationTrend'] * 0.3) +
                                                         (row['ThemePopularity'] * 0.2) +
                                                         (10 if row['Exclusivity'] == 'Exclusive' else 0), axis=1)
    
    return df_lego

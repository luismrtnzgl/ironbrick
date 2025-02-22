def limpiar_df_lego(df):
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

def normalizar_df_lego(df):
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


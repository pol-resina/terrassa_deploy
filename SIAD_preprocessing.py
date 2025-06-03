import pandas as pd
import numpy as np
import random
import re

import warnings
warnings.filterwarnings('ignore')


def siad_preprocess(df):
    """
    Reads dataframe from SIAD and preprocesses it.

    Parameters:
        data (pd.DataFrame): Dataframe containing the SIAD data.

    Returns:
        pandas.DataFrame: Preprocessed discrimination data.
    """

    cols_important = ['ID_UPC', 'COD_CITA', 'DISTRICTE_PH', 'SECCIO_PH', 'BARRI_PH',
                'NOM_BARRI_PH', 'NOM_AREA', 'COD_EXPEDIENT', 'DATA_CITA_INI',
                'NOM_TIPUSCITA', 'DATA_CITA_FI', 'DATA_OBERTURA', 'DESC_DENUNCIA', 'EDAT',
                'EXP_NOM_ORIGEN', 'EXP_NUMFILLS', 'GR_EDAT', 'NOM_EXP_ECIVIL',
                'NOM_EXP_NIVSOCIO', 'NOM_EXP_PROFESIO', 'NOM_EXP_TITULACIO',
                'NOM_FONT_INGRESSOS', 'NOM_MOTIU', 'NOM_PERMIS_RESI', "NOM_PER_TITULACIO",
                'NOM_PERMIS_TREBALL', 'NOM_PER_PAISNAIX', 'TIPUSFAMILIA',
                'GRUPNACIONALITAT', 'COD_MAL', 'DESC_MOTIU_TANCAMENT',
                'DESC_TIPUS_ALTA', 'DISCAPACITAT', 'ORDRE PROTECCIO',
                'DESC_MALTRACTAMENT', 'DESC_AGRESSOR']
    
    missing = []
    try:
        df = df[cols_important]
    except KeyError as e:
        missing = re.findall(r"'(.*?)'", str(e))
        for missing_column in missing:
            df[missing_column] = np.nan
        df = df[cols_important]

    #Create column DURADA_CITA
    df["DATA_CITA_INI"] = pd.to_datetime(df["DATA_CITA_INI"], format='mixed')
    df["DATA_CITA_FI"] = pd.to_datetime(df["DATA_CITA_FI"], format='mixed')
    df["DATA_OBERTURA"] = pd.to_datetime(df["DATA_OBERTURA"], format='mixed')

    df["DURADA_CITA"] = df.apply(lambda col: col["DATA_CITA_FI"]-col["DATA_CITA_INI"]
                             if col["DATA_CITA_FI"] > col["DATA_CITA_INI"]
                             else col["DATA_CITA_INI"]-col["DATA_CITA_FI"], axis=1)
    if df["DURADA_CITA"].notna().all():
      df["DURADA_CITA"] = df['DURADA_CITA'].dt.total_seconds()/60

    df = df.drop(columns = ["DATA_CITA_FI"])

    #Get only the rows >2015
    df = df[(df["DATA_OBERTURA"] >= '01-01-2015')|df["DATA_OBERTURA"].isna()]

    #Drop columns defined previously
    df["NOM_EXP_TITULACIO"] = df["NOM_EXP_TITULACIO"].fillna(df["NOM_PER_TITULACIO"])
    df = df.drop(columns=["NOM_PER_TITULACIO"])


    #Change NAs of categoricals to other vales or Sense dades
    df["DISCAPACITAT"] = df["DISCAPACITAT"].fillna('No')
    df["ORDRE PROTECCIO"] = df["ORDRE PROTECCIO"].fillna('No')
    df["DESC_MALTRACTAMENT"] = df["DESC_MALTRACTAMENT"].fillna('No aplica')
    df["DESC_AGRESSOR"] = df["DESC_AGRESSOR"].fillna('No aplica')
    
    for column in df.columns:
        if df[column].dtype == 'object':
            df[column] = df[column].fillna('Sense dades')

    #Fix wrong EDAT values
    ##Create function that gets the min and max from GR_EDAT
    def extract_min_max(range_str):
        if '-' in range_str:
            start, end = map(int, range_str.split('-'))
            return start, end
        elif '>' in range_str:
            limit = int(range_str.strip('>'))
            return limit, 123
        else:
            return np.nan, np.nan 
        
    df[['MIN_EDAT', 'MAX_EDAT']] = df['GR_EDAT'].apply(
    lambda x: pd.Series(extract_min_max(x)))

    ##Fix EDAT based on min and max extracted
    df["EDAT"] = df.apply(
    lambda row: row['EDAT']
    if pd.isna(row['MIN_EDAT']) or pd.isna(row['MAX_EDAT']) or row['MIN_EDAT'] <= row['EDAT'] <= row['MAX_EDAT']
    else random.randint(row['MIN_EDAT'], row['MAX_EDAT']),
    axis=1
    )
    
    ##Drop columns min and max
    df = df.drop(columns = ["MIN_EDAT", "MAX_EDAT"])
            
    #Replace non-informative values to Sense dades
    df["DESC_DENUNCIA"] = df["DESC_DENUNCIA"].replace('NS', 'Sense dades')
    df["NOM_EXP_ECIVIL"] = df["NOM_EXP_ECIVIL"].replace('Sense informacio', 'Sense dades')
    df["NOM_EXP_PROFESIO"] = df["NOM_EXP_PROFESIO"].replace('Sense informacio', 'Sense dades')
    df["NOM_PERMIS_RESI"] = df["NOM_PERMIS_RESI"].replace('NS', 'Sense dades')
    df["NOM_PERMIS_TREBALL"] = df["NOM_PERMIS_TREBALL"].replace('Sense informacio', 'Sense dades')
    df["DESC_TIPUS_ALTA"] = df["DESC_TIPUS_ALTA"].replace('Sense informacio', 'Sense dades')
    df["GR_EDAT"] = df["GR_EDAT"].replace('>80', '80+')

    ##Caps
    df["NOM_PER_PAISNAIX"] = df["NOM_PER_PAISNAIX"].str.capitalize()
        
    #Drop rows in which we have no information about DATA_OBERTURA
    df = df[(df["DATA_OBERTURA"] != '1900-01-01')]
    df = df.dropna(subset='DATA_OBERTURA')
    
    return (df, missing)

# %% [markdown]
# 

# %%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
scaler = MinMaxScaler()
from scipy.optimize import curve_fit

# %%
# Read the Excel file
Raw_GDP_df = pd.read_excel(r"Comparitive Analysis LR$GDP/dataset/API_NY.GDP.MKTP.CD_DS2_en_excel_v2_5447810.xls")
Raw_TotalUnemployment_df = pd.read_excel(r"Comparitive Analysis LR$GDP/dataset/API_SL.UEM.TOTL.ZS_DS2_en_excel_v2_5358380.xls")
Raw_metadata_df = pd.read_excel(r'Comparitive Analysis LR$GDP/dataset/API_SE.ADT.LITR.ZS_DS2_en_excel_v2_5358601.xls', sheet_name='Metadata - Countries')
# Delete the first three rows using excel



# %%
Raw_GDP_df.columns

# %%
# Analysis Unemployment in Lower Middle Income Countries with Respect to GDP

# %%
# Selecting the from 2015 - 2019
GDP_df = Raw_GDP_df[['Country Name', 'Country Code',"2015", "2016", "2017", "2018", "2019"]]
TotalUnemployment_df = Raw_TotalUnemployment_df[['Country Name', 'Country Code',"2015", "2016", "2017", "2018", "2019"]]
# Select only the 'country code' and 'income group' columns
metadata_df = Raw_metadata_df[['Country Code', 'IncomeGroup']]

# %%
TotalUnemployment_df.columns


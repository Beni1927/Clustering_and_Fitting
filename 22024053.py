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

# %%
def process_dataset(dataset, metadata_df):
    # Merge the dataset with the metadata on the 'Country Code' column
    merged_df = pd.merge(dataset, metadata_df, on='Country Code', how='left')

    # Filter the merged DataFrame for rows with income group 'Low income'
    low_income_df = merged_df[merged_df['IncomeGroup'] == 'Lower middle income']
    #low_income_df = low_income_df.dropna()
    low_income_df = low_income_df.fillna(value=True)
    low_income_df = low_income_df.drop(['Country Code', 'IncomeGroup'], axis=1)
    # Reset the index of the filtered DataFrame
    low_income_df = low_income_df.reset_index(drop=True)

    return low_income_df

# %%
GDP_df = process_dataset(GDP_df, metadata_df)
TotalUnemployment_df = process_dataset(TotalUnemployment_df, metadata_df)

# %%

def perform_clustering(dataset, num_clusters=3):
    # Normalize data
    scaler = MinMaxScaler()
    data_norm = scaler.fit_transform(dataset.iloc[:, 1:])

    # Determine optimal number of clusters using elbow method
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data_norm)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('Within-Cluster Sum of Squares')
    plt.show()

    # Fit KMeans model to data
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_norm)

    # Get cluster labels and add to original data
    dataset['cluster'] = kmeans.labels_

    # Evaluate clustering with silhouette score
    score = silhouette_score(data_norm, kmeans.labels_)
    print(f'Silhouette Score: {score}')

    return dataset


# %%
perform_clustering(GDP_df)

# %%

def scatterplot(dataset,data_norm,n_components):
    # Fit KMeans model to data
    kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(data_norm)

    # Get cluster labels
    labels = kmeans.labels_

    # Perform PCA
    pca = PCA(n_components)
    data_pca = pca.fit_transform(data_norm)

    # Store country codes for remaining rows in a separate variable
    country_codes = dataset['Country Name']
    country_codes_cleaned = country_codes[dataset.index]

    # Add country codes and cluster labels to data_plot dataframe
    data_plot = pd.DataFrame({'x': data_pca[:, 0], 'y': data_pca[:, 1], 'cluster': labels})

    # Visualize clusters
    sns.scatterplot(data=data_plot, x='x', y='y', hue='cluster', palette='Set1')
    plt.title('Clusters')

    # Add legend
    legend_labels = ['Group ' + str(i+1) for i in range(len(set(kmeans.labels_)))]
    plt.legend(title='Cluster', labels=legend_labels, loc='lower left')
    plt.show()


# %%

data_norm = scaler.fit_transform(GDP_df[['2015', '2016', '2017', '2018', '2019']])
n_components = 3
scatterplot(GDP_df,data_norm,n_components)

# %%
GDP2019_DF = GDP_df[["Country Name","2019"]]

# %%
UE2019_DF = TotalUnemployment_df[["Country Name","2019"]]

# %%
DF_2019 = pd.merge(GDP2019_DF, UE2019_DF, on="Country Name", how="outer")

# %%
DF_2019.columns

# %%

data_norm = scaler.fit_transform(DF_2019[['2019_x', '2019_y',]])
n_components= 2
scatterplot(DF_2019,data_norm,n_components)

# %%
DF_2019

# %%
perform_clustering(DF_2019,4)

# %%
# Select data for India
def timeseries_visualization(country):
    CountryDataVis = Raw_GDP_df[Raw_GDP_df['Country Name'] == country]

    # Remove unnecessary columns
    CountryDataVis = CountryDataVis.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], axis=1)

    # Convert the years to datetime format
    CountryDataVis = CountryDataVis.melt(id_vars=['1960'], var_name='Year', value_name='Value')
    CountryDataVis['Year'] = pd.to_datetime(CountryDataVis['Year'], format='%Y')

    # Set the 'Year' column as the index
    CountryDataVis = CountryDataVis.set_index('Year')
    # Select data for India
    UE_CountryDataVis = Raw_TotalUnemployment_df[Raw_TotalUnemployment_df['Country Name'] == 'India']

    # Remove unnecessary columns
    UE_CountryDataVis = UE_CountryDataVis.drop(['Country Name', 'Country Code', 'Indicator Name', 'Indicator Code'], axis=1)

    # Convert the years to datetime format
    UE_CountryDataVis = UE_CountryDataVis.melt(id_vars=['1960'], var_name='Year', value_name='Value')
    UE_CountryDataVis['Year'] = pd.to_datetime(UE_CountryDataVis['Year'], format='%Y')

    # Set the 'Year' column as the index
    UE_CountryDataVis = UE_CountryDataVis.set_index('Year')
    plt.plot(UE_CountryDataVis.index, UE_CountryDataVis['Value'])
    plt.title('Unemployement for '+str(country) )
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.show()
    # Plot the time series
    plt.plot(CountryDataVis.index, CountryDataVis['Value'])
    plt.title('GDP for '+ str(country))
    plt.xlabel('Year')
    plt.ylabel('Value')
    plt.show()

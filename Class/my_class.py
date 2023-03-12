"""
This module provides a class with 6 methods
"""

import os
import pandas as pd
import geopandas as gpd
import requests
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing

class Group22:
    """
    A class to examine a dataset on agriculture.

    ...

    Attributes
    ----------
    url : str
        url of dataset
    filename : str
        filename of dataset

    Methods
    -------
    download_data:
        downloads two datasets and turns the first one into a pandas dataframe, the second one is a geopandas dataframe
    get_countries:
        creates a list of all countries in the dataframe
    plot_quantity_correlation:
        plots a correlation matrix of quantity columns
    plot_output_area_chart:
        plots an area chart of output columns
    compare_countries_output:
        plots a line graph of output columns for given countries
    _gapminder_:
        plots agricultural production data for given year
    choropleth:
        
    """

    def __init__(self, url, filename):

        self.url = url
        self.filename = filename

    def download_data(self):
        """
        Returns agricultural dataframe from the dataset and geodataframe from Natural Earth

        Parameters
        ----------
        None

        Returns
        -------
        dataframes
        """
        if not os.path.exists("../downloads"):
            os.makedirs("../downloads")
            
        if not os.path.exists(os.path.join("../downloads", self.filename)):
            response = requests.get(self.url)
            if response.status_code == 200:
                with open(os.path.join("../downloads", self.filename), "wb") as my_f:
                    my_f.write(response.content)
                print(f"{self.filename} has been downloaded")
            else:
                print(f"Error downloading {self.filename}: {response.status_code}")
        else:
            print(f"{self.filename} already exists")

        data_path = os.path.join("../downloads", self.filename)
        my_df = pd.read_csv(data_path, on_bad_lines="skip")
        
        # Remove aggregated columns
        agg_cols = ['Asia', 'Sahel', 'Central Africa', 'Central America', 'Central Asia','Central Europe', 
                    'Developed Asia', 'Developed countries', 'East Africa', 'Eastern Europe', 'Europe', 
                    'High income', 'Horn of Africa', 'Latin America and the Caribbean', 'Least developed countries', 
                    'Low income', 'Lower-middle income', 'Micronesia', 'North Africa', 'North America', 
                    'Northeast Asia', 'Northern Europe', 'Oceania', 'Pacific', 'South Asia', 
                    'Southeast Asia', 'Southern Africa', 'Southern Europe', 'Sub-Saharan Africa', 
                    'Upper-middle income', 'West Africa', 'West Asia','Western Europe', 'World']
        my_df = my_df[~my_df.Entity.isin(agg_cols)]
        
        # Download and read geographical dataset
        geo_filename = "geo_data"
        if not os.path.exists(os.path.join("../downloads", geo_filename)):
            geo_path = os.path.join("../downloads", geo_filename)
            geo_df = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        else:
            print(f"{geo_filename} already exists")
            
        self.my_df = my_df
        self.geo_df = geo_df

        return my_df, geo_df

    def get_countries(self):
        """
        Returns a list of unique country names from the
        "Entity" column of the agricultural dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        list of unique country names
        """
        return list(self.my_df["Entity"].unique())

    def plot_quantity_correlation(self):
        """
        Plots a correlation matrix of quantity columns from the
        input dataframe.

        Parameters
        ----------
        None

        Returns
        -------
        None
            Displays a heatmap of the correlation matrix
            of quantity columns.
        """
        quantity_cols = [col for col in self.my_df.columns if "_quantity" in col]
        quantity_df = self.my_df[quantity_cols]
        corr_matrix = quantity_df.corr()
        
        # Create a boolean mask for the upper triangle of the matrix
        mask = np.zeros_like(corr_matrix)
        mask[np.triu_indices_from(mask)] = True

        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", mask=mask)
        plt.title("Correlation Matrix of Quantity Columns")
        plt.annotate('Source: Agricultural total factor productivity (USDA), OWID 2021', (0,0), (-100,-150), fontsize=10, 
                     xycoords='axes points', textcoords='offset points', va='top')
        plt.show()

    def plot_output_area_chart(self, country=None, normalize=False):
        """
        Plots an area chart of the distinct "_output_" columns.

        Parameters
        ----------
        country : str, optional
            The name of the country to plot the output for.
            If None or 'World', plots the sum for all countries.
            Default is None.
        normalize : bool, optional
            If True, normalizes the output in relative terms:
            each year, output should always be 100%.
            Default is False.

        Raises
        ------
        ValueError
            If no output columns are found in the dataset.
            If the given country name is not valid.
        
        Returns
        -------
        None
        """
        output_cols = [col for col in self.my_df.columns if "_output_" in col]
        if not output_cols:
            raise ValueError("No output columns found in the dataset.")

        if country is None or country == "World":
            df_country = (
                pd.DataFrame(self.my_df)
                .groupby("Year")[output_cols]
                .sum()
                .reset_index()
            )
            title = "World Output"
        else:
            if country not in self.get_countries():
                raise ValueError(f"{country} is not a valid country name.")
            df_country = self.my_df[self.my_df["Entity"] == country][
                ["Year"] + output_cols
            ]
            title = f"{country} Output"

        if normalize:
            df_country[output_cols] = (
                df_country[output_cols].div(df_country[output_cols].sum(axis=1), axis=0)
                * 100
            )

        df_country.set_index("Year").plot(kind="area", stacked=True)
        plt.title(title)
        plt.xlabel("Year")
        plt.ylabel("Output")
        plt.annotate('Source: Agricultural total factor productivity (USDA), OWID 2021', (0,0), (0,-35), fontsize=10, 
                     xycoords='axes points', textcoords='offset points', va='top')
        plt.show()

    def compare_output_countries(self, countries):
        """
        Plots a comparison of the total of the '_output_' column for each of the given countries.

        Parameters:
        ----------
        countries : str or list
            Country or list of countries to compare

        Raises
        ------
        ValueError
            If no output columns are found in the dataset.
            If the provided country/countries is not in the list of valid country names.

        Returns
        -------
        None
        """
        # Create total output column
        output_cols = [col for col in self.my_df.columns if "_output_" in col]
        if not output_cols:
            raise ValueError("No output columns found in the dataset.")
        else:
            self.my_df["total_output"] = pd.DataFrame(self.my_df)[output_cols].sum(axis=1)

        # Transform input to list
        if not isinstance(countries, list):
            countries = countries.split()

        # Plot each given countries output
        for i in countries:
            if i not in self.get_countries():
                raise ValueError(f'{i} is not a valid country name.')
            else:
                country_selected = self.my_df[self.my_df['Entity'].isin([i])][['Entity', 'Year', 'total_output']]
                plt.plot(country_selected['Year'], country_selected['total_output'], label=i)

        plt.title(f'Comparison of Output Totals for selected Countries ({countries})')
        plt.xlabel('Year')
        plt.ylabel('Total Output')
        plt.legend()
        plt.annotate('Source: Agricultural total factor productivity (USDA), OWID 2021', (0,0), (0,-35), fontsize=10, 
                     xycoords='axes points', textcoords='offset points', va='top')
        plt.show()

    def __gapminder__(self, year: int):
        """
        Plots a scatter plot to visualize agricultural production data for a given year.

        Parameters
        ----------
        year : int
            The year for which to plot the agricultural production data.

        Raises
        ----------
        TypeError: If year is not an integer.
        ValueErrof: If year is not present in the DataFrame.

        Returns
        ----------
        None

        The scatter plot shows the relationship between the quantity of fertilizer
        used and the quantity of agricultural output,
        with the size of each dot indicating the amount of capital invested.
        The x-axis is on a logarithmic scale.
        """
        if not isinstance(year, int):
            raise TypeError("Year must be an integer.")

        if year not in self.my_df["Year"].unique():
            raise ValueError("Year not present in DataFrame.")

        data = self.my_df[self.my_df["Year"] == year]

        my_x = data["fertilizer_quantity"]
        my_y = data["output_quantity"]
        size = data["capital_quantity"] / 10000  # set size based on capital_quantity

        plt.scatter(my_x, my_y, s=size)
        plt.xscale("log") # set x-axis scale to logarithmic
        plt.yscale("log") # set y-axis scale to logarithmic
        plt.xlabel("Fertilizer Quantity (log scale)")
        plt.ylabel("Output Quantity (log scale)")
        plt.title(f"Agricultural Production ({year})")
        plt.gca().legend(("Capital Quantity",), scatterpoints=1, fontsize=10)
        plt.annotate('Source: Agricultural total factor productivity (USDA), OWID 2021', (0,0), (0,-40), fontsize=10, 
                     xycoords='axes points', textcoords='offset points', va='top')
        plt.show()

    def choropleth(self, year: int):
        """
        Plots a choropleth map to visualize agricultural yield data for a given year.

        Parameters
        ----------
        year : int
            The year for which to plot the agricultural yield data.

        Raises
        ----------
        TypeError: If year is not an integer.
        ValueError: If year is not present in the DataFrame.

        Returns
        ----------
        None
        """
        # Check if the year is an integer
        if not isinstance(year, int):
            raise TypeError('Year must be an integer.')
    
        if year not in self.my_df["Year"].unique():
            raise ValueError('Year not present in DataFrame.')
    
        # Merge the dataframes on the country names
        merge_dict = {'Bosnia and Herzegovina': 'Bosnia and Herz.', 'Burma': 'Myanmar', 'Eswatini': 'eSwatini', 
                      'United States': 'United States of America', 'North Macedonia': 'Macedonia', 
                      'Dominican Republic': 'Dominican Rep.', 'Equatorial Guinea': 'Eq. Guinea', 'South Sudan': 'S. Sudan',
                      'Democratic Republic of Congo': 'Dem. Rep. Congo', 'Solomon Islands': 'Solomon Is.', 'Timor': 'Timor-Leste',
                      'Central African Republic': 'Central African Rep.', 'Macedonia': 'North Macedonia'}
        agr_data = self.my_df.replace({'Entity': merge_dict})
        merged_data = self.geo_df.merge(agr_data, left_on='name', right_on='Entity', how='left')
    
        # Select the data for the specified year
        merged_data_year = merged_data[merged_data.Year == year][['Year', 'geometry', 'tfp', 'pop_est',
                                                                  'continent', 'name', 'iso_a3', 'gdp_md_est']]
        
        # Plot the data on a world map
        fig, ax = plt.subplots(figsize=(20, 15))
        merged_data_year.plot(column='tfp', cmap='YlGnBu', linewidth=1, ax=ax, legend=False)
    
        # Add a title and remove the axis
        ax.set_title(f'Agricultural Yield in {year}', fontdict={'fontsize': '15', 'fontweight': '2'})
        ax.axis('off')
    
        # Add a colorbar
        vmin, vmax = merged_data_year['tfp'].min(), merged_data_year['tfp'].max()
        sm = plt.cm.ScalarMappable(cmap='YlGnBu', norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm._A = []
        cbar = fig.colorbar(sm, shrink=0.3)
    
        # Show the plot
        plt.annotate('Source: Natural Earth, 2023', (0,0), (50,0), fontsize=10, 
                     xycoords='axes points', textcoords='offset points', va='top')
        plt.show()
        
    def predictor(self, countries):
        """
        Plots the actual and predicted Total Factor Productivity (TFP) values for up to three specified countries 
        from 1960 to 2050.

        Parameters
        ----------
        countries : list
            A list of country names for which the TFP data should be plotted.

        Raises
        ----------
        TypeError: If countries is not a list.
        ValueError: If all of the specified countries are not present in the dataset. Suggests alternative country options.

        Returns
        ----------
        None
        
        Generates a plot of actual and predicted TFP values for the specified countries.
        """
        # Check if countries is a list
        if not isinstance(countries, list):
            raise TypeError('Input must be a list of countries.')
        
        data = self.my_df
        
        # Filter the data for the selected countries
        filtered_data = data[data['Entity'].isin(countries)]

        # Check if any country is valid
        if len(filtered_data['Entity'].unique()) == 0:
            raise ValueError(f'Non of the given countries are in the dataframe. Please choose any of the following countries:\n{data.Entity.unique()}')
        
        # Check if any country is missing
        if len(filtered_data['Entity'].unique()) < len(countries):
            missing_countries = set(countries) - set(filtered_data['Entity'].unique())
            print(f"Warning: The following countries are missing from the dataset and will be ignored: {missing_countries}")
            countries = list(set(countries) - missing_countries)

        if len(countries) > 3:
            countries = countries[:3]
            print(f"Only the first three suggested countries {', '.join(countries)} will be taken into account for the prediciton.")

        # Iterate over the specified countries and plot their actual TFP data
        fig, ax = plt.subplots()
        
        # Iterate over the specified countries and plot their actual TFP data
        for country in countries:
            country_data = data[data['Entity']==country]
            
            # Compute weights for each year based on its distance from the most recent year
            weights = np.linspace(1, 0.5, len(country_data))
            tfp_data = country_data['tfp'].values
            weighted_tfp = np.average(tfp_data, weights=weights)
            years = country_data['Year'].values
            ax.plot(years, tfp_data, label=country)
            
            # Fit an ETS model to the data and predict future values
            model = ExponentialSmoothing(tfp_data, trend='add')
            model_fit = model.fit()
            future_years = range(max(years)+1, 2051)
            predicted_values = model_fit.predict(start=len(tfp_data), end=len(tfp_data)+len(future_years)-1)
            
            # Plot the predicted values for the country
            ax.plot(future_years, predicted_values, linestyle='--', color=ax.lines[-1].get_color())
        
        ax.legend()
        ax.set_xlabel('Year')
        ax.set_ylabel('TFP')
        ax.set_title('TFP per Year by Country')
        plt.annotate('Source: Natural Earth, 2023', (0,0), (0,-40), fontsize=10, 
                     xycoords='axes points', textcoords='offset points', va='top')
        plt.show()
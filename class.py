"""
This module provides a class with 6 methods
"""

import os
import pandas as pd
import requests
import seaborn as sns
import matplotlib.pyplot as plt

class Group22:
    """
    A class to to examine a dataset on agriculture.

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
        downloads the dataset and turns it into a pandas dataframe
    """
    def __init__(self, url, filename):
        """
        Constructs all the necessary attributes for the class object.

        Parameters
        ----------
        url : str
            url of dataset
        filename : str
            filename of dataset
        """
        self.url = url
        self.filename = filename

    def download_data(self):
        """
        Returns a dataframe from the dataset

        Parameters
        ----------
        None

        Returns
        -------
        dataframe
        """
        if not os.path.exists(os.path.join("downloads", self.filename)):
            response = requests.get(self.url)
            if response.status_code == 200:
                with open(os.path.join("downloads", self.filename), "wb") as f:
                    f.write(response.content)
                print(f"{self.filename} has been downloaded")
            else:
                print(f"Error downloading {self.filename}: {response.status_code}")
        else:
            print(f"{self.filename} already exists")

        data_path = os.path.join("downloads", self.filename)
        df = pd.read_csv(data_path, on_bad_lines="skip")
        
        self.df = df

        return df
    
    
    def get_countries(self, df):
        """
        Returns a list of all the countries of the dataset

        Parameters
        ----------
        df : pandas dataframe
            dataframe of the dataset

        Returns
        -------
        list
        """
        return list(df['Entity'].unique())
    
    def plot_quantity_correlation(self, df):
        """
        Returns a correlation matrix of the quantity columns

        Parameters
        ----------
        df : pandas dataframe
            dataframe of the dataset

        Returns
        -------
        correlation matrix
        """
        quantity_cols = [col for col in df.columns if '_quantity' in col]
        quantity_df = df[quantity_cols]
        corr_matrix = quantity_df.corr()

        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix of Quantity Columns')
        plt.show()
        
    def plot_output_area_chart(self, df, country=None, normalize=False):
        """
        Plots an area chart of the distinct "_output_" columns.

        Parameters
        ----------
        df : pandas dataframe
            dataframe of the dataset
        country : str or None
            Country name. If None or 'World', plots the sum for all countries.
        normalize : bool
            If True, normalizes the output in relative terms: each year, output should always be 100%.

        Returns
        -------
        area chart of the distinct "_output_" columns
        """
        output_cols = [col for col in self.df.columns if '_output_' in col]
        if not output_cols:
            raise ValueError("No output columns found in the dataset.")

        if country is None or country == 'World':
            df_country = self.df.groupby('Year')[output_cols].sum().reset_index()
            title = 'World Output'
        else:
            if country not in self.get_countries(df):
                raise ValueError(f"{country} is not a valid country name.")
            df_country = self.df[self.df['Entity'] == country][['Year'] + output_cols]
            title = f"{country} Output"

        if normalize:
            df_country[output_cols] = df_country[output_cols].div(df_country[output_cols].sum(axis=1), axis=0) * 100

        df_country.set_index('Year').plot(kind='area', stacked=True)
        plt.title(title)
        plt.xlabel('Year')
        plt.ylabel('Output')
        plt.show()
        
    def compare_output_countries(self, df, countries):
        """
        Plots a comparison of the total of the '_output_' column for each of the given countries.

        Parameters:
        ----------
        df : pandas dataframe
            dataframe of the dataset
        country : str or list
            country names to compare

        Returns
        -------
        line graph of total '_output_' columns of given countries over the years
        """
        # Create total output column
        output_cols = [col for col in self.df.columns if '_output_' in col]
        df['total_output'] = self.df[output_cols].sum(axis=1)

        # Transform input to list
        if not isinstance(countries, list):
            countries = [countries]

        if len(countries) == 1:
            if countries not in self.get_countries(df):
                raise ValueError(f'{countries} is not a valid country name.')
            else: 
                country_selected = self.df[df['Entity'].isin(countries)][['Entity', 'Year', 'total_output']]
                plt.plot(country_selected['Year'], country_selected['total_output'], label=countries)
        else:
            for i in countries:
                if i not in self.get_countries(df):
                    raise ValueError(f'{i} is not a valid country name.')
                else:
                    country_selected = self.df[df['Entity'].isin([i])][['Entity', 'Year', 'total_output']]
                    plt.plot(country_selected['Year'], country_selected['total_output'], label=i)

        plt.title('Comparison of Output Totals for selected Countries')
        plt.xlabel('Year')
        plt.ylabel('Total Output')
        plt.legend()
        plt.show()
        
    def __gapminder__(self, year):
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

        The scatter plot shows the relationship between the quantity of fertilizer used and the quantity of agricultural output,
        with the size of each dot indicating the amount of capital invested. The x-axis is on a logarithmic scale.
        """
        if not isinstance(year, int):
            raise TypeError("Year must be an integer.")
            
        if year not in self.df['Year'].unique():
            raise ValueError("Year not present in DataFrame.")

        data = self.df[self.df['Year'] == year]

        x = data['fertilizer_quantity']
        y = data['output_quantity']
        size = data['capital_quantity'] / 10000 # set size based on capital_quantity

        plt.scatter(x, y, s=size)
        plt.xscale('log')  # set x-axis scale to logarithmic
        plt.xlabel('Fertilizer Quantity (log scale)')
        plt.ylabel('Output Quantity')
        plt.title(f'Agricultural Production ({year})')
        plt.gca().legend(('Capital Quantity',), scatterpoints=1, fontsize=10)
        plt.show()
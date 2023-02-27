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
                with open(os.path.join("downloads", self.filename), "wb") as my_f:
                    my_f.write(response.content)
                print(f"{self.filename} has been downloaded")
            else:
                print(f"Error downloading {self.filename}: {response.status_code}")
        else:
            print(f"{self.filename} already exists")

        data_path = os.path.join("downloads", self.filename)
        my_df = pd.read_csv(data_path, on_bad_lines="skip")

        self.my_df = my_df

        return my_df

    def get_countries(self, my_df):
        """
        Returns a list of unique country names from the
        "Entity" column of the input dataframe.

        Parameters
        ----------
        my_df : pandas dataframe
            The input dataframe to extract unique
            country names from.

        Returns
        -------
        list of unique country names
        """
        return list(my_df["Entity"].unique())

    def plot_quantity_correlation(self, my_df):
        """
        Plots a correlation matrix of quantity columns from the
        input dataframe.

        Parameters
        ----------
        my_df : pandas dataframe
            The input dataframe to extract quantity columns from.

        Returns
        -------
        None
            Displays a heatmap of the correlation matrix
            of quantity columns.
        """
        quantity_cols = [col for col in my_df.columns if "_quantity" in col]
        quantity_df = my_df[quantity_cols]
        corr_matrix = quantity_df.corr()

        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.title("Correlation Matrix of Quantity Columns")
        plt.show()

    def plot_output_area_chart(self, my_df, country=None, normalize=False):
        """
        Plots an area chart of the distinct "_output_" columns.

        Parameters
        ----------
        my_df : pandas dataframe
            dataframe of the dataset
        country : str, optional
            The name of the country to plot the output for.
            If None or 'World', plots the sum for all countries.
            Default is None.
        normalize : bool, optional
            If True, normalizes the output in relative terms:
            each year, output should always be 100%.
            Default is False.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no output columns are found in the dataset
            or if the given country name is not valid.
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
            if country not in self.get_countries(my_df):
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
        plt.show()

    def compare_output_countries(self, my_df, countries):
        """
        Plots a comparison of the total of the '_output_' column for each of the given countries.

        Parameters:
        ----------
        my_df : pandas dataframe
            dataframe of the dataset
        country : str or list
            Country or list of countries to compare

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If no output columns are found in the dataset.
            If the provided country/countries is not
            in the list of valid country names.
        """
        # Create total output column
        output_cols = [col for col in self.my_df.columns if "_output_" in col]
        my_df["total_output"] = pd.DataFrame(self.my_df)[output_cols].sum(axis=1)

        # Transform input to list
        if not isinstance(countries, list):
            countries = [countries]

        if len(countries) == 1:
            if countries not in self.get_countries(my_df):
                raise ValueError(f"{countries} is not a valid country name.")
            country_selected = pd.DataFrame(self.my_df)[my_df["Entity"].isin(countries)][
                ["Entity", "Year", "total_output"]
            ]
            plt.plot(
                country_selected["Year"],
                country_selected["total_output"],
                label=countries,
            )
        else:
            for i in countries:
                if i not in self.get_countries(my_df):
                    raise ValueError(f"{i} is not a valid country name.")
                country_selected = pd.DataFrame(self.my_df)[my_df["Entity"].isin([i])][
                    ["Entity", "Year", "total_output"]
                ]
                plt.plot(
                    country_selected["Year"],
                    country_selected["total_output"],
                    label=i,
                )

        plt.title("Comparison of Output Totals for selected Countries")
        plt.xlabel("Year")
        plt.ylabel("Total Output")
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
        plt.xscale("log")  # set x-axis scale to logarithmic
        plt.xlabel("Fertilizer Quantity (log scale)")
        plt.ylabel("Output Quantity")
        plt.title(f"Agricultural Production ({year})")
        plt.gca().legend(("Capital Quantity",), scatterpoints=1, fontsize=10)
        plt.show()

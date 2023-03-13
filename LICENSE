# Agricultural Data Analysis Project

This project analyzes a dataset on agricultural production using Python programming language. The project includes a custom environment, a Python class with 6 methods, a__docs__ directory with more information on the documentation, and a Jupyter notebook for demonstration and analysis.

## Installation

To run this project, clone the repository and create a custom environment using the provided environment.yml file:

```bash
conda env create -f environment.yml
```

Activate the environment: 
```bash
conda activate adpro_project
```

## Usage

The my_class.py file contains the Group22 class with 6 methods that allow for the analysis of the agricultural data.
```python
from my_class import Group22

# create an instance of the class
group = Group22(url="https://example.com/dataset.csv", filename="dataset.csv")

# download the data and create dataframes
group.download_data()

# get a list of all countries in the dataframe
countries = group.get_countries()

# plot a correlation matrix of quantity columns
group.plot_quantity_correlation()

# plot an area chart of output columns
group.plot_output_area_chart()

# plot a line graph of output columns for specified countries
group.compare_countries_output(['USA', 'China'])

# plot agricultural production data for a given year
group._gapminder_(year=2000)

# plot a choropleth map to visualize agricultural yield data for a given year
group.choropleth(year=2010)

# plot the actual and predicted TFP values for up to three specified countries from 1960 to 2050
group.predictor(['USA', 'China', 'India'])
```

## Dataset

The dataset used in this project from the [Our World in Data](https://ourworldindata.org/). The dataset can be found [here](https://github.com/owid/owid-datasets/blob/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)/Agricultural%20total%20factor%20productivity%20(USDA).csv), which provides extensive information on agriculture from 1961 to 2018. The dataset contains information on agricultural production, trade, food balance sheets, prices, and indices for various countries and regions.

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[LINK](https://gitlab.com/annaziegert/group_22/-/blob/4de67b8144a70dcde74bf968068d653c53add3fc/LICENSE)
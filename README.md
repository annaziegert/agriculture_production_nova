# Welcome to Group 22 of Project Agros
---

## Scenario

Your company is participating in a two-day hackathon promoted to study the agricultural output of several countries. They send your group, the best team of Data Scientists in the company's roster. By undertaking this task, your company expects to contribute to the green transition by having a more savvy taskforce. You decide to create a python class for the challenge.

## Goal

For this project, we will be using data from [Our World in Data](https://ourworldindata.org/). The dataset can be found [here](https://github.com/owid/owid-datasets/blob/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)/Agricultural%20total%20factor%20productivity%20(USDA).csv).

Go over the set. Analyze all the data [here](https://github.com/owid/owid-datasets/tree/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)).

<div class="alert alert-danger">
    <b> THE MOST IMPORTANT TOOLS FOR A DATA SCIENTIST IS PATIENCE AND COMMUNICATION</b>
    <br>
    <b> Discuss the contents of the dataset with your colleagues. Understanding the data is a priority. </b>
</div>

Use whatever python tools you find apropriate. We recommend seaborn for plotting.

## Structure of the project

Keep the .py files in a separate directory. The only files in the main directory of the project should be the **Showcase Notebook** and the several configuration files (.yml, .gitignore, and others). Everything else should have their own directories.

### Day 1, Phase 1

- [ ] One of you will create a gitlab repository (it does not matter who). __THE NAME OF THE REPOSITORY MUST BE "Group_XX" where XX is the number of your group! If you are group 3, then XX must be 03. Always use two digits!__
- [ ] Initialize the repo with a README.md file, a proper license, and a .gitignore for the programming language you will use.
- [ ] The one who created the repository will then give __Maintainer__ permissions to the rest of the group. Check under "Project Information" > "Members".
- [ ] Every element of the group clones the repository to their own laptops.

### Day 1, Phase 2

- [ ] The class you decide the create for the project has finally been named after a brief fight and is __PEP8 compliant, like the entire project__.

The class will have several methods, which you will __not__ develop in the master branch.  
Document everything!  
Make your calls compliant with __Static Type Checking__.

- [ ] One method will _download_ the data file into a __downloads/__ directory in the root directory of the project (main project directory). If the data file already exists, the method will not download it again.
- [ ] This method must also read the dataset into a pandas dataframe which is an attribute of your class.

## Day 1, Phase 3

- [ ] Develop a second method that outputs a list of the available countries in the data set.
- [ ] Develop a third method that plots a way to correlate the "\_quantity" columns.
- [ ] Develop a fourth method that plots an area chart of the *distinct* "\_output_" columns. This method should have two arguments: a __country__ argument and a __normalize__ argument. The former, when receiving *NONE* or 'World' should plot the sum for all *distinct* countries. The latter, if True, normalizes the output in relative terms: each year, output should always be 100%. The X-axis should be the Year. The method should return a ValueError when the chosen country does not exist.
- [ ] Develop a fifth method that may receive a string with a country or a list of country strings. This method should compare the total of the "\_output_" columns for each of the chosen countries and plot it, so a comparison can be made. The X-axis should be the Year.
- [ ] Develop a sixth method that must be called __gapminder__. This is a reference to the famous [gapminder tools](https://www.gapminder.org/tools/#$chart-type=bubbles&url=v1). This method should receive an argument __year__ which must be an __int__. If the received argument is not an int, the method should raise a TypeError. This method should plot a scatter plot where __x__ is __fertilizer_quantity__, y is __output_quantity__, and the area of each dot should be a third relevant variable you find with exploration of the data.

### Day 1, Phase 4

- [ ] Make a "showcase notebook" where you import your __Class__ and showcase all the methods you developed. Tell a story about your analysis and findings in the showcase notebook:

Let's begin by telling a story.

Start by an overall analysis with the gapminder plot for the most recent year. How do you see the world's agricultural production? Write 1-2 short paragraphs about your analysis.

Choose **three** countries in the list. Any countries will do, but choose one from each continent.

Use the fourth method to illustrate each country chosen and point out the main differences.

Use the fifth method to illustrate each country chosen and point out the main differences.

How do the variables correlate with each other? Run your third method and briefly analyse it.

Finish day one with a very short overall analysis between quantities and outputs.


### Day 2, Phase 1: Extending the scope

- [ ] Make sure you are not using aggregated columns. It makes no sense in using an Asian country and then also use *Asia* in your plots, for example. You did not notice this on day one (well, if you did, the real group, not the group from the scenario hackathon, congratulatios. Data Science is also about detail).
- [ ] ALL methods at the end of day 2 should have a descriptive title, correct labels, and an annotation stating the source of the data, like the **Source** in this [example](https://raw.githubusercontent.com/guimarais/iaea_reactors/main/Operations_Shutdowns.png). Don't lose time by formatting the rest of the plot, leave as is, but the source of the data is fundamental.


## Day 2, Phase 2: World view

- [ ] Make a module called **choropleth**. This method should receive a year as input, which must be an integer. **Raise** otherwise. We're going deep with our analysis and we're going to use geodata. We recommend you install geopandas. Alter the method where you download the data to also download and read a geographical dataset. The geo dataset must have the polygons for as many countries as possible. You can get such a datafile [here](https://www.naturalearthdata.com) (use cultural data, as it refers to countries). If you download a zip file, remember you want to access the shapefile (.shp) inside. [Read the documentation for geopandas for examples on how to read data](https://geopandas.org/en/stable/docs/user_guide/io.html).
- [ ] **Merge** (pandas equivalent to SQL JOIN) the agricultural data with the geodata on the countries. Make sure the **left** dataframe is the geopandas dataframe. When you plot the result of the merge, you may notice some important country or countries missing. That is because their names don't match. Make a **VARIABLE** of the class called *merge_dict* which is a dictionary that renames at least one country
- [ ] Plot the **tfp** variable on a world map. Make sure you use a colorbar. [This example should help](https://medium.com/geekculture/three-ways-to-plot-choropleth-map-using-python-f53799a3e623).

<div class="alert alert-danger">
    <b> Treat the geopandas dataframe as a regular dataframe. </b>
    <br>
    <b> Keep the "geometry" column untouched, as that is the special column for plots.</b>
</div>


### Day 2, Phase 3: Predictor

- [ ] Make a **predictor** method that receives a list of countries as input, up to three. If one or more countries on the list is not present in the Agricultural dataframe, it should be ignored. If none is, **raise** an error message reminding the user what countries are available. Maybe you already have that from yesterday?
- [ ] The **predictor** method should plot the **tfp** in the dataset and then complement it with an **ARIMA** prediction up to 2050. Use the same color for each country's actual and predicted data, but a different line style.

### Day 2, Phase 4: Cleaning up

- [ ] Add a yaml file to git with all the packages you used, a conda environment file. This file will be used to generate an environment where your code will be ran. Remember to make it OS independent.
- [ ] Use sphinx to generate a __docs__ directory that will shocase the documentation of your code. Remember to comment .gitignore appropriately so everything is included. Update README.md to tell the user how to start using the project.

___
### Finishing up:

Let's begin by telling a story.

Start by an overall analysis with the gapminder plot for the most recent year and the year 1980. What can you see as the major changes regarding the variables you considered the most relevant on day 1?

Choose **three** countries in the list. Any countries will do, but at least one must be a member of the EU and one must be outside the EU.  
Use all other analysis methods (all you have developed except the **gapminder**, the **choropleth**, and the **predictor**) to analyse the evolution of each country's agricultural data. Write a small story about it.

End by showing **choropleth** for the most recent year and 2000.

Finish by analysing the output of **predictor**.

---
## Grading

Between the two parts, there are 20 gradable items in both Part 1 and 2. The last two items of Day 1, Phase 1 are not gradable.

<div class="alert alert-danger">
    <b> REMEMBER: IT IS OK TO PROTOTYPE CODE IN NOTEBOOKS, BUT THE FINAL CLASS MUST BE IN A SINGLE .py FILE! </b>
    <br>
    <b> The final delivery of the project is the "showcase" notebook. Don't place this notebook together with prototyping notebooks.</b>
    <br>
    <b> Prototyping notebooks must have their own separate directory.</b>
    <br>
    <b> We will only consider contents in your "master" repository before the end of the deadline.</b>
</div>
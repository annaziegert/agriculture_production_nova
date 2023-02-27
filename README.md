# Welcome to project Agros
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

<div class="alert alert-danger">
    <b> REMEMBER: IT IS OK TO PROTOTYPE CODE IN NOTEBOOKS, BUT THE FINAL CLASS MUST BE IN A SINGLE .py FILE! </b>
    <br>
    <b> The final delivery of the project is the "showcase" notebook from Phase 4. Don't place this notebook together with prototyping notebooks.</b>
    <br>
    <b> Prototyping notebooks must have their own separate directory.</b>
    <br>
    <b> We will only consider contents in your "master" repository.</b>
</div>

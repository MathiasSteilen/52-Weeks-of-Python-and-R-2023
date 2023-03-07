# 2023: 52 Weeks of Working with Data in  `Python` and `R`

## Content of this respository

This repo holds the contents of the challenge I set myself for 2023:

> Finish one machine learning, data wrangling or data viz project each week of 2023 in both Python and R.

Given that both languages are complements nowadays, with R being strongest for data wrangling and visualisation and Python being the predominant language for machine learning, I believe that this challenge will be incredibly useful to gain hands-on experience.

<br>

## Conclusion

To be written at the end of 2023...

<br>

## Summary of Weekly Projects

***

### Week 1: Predicting Olympic Athletes' weights 🏅🤸

|  | Description |
| :------ | :------ |
| **Data Source**      | [TidyTuesday](https://github.com/rfordatascience/tidytuesday/blob/master/data/2021/2021-07-27/readme.md)       |
| **Goal**   | Predicting athlete weight with Elastic Net with `scikit-learn` (Python) and `tidymodels` (R) |
| **Keywords**   | Supervised Learning, Elastic Net, K-Fold Cross Validation, Hyperparameter Tuning, Parallel Processing |
| **Results**   | $R^2 \approx 74.6\%$ for `scikit-learn` and $R^2 \approx 73.5\%$ for `tidymodels`. Note that the differences are likely due to the splits and can be ignored. |
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/01%20-%20Olympic%20Athletes/OlympicsPython.ipynb), [R](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/01%20-%20Olympic%20Athletes/OlympicsR.ipynb) |

<p align="center">
  <img src="01 - Olympic Athletes/python.png" height="300" />
  <img src="01 - Olympic Athletes/r.png" height="300" />
</p>

***

### Week 2: Visualising NYC School Bus Breakdowns and Delays 🚌💥

| | Description |
| :----------- | :----------- |
| **Data Source** | [Kaggle](https://www.kaggle.com/datasets/mattop/new-york-city-bus-breakdown-and-delays) |
| **Goal** | Wrangling the medium sized (>500K observations) dataset on schoolbus breakdowns in NYC to understand more about the data and make visualisations about potential time trends |
| **Keywords** | Data Wrangling, Data Visualisation, `Pandas`, `Tidyverse`, `ggplot2`, `seaborn`, `matplotlib` |
| **Results** | `ggplot2` and the `Tidyverse` are a godsend for data wrangling and visualisation, however the combination of `Pandas` and `seaborn` also gets the job done very well, once you get used to the different way of thinking. Also: Manhattan's school bus delays are getting worse.|
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/02%20-%20Bus%20Breakdowns%20NYC/BusesPython.ipynb), [R](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/02%20-%20Bus%20Breakdowns%20NYC/BusesR.ipynb) |

<p align="center">
  <img src="02 - Bus Breakdowns NYC/python.png" height="250" />
  <img src="02 - Bus Breakdowns NYC/r.png" height="250" width="500"/>
</p>

***

### Week 3: Comparing Gradient Boosting Methods on Insurance Data 🌲🚀

| | Description |
| :----------- | :----------- |
| **Data Source** | [Kaggle](https://www.kaggle.com/datasets/thedevastator/prediction-of-insurance-charges-using-age-gender?datasetId=2792769&sortBy=dateRun&tab=profile) |
| **Goal** | Compare the performance of two gradient boosting methods (`LightGBM` and `XGBoost`) on insurance data, namely charges in USD for each individual based on predictors such as age, region or smoker. Implementation in `sklearn` (Python) and `tidymodels` (R). |
| **Keywords** | Supervised Learning, Gradient Boosting, LightGBM, XGBoost, K-Fold Cross Validation, Randomised Grid Search, Hyperparameter Tuning |
| **Results** | Tuning times in Python's `sklearn` are significantly faster than in `tidymodels`, however ease of use for preprocessing is much better. For large projects, initial prototype models could be made in R, then shift to Python, once the proof-of-concept stands? |
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/03%20-%20Boosting%20Methods/Insurance-Python.ipynb), [R](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/03%20-%20Boosting%20Methods/Insurance-R.ipynb) |

<p align="center">
  <img src="03 - Boosting Methods/python.png" height="250" />
  <img src="03 - Boosting Methods/R.png" height="250" />
  <img src="03 - Boosting Methods/ridges.png" height="250" width="500" />
</p>

***

### Week 4: Predicting Concrete Compressive Strength 👷🏗️

| | Description |
| :----------- | :----------- |
| **Data Source** | [Kaggle](https://www.kaggle.com/datasets/sinamhd9/concrete-comprehensive-strength) |
| **Goal** | Predict the compressive strength of concrete in megapascals from its separate components (Regression Setting). |
| **Keywords** | Supervised Learning, `Random Forest`, `Elastic Net`, Randomised Grid Search, K-Fold Cross Validation, Hyperparameter Tuning, Model Evaluation Metrics |
| **Results** | The improvement of the regularised regression (elastic net) was negligible over OLS. The performance of the tuned random forest was most impressive with $R^2 \approx 0.9$ and $\text{MAPE} \approx 0.13$.   |
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/04%20-%20Concrete%20Strength/ConcretePython.ipynb) |

<p align="center">
  <img src="04 - Concrete Strength/python.png" width="500" />
  <img src="04 - Concrete Strength/predictions.png" width="500" />
</p>

***

### Week 5: Scraping Belgian Rent Prices 🏠💶🇧🇪

| | Description |
| :----------- | :----------- |
| **Data Source** | Website of a major Belgian real estate platform |
| **Goal** | Scrape rent prices for all of Brussels from the website using the dynamic web scraping library `Selenium` and make it available for download. In a later week, train a model that uses tokenisation on the descriptions for price predictions. |
| **Keywords** | Dynamic Web Scraping, `HTML`,  `Selenium`, `RSelenium`, Data Cleaning |
| **Results** | The scraped data set encompasses 3,477 listings and has 109 columns. It is available for download on [Kaggle](https://www.kaggle.com/datasets/mathiassteilen/monthly-rent-of-rented-flats-in-brussels). |
| **Notebooks** | [R Script]() |

<p align="center">
  <img src="05 - Brussels Rent Prices/Kaggle Picture.png" width="500" />
</p>


***

### Week 6: Luxury Watches ⌚💸

| | Description |
| :----------- | :----------- |
| **Data Source** | [chrono24.at](https://www.chrono24.at/) |
| **Goal** | Filtering down for some specific brands, I want to scrape the information off of each watch listing (N = 45,000) pertaining to price, brand, colour, age and so forth. With the data, I can then create visualisations and train a model to create valuations for new watches. |
| **Keywords** | Web Scraping, `HTML`,  `Selenium`, `RSelenium`, `rvest`, Data Cleaning |
| **Results** | The watch prices are much harder to predict than rental prices due to the sparsity of the data on some brands. Brand and model are much more important than exterior factors, so the performance is not impressive. Additionally, omitted variables play a factor, for instance rarity or collector edition tags. |
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/06%20-%20Luxury%20Watches/Watches-Python.ipynb), [R](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/06%20-%20Luxury%20Watches/Selenium%20%2B%20Cleaning.R) |

<p align="center">
  <img src="06 - Luxury Watches/python.png" width="250" />
  <img src="06 - Luxury Watches/r.png" width="500" />
</p>

***

### Week 7: New York City Open Data 🗽🌇

| | Description |
| :----------- | :----------- |
| **Data Source** | [NYC Open Data](https://opendata.cityofnewyork.us/) |
| **Goal** | The NYC Open Data platform holds almost endless amounts of free data. I am especially interested in `All Arrests in NYC since 2006`, `Citywide Payroll Data`, `Causes of Death`, `Restaurant Violations` and `Dog Registrations`. |
| **Keywords** | `pandas`, `ggplot2`, `tidyverse`, `Data Wrangling`, `Data Visualisation`, `Exploratory Data Analysis` |
| **Results** | Some useless fun facts, like that some dogs were named Mozzarella in 1994 and then less fun facts like that firemen in NYC pull the most overtime of all agencies.  |
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/07%20-%20New%20York%20Open%20Data/NYC-Python.ipynb), [R](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/07%20-%20New%20York%20Open%20Data/NYC-R.ipynb) |

<p align="center">
  <img src="07 - New York Open Data/r.png" width="500" />
  <img src="07 - New York Open Data/python.png" width="500" />
  <img src="07 - New York Open Data/python2.png" width="500" />
</p>

***

### Week 8: Using Tokenisation on Rental Listing Titles to Improve Predictions 📜🗣️

| | Description |
| :----------- | :----------- |
| **Data Source** | [willhaben.at](https://www.willhaben.at) (Scraped >7,000 rental listings before modelling) |
| **Goal** | First, I want to train a LGBM (boosted trees) model on the monthly rent data in Vienna (predictors: size of apartment, postcode, landlord, number of rooms). Then, I want to use a natural-language-processing approach, namely tokenisation, on the free text titles (Example: 'Provisionsfreie 2-Zimmer-Wohnung mit ca. 10 m² Terrasse') of the listings to see if the model performance can be improved with the additional information. |
| **Keywords** | `NLP`, `Tokenization`, `Supervised Learning`, `Gradient Boosting`, `LightGBM`, `XGBoost`, `K-Fold Cross Validation`, `Randomised Grid Search`, `Hyperparameter Tuning` |
| **Results** | The prediction performance got 4 ppts better $R^2$ (64% -> 68%) and 16 EUR/month less absolut error on average. The CountVectoriser makes it ridiculously easy to apply tokenisation, to the point where I'd even give the `sklearn` approach more credit than the `tidymodels` one. All in all, tokenisation improved the performance of this model significantly and can therefore be recommended in cases like this. |
| **Notebooks** | [Python](), [R]() |

<p align="center">
  <img src="08 - Tokenisation/python.png" width="300" />
  <img src="08 - Tokenisation/R.png" width="300" />
  <img src="08 - Tokenisation/python1.png" width="300" />
  <img src="08 - Tokenisation/python2.png" width="300" />
</p>

***

### Week 9: Coming up...

| | Description |
| :----------- | :----------- |
| **Data Source** | TBD |
| **Goal** | TBD |
| **Keywords** | TBD |
| **Results** | TBD |
| **Notebooks** | TBD |

***

### Week 10: Coming up...

| | Description |
| :----------- | :----------- |
| **Data Source** | TBD |
| **Goal** | TBD |
| **Keywords** | TBD |
| **Results** | TBD |
| **Notebooks** | TBD |
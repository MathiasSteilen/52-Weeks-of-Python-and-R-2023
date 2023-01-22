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

All results are visible as Jupyter Notebooks in the respective folders, if you are interested in the details. You can also find the overview below:

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

### Week 4: Coming up...

| | Description |
| :----------- | :----------- |
| **Data Source** | TBD |
| **Goal** | TBD |
| **Keywords** | TBD |
| **Results** | TBD |
| **Notebooks** | TBD |

***

### Week 5: Coming up...

| | Description |
| :----------- | :----------- |
| **Data Source** | TBD |
| **Goal** | TBD |
| **Keywords** | TBD |
| **Results** | TBD |
| **Notebooks** | TBD |

***

### Week 6: Coming up...

| | Description |
| :----------- | :----------- |
| **Data Source** | TBD |
| **Goal** | TBD |
| **Keywords** | TBD |
| **Results** | TBD |
| **Notebooks** | TBD |

***

### Week 7: Coming up...

| | Description |
| :----------- | :----------- |
| **Data Source** | TBD |
| **Goal** | TBD |
| **Keywords** | TBD |
| **Results** | TBD |
| **Notebooks** | TBD |

***

### Week 8: Coming up...

| | Description |
| :----------- | :----------- |
| **Data Source** | TBD |
| **Goal** | TBD |
| **Keywords** | TBD |
| **Results** | TBD |
| **Notebooks** | TBD |

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
# 2023: 52 Weeks of Working with Data in  `Python` and `R`

This repo holds the contents of the challenge I set myself for 2023:

> **Finish one machine learning, data wrangling or data viz project each week of 2023 in both Python and R.**

Given that both languages are complements nowadays, with R being strongest for data wrangling and visualisation and Python being the predominant language for machine learning, I believe that this challenge will be incredibly useful to gain hands-on experience.

<br>

<!-- ## Conclusion

To be written at the end of 2023...

<br> -->

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
  <img src="01 - Olympic Athletes/python.png" height="250" />
  <img src="01 - Olympic Athletes/r.png" height="250" />
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
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/08%20-%20Tokenisation/Python.ipynb), [R](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/08%20-%20Tokenisation/R.ipynb) |

<p align="center">
  <img src="08 - Tokenisation/python.png" width="300" />
  <img src="08 - Tokenisation/R.png" width="300" />
  <img src="08 - Tokenisation/python1.png" width="300" />
  <img src="08 - Tokenisation/python2.png" width="300" />
</p>

***

### Week 9: Day-Ahead Electricity Prices (Time Series Forecasting) 📈⚡🔮

| | Description |
| :----------- | :----------- |
| **Data Source** | [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/dashboard/show) (Scraped using `Selenium`) |
| **Goal** | This week, I want to try to use two different supervised ML methods on time series data, specifically on Day-Ahead Electricity Prices. The goal is to predict the prices of the day ahead. Then, I want to demonstrate the importance of retraining models frequently by investigating if retraining increases predictive performance. |
| **Keywords** | `Retraining`, `Time Series Forecasting`, `Autocorrelation`, `Supervised Learning`, `Random Forest`, `Elastic Net` |
| **Results** | The random forest model benefitted a lot from retraining! Generally, elastic net seems to stick more strongly to the autocorrelation component than the random forest does. The retrained random forest seems to be the best solution, but I'm sure that this model can be expaned further (more variables, more data, longer lookback than 3 years). As the point of this week was only to show the benefit of retraining, I won't dive deeper into increasing model performance and conclude with the images shown. |
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/09%20-%20Time%20Series%20and%20Retraining/Python.ipynb), [R](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/09%20-%20Time%20Series%20and%20Retraining/R.ipynb) |

<p align="center">
  <img src="09 - Time Series and Retraining/electricity prices.png" width="500" />
  <img src="09 - Time Series and Retraining/RF_retraining_time series.png" width="500" />
  <img src="09 - Time Series and Retraining/Preds_v_Actuals.png" width="500" />
</p>

***

### Week 10: Neural Network with PyTorch 🔥

| | Description |
| :----------- | :----------- |
| **Data Source** | [Swissgrid](https://www.swissgrid.ch/en/home/operation/grid-data/generation.html#downloads) |
| **Goal** | Create a Neural Network (Multilayer Perceptron) for the regression task of predicting daily energy consumption in Switzerland in both regular PyTorch and the newer framework PyTorch Lightning. Predictors are weather data, as well as date information. I split the data from 2012 to 2019 into training, validation and holdout sets for the training and then used an artifical neural network with one hidden layer of 128 nodes to predict the next day. |
| **Keywords** | `PyTorch`, `PyTorch Lightning`, `Artificial Neural Network`, `Regression`, `Supervised learning`, `Overfitting`, `Underfitting` |
| **Results** | The model achieves an $R^2$ of over 90% on the holdout data set, which wasn't used in the training process or validation process. |
| **Notebooks** | [PyTorch](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/10%20-%20Pytorch/Newer%20ANN%20Test.ipynb), [PyTorch Lightning](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/10%20-%20Pytorch/Pytorch%20Lightning.ipynb) |

<p align="center">
  <img src="10 - Pytorch/Picture1.png" width="500" />
</p>

***

### Week 11: Kaggle Competition on Sleep Prediction 😴🛏️

| | Description |
| :----------- | :----------- |
| **Data Source** | [Kaggle](https://www.kaggle.com/competitions/kaggle-pog-series-s01e04/overview) |
| **Goal** | [Rob Mulla](https://www.youtube.com/@robmulla/streams) hosted the fourth installment of Pogchamps competition. This time, it's about predicting his sleep pattern (sleep in hours) from a set of health data (XML) and sleep data from 2015 to 2021. The goal is to land somewhere in the top half of the current leaderboard with a supervised model using `sklearn`.  |
| **Keywords** | `XML`, `Supervised learning`, `LightGBM`, `Gradient Boosting`, Randomised Grid Search, K-Fold Cross Validation, Hyperparameter Tuning, Model Evaluation Metrics, `RMSE` |
| **Results** | I first parsed the XML and aggregated all useable data to daily values, then left-joined on the training and testing dataset. Using LightGBM on a subset of the full training data, I managed to land in the top 25% of participants of 100 participants in the public leaderboard (20% of the data) in the last week of March. Of course, things might still change in the leaderboard, but my new year's resolution must go on. |
| **Notebooks** | [Data Preparation](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/11%20-%20Kaggle%20Competition%20Rob%20Mulla/Data%20Preparation.ipynb), [EDA](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/11%20-%20Kaggle%20Competition%20Rob%20Mulla/EDA.ipynb), [Modelling](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/11%20-%20Kaggle%20Competition%20Rob%20Mulla/Modelling.ipynb) |

<p align="center">
  <img src="11 - Kaggle Competition Rob Mulla/Ranking.png" width="500" />
</p>

***

### Week 12: Hyperparameter Tuning in with Raytune in Pytorch Lightning 🔥🛠️

| | Description |
| :----------- | :----------- |
| **Data Source** | [Swissgrid](https://www.swissgrid.ch/en/home/operation/grid-data/generation.html#downloads) |
| **Goal** | Previously, I have created an MLP in PyTorch and PyTorch Lightning to predict Swiss Energy Demand. The hyperparameters for the latter were handpicked (e.g. hidden layer nodes, batch size and learning rate). The goal for this week is to pick up `RayTune`, a library for experiment execution and hyperparameter tuning at any scale, and tune hyperparameters parallely on my CPU. Further, I want to implement an early stopping technique to cancel unpromising trials for time efficiency. |
| **Keywords** | `RayTune`, `Hyperparameter Tuning`, `Parallel Computation`, `ASHAScheduler`, `PyTorch`, `PyTorch Lightning`, `Artificial Neural Network`, `Regression`, `Supervised learning`, `Overfitting`, `Underfitting` |
| **Results** | The hyperparameters to be tuned were `learning rate` (lr), `batch size` (batch_size), and the `number of nodes` in the first (nodes1) and second (nodes2) layers. I used callbacks for saving the states of the models during training, in order to be able to reload checkpoints of the best model. I used the ASHAScheduler to end unpromising trials early for time efficiency. The tuning process improved the R squared by about 3 ppts and significantly reduced RMSE, indicating that the tuning was a success. |
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/12%20-%20PyTorch%20Hyperparameter%20Tuning/Pytorch%20Lightning.ipynb) |

<p align="center">
  <img src="12 - PyTorch Hyperparameter Tuning/Picture1.png" width="500" />
  <img src="12 - PyTorch Hyperparameter Tuning/python4.png" width="500" />
  <img src="12 - PyTorch Hyperparameter Tuning/python.png" width="500" />
  <img src="12 - PyTorch Hyperparameter Tuning/meme.png" width="150" />
</p>

***

### Week 13: Image Classification - Detecting Cracks in Concrete

| | Description |
| :----------- | :----------- |
| **Data Source** | TBD |
| **Goal** | TBD |
| **Keywords** | TBD |
| **Results** | TBD |
| **Notebooks** | TBD |

***

### Week 14: Fear and Greed

| | Description |
| :----------- | :----------- |
| **Data Source** | TBD |
| **Goal** | TBD |
| **Keywords** | TBD |
| **Results** | TBD |
| **Notebooks** | TBD |

***

### Next Week: Coming up...

| | Description |
| :----------- | :----------- |
| **Data Source** | TBD |
| **Goal** | TBD |
| **Keywords** | TBD |
| **Results** | TBD |
| **Notebooks** | TBD |

***

### Next Week: Coming up...

| | Description |
| :----------- | :----------- |
| **Data Source** | TBD |
| **Goal** | TBD |
| **Keywords** | TBD |
| **Results** | TBD |
| **Notebooks** | TBD |
# 2023: 52 Weeks of Working with Data in  `Python` and `R`

This repo holds the contents of the challenge I set myself for 2023:

> **Finish one machine learning, data wrangling or data viz project every other week of 2023 in both Python and R.**

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
| **Notebooks** | [R Script](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/05%20-%20Brussels%20Rent%20Prices/Selenium.R) |

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
  <img src="10 - Pytorch/Plots for GIF/NN Animation.gif" width="500" />
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

### Week 13: 2023 Easter Weather

| | Description |
| :----------- | :----------- |
| **Data Source** | [Swiss National Basic Climatological Network](https://www.meteoswiss.admin.ch/weather/measurement-systems/land-based-stations/swiss-national-basic-climatological-network.html) |
| **Goal** | Initially, I wanted to do image classification, but started an internship, which caught up most of my time. During Easter I thought it was unusually cold, so this week, I'll check this assumption. |
| **Keywords** | Data Visualisation, `ggplot2`, Nested `dataframe`, `pandas`, `tidyverse`, `Boxplots` |
| **Results** | I used `pandas` to download the csv data into a nested dataframe (using dictionaries) for each weather station. Additionally, I used the `holidays` package to make out the dates of Easter since 1960. After some cleaning and extracting the data from its nested form, I then proceeded to experiment with some charts in `ggplot2` in R and decided on the final product below. The conclusion: Not only was I wrong in thinking that it was extraordinarily cold, I also did not notice how lucky we got with the sunshine. |
| **Notebooks** | [Python](), [R]() |

<p align="center">
  <img src="13 - Easter Weather/R.png" width="500" />
  <img src="13 - Easter Weather/Python.png" width="500" />
</p>

***

### Week 14: Playing Cartpole with the Cross Entropy Method (Reinforcement Learning) 🤖🎮

| | Description |
| :----------- | :----------- |
| **Data Source** | [Open AI Gym](https://gymnasium.farama.org/) |
| **Goal** | I want to train an agent (ANN with one hidden layer) to play Cartpole, i.e. to learn to balance the pole by itself. For this, I will use the Cross Entropy Method. The method will be implemented without higher level libraries, except for PyTorch. |
| **Keywords** | Reinforcement Learning, `PyTorch`, Artificial Neural Network, Cross Entropy Method, Loss Function, Exploration, Exploitation, `gym` |
| **Results** | The Cross Entropy Method uses an ANN like in a classification setting. Inputs are observations and outputs are the probabilities of certain actions given the inputs. The actions are then sampled according to the probabilities of the ANN, which enables exploration vs. exploitation. After a batch, i.e. a certain number of episodes, all episodes with a reward smaller than some percentile are discarded. In other words, the episodes from the batch that worked out best for the agent are kept and the data (input observations made, output subsequent actions taken) are used as a batch to compute a loss and optimise the ANN's weights. Iteratively, for a number of batches (which each contain a number of episodes which each contain a number of steps, observations, actions and rewards), the model is trained on the top percentiles of the batches, in turn getting better and better at mapping observations to a desirable policy. |
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/14%20-%20RL%20Cross%20Entropy%20Method/Cartpole%20Cross%20Entropy%20Method.ipynb) |

<p align="center">
  <img src="14 - RL Cross Entropy Method/Cartpole.gif" width="500" />
  <img src="14 - RL Cross Entropy Method/python.png" width="500" />
</p>

***

### Week 15: Playing FrozenLake with Value Iteration (Reinforcement Learning) 🦿🏞️

| | Description |
| :----------- | :----------- |
| **Data Source** | [Open AI Gym](https://gymnasium.farama.org/) |
| **Goal** | I want to train an agent to play FrozenLake, i.e. to learn to cross a lake and get to the gift itself. For this, I will use the tabular value iteration method. |
| **Keywords** | Reinforcement Learning, Transition Probabilities, Value Iteration, Loss Function, Exploration, Exploitation, `gym` |
| **Results** | The value iteration method stores state values in a table (first picture below) that are iteratively recomputed using the Bellman update while getting new information about rewards from the environment through experience. Experience is gathered from random runs through the environment. The value of a state is the expected future total reward (from discounted individual step rewards) from that state. For FrozenLake, the value iteration works very fast (less than 30 seconds) and finds the optimal solution even for a larger environment that I custom-made to be entirely solvable, despite the agent slipping 2/3s of the time (second picture below). |
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/15%20-%20RL%20Value%20Iteration/FrozenLake%20Value%20Iteration%20Method.ipynb) |

<p align="center">
  <img src="15 - RL Value Iteration/state_values.png" width="250" />
  <img src="15 - RL Value Iteration/solvable_env.png" width="250" />
</p>

***

### Week 16: Playing FrozenLake with Tabular Q-Learning (Reinforcement Learning) 📋🏞️

| | Description |
| :----------- | :----------- |
| **Data Source** | [Open AI Gym](https://gymnasium.farama.org/) |
| **Goal** | I want to train an agent to play FrozenLake, i.e. to learn to cross a lake and get to the gift itself. For this, I will use the tabular Q-Learning method. |
| **Keywords** | Reinforcement Learning, Transition Probabilities, Value Iteration, Loss Function, Exploration, Exploitation, `gym` |
| **Results** | Tabular Q-learning is a model-free reinforcement learning algorithm that learns an optimal policy by iteratively updating Q-values for state-action pairs. The Q-values are updated using the Bellman equation, which combines the observed reward with the maximum expected future reward from the next state-action pair. Transition probabilities are frequentist, i.e. they are deducted from experience. By repeated exploration and exploitation, Q-learning hopefully converges to the optimal Q-values, allowing the agent to select actions that maximize its expected return. For FrozenLake, tabular Q-learning took a bit longer than value iteration, but finds the optimal solution even for the larger environment like before, too. The picture below shows how we are now dealing with state-action pair values (Q-values) instead of state values. The trained agent has a win rate a little lower than 3/4, which is pretty good for a non-deterministic environment which can be failed with a little bit of bad luck. |
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/16%20-%20RL%20Tabular%20Q-Learning/FrozenLake%20Tabular%20Q-Learning.ipynb) |

<p align="center">
  <img src="16 - RL Tabular Q-Learning/state_action_values.png" width="500" />
  <img src="16 - RL Tabular Q-Learning/rolling_wins.png" width="500" />
</p>

***

### Week 17: Playing Lunar Lander with Deep Q-Networks (Reinforcement Learning) 🤖🧑‍🚀

| | Description |
| :----------- | :----------- |
| **Data Source** | [Open AI Gym](https://gymnasium.farama.org/) |
| **Goal** | I want to train an agent to land a spacecraft in a toy environment, i.e. to play OpenAI Gym's Lunar Lander. For this, I will use the DQN method. The method will be implemented without higher level libraries, except for PyTorch. |
| **Keywords** | Reinforcement Learning, `PyTorch`, Feed Forward Neural Network, Value Iteration, Loss Function, Exploration, Exploitation, `gym` |
| **Results** | Deep Q-Network (DQN) is a reinforcement learning algorithm that combines Q-learning with deep neural networks to handle high-dimensional state spaces. DQN employs an epsilon-greedy strategy for action selection, balancing exploration and exploitation. During training, the agent interacts with the environment, and the neural network approximates the Q-values for different state-action pairs. Initially, the agent explores the environment with a higher probability of selecting random actions (epsilon). As training progresses, epsilon is gradually decreased, shifting the agent towards exploiting the learned Q-values. This way, DQN achieves a balance between exploring new actions and exploiting the current knowledge to learn an optimal policy. For this lunar lander case, the training progress took about 40 minutes on a laptop with an i5 processor, which is a bit longer than previous toy examples. However, it does not come as a surprise given the amount of forwards and backward passes required in the neural net. |
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/17%20-%20RL%20DQN/Moonlander%20DQN.ipynb) |

<p align="center">
  <img src="17 - RL DQN/trained lander.gif" width="250" />
  <img src="17 - RL DQN/Training Process.png" width="250" />
  <img src="17 - RL DQN/reward_progress.png" width="500" />
</p>

***

### Week 18: Playing CartPole with Advantage Actor Critic (Reinforcement Learning) 🤖🧑‍🚀


| | Description |
| :----------- | :----------- |
| **Data Source** | [Open AI Gym](https://gymnasium.farama.org/) |
| **Goal** | Shifting away from value learning, I want to turn to policy gradients and train an agent (A2C neural network with common body and two heads) to play Cartpole, i.e. to learn to balance the pole by itself. For this, I will use the Advantage Actor Critic method with entropy bonus. The method will be implemented without higher level libraries, except for PyTorch. |
| **Keywords** | Reinforcement Learning, `PyTorch`, Feed Forward Neural Network, Value Iteration, Loss Function, Exploration, Exploitation, `gym` |
| **Results** | The Advantage Actor Critic (A2C) method with an entropy bonus is a reinforcement learning algorithm that combines elements of policy-based and value-based methods. It uses an actor network to estimate the policy, determining the probabilities of selecting different actions based on the current state, and a critic network to estimate the state value function. The advantage function, which represents the advantage of taking a specific action in a given state compared to the average value of that state, is also calculated. The entropy bonus, which encourages exploration by "rewarding" the actor for being unsure about a certain action, is added to the actor's loss function to increase the policy's stochasticity. The flow diagram below shows the training process without entropy bonus. Additionally, you can see the training process for two runs, during the second of which the agent's performance inexplicably plummeted, which I found quite surprising. |
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/18%20-%20RL%20A2C/Cartpole%20A2C%20with%20entropy%20bonus.ipynb) |

<p align="center">
  <img src="18 - RL A2C/A2C Visualisation.png" width="750" />
  <img src="18 - RL A2C/training_process.png" width="375" />
  <img src="18 - RL A2C/Instability A2C.png" width="375" />
</p>

***

### Week 19: Visualising Rent Prices in Bern, Switzerland 🏠💶

| | Description |
| :----------- | :----------- |
| **Data Source** | Websites of two major Swiss real estate platforms |
| **Goal** | Scrape rent prices for all of Switzerland from the websites using the dynamic web scraping library `Selenium` and the static `rvest` and visualise the results. |
| **Keywords** | Dynamic Web Scraping, `HTML`,  `rvest`, `RSelenium`, Data Cleaning, Data Visualisation |
| **Results** | There are very strong differences in rent prices between Swiss cities and even within Bern itself. |
| **Notebooks** | [R Script](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/19%20-%20Housing%20Bern/Bern.R) |

<p align="center">
  <img src="19 - Housing Bern/SwissRent.png" width="500" />
  <img src="19 - Housing Bern/BernRent.png" width="500" />
</p>

***

### Week 20: Velo Zurich 🚲🇨🇭

| | Description |
| :----------- | :----------- |
| **Data Source** | [Zurich Open Data](https://data.stadt-zuerich.ch/) |
| **Goal** | On my way to work, I noticed bike counting stations in many locations in Bern, which led me on a quest of finding the data that they collect. |
| **Keywords** | Data Visualisation, `ggplot2` |
| **Results** | Unfortunately, I didn't find the data for Bern, but there were years and years of data available from Zurich, so I'm going to share my findings of patterns below. |
| **Notebooks** | [R Script](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/20%20-%20Velo%20Zurich/Velo.R) |

<p align="center">
  <img src="20 - Velo Zurich/velo_barometer.jpg" width="250" />
  <img src="20 - Velo Zurich/week_dist.png" width="500" />
  <img src="20 - Velo Zurich/hour_dist.png" width="500" />
  <img src="20 - Velo Zurich/hour_distnormalised.png" width="500" />
  <img src="20 - Velo Zurich/weekday_dist.png" width="500" />
</p>

***

### Week 21: Visualising Entry-Level Motorcycle Prices 

| | Description |
| :----------- | :----------- |
| **Data Source** | Swiss Used Motorcycle Platform |
| **Goal** | This week's project happened on the basis of some private research on used motorcycle prices. In order to compare the models of interest, I want to scrape the data from a Swiss online platform and compare prices. |
| **Keywords** | Web Scraping, Data Visualisation, `Timeouts`, HTTP Requests, `HTML` |
| **Results** | In the chart below, the price premium for the well-known MT07 is clearly discernible. Of course, older bikes and bikes with more mileage get lower prices. Little fun fact: The final model that was bought was a 2021 CB 500 F in black. |
| **Notebooks** | [R Script]() |

<p align="center">
  <img src="21 - Motorcycles/moto_viz.png" width="500" />
</p>

***

### Week 22: Pedestrians in Bahnhofstrasse Zurich 🚶🚶‍♀️

| | Description |
| :----------- | :----------- |
| **Data Source** | [Open Data Swiss](https://opendata.swiss/de/dataset?q=fussg%C3%A4nger) |
| **Goal** | The goal of this analysis is to understand pedestrian traffic patterns along Zurich's Bahnhofstrasse. I aim to examine various factors influencing pedestrian behavior, such as weather conditions, time of day, day of the week, and even seasonal variations like the Christmas season. By aggregating and analyzing pedestrian counts from open data sources, I seek insights into how external factors impact pedestrian flow. |
| **Keywords** | Pedestrian frequency, Zurich, Bahnhofstrasse, Urban analysis, Traffic patterns, Data visualization, `plotnine`, `pandas` |
| **Results** | Clearly, pedestrian traffic is driven by two main factors: Commuters and shoppers. Commuters make up the bigger part and are invariant to bad weather. The traffic throughout the week is relatively constant, with higher levels on a Friday. As shops are closed on Sundays, the street experiences less than half of traffic compared to regular days. Saturdays are most crowded due to shoppers. Additionally, shops stay open longer and the street remains crowded for longer into the evening on Saturdays. |
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/22%20-%20Bahnhofstrasse%20Pedestrians/pedestrians.ipynb) |

<p align="center">
  <img src="22 - Bahnhofstrasse Pedestrians/boxplot.png" width="500" />
  <img src="22 - Bahnhofstrasse Pedestrians/heatmap.png" width="500" />
</p>

***

### Week 23: Swiss-End User Electricity Consumption

| | Description |
| :----------- | :----------- |
| **Data Source** | Swissgrid |
| **Goal** | The goal for this week is very simple: I need a new LinkedIn header, so I just want to make a visually appealing chart without regard for technical details. |
| **Keywords** | `plotnine`, `ggplot2`  |
| **Results** | The picture below shows the end-user electricity consumption of Switzerland for each day in the year (1-365) over the past decade with a colourful distinction between workdays, Saturdays and Sundays. But: The only thing that really counts for the banner is the artistic touch. :-) |
| **Notebooks** | [RMarkdown](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/23%20-%20Swiss%20End%20User%20Electricity%20Consumption/Swiss%20End-User%20Electricity.Rmd) |

<p align="center">
  <img src="23 - Swiss End User Electricity Consumption/LinkedIn Banner.png" width="500" />
</p>

***

### Week 24: Scraping Raw Smartphone Specifications and Prices 📱

| | Description |
| :----------- | :----------- |
| **Data Source** | Largest Swiss Tech Retailer |
| **Goal** | The goal of this week is to systematically collect smartphone specifications and historical prices from the website of the biggest tech retailer in Switzerland. I want to have a comprehensive dataset that can be used for market analysis, product comparisons, or any other insights related to current smartphone offerings in 2023. Especially the historic chart on the website (shown below) is interesting to me, because I want to analyse fluctuations in the prices and depreciation patterns of various smartphone providers. |
| **Keywords** | Web Scraping, `Selenium`, `Pandas`, Chrome WebDriver, `JSON` Parsing, Randomized Delays, `HTTP` |
| **Results** | The final notebook gets all devices in the smartphone category and extracts details like specifications and price history. The end result is a set of CSV files on price history and specifications of 1887 smartphones with over 100 variables, which will be analysed another week. |
| **Notebooks** | [Python](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/24%20-%20Digitec/Price%20Scraping.ipynb) |

<p align="center">
  <img src="24 - Digitec/article.png" width="500" />
</p>

***

### Week 25: PubliBike Visualisations 🚲

| | Description |
| :----------- | :----------- |
| **Data Source** | PubliBike API |
| **Goal** | The goal for this week was to use the PubliBike API to retrieve information about the position of the shared bicycles at regular intervals and analyse the movement of them in the city. In order to do so, I wrote a script that regularly fetches data and stores it in a csv file. I then ran it for a whole month and analysed the data. |
| **Keywords** | `API`, `requests`, data wrangling, data visualisation, `plotnine`, `plotly`, `animation`, `spatial data` |
| **Results** | The final visualisations can be seen below for the cities of Zurich, Bern and Fribourg. They show the distribution of bikes across their stations by time of day. One interesting finding: university students are among the biggest users, and many people rely on these bikes for their daily commute to the train station after work. |
| **Notebooks** | [Data Miner](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/25%20-%20Publibike/miner.py), [Visualisations](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/25%20-%20Publibike/publibike.ipynb)  |

<p align="center">
  <img src="25 - Publibike/test_zürich.gif" width="250" />
  <img src="25 - Publibike/test_bern.gif" width="250" />
  <img src="25 - Publibike/test_fribourg.gif" width="250" />
</p>

***

### Week 26: Predicting Day-Ahead Active Losses in the Transmission Grid ⚡

| | Description |
| :----------- | :----------- |
| **Data Source** | SwissGrid |
| **Keywords** | Hackathon, `Python`, `Optuna`, `elastic net`, time series |
| **Results** | I participated in Swissgrid's challenge of predicting day-ahead active losses in the transmission grid at the Energy Data Hackdays 2023 and managed to win the challenge out of 20 participants together with my colleague with an Elastic Net model finetuned with Optuna in Python. |
| **Notebooks** | [Best Model](https://github.com/MathiasSteilen/52-Weeks-of-Python-and-R-2023/blob/main/26%20-%20Swissgrid%20Time%20Series%20Optuna/1_ElasticNet.ipynb) |

<p align="center">
  <img src="26 - Swissgrid Time Series Optuna/Split.png" width="500" />
</p>

***
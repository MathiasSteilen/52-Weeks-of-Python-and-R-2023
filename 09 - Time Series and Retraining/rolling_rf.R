# Packages and Admin
library(tidyverse)
library(lubridate)
library(doParallel)
library(broom)
library(zoo)
library(lubridate)
library(tidymodels)

# Read and modify data
dt <- read_csv("C:/Users/mathi/OneDrive/Python/52-Weeks-of-Python-and-R-2023/09 - Time Series and Retraining/day_ahead.csv")

data <- dt %>% 
  mutate(
    year = year(datetime),
    month = month(datetime),
    dow = wday(datetime, week_start = 1),
    hour = hour(datetime),
    month = lubridate::month(datetime),
    year_half = lubridate::semester(datetime) %>% as.factor,
    week_day = lubridate::wday(datetime),
    week_in_year = lubridate::week(datetime),
    quarter = lubridate::quarter(datetime),
    doy = lubridate::yday(datetime),
    dom = lubridate::mday(datetime),
    date_christmas = ifelse(between(doy, 358, 366), 1, 0)
  )

# Include lags up to n
for (i in 25:48){
  var_name <- paste0("price_lag", i)
  data[,var_name] <- lag(data$price, i)
}
rm(i)

# Tibble to store rolling predictions
predictions <- data %>% 
  sample_n(0) %>%  
  mutate(.pred = numeric())

# Function for parallel processing
unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

# Loop to retrain every day
for (i in 1:431){
  
  # Three years lookback
  dt_train <- data %>% 
    filter(ymd(substr(datetime, 1, 10)) >= (as.Date("2021-12-31") - 365*3 + i - 1),
           ymd(substr(datetime, 1, 10)) < (as.Date("2021-12-31") + i - 1),)
  
  # Make Predictions on the next day
  dt_test <- data %>% 
    filter(ymd(substr(datetime, 1, 10)) == (as.Date("2021-12-31") + i))
  
  # Recipe
  rf_rec <- recipe(price ~ ., data = dt_train) %>%
    step_naomit(price) %>% 
    step_date(datetime) %>% 
    step_holiday(datetime, holidays = timeDate::listHolidays("CH")) %>%
    step_impute_mean(all_numeric_predictors()) %>% 
    step_rm(day)
  
  # rf_rec %>% prep %>% juice %>% glimpse()
  
  # Retrain on first day and on desired interval
  if (i %% 1 == 0 | i == 1){
    cl <- makePSOCKcluster(7)
    registerDoParallel(cl)
    
    rf_fit <- workflow() %>% 
      add_recipe(rf_rec) %>% 
      add_model(rand_forest() %>% 
                  set_mode("regression") %>% 
                  set_engine("ranger", importance = "permutation")) %>% 
      fit(dt_train)
    
    stopCluster(cl)
    unregister_dopar()
  }
  
  # Make predictions
  predictions <- predictions %>% 
    bind_rows(
      rf_fit %>% 
        augment(dt_test)
    )
  
  # Verbose
  print(paste("Day", i, "done"))
}

# Save predictions
write_csv(predictions, "C:/Users/mathi/OneDrive/Python/52-Weeks-of-Python-and-R-2023/09 - Time Series and Retraining/Predictions/preds_rf_retrain1.csv")

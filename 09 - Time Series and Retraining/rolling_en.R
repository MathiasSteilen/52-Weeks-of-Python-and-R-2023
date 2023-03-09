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
    day = wday(datetime, week_start = 1),
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
  
  # Training: all years lookback
  dt_train <- data %>% 
    filter(
      # ymd(substr(datetime, 1, 10)) >= (as.Date("2021-12-31") - 365*5 + i - 1),
      ymd(substr(datetime, 1, 10)) < (as.Date("2021-12-31") + i - 1)
    )
  
  # Make Predictions on the next day
  dt_test <- data %>% 
    filter(ymd(substr(datetime, 1, 10)) == (as.Date("2021-12-31") + i))
  
  # Recipe
  en_rec <- recipe(price ~ ., data = dt_train) %>%
    step_naomit(price) %>% 
    step_date(datetime) %>% 
    step_holiday(datetime, holidays = timeDate::listHolidays("CH")) %>%
    step_rm(datetime) %>% 
    step_impute_mean(all_numeric_predictors()) %>% 
    step_normalize(all_numeric_predictors()) %>% 
    step_dummy(all_nominal_predictors()) %>% 
    step_zv(all_predictors())
  
  
  
  # Retrain every day
  if (i %% 1 == 0 | i == 1){
    cl <- makePSOCKcluster(6)
    registerDoParallel(cl)
    
    en_fit <- workflow() %>% 
      add_recipe(en_rec) %>% 
      add_model(linear_reg(mixture = 0.5, 
                           penalty = 0.02) %>% 
                  set_engine("glmnet")) %>% 
      fit(dt_train)
    
    stopCluster(cl)
    unregister_dopar()
  }
  
  # Make predictions
  predictions <- predictions %>% 
    bind_rows(
      en_fit %>% 
        augment(dt_test)
    )
  
  # Verbose
  print(paste("Day", i, "done"))
}

# Save predictions
write_csv(predictions, "C:/Users/mathi/OneDrive/Python/52-Weeks-of-Python-and-R-2023/09 - Time Series and Retraining/Predictions/preds_en_retrain1.csv")

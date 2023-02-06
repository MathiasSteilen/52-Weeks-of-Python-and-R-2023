# Libraries
library(tidyverse)
library(tidymodels)
library(doParallel)
library(vip)
library(tidytext)
library(lubridate)
library(broom)
library(scales)

# Working Directory To Source File Location
setwd(dirname(rstudioapi::getActiveDocumentContext()[[2]]))

# Chart Theme
theme_set(theme_bw() +
            theme(plot.title = element_text(size = 14, face = "bold"),
                  plot.subtitle = element_text(size = 10, face = "italic",
                                               colour = "grey50")))

# Loading Data
data <- read_csv("rental_df_dirty.csv")

# Monthly rental price by postcode
data %>% 
  drop_na(monthly_rental_price, living_area) %>% 
  mutate(price_sqm = monthly_rental_price/living_area) %>% 
  filter(postcode %in% (data %>% 
                          drop_na(postcode) %>% 
                          count(postcode, sort = T) %>% 
                          head(20) %>% 
                          pull(postcode))) %>% 
  add_count(postcode) %>% 
  mutate(postcode = paste0(postcode, " (n=", n, ")")) %>% 
  ggplot(aes(y = postcode %>% 
               as.factor() %>%
               fct_reorder(price_sqm),
             x = price_sqm)) +
  geom_boxplot(outlier.color = NA) +
  labs(title = "Rent in Antwerpen: Price per Square Metre",
       subtitle = "20 most frequent postcodes shown. Number of listings in brackets.",
       y = "Postcode",
       x = "Price per Square Metre") +
  scale_x_continuous(labels = comma_format(suffix = "€")) +
  coord_cartesian(xlim = c(0, 25))

# Training a model
dt_split <- data %>% 
  drop_na(monthly_rental_price) %>% 
  mutate(across(where(is.character), as.factor),
         monthly_rental_price = log(monthly_rental_price)) %>%
  select(-c(monthly_costs)) %>% 
  initial_split()

dt_train <- training(dt_split)
dt_test <- testing(dt_split)

folds <- vfold_cv(dt_train, v = 5)

rf_fit <- workflow() %>% 
  add_recipe(
    recipe(monthly_rental_price ~ ., data = dt_train) %>% 
      step_impute_knn(all_predictors()) %>% 
      step_other(all_nominal_predictors(), threshold = 0.01)
  ) %>% 
  add_model(
    rand_forest() %>% 
      set_mode("regression") %>% 
      set_engine("ranger", importance = "impurity")
  ) %>% 
  fit(dt_train)

rf_fit %>% 
  extract_fit_parsnip() %>% 
  vip::vi() %>% 
  ggplot(aes(x = Importance, 
             y = Variable %>% fct_reorder(Importance))) +
  geom_col()

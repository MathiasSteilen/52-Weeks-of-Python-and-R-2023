# Libraries
library(tidytext)
library(ggridges)
library(tidyverse)
library(tidymodels)
library(doParallel)
library(vip)
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
dt_train <- read_csv("train.csv")
dt_test <- read_csv("test.csv")
folds <- vfold_cv(dt_train, v = 3)

# One chart
dt_train %>% 
  select(where(is.character), charges) %>% 
  pivot_longer(-charges) %>% 
  ggplot(aes(charges, 
             value %>% reorder_within(charges, name))) +
  geom_boxplot() +
  facet_wrap(~ name, scales = "free") +
  scale_y_reordered()

dt_train %>% 
  select(where(is.character), charges) %>% 
  pivot_longer(-charges) %>% 
  mutate(charges = charges/1000) %>% 
  ggplot(aes(charges, 
             value %>% reorder_within(charges, name),
             fill = name)) +
  geom_density_ridges(scale = 1.5, alpha = 0.5, size = 0.25) +
  facet_wrap(~ name, scales = "free") +
  scale_y_reordered() +
  scale_x_continuous(labels = dollar_format(suffix = "k")) +
  ggsci::scale_fill_futurama() +
  labs(title = "Density of Charges by Nominal Predictor Value",
       y = NULL, x = "Charges") +
  theme(legend.position = "none",
        strip.background = element_rect(fill = alpha("black", 0.1)),
        strip.text = element_text(face = "bold", size = 10))

ggsave(file = "ridges.png", dpi = 350, width = 6, height = 3)

# Making the models
xg_wflow <- workflow() %>% 
  add_model(
    boost_tree(trees = tune(),
               tree_depth = tune(),
               min_n = tune(),
               loss_reduction = tune(),
               sample_size = tune(),
               mtry = tune(),
               learn_rate = tune()) %>%
      set_engine("xgboost", importance = "impurity") %>%
      set_mode("regression")
  ) %>% 
  add_recipe(
    recipe(charges ~ ., data = dt_train) %>% 
      step_novel(all_nominal_predictors()) %>% 
      step_normalize(all_numeric_predictors()) %>% 
      step_dummy(all_nominal_predictors(), one_hot = TRUE),
    blueprint = hardhat::default_recipe_blueprint(allow_novel_levels = TRUE)
  )

lgbm_wflow <- workflow() %>% 
  add_model(
    boost_tree(trees = tune(),
               tree_depth = tune(),
               min_n = tune(),
               loss_reduction = tune(),
               sample_size = tune(),
               mtry = tune(),
               learn_rate = tune()) %>%
      set_engine("lightgbm") %>%
      set_mode("regression")
  ) %>% 
  add_recipe(
    recipe(charges ~ ., data = dt_train) %>% 
      step_novel(all_nominal_predictors()) %>% 
      step_normalize(all_numeric_predictors()) %>% 
      step_dummy(all_nominal_predictors(), one_hot = TRUE),
    blueprint = hardhat::default_recipe_blueprint(allow_novel_levels = TRUE)
  )

# Hyperparameter Tuning XGBoost
start_time <- Sys.time()
unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

cl <- makePSOCKcluster(7)
registerDoParallel(cl)

xg_tune <- tune_grid(object = xg_wflow,
                     resamples = folds,
                     grid = grid_latin_hypercube(
                       trees(),
                       tree_depth(),
                       min_n(),
                       loss_reduction(),
                       sample_size = sample_prop(),
                       finalize(mtry(), dt_train),
                       learn_rate(),
                       size = 100
                     ))

stopCluster(cl)
unregister_dopar()
Sys.time() - start_time

xg_tune %>% 
  show_best(metric = "rsq") %>% 
  select(.metric, mean, std_err, n)

# Hyperparameter Tuning LGBM
start_time <- Sys.time()
unregister_dopar <- function() {
  env <- foreach:::.foreachGlobals
  rm(list=ls(name=env), pos=env)
}

cl <- makePSOCKcluster(7)
registerDoParallel(cl)

lgbm_tune <- tune_grid(object = lgbm_wflow,
                       resamples = folds,
                       grid = grid_latin_hypercube(
                         trees(),
                         tree_depth(),
                         min_n(),
                         loss_reduction(),
                         sample_size = sample_prop(),
                         finalize(mtry(), dt_train),
                         learn_rate(),
                         size = 100
                       ))

stopCluster(cl)
unregister_dopar()
Sys.time() - start_time

lgbm_tune %>% 
  show_best(metric = "rsq") %>% 
  select(.metric, mean, std_err, n)

# Fit the best models
xg_fit <- xg_wflow %>% 
  finalize_workflow(select_best(xg_tune, metric = "rsq")) %>% 
  fit(dt_train)

lgbm_fit <- lgbm_wflow %>% 
  finalize_workflow(select_best(lgbm_tune, metric="rsq")) %>% 
  fit(dt_train)

# Evaluate model performance
eval_metrics <- metric_set(rsq, mae, mape)

xg_fit %>% 
  augment(dt_test) %>% 
  eval_metrics(truth = charges, estimate = .pred)

lgbm_fit %>% 
  augment(dt_test) %>% 
  eval_metrics(truth = charges, estimate = .pred)

bind_rows(
  xg_fit %>% 
    augment(dt_test) %>% 
    mutate(model = "XGBoost"),
  lgbm_fit %>% 
    augment(dt_test) %>% 
    mutate(model = "LGBM")
) %>% 
  ggplot(aes(x = charges, y = .pred)) +
  geom_point(alpha = 0.25, colour = "midnightblue") +
  geom_abline() +
  labs(title = "R: Predicted vs. Actuals",
       x = "Actuals", y = "Predictions") +
  facet_wrap(~ model) +
  scale_y_continuous(labels = dollar_format()) +
  scale_x_continuous(labels = dollar_format()) +
  theme_bw() +
  theme(axis.text = element_text(size = 7))

ggsave(file = "R.png", dpi = 350, width = 4, height = 4)

# Libraries
library(tidyverse)
library(tidymodels)
library(textrecipes)
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
data <- read_csv("watches_clean.csv")

# EDA ----

# Preisverteilung nach Marke
data %>%
  group_by(marke) %>% 
  summarise(lower = quantile(price, 0.1, na.rm = T),
            higher = quantile(price, 0.9, na.rm = T),
            median = median(price, na.rm = T)) %>% 
  ggplot(aes(x = marke %>% fct_reorder(-median))) +
  geom_point(aes(y = median),
             shape = "square", size = 2) +
  geom_segment(aes(xend = marke %>% fct_reorder(-median),
                   y = lower, yend = higher),
               alpha = 0.6) +
  labs(title = "Luxury Watch Prices by Brand",
       subtitle = "Source: Chrono24.at, February 2023 | N = 45,520\nSquare represents median, errorbars show 10th to 90th percentiles.",
       y = "EUR (Log scale)", x = "Brand", colour = NULL) +
  scale_y_log10(labels = comma_format()) +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5,
                                   size = 7.5),
        legend.position = "bottom",
        plot.background = element_rect(fill = "grey98"),
        panel.background = element_blank(),
        legend.background = element_blank(),
        legend.margin = margin(-0.25, 0, 0, 0, unit = "cm"),
        legend.key = element_blank(),
        plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        panel.grid.minor.y = element_blank())

ggsave(filename = "r.png", dpi = 300, width = 6, height = 4)

# Preisverteilung von Modellen
plot_models <- function(tbl, brand){
  
  tbl %>% 
    filter(marke == brand) %>% 
    add_count(modell) %>% 
    filter(n > 30) %>% 
    mutate(modell = paste0(modell, " (N=", n, ")")) %>% 
    drop_na(modell, price) %>% 
    group_by(modell) %>% 
    summarise(lower = quantile(price, 0.1, na.rm = T),
              higher = quantile(price, 0.9, na.rm = T),
              median = median(price, na.rm = T)) %>% 
    ggplot(aes(y = modell %>% fct_reorder(-median))) +
    geom_point(aes(x = median),
               shape = "square", size = 2) +
    geom_segment(aes(yend = modell %>% fct_reorder(-median),
                     x = lower, xend = higher),
                 alpha = 0.6) +
    labs(title = paste0("Luxury Watch Prices: ", brand, " Models"),
         subtitle = "Source: Chrono24.at, February 2023 | N = 45,520\nSquare represents median, errorbars show 10th to 90th percentiles.",
         x = "Euro", y = NULL) +
    scale_x_continuous(labels = comma_format()) +
    theme(legend.position = "bottom",
          plot.background = element_rect(fill = "grey98"),
          panel.background = element_blank(),
          legend.background = element_blank(),
          legend.margin = margin(-0.25, 0, 0, 0, unit = "cm"),
          legend.key = element_blank(),
          plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
          panel.grid.minor.y = element_blank())
  
}

plot_models(data, "Rolex")
plot_models(data, "Omega")
plot_models(data, "Patek Philippe")

# Training a model ----
dt_split <- data %>%
  drop_na(price) %>% 
  select(-c(response_time, inseratscode, 
            referenznummer, code_des_handlers)) %>%
  initial_split()

dt_train <- training(dt_split)
dt_test <- testing(dt_split)

# Recipe
model_rec <- recipe(price ~ ., data = dt_train) %>%
  step_novel(all_nominal_predictors()) %>% 
  step_impute_median(all_numeric_predictors()) %>% 
  step_unknown(all_nominal_predictors()) %>% 
  step_tokenize(header) %>%
  step_stopwords(header) %>%
  step_tokenfilter(header, max_tokens = 100) %>%
  step_tf(header) %>%
  step_other(all_nominal_predictors(), threshold = 0.01) %>% 
  # step_normalize(all_numeric_predictors()) %>% 
  step_dummy(all_nominal_predictors())
# prep() %>% juice() %>% glimpse()

# Specification
gb_spec <- boost_tree() %>% 
  set_engine("xgboost") %>% 
  set_mode("regression")

# Workflow
gb_wflow <- workflow() %>% 
  add_recipe(
    model_rec,
    blueprint = hardhat::default_recipe_blueprint(allow_novel_levels = T)
  ) %>% 
  add_model(gb_spec)

gb_fit <- gb_wflow %>% 
  fit(dt_train)

# Check OOS fit
eval_metrics <- metric_set(rsq, mae, mape)

gb_fit %>% 
  augment(dt_test) %>% 
  # eval_metrics(truth = price, estimate = .pred)
  ggplot(aes(price, 
             .pred)) +
  geom_point(alpha = 0.3) +
  geom_abline()
  # facet_wrap(~ marke, scales = "free")

# Conclusion: Model doesn't work very well, try focusing on EDA
# and on learning how to do tokenisation in sklearn,
# potentially even using a neural net?



# Libraries
library(tidyverse)
library(tidymodels)
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
df <- read_csv("../../Data too Large to Push/bus-breakdowns-delays.csv") %>% 
  janitor::clean_names()

data <- df %>%
  mutate(contains_min = str_detect(tolower(how_long_delayed), "min")) %>% 
  filter(contains_min) %>% 
  mutate(
    across(c(school_year, how_long_delayed), ~ parse_number(.x)),
    across(c(occurred_on, created_on, informed_on, last_updated_on),
           ~ mdy_hms(.x))
  )

# Which breakdown reasons occur most frequently at which times 
# throughout the week?

glimpse(data)

data %>% count(reason, sort = T)

# Average delay by reason of delay
data %>% 
  group_by(reason) %>% 
  summarise(mean_delay = mean(how_long_delayed, na.rm = T)) %>% 
  ggplot(aes(mean_delay, 
             reason %>% reorder(mean_delay))) +
  geom_col(fill = "midnightblue", alpha = 0.6) +
  labs(title = "Accidents are terrible news for arriving on time...\nWho could have guessed?",
       y = NULL, x = "Average Delay")
# Libraries
library(tidyverse)
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

# WEATHER --------------------------------------------------------------------

# Loading Data
weather <- bind_rows(
  read_csv("Weather/Findel Daily Weather since 1947.csv") %>% 
    janitor::clean_names() %>% 
    transmute(date = dmy(date),
              precipitation = precipitation_mm,
              location = "LU"),
  read_csv("Weather/weather_data.csv") %>% 
    transmute(date = dmy(date), 
              precipitation = STG_precipitation,
              location = "CH")
)

# Monthly amount of precipitation compared by month and location
weather %>% 
  filter(year(date) >= 2000,
         year(date) <= 2020) %>% 
  mutate(month = month(date)) %>% 
  group_by(location, month) %>% 
  summarise(precipitation = sum(precipitation, na.rm = T)) %>% 
  ggplot(aes(x = factor(month), y = precipitation, fill = location)) +
  geom_col(position = "dodge") +
  labs(title = "Total Monthly Precipitation from 2000 to 2020",
       subtitle = "CH: St. Gallen, LU: Findel",
       x = "Month", y = "Sum of Precipitation", fill = NULL) +
  scale_y_continuous(labels = comma_format(suffix = " mm")) +
  ggsci::scale_fill_jama()

# Rainy days by year over time
weather %>% 
  filter(year(date) >= 2000,
         year(date) <= 2020) %>% 
  mutate(year = year(date),
         rainy_day = ifelse(precipitation >= 2.5, 1, 0)) %>% 
  group_by(year, location) %>% 
  summarise(rainy_days = sum(rainy_day)) %>% 
  ggplot(aes(year, rainy_days, colour = location)) +
  geom_point() +
  geom_line() +
  labs(title = "Rainy Days (>2.5mm) for each Year",
       subtitle = "CH: St. Gallen, LU: Findel",
       x = NULL, y = "Days", colour = NULL) +
  ggsci::scale_colour_jama()

# Zero rain days by year
weather %>% 
  filter(year(date) >= 2000,
         year(date) <= 2020) %>% 
  mutate(year = year(date),
         not_rainy_day = ifelse(precipitation == 0, 1, 0)) %>% 
  group_by(year, location) %>% 
  summarise(not_rainy_days = sum(not_rainy_day)) %>% 
  ggplot(aes(year, not_rainy_days, colour = location)) +
  geom_point() +
  geom_line() +
  labs(title = "Zero Rain Days for each Year",
       subtitle = "CH: St. Gallen, LU: Findel",
       x = NULL, y = "Days", colour = NULL) +
  ggsci::scale_colour_jama()


# Amount of rain by year over time
weather %>% 
  group_by(year = year(date), location) %>% 
  summarise(precipitation = sum(precipitation)) %>% 
  ggplot(aes(year, precipitation, colour = location)) +
  geom_line() +
  geom_point() +
  labs(title = "Total Precipitation by Year over Time",
       subtitle = "CH: St. Gallen, LU: Findel",
       y = "Total Precipitation", x = "Year", colour = NULL) +
  scale_y_continuous(labels = comma_format(suffix = " mm")) +
  ggsci::scale_colour_jama()

# How where the last 3 years compared to historical data?
weather %>% 
  group_by(year = year(date), location) %>% 
  summarise(precipitation = sum(precipitation)) %>% 
  ungroup() %>% 
  group_by(location) %>% 
  mutate(ntile = ntile(precipitation, n = 100)) %>% 
  filter(between(year, 2019, 2021)) %>% 
  ggplot(aes(factor(year), ntile, fill = location)) +
  geom_col(position = "dodge") +
  labs(title = "Percentile of Precipitation for Last 3 Years",
       subtitle = "CH: St. Gallen, LU: Findel", 
       y = "Percentile Compared to Historical Data",
       x = NULL, fill = NULL) +
  ggsci::scale_fill_jama()


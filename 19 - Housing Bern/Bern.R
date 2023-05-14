# Libraries
library(tidyverse)
library(stringr)
library(tidytext)
library(lubridate)
library(broom)
library(scales)
library(patchwork)

# Working Directory To Source File Location
setwd(dirname(rstudioapi::getActiveDocumentContext()[[2]]))

# Chart Theme
theme_set(theme_bw() +
            theme(plot.title = element_text(size = 15, face = "bold"),
                  plot.subtitle = element_text(size = 10, face = "italic",
                                               colour = "grey50")))

# Loading Data
data <- read_csv("Homegate_scrape_clean_2023-05-06.csv") %>% 
  as_tibble() %>% 
  janitor::clean_names() %>% 
  mutate(price_sqm = price/sq_m) %>% 
  select(-x1)

data <- read_csv("apartments_clean.csv") %>% 
  as_tibble() %>% 
  janitor::clean_names() %>% 
  mutate(price_sqm = price_chf/space_sqm,
         listing_type = "rent") %>% 
  rename(price = price_chf, zip_code = postcode, nr_rooms = rooms)

data %>% glimpse()

data %>% count(listing_type)

data %>% 
  summarise(across(everything(), ~ sum(is.na(.x)) / n())) %>% 
  pivot_longer(everything(), names_to = "column", values_to = "missing") %>% 
  arrange(-missing)

# Wohnungspreise Bern für verschiedene Raumgrössen
data %>% 
  filter(between(zip_code, 3000, 3100)) %>% 
  filter(listing_type == "rent") %>% 
  filter(between(nr_rooms, 3.5, 4.5)) %>% 
  add_count(zip_code) %>% 
  filter(n > 5) %>% 
  group_by(zip_code) %>% 
  summarise(lower = quantile(price_sqm, 0.1, na.rm = T),
            median = median(price_sqm, 0.5, na.rm = T),
            higher = quantile(price_sqm, 0.9, na.rm = T),
            n = n()) %>% 
  ungroup() %>% 
  ggplot(aes(x = zip_code %>% as.factor)) +
  geom_point(aes(y = median),
             shape = "square", size = 2) +
  geom_segment(aes(xend = zip_code %>% as.factor,
                   y = lower, yend = higher),
               alpha = 0.6) +
  labs(title = "Rent Prices in Bern for 3.5 to 4.5 Rooms",
       subtitle = "Source: Anibis, May 2022 | N = 489\nOnly postal codes with more than 5 observations are considered for calculation.\nSquare representing median, errorbars showing 10th to 90th percentile.",
       y = bquote("CHF per" ~ m^2), x = "Postal Code", colour = NULL) +
  coord_cartesian(ylim = c(10, 35)) +
  scale_y_continuous(labels = comma_format(),
                     breaks = seq(10, 105, 10)) +
  scale_colour_manual(values = c("#0073C2FF", "#EFC000FF", "#CD534CFF",
                                 "#7AA6DCFF", "#003C67FF", "#8F7700FF",
                                 "#3B3B3BFF", "#4A6990FF")) +
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
  
ggsave("BernRent.png", dpi = 350, width = 7, height = 4.5)
  

# Price per square metre across major cities
# 10th to 90th percentile by zip code for the cities:
data %>% 
  filter(listing_type == "rent") %>% 
  separate(location, into = c("street", "location"), sep = ", ") %>% 
  drop_na(location) %>% 
  mutate(city = case_when(
    grepl("Zürich", location) ~ "Zürich",
    grepl("Genève", location) ~ "Genf",
    grepl("Basel", location) ~ "Basel",
    grepl("Lausanne", location) ~ "Lausanne",
    grepl("Bern", location) ~ "Bern",
    grepl("Winterthur", location) ~ "Winterthur",
    grepl("St. Gallen", location) ~ "St. Gallen",
    grepl("Baden", location) ~ "Baden"
  )) %>% 
  filter(city %in% c("Zürich",
                     "Genf",
                     "Basel",
                     "Lausanne",
                     "St. Gallen",
                     "Bern",
                     "Baden",
                     "Winterthur")) %>% 
  add_count(zip_code, name = "zip_count") %>% 
  filter(zip_count > 15) %>% 
  filter(!zip_code %in% c(9012, 1233)) %>% 
  group_by(zip_code) %>% 
  summarise(city = unique(city),
            lower = quantile(price_sqm, 0.1, na.rm = T),
            median = median(price_sqm, 0.5, na.rm = T),
            higher = quantile(price_sqm, 0.9, na.rm = T)) %>%  
  ggplot(aes(x = zip_code %>% as.factor)) +
  geom_point(aes(y = median, colour = city),
             shape = "square", size = 2) +
  geom_segment(aes(xend = zip_code %>% as.factor,
                   y = lower, yend = higher, 
                   colour = city),
               alpha = 0.6) +
  labs(title = "Rent Prices in Swiss Cities",
       subtitle = "Source: Homegate, November 2022 | N = 2,548\nOnly postal codes with more than 15 observations are considered for calculation.\nSquare representing median, errorbars showing 10th to 90th percentile.",
       y = bquote("CHF per" ~ m^2), x = "Postal Code", colour = NULL) +
  coord_cartesian(ylim = c(10, 100)) +
  scale_y_continuous(labels = comma_format(),
                     breaks = seq(10, 105, 10)) +
  scale_colour_manual(values = c("#0073C2FF", "#EFC000FF", "#CD534CFF",
                                 "#7AA6DCFF", "#003C67FF", "#8F7700FF",
                                 "#3B3B3BFF", "#4A6990FF")) +
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

ggsave("SwissRent.png", dpi = 350, width = 7, height = 4.5)
ggsave("SwissRent.pdf", width = 7, height = 4.5)

# Price per squatre metre by canton for each room number
data %>% 
  filter(listing_type == "rent") %>% 
  # filter(canton %in% c("zurich", "stgallen", "bern", "baselstadt")) %>% 
  filter(nr_rooms < 6) %>% 
  drop_na(canton, nr_rooms, price_sqm) %>% 
  group_by(canton, nr_rooms) %>% 
  summarise(canton = unique(canton),
            lower = quantile(price_sqm, 0.1, na.rm = T),
            median = median(price_sqm, 0.5, na.rm = T),
            higher = quantile(price_sqm, 0.9, na.rm = T)) %>% 
  ggplot(aes(x = nr_rooms %>% as.factor)) +
  geom_point(aes(y = median, colour = nr_rooms %>% as.factor),
             shape = "square", size = 2) +
  geom_segment(aes(xend = nr_rooms %>% as.factor,
                   y = lower, yend = higher, 
                   colour = nr_rooms %>% as.factor),
               alpha = 0.6) +
  facet_wrap(~ canton, scales = "free_y") +
  labs(title = "Rent Prices in Swiss Cities",
       subtitle = "Source: Homegate, November 2022 | N = 2,548\nOnly postal codes with more than 15 observations are considered for calculation.\nSquare representing median, errorbars showing 10th to 90th percentile.",
       y = bquote("CHF per" ~ m^2), x = "Postal Code", colour = NULL) +
  # coord_cartesian(ylim = c(10, 100)) +
  # scale_y_continuous(labels = comma_format(),
  #                    breaks = seq(10, 105, 10)) +
  # scale_colour_manual(values = c("#0073C2FF", "#EFC000FF", "#CD534CFF",
  #                                "#7AA6DCFF", "#003C67FF", "#8F7700FF",
  #                                "#3B3B3BFF", "#4A6990FF")) +
  theme(legend.position = "none",
        plot.background = element_rect(fill = "grey98"),
        panel.background = element_blank(),
        legend.background = element_blank(),
        legend.margin = margin(-0.25, 0, 0, 0, unit = "cm"),
        legend.key = element_blank(),
        plot.margin = margin(0.5, 0.5, 0.5, 0.5, unit = "cm"),
        panel.grid.minor.y = element_blank())

# Bern (Time series)
load("Homegate Time Series/Homegate_scrape_clean_2022-11-26.RData")
homegate_data_1 <- homegate_data

load("Homegate Time Series/Homegate_scrape_clean_2022-12-03.RData")
homegate_data_2 <- homegate_data

load("Homegate Time Series/Homegate_scrape_clean_2022-12-10.RData")
homegate_data_3 <- homegate_data

load("Homegate Time Series/Homegate_scrape_clean_2022-12-17.RData")
homegate_data_4 <- homegate_data

load("Homegate Time Series/Homegate_scrape_clean_2022-12-24.RData")
homegate_data_5 <- homegate_data

load("Homegate Time Series/Homegate_scrape_clean_2022-12-31.RData")
homegate_data_6 <- homegate_data

data <- bind_rows(
  homegate_data_1,
  homegate_data_2,
  homegate_data_3,
  homegate_data_4,
  homegate_data_5,
  homegate_data_6
) %>% 
  as_tibble() %>% 
  janitor::clean_names() %>% 
  mutate(price_sqm = price/sq_m) %>% 
  distinct(listing_id, .keep_all = T)

data %>% 
  filter(canton == "bern",
         listing_type == "rent") %>%  
  filter(between(substr(zip_code, 1, 4), 3000, 3100),
         between(nr_rooms, 3, 3.5)) %>% 
  mutate(nr_rooms = "3/3.5 Rooms") %>% 
  add_count(zip_code) %>% 
  filter(n > 5) %>% 
  group_by(zip_code, nr_rooms) %>% 
  summarise(value = quantile(price_sqm, c(0.1, 0.5, 0.9), na.rm = T),
            quantiles = c("lower", "median", "higher")) %>% 
  ungroup() %>% 
  pivot_wider(names_from = quantiles, values_from = value) %>% 
  ggplot(aes(x = factor(zip_code))) +
  geom_point(aes(y = median),
             shape = "square", size = 2) +
  geom_segment(aes(xend = factor(zip_code),
                   y = lower, yend = higher),
               alpha = 0.6) +
  labs(title = "Square Metre Prices for 3 and 3.5 room flats in Bern",
       subtitle = "Source: Homegate, November 2022 - December 2022 \nOnly postal codes with more than 5 observations are considered for calculation.\nSquare representing median, errorbars showing 10th to 90th percentile.",
       y = bquote("CHF per" ~ m^2), x = "Postal Code", colour = NULL) +
  coord_cartesian(ylim = c(10, 40)) +
  scale_y_continuous(labels = comma_format(),
                     breaks = seq(10, 105, 10)) +
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


  ggplot(aes(x = price_sqm,
             y = factor(zip_code))) +
  geom_boxplot() +
  facet_wrap(~ nr_rooms, scales = "free_y") +
  scale_y_reordered()
  
ggsave("BernRent3.5.png", dpi = 350, width = 7, height = 4.5)
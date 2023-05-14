library(tidyverse)

setwd(dirname(rstudioapi::getActiveDocumentContext()[[2]]))

data = read_csv("züri_bikes.csv") %>% 
  janitor::clean_names()

# Remove counters that don't count bikes
velo_zaehler = data %>% 
  mutate(velo_na = is.na(velo_in) + is.na(velo_out)) %>% 
  group_by(fk_zaehler) %>% 
  summarise(velo_na = mean(velo_na)) %>% 
  filter(velo_na == 0) %>% 
  pull(fk_zaehler)

data = data %>% 
  filter(fk_zaehler %in% velo_zaehler) %>% 
  group_by(datum, fk_zaehler, ost, nord) %>% 
  summarise(bike_count = sum(velo_in, na.rm = T) + 
                          sum(velo_out, na.rm = T)) %>% 
  ungroup()

# write to csv
data %>% 
  mutate(year = year(datum),
         month = month(datum),
         day = yday(datum),
         wday = wday(datum, week_start = 1)) %>% 
  write_csv("züri_bikes.csv")

data = read_csv("züri_bikes.csv")

data

# Look at relative patterns throughout year, week, day
data %>% 
  filter(year < 2022) %>% 
  group_by(year, week = week(datum)) %>% 
  summarise(bike_count = sum(bike_count, na.rm = T)) %>% 
  ungroup() %>% 
  group_by(week) %>% 
  summarise(quantile = c("zero", "ten", "twentyfive", "fifty", "seventyfive",
                         "ninety", "hundred"),
            bike_count = quantile(bike_count, c(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1),
                                na.rm = T)) %>% 
  pivot_wider(names_from = quantile, values_from = bike_count) %>% 
  ungroup() %>% 
  ggplot(aes(x = week)) +
  # Q25 - Q75
  geom_ribbon(aes(ymin = twentyfive, ymax = seventyfive, fill = "25%-75%")) +
  # Q10 - Q90
  geom_ribbon(aes(ymin = seventyfive, ymax = ninety, fill = "10%-90%")) +
  geom_ribbon(aes(ymin = ten, ymax = twentyfive, fill = "10%-90%")) +
  # Min Max
  geom_ribbon(aes(ymin = zero, ymax = ten, fill = "Min-Max")) +
  geom_ribbon(aes(ymin = ninety, ymax = hundred, fill = "Min-Max")) +
  # Median Line
  geom_line(aes(y = fifty, colour = "Median"), lty = "dotted") +
  # Line for 2022
  # Line for 2022
  geom_line(data = data %>%
              filter(year == 2022) %>% 
              group_by(week = week(datum)) %>% 
              summarise(bike_count = sum(bike_count, na.rm = T)) %>% 
              ungroup(),
            aes(x = week, y = bike_count, colour = "2022")) +
  scale_fill_manual(values = c("25%-75%" = "#87CEEB",
                               "10%-90%" = "#ADD8E6",
                               "Min-Max" = "#CCCCCC")) +
  scale_colour_manual(values = c("Median" = "dodgerblue",
                                 "2022" = "midnightblue")) +
  labs(title = "Yearly Bike Traffic in Zurich by Calendar Week",
       subtitle = "Data from 2015-2022",
       caption = "Data: Open Data Zurich | 14/05/2023",
       y = "Bike Count", x = "Calendar Week",
       fill = NULL, colour = NULL) +
  scale_x_continuous(breaks = c(1, seq(5, 52, 5))) +
  scale_y_continuous(labels = comma_format()) +
  theme(plot.title = element_text(colour = "#3F68E6", size = 14),
        plot.subtitle = element_text(colour = "black", face = "plain",
                                     size = 10),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        plot.margin = margin(15, 15, 15, 15))

ggsave(width = 8, height = 4, dpi = 300,
       filename = "week_dist.png")

# Weekday
data %>% 
  filter(year < 2022) %>% 
  group_by(year, wday) %>% 
  summarise(bike_count = sum(bike_count, na.rm = T)) %>% 
  ungroup() %>% 
  group_by(wday) %>% 
  summarise(quantile = c("zero", "ten", "twentyfive", "fifty", "seventyfive",
                         "ninety", "hundred"),
            bike_count = quantile(bike_count, c(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1),
                                  na.rm = T)) %>% 
  pivot_wider(names_from = quantile, values_from = bike_count) %>% 
  ungroup() %>% 
  ggplot(aes(x = wday)) +
  # Q25 - Q75
  geom_ribbon(aes(ymin = twentyfive, ymax = seventyfive, fill = "25%-75%")) +
  # Q10 - Q90
  geom_ribbon(aes(ymin = seventyfive, ymax = ninety, fill = "10%-90%")) +
  geom_ribbon(aes(ymin = ten, ymax = twentyfive, fill = "10%-90%")) +
  # Min Max
  geom_ribbon(aes(ymin = zero, ymax = ten, fill = "Min-Max")) +
  geom_ribbon(aes(ymin = ninety, ymax = hundred, fill = "Min-Max")) +
  # Median Line
  geom_line(aes(y = fifty, colour = "Median"), lty = "dotted") +
  # Line for 2022
  # Line for 2022
  geom_line(data = data %>%
              filter(year == 2022) %>% 
              group_by(wday) %>% 
              summarise(bike_count = sum(bike_count, na.rm = T)) %>% 
              ungroup(),
            aes(x = wday, y = bike_count, colour = "2022")) +
  scale_fill_manual(values = c("25%-75%" = "#87CEEB",
                               "10%-90%" = "#ADD8E6",
                               "Min-Max" = "#CCCCCC")) +
  scale_colour_manual(values = c("Median" = "dodgerblue",
                                 "2022" = "midnightblue")) +
  labs(title = "Yearly Bike Traffic in Zurich by Weekday",
       subtitle = "Data from 2015-2022",
       caption = "Data: Open Data Zurich | 14/05/2023",
       y = "Bike Count", x = "Weekday",
       fill = NULL, colour = NULL) +
  scale_x_continuous(breaks = c(1, seq(2, 7, 1))) +
  scale_y_continuous(labels = comma_format()) +
  theme(plot.title = element_text(colour = "#3F68E6", size = 14),
        plot.subtitle = element_text(colour = "black", face = "plain",
                                     size = 10),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        plot.margin = margin(15, 15, 15, 15))

ggsave(width = 8, height = 4, dpi = 300,
       filename = "weekday_dist.png")

# Hour in the day
data %>% 
  filter(year < 2022) %>% 
  group_by(year, hour = hour(datum)) %>% 
  summarise(bike_count = sum(bike_count, na.rm = T)) %>% 
  ungroup() %>% 
  group_by(hour) %>% 
  summarise(quantile = c("zero", "ten", "twentyfive", "fifty", "seventyfive",
                         "ninety", "hundred"),
            bike_count = quantile(bike_count, c(0, 0.1, 0.25, 0.5, 0.75, 0.9, 1),
                                  na.rm = T)) %>% 
  pivot_wider(names_from = quantile, values_from = bike_count) %>% 
  ungroup() %>% 
  ggplot(aes(x = hour)) +
  # Q25 - Q75
  geom_ribbon(aes(ymin = twentyfive, ymax = seventyfive, fill = "25%-75%")) +
  # Q10 - Q90
  geom_ribbon(aes(ymin = seventyfive, ymax = ninety, fill = "10%-90%")) +
  geom_ribbon(aes(ymin = ten, ymax = twentyfive, fill = "10%-90%")) +
  # Min Max
  geom_ribbon(aes(ymin = zero, ymax = ten, fill = "Min-Max")) +
  geom_ribbon(aes(ymin = ninety, ymax = hundred, fill = "Min-Max")) +
  # Median Line
  geom_line(aes(y = fifty, colour = "Median"), lty = "dotted") +
  # Line for 2022
  # Line for 2022
  geom_line(data = data %>%
              filter(year == 2022) %>% 
              group_by(hour = hour(datum)) %>% 
              summarise(bike_count = sum(bike_count, na.rm = T)) %>% 
              ungroup(),
            aes(x = hour, y = bike_count, colour = "2022")) +
  scale_fill_manual(values = c("25%-75%" = "#87CEEB",
                               "10%-90%" = "#ADD8E6",
                               "Min-Max" = "#CCCCCC")) +
  scale_colour_manual(values = c("Median" = "dodgerblue",
                                 "2022" = "midnightblue")) +
  labs(title = "Yearly Bike Traffic in Zurich by Hour",
       subtitle = "Data from 2015-2022",
       caption = "Data: Open Data Zurich | 14/05/2023",
       y = "Bike Count", x = "Hour in Day",
       fill = NULL, colour = NULL) +
  scale_x_continuous(breaks = c(0, seq(1, 23, 1))) +
  scale_y_continuous(labels = comma_format()) +
  theme(plot.title = element_text(colour = "#3F68E6", size = 14),
        plot.subtitle = element_text(colour = "black", face = "plain",
                                     size = 10),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        plot.margin = margin(15, 15, 15, 15))

ggsave(width = 8, height = 4, dpi = 300,
       filename = "hour_dist.png")


# Hour in the day normalised
data %>% 
  group_by(year, hour = hour(datum)) %>% 
  summarise(bike_count = sum(bike_count, na.rm = T)) %>% 
  ungroup() %>% 
  group_by(year) %>% 
  mutate(bike_count = bike_count/max(bike_count)) %>% 
  ungroup() %>% 
  group_by(hour) %>% 
  ggplot(aes(x = hour, y = bike_count, colour = year, group = year)) +
  geom_line() +
  labs(title = "Normalised Yearly Bike Traffic in Zurich by Hour",
       subtitle = "Data from 2015-2022",
       caption = "Data: Open Data Zurich | 14/05/2023",
       y = "Relative Frequency", x = "Hour in Day",
       fill = NULL, colour = NULL) +
  scale_x_continuous(breaks = c(0, seq(1, 23, 1))) +
  scale_y_continuous(labels = percent_format()) +
  scale_colour_gradient(low = "midnightblue", high = "dodgerblue") +
  theme(plot.title = element_text(colour = "#3F68E6", size = 14),
        plot.subtitle = element_text(colour = "black", face = "plain",
                                     size = 10),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        plot.margin = margin(15, 15, 15, 15))

ggsave(width = 8, height = 4, dpi = 300,
       filename = "hour_distnormalised.png")

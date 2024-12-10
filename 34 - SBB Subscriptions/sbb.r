library(tidyverse)


tibble(
  raw_spending = 1:10000,
  None = raw_spending,
  Halbtax = 190 + raw_spending/2,
  HalbtaxPlus1000 = case_when(
    raw_spending <= 800*2 ~ 190 + raw_spending/2,
    raw_spending > 800*2 & raw_spending <= 1000*2 ~ 190 + 800,
    raw_spending > 1000*2 ~ 190 - 200 + raw_spending/2
  ),
  HalbtaxPlus2000 = case_when(
    raw_spending <= 1500*2 ~ 190 + raw_spending/2,
    raw_spending > 1500*2 & raw_spending <= 2000*2 ~ 190 + 1500,
    raw_spending > 2000*2 ~ 190 - 500 + raw_spending/2
  ),
  HalbtaxPlus3000 = case_when(
    raw_spending <= 2100*2 ~ 190 + raw_spending/2,
    raw_spending > 2100*2 & raw_spending <= 3000*2 ~ 190 + 2100,
    raw_spending > 3000*2 ~ 190 - 900 + raw_spending/2
  ),
  Generalabonnement = 3995
) |> 
  pivot_longer(-raw_spending) |> 
  ggplot(aes(raw_spending, value, colour = name)) +
  geom_line() +
  labs(
    title = "SBB Subscriptions",
    x = "Theoretical Spending Without Subscription",
    y = "Theoretical Spending With Subscription",
    subtitle = "Adults Aged 26-65",
    colour = "Type"
  ) +
  scale_x_continuous(labels = scales::comma_format(prefix = "CHF ")) +
  scale_y_continuous(labels = scales::comma_format(prefix = "CHF ")) +
  ggsci::scale_colour_jama() +
  theme_light() +
  theme(plot.title = element_text(face = "bold", size = 14),
        plot.subtitle = element_text(colour = "grey50"))


ggsave(
  "sbb.png", width = 8, height = 4
)
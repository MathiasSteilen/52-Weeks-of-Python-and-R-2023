rm(list = ls())

library(RSelenium)
library(tidyverse)

setwd(dirname(rstudioapi::getActiveDocumentContext()[[2]]))

chromeDr <- rsDriver(browser = "chrome",
                     port = 4569L, 
                     chromever = "108.0.5359.22",
                     extraCapabilities = list(
                       chromeOptions = list(
                         args = c('--disable-gpu',
                                  '--window-size=1280,800'),
                         prefs = list(
                           "profile.default_content_settings.popups" = 0L,
                           "download.prompt_for_download" = FALSE,
                           "directory_upgrade" = TRUE
                         ))))

remDr <- chromeDr[["client"]]
remDr$extraCapabilities$pageLoadStrategy <- "eager"

# Navigate to main page
remDr$navigate("https://www.chrono24.at/search/index.htm?currencyId=EUR&dosearch=true&listingType=BuyItNow&manufacturerIds=1&manufacturerIds=107&manufacturerIds=117&manufacturerIds=118&manufacturerIds=124&manufacturerIds=127&manufacturerIds=149&manufacturerIds=163&manufacturerIds=167&manufacturerIds=168&manufacturerIds=172&manufacturerIds=18&manufacturerIds=187&manufacturerIds=194&manufacturerIds=211&manufacturerIds=221&manufacturerIds=236&manufacturerIds=243&manufacturerIds=245&manufacturerIds=247&manufacturerIds=252&manufacturerIds=30&manufacturerIds=319&manufacturerIds=32&manufacturerIds=601&manufacturerIds=711&maxAgeInDays=28&pageSize=120&redirectToSearchIndex=true&resultview=list&sortorder=0&watchTypes=U")

# Click away annoying cookie message
unlist(
  lapply(
    remDr$findElements(
      using = "css selector",
      value = ".wt-consent-layer-accept-all"
    ), 
    function(x){x$clickElement()}
  )
)

# Get links of all listings
watch_links <- tibble(watch_link = character())

for(site_number in 312:383){
  
  # Navigate to site and wait to load
  url <- paste0(
    "https://www.chrono24.at/search/index.htm?currencyId=EUR&dosearch=true&listingType=BuyItNow&manufacturerIds=172&manufacturerIds=194&manufacturerIds=252&manufacturerIds=211&manufacturerIds=236&manufacturerIds=117&manufacturerIds=711&manufacturerIds=118&manufacturerIds=319&manufacturerIds=30&manufacturerIds=32&manufacturerIds=18&manufacturerIds=163&manufacturerIds=187&manufacturerIds=1&manufacturerIds=221&manufacturerIds=243&manufacturerIds=167&manufacturerIds=124&manufacturerIds=168&manufacturerIds=245&manufacturerIds=247&manufacturerIds=127&manufacturerIds=149&manufacturerIds=601&manufacturerIds=107&maxAgeInDays=28&pageSize=120&redirectToSearchIndex=true&resultview=list&showpage=",
    site_number,
    "&sortorder=0&watchTypes=U")
  
  remDr$navigate(url)
  Sys.sleep(2)
  
  # Extract all links to individual watch listings
  tmp_links <- unlist(
    lapply(
      remDr$findElements(
        using = "css selector",
        value = "#wt-watches > div > a"
      ), 
      function(x){x$getElementAttribute("href")}
    )
  )
  
  # Append the temporary links to the watch links data frame
  watch_links <- watch_links %>% 
    bind_rows(tibble(watch_link = tmp_links))
  
  print(paste("Listing", site_number, "done"))
  
}

watch_links <- bind_rows(
  read_csv("watch_links_first17k.csv"),
  read_csv("watch_links_second10k.csv"),
  read_csv("watch_links_third8k.csv"),
  read_csv("watch_links_fourth3k.csv"),
  read_csv("watch_links_last8k.csv")
) %>% 
  distinct()

# Save watch links as csv
watch_links %>% distinct() %>% write_csv("watch_links.csv")
watch_links <- watch_links %>% distinct()

# Trying RVEST for actual listings
library(rvest)
library(httr)

watch_infos = tibble()

for (link_number in 2441:nrow(watch_links)){
  
  url_tmp <- watch_links %>% 
    slice(link_number) %>% 
    pull()
  
  tryCatch(
    html_tmp <- url_tmp %>%
      GET(., timeout(90)) %>% 
      read_html(.),
    error = function(e){NA}
  )
  
  # Header and subheader
  header = html_tmp %>% 
    html_nodes(".h3.m-y-0") %>% 
    html_text()
  
  # skip to next iteration if header is not available
  if (identical(header, character(0))){
    next
  }
  
  # Price
  price = html_tmp %>% 
    html_nodes(".js-price-shipping-country") %>% 
    html_text()
  
  # Antwortzeit
  response_time = html_tmp %>% 
    html_nodes(".m-b-5 .text-muted") %>% 
    html_text()
  
  # Infos
  infos = tibble(
    header = html_tmp %>% 
      html_nodes(".p-r-2 strong") %>% 
      html_text()
  ) %>% 
    bind_cols(
      tibble(
        body = html_tmp %>% 
          html_nodes(".p-r-2+ td") %>% 
          html_text()
      )
    ) %>% 
    pivot_wider(names_from = header, 
                values_from = body,
                values_fn = list) %>%
    unnest(everything()) %>%
    distinct()
  
  watch_infos <- watch_infos %>%
    bind_rows(
      tibble(
        header = ifelse(identical(header, character(0)),
                        NA, header),
        price = ifelse(identical(price, character(0)),
                       NA, price),
        response_time = ifelse(identical(response_time, character(0)), 
                               NA, response_time)
      ) %>% 
        bind_cols(infos)
    )
  
  print(paste("Link", link_number, "retrieved"))
  Sys.sleep(runif(n = 1, min = 0.5, max = 1))
  
}

# watch_infos %>% save(file = "watches.RData")

watch_infos <- read_csv("watches_dirty.csv")

watch_infos %>% 
  head(5) %>% 
  pull(price)

# Cleaning
watches <- watch_infos %>% 
  janitor::clean_names() %>% 
  select(which(colMeans(is.na(.)) < 0.8), 
         -c(preis)) %>% 
  mutate(header = str_replace_all(header, "\n", "") %>% 
           gsub('[[:punct:] ]+', ' ', .),
         across(c(response_time, material_armband, herstellungsjahr,
                  zustand, lieferumfang, durchmesser, schliesse,
                  geschlecht),
                ~ str_replace_all(.x, "\n", "")),
         price = ifelse(price == "Preis auf Anfrage", NA, price),
         zustand = sub("\\(.*", "", zustand),
         across(c(zustand, geschlecht, lieferumfang),
                ~ trimws(.x)),
         across(c(price),
                ~ str_replace_all(.x, intToUtf8(160), "")),
         across(c(price),
                ~ str_replace_all(.x, " ", "")),
         across(c(price, herstellungsjahr, gangreserve, durchmesser),
                ~ parse_number(.x))) %>%
  rename(gangreserve_h = gangreserve)

watches %>% glimpse()
watches %>% count(sort = T, response_time) %>% 
  print(n = 30)

watches %>% write_csv("watches_clean.csv")
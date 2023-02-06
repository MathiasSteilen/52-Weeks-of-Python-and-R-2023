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

# First get the rental listings' links (have to go through 50 pages)
rental_links <- tibble(rental_link = character())

for(site_number in 1:117){
  
  # Navigate to site and wait 5 seconds to load
  url <- paste0(
    "https://www.immoweb.be/en/search/house-and-apartment/for-rent/brussels/province?countries=BE&page=",
    site_number,
    "&orderBy=newest")
  
  remDr$navigate(url)
  Sys.sleep(2)
  
  # Extract all links to individual rental listings
  tmp_links <- remDr$findElements(
    using = "css selector",
    value = '.card__title-link'
  )
  
  tmp_links <- unlist(
    lapply(tmp_links, function(x){x$getElementAttribute("href")})
  )
  
  # Append the temporary links to the rental links data frame
  rental_links <- rental_links %>% 
    bind_rows(tibble(rental_link = tmp_links))
  
  print(paste("Listing", site_number, "done"))
  
}

# Save rental links as csv
rental_links %>% distinct() %>% write_csv("listings_links.csv")
rental_links <- rental_links %>% distinct()

# ACTUAL LISTINGS --------------------------------------------
# Go through each individual element and pull out all the information I can
rental_df <- tibble()

# rental_links <- tibble(
#   url = "https://www.immoweb.be/en/classified/apartment/for-rent/anderlecht/1070/10357860"
# )

for (listing_number in 1:nrow(rental_links)){
  
  remDr$navigate(
    url = rental_links %>% slice(listing_number) %>% pull()
  )
  Sys.sleep(5)
  
  # If the page shows an error, go to the next one without stopping
  if (!is.null(
    unlist(
      lapply(
        remDr$findElements(
          using = "css selector",
          value = '.page-error__title'
        ), 
        function(x){x$getElementText()}
      )
    )
  )){
    print(paste("Listing", listing_number, "shows an error..."))
    next
  }
  
  # Get price, wait until page to load
  price <- unlist(
    lapply(
      remDr$findElements(
        using = "css selector",
        value = '.classified__price'
      ), 
      function(x){x$getElementText()}
    )
  )
  
  # Agency
  agency <- unlist(
    lapply(
      remDr$findElements(
        using = "css selector",
        value = '#customer-card .button__label'
      ), 
      function(x){x$getElementText()}
    )
  )
  
  # Description
  description <- unlist(
    lapply(
      remDr$findElements(
        using = "css selector",
        value = '.classified__description'
      ), 
      function(x){x$getElementText()}
    )
  )
  
  # Title
  listing_title <- unlist(
    lapply(
      remDr$findElements(
        using = "css selector",
        value = '.classified__title'
      ), 
      function(x){x$getElementText()}
    )
  )
  # Living Space
  space <- unlist(
    lapply(
      remDr$findElements(
        using = "css selector",
        value = '.classified__information--property'
      ), 
      function(x){x$getElementText()}
    )
  )
  
  # Address
  address <- unlist(
    lapply(
      remDr$findElements(
        using = "css selector",
        value = '.classified__information--address-row+ .classified__information--address-row'
      ), 
      function(x){x$getElementText()}
    )
  )
  
  # Details
  details <- unlist(
    lapply(
      remDr$findElements(
        using = "css selector",
        value = '.classified-table__data , .classified-table__header'
      ), 
      function(x){x$getElementText()}
    )
  )
  
  rental_df <- rental_df %>%
    bind_rows(
      tibble(
        header = details[c(TRUE, FALSE)],
        values = details[c(FALSE, TRUE)]
      ) %>% 
        distinct() %>% 
        pivot_wider(names_from = header, values_from = values) %>% 
        mutate(price = price,
               agency = agency,
               space = space,
               address = address,
               description = description,
               listing_type = listing_title)
    )
  
  print(paste("Listing", listing_number, "done..."))
  
}

# Close driver
remDr$close()
chromeDr[["server"]]$stop()

rental_df %>% 
  write_csv("rental_df_dirty.csv")

# Cleaning
colMeans(is.na(rental_df)) %>% 
  enframe() %>% 
  arrange(-value) %>% 
  print(n = 1000)

rental_df %>% 
  select(-(colMeans(is.na(rental_df)) %>% 
             enframe() %>% 
             filter(value > 0.9) %>% 
             pull(name))) %>% 
  select(-c(`CO₂ emission`, `Yearly theoretical total energy consumption`, 
            `Tenement building`, `Reference number of the EPC report`,
            space, `External reference`, Address, price)) %>% 
  janitor::clean_names() %>%
  rename(postcode = address,
         consumption_kwh_m2 = primary_energy_consumption) %>% 
  mutate(across(c(floor, monthly_rental_price, living_area, bedrooms,
                  bathrooms, terrace_surface, monthly_costs, 
                  construction_year, toilets, postcode, 
                  consumption_kwh_m2, number_of_floors, number_of_frontages),
                ~ parse_number(.x))) %>% 
  relocate(monthly_rental_price, monthly_costs, living_area, postcode,
           bedrooms, bathrooms, toilets, construction_year) %>% 
  glimpse()


  write_csv("immo_listings_clean.csv")
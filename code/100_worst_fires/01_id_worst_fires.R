# Find worst 100 fires by number of people evacuated

# Deciding to do it by evacuation because if there was a large evacuation, this 
# means that there were a lot of people experiencing the stress of thinking a
# wildfire might destroy their community + the dangers of evacuation.

pacman::p_load(tidyverse, here, sfarrow, sf)


wf_dat <- 
  st_read(here('local_data', 'raw_data', 'wfbz_disasters_2000-2025.geojson'))

wf_dat <-
  wf_dat %>%
  mutate(year = lubridate::floor_date(wildfire_ignition_date, unit = 'year')) %>%
  filter(year >= as.Date("2000-01-01") & year < as.Date("2019-01-01"))



wf_dat <- wf_dat %>% arrange(desc(wildfire_civil_evacuation))

wf_dat <- wf_dat %>% head(100)

st_write(wf_dat, here("local_data", "intermediate_data", 'worst_100_fires.geojson'))




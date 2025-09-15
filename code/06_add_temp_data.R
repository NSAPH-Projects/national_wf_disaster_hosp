# Add temperature data to wildfire zcta exposure data


# Libraries ---------------------------------------------------------------

pacman::p_load(tidyverse, here)

# Read --------------------------------------------------------------------

exp <- read_csv(here(
  "local_data",
  'intermediate_data',
  'binary_exposures_by_zcta.csv'
))

conus_temp_path = '/Volumes/squirrel-utopia/era_5/data/output/results_conus.csv'
conus_temp <- read_csv(conus_temp_path)

ak_temp_path = '/Volumes/squirrel-utopia/era_5/data/output/results_ak.csv'
ak_temp <- read_csv(ak_temp_path) %>% 
  mutate(ID_admin_unit = as.character(ID_admin_unit))

hi_temp_path = '/Volumes/squirrel-utopia/era_5/data/output/results_hi.csv'
hi_temp <- read_csv(hi_temp_path) %>% 
  mutate(ID_admin_unit = as.character(ID_admin_unit))


# Do ----------------------------------------------------------------------

all_temp <- bind_rows(list(conus_temp, ak_temp, hi_temp))

all_temp <- all_temp %>%
  pivot_longer(
    cols = starts_with("t2m_"),
    names_to = "date",
    values_to = "temperature"
  ) %>%
  mutate(month = substr(sub("t2m_", "", date), start = 1, stop = 7)) %>%
  select(-date)

exp <- exp %>% left_join(all_temp)


# Write -------------------------------------------------------------------

write_csv(exp, here(
  "local_data",
  'intermediate_data',
  'exposures_with_temp_for_upload.csv'
))

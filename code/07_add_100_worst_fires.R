# Add 100 worst fires analysis

# Libraries ---------------------------------------------------------------

pacman::p_load(tidyverse, here)

# Read --------------------------------------------------------------------

exp <- read_csv(here(
  "local_data",
  'intermediate_data',
  'exposures_with_temp_for_upload.csv'
))

worst_fires <- read_csv(here(
  "local_data",
  'intermediate_data',
  'binary_exposures_by_zcta_100_worst_fires.csv'
)) %>%
  select(ID_admin_unit, month, worst_fires_exposed_1, worst_fires_exposed_more_than_0_1, worst_fires_unexposed_1)

exp <- exp %>% left_join(worst_fires)

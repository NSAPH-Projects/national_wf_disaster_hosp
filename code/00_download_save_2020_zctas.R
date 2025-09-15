
# Load the tigris package
library(here)
library(tigris)
library(sf)
library(tidyverse)

# Set options to cache the data locally
options(tigris_use_cache = TRUE)

# Download 2020 ZCTAs
zctas_2020 <- zctas(year = 2020)

# View the first few rows
head(zctas_2020)

zctas_2020 <- zctas_2020 %>% select(ID_admin_unit = ZCTA5CE20)


st_write(
  zctas_2020,
  here(
    'local_data',
    'zctas_2020.geojson'
  ),
  driver = "GeoJSON",
  delete_dsn = TRUE
)

zcta_list <- sort(unique(zctas_2020$ID_admin_unit))

write_rds(zcta_list, here("data_for_upload", 'zcta_list.RDS'))


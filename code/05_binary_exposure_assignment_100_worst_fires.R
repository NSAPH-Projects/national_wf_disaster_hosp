# Assign exposure for each of the 9 analyses 

pacman::p_load(tidyverse, here)

exp <- read_rds(
  here(
    "local_data",
    "intermediate_data",
    "percent_exposed_by_zcta_month_to_100_worst_fires.RDS"
  )
)


exp <- exp %>% mutate(
  worst_fires_exposed_1 = ifelse(p_exp_main >= 0.1, 1, 0),
  worst_fires_exposed_more_than_0_1 = ifelse(p_exp_main > 0, 1, 0),
  worst_fires_unexposed_1 = ifelse(p_exp_main == 0, 1, 0))

write_csv(exp, here("local_data", 'intermediate_data', 'binary_exposures_by_zcta_100_worst_fires.csv'))



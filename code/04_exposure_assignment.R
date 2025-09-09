# Assign exposure for each of the 9 analyses 

pacman::p_load(tidyverse, here)

exp <- read_rds(here("local_data", "percent_exposed_by_zcta_month.RDS"))


exp <- exp %>% mutate(
  primary_exposed_1 = ifelse(p_exp_main >= 0.1, 1, 0),
  primary_exposed_more_than_0_1 = ifelse(p_exp_main > 0, 1, 0),
  primary_unexposed_1 = ifelse(p_exp_main == 0, 1, 0),
  less_misclas_exposed_2 = ifelse(p_exp_main > 0.5, 1, 0),
  less_misclas_exposed_more_than_0_2 = ifelse(p_exp_main > 0, 1, 0),
  less_misclas_unexposed_2 = ifelse(p_exp_main == 0, 1, 0),
  larger_exposed_1 = ifelse(p_exp_larger >= 0.1, 1, 0),
  larger_exposed_more_than_0_1 = ifelse(p_exp_larger > 0, 1, 0),
  larger_unexposed_1 = ifelse(p_exp_larger == 0, 1, 0),
  smaller_exposed_1 = ifelse(p_exp_smaller >= 0.1, 1, 0),
  smaller_exposed_more_than_0_1 = ifelse(p_exp_smaller > 0, 1, 0),
  smaller_unexposed_1 = ifelse(p_exp_smaller == 0, 1, 0))


write_csv(exp, here("local_data", 'binary_exposures_by_zcta.csv'))



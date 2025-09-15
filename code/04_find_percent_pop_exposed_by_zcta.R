# Find percent of ZCTA exposed for each 3 exposure levels for each month

pacman::p_load(tidyverse, here)

# Load data ---------------------------------------------------------------

exp <- read_csv(here(
  "local_data",
  "intermediate_data",
  "exposed_population_counts_by_zcta.csv"
))

pop <- read_csv(here(
  "local_data",
  "intermediate_data",
  "total_population_counts_by_zcta.csv"
))


# Get right pop data for each year ----------------------------------------

pop_list <- 
  pop %>%
  group_by(ghsl_used) %>%
  group_split()

month_vec <-
  format(seq.Date(
    from = as.Date("2000-01-01"),
    by = "month",
    length.out = 228
  ), "%Y-%m")

month_ranges <- list(
  1:36,      # ghsl 2000
  37:96,     # ghsl 2005
  97:156,    # ghsl 2010
  157:216,   # ghsl 2015
  217:228    # ghsl 2020
)

result_list <- list()

for (i in seq_along(pop_list)) {
  for (m in month_ranges[[i]]) {
    temp_df <- pop_list[[i]]
    temp_df$month <- month_vec[m]
    result_list[[length(result_list) + 1]] <- temp_df
  }
}

pop_data_by_month <- do.call(rbind, result_list)

# Join exposure data ------------------------------------------------------

pop_data_by_month <- pop_data_by_month %>% left_join(exp)

percent_exposed <- pop_data_by_month %>% select(-ID_hazard) 
percent_exposed[is.na(percent_exposed)] <- 0

percent_exposed <- 
  percent_exposed %>%
  mutate(
  p_exp_main = exposed_main / population,
  p_exp_smaller = exposed_smaller / population,
  p_exp_larger = exposed_larger / population
)


write_rds(
  percent_exposed,
  here(
    "local_data",
    "intermediate_data",
    "percent_exposed_by_zcta_month.RDS"
  )
)

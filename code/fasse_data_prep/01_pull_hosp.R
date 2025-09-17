# This file reads in the medpar denominator and hospitalization files for
# 2000-2018 and pulls the hospitalizations related to CVD and respiratory hosp,
# urgent and emergency hospitalizations only

# going to tally them in this script to save memory
# need daily hospitalization counts by day for each county for 2018
# panel

# By: Heather, September 20th, 2024

# Libraries ---------------------------------------------------------------

pacman::p_load(here, tidyverse, arrow, data.table)


# Helper ------------------------------------------------------------------

# function helps identify hospitalizations that contain any icd code from 
# a predetermined list of codes 
contains_any_code <- function(codes, condition) {
  as.integer(Reduce(`|`, lapply(codes, `%chin%`, condition)))
}


# Read files --------------------------------------------------------------

zctas <- read_rds(here('data_for_upload', 'zcta_list.RDS'))
years <- 2000:2001

# read in medicare hospitalizations and beneficiaries for 2018
bene_base_paths <- 
  paste0(here("data"), "/mbsf_medpar_denom_", years, ".parquet")
benes <- lapply(FUN = arrow::read_parquet, X = bene_base_paths)
benes <- lapply(FUN = setDT, X = benes)
benes <- lapply(FUN = sample_n, X = benes, 100000)

hosp_base_paths <- 
  paste0(here("data"), "/medpar_hospitalizations_", years, ".parquet")
hosps <- lapply(FUN = arrow::read_parquet(), X = hosp_base_paths)
hosps <- lapply(FUN = setDT, X = hosps)
hosps <- lapply(FUN = sample_n, X = benes, 100000)

# join metadata to hospitalizations
hosp_info_list <- Map(function(hosp, bene) {
  merge(hosp, bene, by = "bene_id", all.x = TRUE)
}, hosps, benes)


# Pull ICD codes ----------------------------------------------------------

# filter to hospitalizations actually dated in 2018
hosp_info <- hosp_info |>
  mutate(admission_date = as.Date(admission_date)) |>
  filter(admission_date %in% panel_fips$day)

# get the first five ICD codes
hosp_info <-
  hosp_info[, paste0("code_", 1:5) := transpose(lapply(diagnoses, function(x) {
    length(x) <- 5
    x
  }))]


# Define codes from icd code file -----------------------------------------

# all cvd codes, including hypertension and hem stroke
all_cvd <- icd_codes$all_cvd 
# cvd without hypertension
cvd_no_hyp <- icd_codes$cvd_no_hyp
# all respiratory codes
resp <- icd_codes$all_resp
# all hypertension codes
hyp <- icd_codes$hyp_codes

# code cols 
code_columns <- c("code_1", "code_2", "code_3", "code_4", "code_5")

# indicate hospitalizations with the codes
hosp_info <- hosp_info[, contains_any_cvd_code :=
                         contains_any_code(.SD, all_cvd),
                       .SDcols = code_columns]

hosp_info <- hosp_info[, contains_cvd_no_hyp :=
                         contains_any_code(.SD, cvd_no_hyp),
                       .SDcols = code_columns]

hosp_info <- hosp_info[, contains_resp_code :=
                         contains_any_code(.SD, resp),
                       .SDcols = code_columns]

hosp_info <- hosp_info[, contains_hyp_code :=
                         contains_any_code(.SD, hyp),
                       .SDcols = code_columns]

# identify emergency and urgent vs non emergency
hosp_info[, `:=`(
  is_urg_or_emerg = ifelse(admsn_type_cd == 1 | admsn_type_cd == 2, 1, 0)
)]

# identify hosp by age 
hosp_info[, `:=` (
  age = ifelse(age_dob >= 75, 1, 0)
)]


# Summarize ---------------------------------------------------------------

urgent_hosp_by_day_by_county <- hosp_info[is_urg_or_emerg == 1]

# get dataset that is stratified by sex - unfortunately excluding people 
# with sex marker that is not male or female bc this is an error, not recording 
# intersex people
urg_hosp_by_sex <-
  urgent_hosp_by_day_by_county[sex != 0, .(
    n_all_cvd = sum(contains_any_cvd_code),
    n_cvd_no_hyp = sum(contains_cvd_no_hyp),
    n_resp = sum(contains_resp_code)
  ),
  by = .(admission_date, county, state, sex)]

urg_hosp_by_sex <-
  dcast(
    urg_hosp_by_sex,
    admission_date + county + state ~ sex,
    value.var = c(
      "n_all_cvd",
      "n_cvd_no_hyp",
      "n_resp"
    )
  )

# change names to reflect that these are counts by sex 
setnames(urg_hosp_by_sex, old = names(urg_hosp_by_sex)[-(1:3)], 
         new = paste0(names(urg_hosp_by_sex)[-(1:3)], "_sex"))

urg_hosp_by_age <- urgent_hosp_by_day_by_county[, .(
  n_all_cvd = sum(contains_any_cvd_code),
  n_cvd_no_hyp = sum(contains_cvd_no_hyp),
  n_resp = sum(contains_resp_code)
),
by = .(admission_date, county, state, age)]

#do the same for age
urg_hosp_by_age <-
  dcast(
    urg_hosp_by_age,
    admission_date + county + state ~ age,
    value.var = c(
      "n_all_cvd",
      "n_cvd_no_hyp",
      "n_resp"
    )
  )

setnames(urg_hosp_by_age, old = names(urg_hosp_by_age)[-(1:3)], 
         new = paste0(names(urg_hosp_by_age)[-(1:3)], "_age"))

urg_hosp_by_medicaid <- urgent_hosp_by_day_by_county[, .(
  n_all_cvd = sum(contains_any_cvd_code),
  n_cvd_no_hyp = sum(contains_cvd_no_hyp),
  n_resp = sum(contains_resp_code)
),
by = .(admission_date, county, state, dual)]

urg_hosp_by_medicaid <-
  dcast(
    urg_hosp_by_medicaid,
    admission_date + county + state ~ dual,
    value.var = c(
      "n_all_cvd",
      "n_cvd_no_hyp",
      "n_resp"
    )
  )

# change names to reflect that these are counts by sex 
setnames(urg_hosp_by_medicaid, old = names(urg_hosp_by_medicaid)[-(1:3)], 
         new = paste0(names(urg_hosp_by_medicaid)[-(1:3)], "_medicaid"))

# summarize total hospitalizations 
urgent_hosp_by_day_by_county <- urgent_hosp_by_day_by_county[, .(
  n_all_cvd = sum(contains_any_cvd_code),
  n_cvd_no_hyp = sum(contains_cvd_no_hyp),
  n_resp = sum(contains_resp_code),
  n_hyp = sum(contains_hyp_code),
  n_all = .N),
  by = .(admission_date, county, state)
]

# join all together 
urgent_hosp_by_day_by_county <- 
  urgent_hosp_by_day_by_county %>%
  left_join(urg_hosp_by_age) %>% 
  left_join(urg_hosp_by_sex) %>%
  left_join(urg_hosp_by_medicaid)

urgent_hosp_by_day_by_county <-
  urgent_hosp_by_day_by_county[county %chin% panel_fips$five_digit_fips]

# Write  ------------------------------------------------------------------

write_rds(
  urgent_hosp_by_day_by_county,
  here(
    "data",
    "urg_and_emerg_num_hosp_by_day_by_county_inc_state_jan_28.RDS"
  )
)

# write denom, for secondary analyses also
benes_2018[, age_group := ifelse(age_dob >= 75, "older_75", "under_75")]
benes_2018[, sex_group := ifelse(sex == '1', "sex_1", "sex_2")]
benes_2018[, medicaid_group := ifelse(dual == 1, "medicaid_eligible", "non_eligible")]

benes_2018 <- benes_2018[county %chin% panel_fips$five_digit_fips]

# summarize
benes_summary <- benes_2018[, .(
  n_benes = .N,
  n_benes_older_75 = sum(age_group == "older_75"),
  n_benes_under_75 = sum(age_group == "under_75"),
  n_benes_sex_1 = sum(sex_group == "sex_1"),
  n_benes_sex_2 = sum(sex_group == "sex_2"),
  n_benes_medicaid_eligible = sum(medicaid_group == "medicaid_eligible"),
  n_benes_non_eligible = sum(medicaid_group == 'non_eligible')
), by = .(county, state)]

write_rds(benes_summary, here('data', 'benes_by_county_fips_jan_28.RDS'))
# This script sets up FASSE for data cleaning for the national wf disaster and
# hospitalizations project

# libraries
pacman::p_load(here)

# set proxies for GitHub access
Sys.setenv(http_proxy="http://rcproxy.rc.fas.harvard.edu:3128")
Sys.setenv(https_proxy="http://rcproxy.rc.fas.harvard.edu:3128")

# stop R from being a lil bih about small and large numbers
options(scipen = 999)

# add token so we can use GitHub
gitcreds::gitcreds_set()
gitcreds::gitcreds_get()

# create symlinks for 2000-2018 denom and hospitalization files
# denom file
years <- 2000:2018
target_file_denoms <-
  paste0("/n/dominici_nsaph_l3/Lab/projects/analytic/",
         "mbsf_medpar_denom/mbsf_medpar_denom_", years, ".parquet")
link_name_denoms <-paste0(here("data"), "/mbsf_medpar_denom_", years, ".parquet")

# verify the symlinks
mapply(function(from, to) {
  if (!file.exists(to)) {
    file.symlink(from = from, to = to)
    cat("Initial symbolic link created successfully", to, "\n")
  }
  if (file.exists(to)) {
    cat("Symbolic link previously created successfully for", to, "\n")
  }
}, target_file_denoms, link_name_denoms)

# hospitalization file
target_file_hosps <-
  paste0(
    "/n/dominici_nsaph_l3/Lab/projects/analytic/",
    "mbsf_medpar_denom/medpar_hospitalizations_", years, ".parquet"
  )
link_name_hosps <-
  paste0(here("data"), "/medpar_hospitalizations_", years, ".parquet")

# verify the symlinks
mapply(function(from, to) {
  if (!file.exists(to)) {
    file.symlink(from = from, to = to)
    cat("Initial symbolic link created successfully", to, "\n")
  }
  if (file.exists(to)) {
    cat("Symbolic link previously created successfully for", to, "\n")
  }
}, target_file_hosps, link_name_hosps)


# zip to county xwalk
target_file_xwalk <-
  paste0(
    "/n/dominici_nsaph_l3/Lab/exposure/zip2zcta_master_xwalk/",
    "zip2zcta_master_xwalk.csv"
  )
link_name_xwalk <-
  here("data", "zip2zctaxwalk.csv")


# verify the symlinks
if (file.exists(link_name_xwalk)) {
  cat("Symbolic link previously created successfully.\n")
} else {
  # Create the symbolic link
  file.symlink(from = target_file_xwalk, to = link_name_xwalk)
  if (file.exists(link_name_xwalk)) {
    cat("Initial symbolic link created successfully.\n")
  }
}

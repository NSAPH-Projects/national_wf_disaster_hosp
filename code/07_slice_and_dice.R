# Slice and dice the data to create time series for DID.
# for wf disaster -> hospitalizations project

# we're starting with a dataframe that has IDs for each ZCTA, months 2000-01 to 
# 2018-12 for every ZCTA, as well as an indicator of whether a zcta was exposed 
# or not, and an indicator of whether it was exposed enough to not be a control
# but did not meet the threshold of exposure
# want to split it up into time series that are either:
# 1. unexposed
# 2. have a three week pre-intervention period, are exposed, and then have a two-year post washout period
# 3. were near a fire but not near enough to be exposed, so are not controls, so we should exclude them.

# want to have new unique IDs for each of these chunks of data. 


library(data.table)
library(tidyverse)
library(here)
library(lubridate)



extract_windows <- function(dt) {
  dt <- copy(dt)  # Avoid modifying original
  dt[, processed := FALSE] # assigning a column called processed that is false
  result_list <- list() # initializing a results list 
  new_id <- 1 # adding a new ID counter 
  
  repeat { # is this a while loop? 
    # Find first exposure in unprocessed data
    dt_unprocessed <- dt[processed == FALSE] # maybe later in the loop we assign processed to some things
    if (nrow(dt_unprocessed) == 0) break # once it's all processed we end 
    
    dt_unprocessed[, first_exposure := min(month_num[exposed == 1], na.rm = TRUE), by = ID_admin_unit] # we process all IDs together, and find the first exp in unprocessed
    dt_unprocessed[first_exposure == Inf, first_exposure := NA] # if there is no first exp set to NA
    dt_unprocessed[, window_start := first_exposure - 3] # yes good window start 
    dt_unprocessed[, window_end := first_exposure + 24] # and window end 
    
    # Window segment
    window <- dt_unprocessed[!is.na(first_exposure) & month_num >= window_start & month_num <= window_end] # select the rows that are within the window 
    if (nrow(window) > 0) { # if there are no rows do nothing
      window[, new_ID := new_id] # add a new ID to it 
      window[, window_exposed := as.integer(month_num >= first_exposure)] # ah this creates the indicator variable for exposure - 1 after the exp happens
      result_list[[length(result_list) + 1]] <- window # add the window to the results list 
      dt[window, processed := TRUE, on = .(ID_admin_unit, month_num)] # matches rows in dt to rows in window where both ID and month are equal.For those matched rows in dt, it sets processed to TRUE.
      new_id <- new_id + 1 # increment new id 
    }
    
    # Pre-window segment
    pre_window <- dt_unprocessed[!is.na(first_exposure) & month_num < window_start] # window start is first exposure so this is indeed times before that
    if (nrow(pre_window) > 0) { # if empty do nothing
      pre_window[, new_ID := new_id] # add new id 
      pre_window[, window_exposed := 0L] # no exposure - good 
      result_list[[length(result_list) + 1]] <- pre_window # good adding to the thing 
      dt[pre_window, processed := TRUE, on = .(ID_admin_unit, month_num)] # making the processed true in dt
      new_id <- new_id + 1 # increment new id
    }
    
    # No exposure segment
    no_exposure <- dt_unprocessed[is.na(first_exposure)] # good this works! 
    if (nrow(no_exposure) > 0) {
      no_exposure[, new_ID := new_id]
      no_exposure[, window_exposed := 0L]
      result_list[[length(result_list) + 1]] <- no_exposure
      dt[no_exposure, processed := TRUE, on = .(ID_admin_unit, month_num)]
      new_id <- new_id + 1
    }
  }
  
  # Combine all segments
  result <- rbindlist(result_list, use.names = TRUE, fill = TRUE)
  result[, processed := NULL]
  result[, c("first_exposure", "window_start", "window_end") := NULL]
  return(result) # this just removes columns that we don't care about. 
}

# Usage:
# setDT(df)
# result_df <- extract_windows(df)


# Do ----------------------------------------------------------------------

exp <- read_csv(here("local_data", 'binary_exposures_by_zcta.csv'))

exp <- exp %>%
  mutate(month_num = (year(ym(month)) - min(year(ym(
    month
  )))) * 12 + month(ym(month)))

exp_primary <- exp %>% select(ID_admin_unit, month, month_num, exposed = primary_exposed_1, primary_exposed_more_than_0_1)
#exp_primary <- exp_primary %>% filter(ID_admin_unit == '95969' | ID_admin_unit == '96761')

setDT(exp_primary)

result_df <- extract_windows(exp_primary)
result_df <- result_df %>% arrange(ID_admin_unit, month_num)

result_df <- result_df %>% mutate(unique_id = paste(ID_admin_unit, new_ID, sep = '_'))
result_df <- result_df %>%
  mutate(exclude = ifelse(primary_exposed_more_than_0_1 == 1 &
                            exposed == 0 & window_exposed == 0, 1, 0)) %>%
  group_by(unique_id) %>%
  mutate(exclude = max(exclude))

result_df <- result_df %>% filter(exclude == 0)

for_did <- result_df %>% select(unique_id, month_num, intervention = window_exposed)
for_did <- for_did %>% group_by(unique_id) %>% mutate(time = row_number())


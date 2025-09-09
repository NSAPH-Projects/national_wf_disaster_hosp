# EDA for wildfire exposure

pacman::p_load(tidyverse, here, sf, zippeR, geofacet, tigris, gganimate)
options(scipen = 9999)
options(tigris_use_cache = TRUE)

exp <- read_rds(here("local_data", "percent_exposed_by_zcta_month.RDS"))

sum(exp$exposed_main)
# 84,570,242 exposed over the study period. 
# (<10 km from a large fire or <5km from a small fire) 
# this double-counts people if they were exposed in more than one month


# how many people exposed in total by month
# how many people exposed in total in ca by month, hi, ak?
monthly_total <- 
  exp %>%
  group_by(month) %>%
  summarize(n_exp = sum(exposed_main, na.rm = T))

monthly_total %>%
  ggplot() + 
  geom_point(aes(x = month, y = n_exp)) + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

ca <-
  exp %>% filter(as.numeric(substr(ID_admin_unit, 1, 3)) >= 900 &
                   as.numeric(substr(ID_admin_unit, 1, 3)) <= 961) %>%
  group_by(month) %>%
  summarize(n_exp = sum(exposed_main, na.rm = T))

ca %>%
  ggplot() + 
  geom_point(aes(x = month, y = n_exp)) + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

hi <-
  exp %>% filter(as.numeric(ID_admin_unit) >= 96701 &
                   as.numeric(ID_admin_unit) <= 96898) %>%
  group_by(month) %>%
  summarize(n_exp = sum(exposed_main, na.rm = T))

hi %>%
  ggplot() + 
  geom_point(aes(x = month, y = n_exp)) + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

ak <-
  exp %>% filter(as.numeric(ID_admin_unit) >= 99501 &
                   as.numeric(ID_admin_unit) <= 99999) %>%
  group_by(month) %>%
  summarize(n_exp = sum(exposed_main, na.rm = T))

ak %>%
  ggplot() + 
  geom_point(aes(x = month, y = n_exp)) + 
  theme_minimal() + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1))

# how many people exposed in total by year by state?
# what states are the most exposed?

xwalk <- zi_load_crosswalk(year = 2020) %>%
  select(ID_admin_unit = ZIP, state = STATE)

exp <- exp %>% left_join(xwalk) 
exp <- exp %>% mutate(year = substr(month, start = 1, stop = 4))

exp <- exp %>% group_by(state, year) %>% summarize(n_exp = sum(exposed_main, na.rm = T))

exp <- exp %>% drop_na()

exp %>% ggplot(aes(x = year, y = n_exp)) +
  geom_bar(stat = "identity") +
  facet_geo(~ state, grid = "us_state_grid1") +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust = 1)) +
  labs(x = "Month", y = "People Exposed", title = "Yearly wildfire disaster exposure by state 2000-2018")

# animation of monthly exposure by zcta for ca, hi, ak, whole US

states_raw <- tigris::states() 
states <- states_raw %>%
  filter(STATEFP < 57 & STUSPS != 'HI' & STUSPS != 'AK') %>%
  select(state = STUSPS) %>%
  filter(state %in% exp$state)
states <- st_transform(states, crs = 5070)

states %>% ggplot() + geom_sf() + theme_minimal()

exp <- exp %>% filter(state != 'HI' & state != 'AK') %>% left_join(states)
exp <- st_as_sf(exp)

# animate
# whole us 

p <- ggplot(exp) +
  geom_sf(aes(fill = n_exp), color = NA) +
  scale_fill_viridis_c(option = "plasma") +
  theme_void() +
  labs(title = "Yearly Wildfire Exposure", fill = "Exposed Pop") +
  transition_manual(year) +
  labs(subtitle = "Year: {current_frame}")

anim <- animate(p, nframes = length(unique(exp$year)) * 5, fps = 8, renderer = gifski_renderer())

anim_save(here("figures", "wildfire_exposure.gif"), animation = anim)


# how about by zcta? 
zctas <- tigris::zctas()

zctas <- zctas %>% select(zcta = ZCTA5CE20)

xwalk <- zi_load_crosswalk(year = 2020)
xwalk <- xwalk %>% select(zip = ZIP, zcta = ZCTA)


exp <- read_rds(here("local_data", "percent_exposed_by_zcta_month.RDS"))

zip_monthly_total <- 
  exp %>%
  group_by(ID_admin_unit, month) %>%
  summarize(n_exp = sum(exposed_main, na.rm = T))

zip_monthly_total_ca <-
  zip_monthly_total %>%
  filter(as.numeric(substr(ID_admin_unit, 1, 3)) >= 900 &
           as.numeric(substr(ID_admin_unit, 1, 3)) <= 961)

zip_monthly_total_ca <- zip_monthly_total_ca %>%
  rename(zip = ID_admin_unit) %>% 
  left_join(xwalk)

zip_monthly_total_ca <- zip_monthly_total_ca %>% left_join(zctas)
zip_monthly_total_ca <- st_as_sf(zip_monthly_total_ca)

#zip_monthly_total_ca %>% ggplot() + geom_sf()
zip_monthly_total <- zip_monthly_total_ca %>% drop_na()

p <- ggplot(zip_monthly_total_ca) +
  geom_sf(aes(fill = n_exp), color = NA) +
  scale_fill_viridis_c(option = "plasma") +
  theme_void() +
  labs(title = "Monthly Wildfire Exposure", fill = "Exposed Pop") +
  transition_manual(month) +
  labs(subtitle = "Month: {current_frame}")

anim <- animate(p, nframes = length(unique(zip_monthly_total_ca$month)), fps = 12, renderer = gifski_renderer())

anim <- animate(p, nframes = length(unique(zip_monthly_total$month)), fps = 12, renderer = gifski_renderer(), ncores = 4)

anim_save(here("figures", "wildfire_exposure_monthly_zcta.gif"), animation = anim)

# TX

exp <- read_rds(here("local_data", "percent_exposed_by_zcta_month.RDS"))

zip_monthly_total <- 
  exp %>%
  group_by(ID_admin_unit, month) %>%
  summarize(n_exp = sum(exposed_main, na.rm = T))

zip_monthly_total_hi <-
  zip_monthly_total %>%
  filter(as.numeric(ID_admin_unit) >= 73301 &
                     as.numeric(ID_admin_unit) <= 88595) 

zip_monthly_total_hi <- zip_monthly_total_hi %>%
  rename(zip = ID_admin_unit) %>% 
  left_join(xwalk)

zip_monthly_total_hi <- zip_monthly_total_hi %>% left_join(zctas)
zip_monthly_total_hi <- st_as_sf(zip_monthly_total_hi)

#zip_monthly_total_ca %>% ggplot() + geom_sf()
zip_monthly_total_hi <- zip_monthly_total_hi %>% drop_na()

p <- ggplot(zip_monthly_total_hi) +
  geom_sf(aes(fill = n_exp), color = NA) +
  scale_fill_viridis_c(option = "plasma") +
  theme_void() +
  labs(title = "Monthly Wildfire Exposure", fill = "Exposed Pop") +
  transition_manual(month) +
  labs(subtitle = "Month: {current_frame}")


anim <- animate(p, nframes = length(unique(zip_monthly_total_hi$month)), fps = 12, renderer = gifski_renderer(), ncores = 4)

anim_save(here("figures", "wildfire_exposure_monthly_zcta_TX.gif"), animation = anim)


# AK ----------------------------------------------------------------------

exp <- read_rds(here("local_data", "percent_exposed_by_zcta_month.RDS"))

zip_monthly_total <- 
  exp %>%
  group_by(ID_admin_unit, month) %>%
  summarize(n_exp = sum(exposed_main, na.rm = T))

zip_monthly_total_ak <-
  zip_monthly_total %>%
  filter(as.numeric(ID_admin_unit) >= 96501 &
           as.numeric(ID_admin_unit) <= 99999) 

zip_monthly_total_ak <- zip_monthly_total_ak %>%
  rename(zip = ID_admin_unit) %>% 
  left_join(xwalk)

zip_monthly_total_ak <- zip_monthly_total_ak %>% left_join(zctas)
zip_monthly_total_ak <- st_as_sf(zip_monthly_total_ak)

#zip_monthly_total_ca %>% ggplot() + geom_sf()
zip_monthly_total_ak <- zip_monthly_total_ak %>% drop_na()

p <- ggplot(zip_monthly_total_ak) +
  geom_sf(aes(fill = n_exp), color = NA) +
  scale_fill_viridis_c(option = "plasma") +
  theme_void() +
  labs(title = "Monthly Wildfire Exposure", fill = "Exposed Pop") +
  transition_manual(month) +
  labs(subtitle = "Month: {current_frame}")


anim <- animate(p, nframes = length(unique(zip_monthly_total_ak$month)), fps = 12, renderer = gifski_renderer(), ncores = 4)

anim_save(here("figures", "wildfire_exposure_monthly_zcta_AK.gif"), animation = anim)

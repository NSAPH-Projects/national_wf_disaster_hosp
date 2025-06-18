if __name__ == "__main__":

    import pathlib
    from pathlib import Path
    import os
    import sys
    import matplotlib.pyplot as plt
    import geopandas as gpd
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import time

    # get popexposure
    from new_helpers_may_17 import PopEstimator

    start = time.time()

    # Set directories --------------------------------------------------------------
    base_path = pathlib.Path.cwd().parent.parent

    # pop dat
    pop_dat_dir = base_path / "GHSL" / "1km"

    ghsl_2000 = (
        pop_dat_dir
        / "GHS_POP_E2000_GLOBE_R2023A_54009_1000_V1_0"
        / "GHS_POP_E2000_GLOBE_R2023A_54009_1000_V1_0.tif"
    )
    ghsl_2005 = (
        pop_dat_dir
        / "GHS_POP_E2005_GLOBE_R2023A_54009_1000_V1_0"
        / "GHS_POP_E2005_GLOBE_R2023A_54009_1000_V1_0.tif"
    )
    ghsl_2010 = (
        pop_dat_dir
        / "GHS_POP_E2010_GLOBE_R2023A_54009_1000_V1_0"
        / "GHS_POP_E2010_GLOBE_R2023A_54009_1000_V1_0.tif"
    )
    ghsl_2015 = (
        pop_dat_dir
        / "GHS_POP_E2015_GLOBE_R2023A_54009_1000_V1_0"
        / "GHS_POP_E2015_GLOBE_R2023A_54009_1000_V1_0.tif"
    )
    ghsl_2020 = (
        pop_dat_dir
        / "GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0"
        / "GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.tif"
    )

    # wf dat
    wf_dat = (
        base_path / "national_wf_disaster_hosp" / "local_data" / "monthly_wf_exposure"
    )
    all_wf_dat = wf_dat / "all_analysis"

    # zctas
    zctas_2020 = (
        base_path / "national_wf_disaster_hosp" / "local_data" / "zctas_2020.parquet"
    )

    # Create data lists  -----------------------------------------------------------
    # make a list of paths that we're going to use for each month
    ghsl_paths = [ghsl_2000, ghsl_2005, ghsl_2010, ghsl_2015, ghsl_2020]

    # rep pattern
    rep_pattern = [3 * 12, 5 * 12, 5 * 12, 5 * 12, 1 * 12]

    # list of ghsls to use for months
    repeated_paths = [
        path for path, count in zip(ghsl_paths, rep_pattern) for _ in range(count)
    ]

    # create wf dat list
    all_wf_exposure = sorted(
        [all_wf_dat / file for file in os.listdir(all_wf_dat) if "month" in file]
    )

# Do the processing --------------------------------------------------------
est = PopEstimator()

# prep zcta data
print("preparing zcta data")
zctas = est.prep_data(path_to_data=zctas_2020, geo_type="spatial_unit")

# prep wf data
from tqdm import tqdm

wfs = []
for i in tqdm(range(len(all_wf_exposure)), desc="Preparing hazard data"):
    wf = est.prep_data(path_to_data=all_wf_exposure[i], geo_type="hazard")
    wfs.append(wf)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def process_exposed_pop(i):
        exposed_pop_df = est.exposed_pop(
            pop_path=repeated_paths[i], hazards=wfs[i], hazard_specific=False
        )
        exposed_pop_df["month"] = i + 1
        return exposed_pop_df

    exposed_pop = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_exposed_pop, i) for i in range(len(wfs))]
        for f in tqdm(
            as_completed(futures), total=len(wfs), desc="Calculating exposed population"
        ):
            exposed_pop.append(f.result())

end = time.time()
print(f"Elapsed time: {end - start} seconds")

# if __name__ == "__main__":
#     exposed_pop = []
#     for i in tqdm(range(len(wfs)), desc="Calculating exposed population"):
#         exposed_pop_df = est.exposed_pop(
#             pop_path=repeated_paths[i], hazards=wfs[i], hazard_specific=False
#         )
#         exposed_pop_df["month"] = i + 1  # add month column
#         exposed_pop.append(exposed_pop_df)

#     end = time.time()
#     print(f"Elapsed time: {end - start} seconds")
# do
# exposed_pop = []
# for i in tqdm(range(len(wfs)), desc="Calculating exposed population"):
#     exposed_pop_df = est.estimate_exposed_pop(
#         pop_path=repeated_paths[i], hazards=wfs[i], hazard_specific=False
#     )
#     exposed_pop_df["month"] = i + 1  # add month column
#     exposed_pop.append(exposed_pop_df)

# Combine all results into a single DataFrame
combined_df = pd.concat(exposed_pop, ignore_index=True)

# write as csv
combined_df.to_csv(
    "/Volumes/squirrel-utopia/national_wf_disaster_hosp/local_data/num_exposed_monthly_all.csv"
)


# # Import popexposure module
# module_path = Path("/Volumes/squirrel-utopia/Pop_Exp/src/Pop_Exp")
# sys.path.append(str(module_path))
# import find_exposure

# # Set directories
# base_path = pathlib.Path.cwd().parent.parent

# # Pop data
# pop_dat_dir = base_path / "GHSL" / "1km"

# ghsl_2000 = (
#     pop_dat_dir
#     / "GHS_POP_E2000_GLOBE_R2023A_54009_1000_V1_0"
#     / "GHS_POP_E2000_GLOBE_R2023A_54009_1000_V1_0.tif"
# )
# ghsl_2005 = (
#     pop_dat_dir
#     / "GHS_POP_E2005_GLOBE_R2023A_54009_1000_V1_0"
#     / "GHS_POP_E2005_GLOBE_R2023A_54009_1000_V1_0.tif"
# )
# ghsl_2010 = (
#     pop_dat_dir
#     / "GHS_POP_E2010_GLOBE_R2023A_54009_1000_V1_0"
#     / "GHS_POP_E2010_GLOBE_R2023A_54009_1000_V1_0.tif"
# )
# ghsl_2015 = (
#     pop_dat_dir
#     / "GHS_POP_E2015_GLOBE_R2023A_54009_1000_V1_0"
#     / "GHS_POP_E2015_GLOBE_R2023A_54009_1000_V1_0.tif"
# )
# ghsl_2020 = (
#     pop_dat_dir
#     / "GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0"
#     / "GHS_POP_E2020_GLOBE_R2023A_54009_1000_V1_0.tif"
# )

# # Wf data
# wf_dat = base_path / "national_wf_disaster_hosp" / "local_data" / "monthly_wf_exposure"

# main_wf_dat = wf_dat / "main_analysis"
# sensitivity_larger = wf_dat / "sensitivity_larger"
# sensitivity_smaller = wf_dat / "sensitivity_smaller"

# # ZCTA data
# zctas_2020 = (
#     base_path / "national_wf_disaster_hosp" / "local_data" / "zctas_2020.parquet"
# )

# # Do
# # make a list of paths that we're going to use for each month
# ghsl_paths = [ghsl_2000, ghsl_2005, ghsl_2010, ghsl_2015, ghsl_2020]

# # rep pattern
# rep_pattern = [3 * 12, 5 * 12, 5 * 12, 5 * 12, 1 * 12]

# # list of ghsls to use for months
# repeated_paths = [
#     path for path, count in zip(ghsl_paths, rep_pattern) for _ in range(count)
# ]


# # Create main analysis data
# # List paths in the main analysis directory
# main_wf_exposure = sorted(
#     [main_wf_dat / file for file in os.listdir(main_wf_dat) if "month" in file]
# )


# def process_file(i):
#     # Extract the month from the first 15 characters of the file name
#     month = str(main_wf_exposure[i - 1].name[6:16])

#     # Perform the exposure calculation
#     monthly_exposure = find_exposure.find_num_people_affected_by_geo(
#         path_to_hazards=main_wf_exposure[i - 1],
#         path_to_additional_geos=zctas_2020,
#         raster_path=repeated_paths[i - 1],
#         by_unique_hazard=False,
#     )

#     # Add the month column to the result
#     monthly_exposure["month"] = month

#     return monthly_exposure


# # Run the processing in parallel
# if __name__ == "__main__":
#     # Run the processing in parallel
#     results = []
#     with ProcessPoolExecutor() as executor:
#         # Map the process_file function to the range of indices
#         results = list(executor.map(process_file, range(1, 228)))

#     # Combine all results into a single DataFrame
#     final_results = gpd.GeoDataFrame(pd.concat(results, ignore_index=True))
#     # Display the final results
#     print(final_results.head())
#     # save final results
#     final_results.to_csv(
#         "/Volumes/squirrel-utopia/national_wf_disaster_hosp/local_data/num_exposed_monthly_main.csv"
#     )

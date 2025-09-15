from concurrent.futures import ProcessPoolExecutor
import tqdm
import pathlib
import os
import pandas as pd

# Stable local version
from popexposure.pop_estimator import PopEstimator


def process_total_pop(est, pop_data_path, ghsl_used):
    est.pop_data = pop_data_path
    total_pop_df = est.est_total_pop()
    if total_pop_df is not None:
        total_pop_df["ghsl_used"] = ghsl_used
    return total_pop_df


if __name__ == "__main__":
    print("Running find_total_res_pop_by_zcta.")

    # set directories
    base_path = pathlib.Path.cwd().parent

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

    zctas_2020 = (
        base_path
        / "national_wf_disaster_hosp"
        / "local_data"
        / "raw_data"
        / "zctas_2020.parquet"
    )

    # make a list of paths that we're going to use for each month
    ghsl_paths = [ghsl_2000, ghsl_2005, ghsl_2010, ghsl_2015, ghsl_2020]

    total_pop = []
    est = PopEstimator(admin_data=zctas_2020)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_total_pop, est, ghsl_paths[i], i)
            for i in range(len(ghsl_paths))
        ]
        for f in tqdm.tqdm(
            futures, total=len(ghsl_paths), desc="Calculating total population"
        ):
            result = f.result()
            if result is not None:
                total_pop.append(result)

    if total_pop:
        final_df = pd.concat(total_pop, ignore_index=True)
        output_path = "/Volumes/squirrel-utopia/national_wf_disaster_hosp/local_data/intermediate_data/total_population_counts_by_zcta.csv"
        final_df.to_csv(output_path, index=False)

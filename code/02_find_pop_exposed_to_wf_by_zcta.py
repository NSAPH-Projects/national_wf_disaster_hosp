from concurrent.futures import ProcessPoolExecutor
import tqdm
import pathlib
import os
import pandas as pd

# Stable local version
from popexposure.pop_estimator import PopEstimator


def process_exposure(est, pop_data_path, hazard_data_path, month):
    est.pop_data = pop_data_path
    exposed_pop_df = est.est_exposed_pop(
        hazard_data=hazard_data_path, hazard_specific=False
    )
    if exposed_pop_df is not None:
        exposed_pop_df["month"] = month
    return exposed_pop_df


if __name__ == "__main__":
    print("Running find_pop_exposed_to_wf_by_zcta.")

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

    all_wf_dat = (
        base_path / "national_wf_disaster_hosp" / "local_data" / "monthly_wf_exposure"
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

    # rep pattern
    rep_pattern = [3 * 12, 5 * 12, 5 * 12, 5 * 12, 1 * 12]

    # list of ghsls to use for months
    repeated_paths = [
        path for path, count in zip(ghsl_paths, rep_pattern) for _ in range(count)
    ]

    all_wf_exposure = sorted(
        [all_wf_dat / file for file in os.listdir(all_wf_dat) if "month" in file]
    )

    months = pd.date_range("2000-01", "2018-12", freq="MS").strftime("%Y-%m").tolist()

    exposed_pop = []
    est = PopEstimator(admin_data=zctas_2020)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                process_exposure, est, repeated_paths[i], all_wf_exposure[i], months[i]
            )
            for i in range(0, len(all_wf_exposure))
        ]
        for f in tqdm.tqdm(
            futures, total=len(all_wf_exposure), desc="Calculating exposed population"
        ):
            result = f.result()
            if result is not None:
                exposed_pop.append(result)

    if exposed_pop:
        final_df = pd.concat(exposed_pop, ignore_index=True)
        output_path = "/Volumes/squirrel-utopia/national_wf_disaster_hosp/local_data/intermediate_data/exposed_population_counts_by_zcta.csv"
        final_df.to_csv(output_path, index=False)

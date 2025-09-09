from concurrent.futures import ThreadPoolExecutor
import tqdm
import pathlib
import os
import pandas as pd


from popexposure.pop_estimator import PopEstimator


def process_exposure(i, est, months):
    exposed_pop_df = est.est_exposed_pop(
        hazard_data=all_wf_exposure[i],
        hazard_specific=False,
        pop_data=repeated_paths[i],
    )
    if exposed_pop_df is not None:
        exposed_pop_df["month"] = months[i]
    print("completed one iteration")
    print(i)
    return exposed_pop_df


def process_total_pop(i, est):
    total_pop_df = est.est_total_pop(pop_data=ghsl_paths[i])
    total_pop_df["ghsl_used"] = i + 1
    print("completed one iteration")
    print(i)
    return total_pop_df


if __name__ == "__main__":

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

    wf_dat = (
        base_path / "national_wf_disaster_hosp" / "local_data" / "monthly_wf_exposure"
    )

    all_wf_dat = wf_dat / "all_analysis"

    zctas_2020 = (
        base_path / "national_wf_disaster_hosp" / "local_data" / "zctas_2020.parquet"
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

    with ThreadPoolExecutor() as executor:
        # Use a list comprehension to pass est to each call
        futures = [
            executor.submit(process_exposure, i, est, months)
            for i in range(len(all_wf_exposure))
        ]
        for f in tqdm.tqdm(
            futures, total=len(all_wf_exposure), desc="Calculating exposed population"
        ):
            result = f.result()
            if result is not None:
                exposed_pop.append(result)

    if exposed_pop:
        final_df = pd.concat(exposed_pop, ignore_index=True)
        output_path = "/Volumes/squirrel-utopia 1/national_wf_disaster_hosp/local_data"
        final_df.to_csv("exposed_population_counts_by_zcta.csv", index=False)

    total_pop = []
    est = PopEstimator(admin_data=zctas_2020)

    with ThreadPoolExecutor() as executor:
        # Use a list comprehension to pass est to each call
        futures = [
            executor.submit(process_total_pop, i, est) for i in range(len(ghsl_paths))
        ]
        for f in tqdm.tqdm(
            futures, total=len(ghsl_paths), desc="Calculating total population"
        ):
            result = f.result()
            if result is not None:
                total_pop.append(result)

    if total_pop:
        final_df = pd.concat(total_pop, ignore_index=True)
        output_path = "/Volumes/squirrel-utopia 1/national_wf_disaster_hosp/local_data"
        final_df.to_csv("total_population_counts_by_zcta.csv", index=False)

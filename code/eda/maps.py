import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend for safe PDF/image output on macOS
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


if __name__ == "__main__":

    # read exposure data
    df = pd.read_csv(
        "/Volumes/squirrel-utopia 1/national_wf_disaster_hosp/local_data/exposed_population_counts_by_zcta.csv"
    )

    # add month to exposure data
    month_lookup = pd.DataFrame(
        {
            "month": range(1, 229),  # 1 to 228
            "month_str": pd.date_range("2000-01-01", periods=228, freq="MS").strftime(
                "%Y-%m"
            ),
        }
    )
    df = df.merge(month_lookup, on="month", how="left")
    # fix admin unit type
    df["ID_admin_unit"] = df["ID_admin_unit"].astype(str)

    # read shapefile
    gdf = gpd.read_file(
        "/Volumes/squirrel-utopia 1/national_wf_disaster_hosp/local_data/zctas_2020.geojson"
    )
    # fix admin unit type
    gdf["ID_admin_unit"] = gdf["ID_admin_unit"].astype(str)

    # split exposure data into monthly exposure
    dfs_by_month = []
    for m in df["month"].unique():
        dfs_by_month.append(df[df["month"] == m])

    # get a shapefile with exposure info for each month
    merged_by_month = []
    for i in range(1, len(dfs_by_month)):
        merged = gdf.merge(dfs_by_month[i], how="left")
        # fill in unexposed units with 0
        for col in ["exposed_main", "exposed_larger", "exposed_smaller"]:
            merged[col] = merged[col].fillna(0)
        # fill in info on month
        month_val = dfs_by_month[i]["month"].dropna().unique()[0]
        month_str_val = dfs_by_month[i]["month_str"].dropna().unique()[0]
        merged["month"] = month_val
        merged["month_str"] = month_str_val
        # filter to california for ease of plotting
        merged = merged[merged["ID_admin_unit"].str[:3].astype(int).between(900, 961)]
        merged_by_month.append(merged)

    # value columns
    value_cols = ["exposed_main"]  # can add others in future

    for value in value_cols:
        with PdfPages(f"{value}_maps.pdf") as pdf:
            for merged in tqdm(merged_by_month, desc=f"Mapping {value}"):
                t = merged["month_str"].unique()[0]
                fig, ax = plt.subplots(figsize=(8, 8))
                merged.plot(column=value, ax=ax, legend=True, cmap="viridis")
                ax.set_title(f"{value} at time {t}")
                ax.axis("off")
                pdf.savefig(fig)
                plt.close(fig)

import geopandas as gpd
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import rasterio
import rasterio.windows
from tqdm import tqdm
from shapely.validation import make_valid
from shapely.ops import unary_union
from shapely import wkt
from pathlib import Path
from exactextract import exact_extract
import warnings
import functools
from osgeo import gdal

# Suppress warnings about centroid crs but raise exceptions
gdal.UseExceptions()
warnings.filterwarnings("ignore")


class PopEstimator:

    def __init__(self):
        """
        Initialize the PopEstimator class, used to find populations exposed to
        environmental hazards.
        Init with empty attributes for hazard and spatial unit data.
        """
        self.hazard_data = None
        self.spatial_units = None

    def prepare_data(self, path_to_data: str, geo_type: str) -> gpd.GeoDataFrame:
        """
        Read, clean, and preprocess geospatial data to pass to
        estimate_exposed_pop.
        This includes cleaning geometries if there is hazard data, buffering
        that hazard data with specified buffer distances.

        If hazard data is passed it must contain a str column called
        'ID_hazard', a geometry column called 'geometry', and numeric buffer
        distance columns beginning with 'buffer_dist_' for any buffer distances
        to be applied to hazards.

        If spatial unit data is passed it must contain a str column called
        'ID_spatial_unit' and a geometry column called 'geometry'.

        Results set PopEstimator class attributes and are also returned.
        If the file is empty, None is returned.

        :param path_to_data: Path to the input geospatial data file, either a
        hazard file or spatial unit file in .geojson or .parquet format, with
        appropriate columns.
        :type path_to_data: str
        :param geo_type: Type of data to process ('hazard' or 'spatial_unit').
        :type geo_type: str
        :returns: Cleaned and processed GeoDataFrame containing ID, geometries,
        and if hazard data, buffered hazard columns.
        :rtype: geopandas.GeoDataFrame
        """
        # self._print_geo_message(geo_type)
        shp_df = self._read_data(path_to_data)
        if shp_df.empty:
            return None
        shp_df = self._remove_missing_geometries(shp_df)
        shp_df = self._make_geometries_valid(shp_df)
        shp_df = self._reproject_to_wgs84(shp_df)

        if geo_type == "hazard":
            shp_df = self._add_utm_projection(shp_df)
            shp_df = self._add_buffered_geoms(shp_df)
            # Find all buffered hazard columns
            buffered_cols = [
                col for col in shp_df.columns if col.startswith("buffered_hazard")
            ]
            cols = ["ID_hazard"] + buffered_cols
            selected = shp_df[cols].copy()
            # Set the first buffered hazard column as the geometry
            if buffered_cols:
                selected = selected.set_geometry(buffered_cols[0], crs=shp_df.crs)
            self.hazard_data = selected

        elif geo_type == "spatial_unit":
            selected = shp_df
            self.spatial_units = selected

        return selected

    def estimate_exposed_pop(
        self,
        pop_path: str,
        hazard_specific: bool,
        hazards: gpd.GeoDataFrame = None,
        spatial_units: gpd.GeoDataFrame = None,
    ) -> gpd.GeoDataFrame:
        """
        Estimate the population exposed to hazards, optionally within spatial
        units.

        This method calculates the sum of raster values (e.g., population)
        within hazard geometries,
        or within the intersection of hazard geometries and spatial units. It supports both hazard-specific
        and combined hazard analyses, and can use pre-loaded or provided hazard and spatial unit data.

        Parameters
        ----------
        pop_path : str
            Path to the population raster file.
        hazard_specific : bool
            If True, exposure is calculated for each hazard individually.
            If False, hazard geometries are combined before exposure calculation.
        hazards : geopandas.GeoDataFrame, optional
            GeoDataFrame containing hazard geometries and buffer columns.
            If None, uses self.hazard_data.
        spatial_units : geopandas.GeoDataFrame, optional
            GeoDataFrame containing spatial unit geometries.
            If None, uses self.spatial_units.

        Returns
        -------
        geopandas.GeoDataFrame
            GeoDataFrame with ID columns and one or more 'exposed' columns indicating
            the sum of raster values within each geometry or intersection.
        """

        if hazards is None:
            hazards = self.hazard_data
        if spatial_units is None:
            spatial_units = self.spatial_units
        if hazards is None:
            return None

        if spatial_units is None:
            if not hazard_specific:
                hazards = self._combine_geometries(hazards)
            exposed = self._mask_raster_partial_pixel(hazards, pop_path)
            self.exposed = exposed
            return exposed

        else:
            if not hazard_specific:
                hazards = self._combine_geometries(hazards)
            intersected_hazards = self._get_unit_hazard_intersections(
                hazards=hazards, spatial_units=spatial_units
            )
            exposed = self._mask_raster_partial_pixel(
                intersected_hazards, raster_path=pop_path
            )
            self.exposed = exposed
            return exposed

    # def estimate_pop(self, pop_path: str, spatial_unit: str):

    # --- Helper methods below ---

    def _print_geo_message(self, geo_type: str):
        """
        Print a message describing the type of geospatial data being processed.

        :param geo_type: Type of data ('hazard' or 'spatial_unit').
        :type geo_type: str
        """
        if geo_type == "hazard":
            print("Reading data and finding best UTM projection for hazard geometries")
        elif geo_type == "spatial_unit":
            print("Reading spatial unit geometries")

    def _read_data(self, path: str) -> gpd.GeoDataFrame:
        """
        Read geospatial data from a file.

        :param path: Path to the data file (.geojson or .parquet).
        :type path: str
        :returns: Loaded GeoDataFrame.
        :rtype: geopandas.GeoDataFrame
        :raises FileNotFoundError: If the file type is unsupported.
        """
        path = Path(path)
        if path.suffix == ".geojson":
            shp_df = gpd.read_file(path)
        elif path.suffix == ".parquet":
            shp_df = gpd.read_parquet(path)
        else:
            raise FileNotFoundError(f"File not found or unsupported file type: {path}")
        return shp_df

    def _remove_missing_geometries(self, shp_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Remove rows from hazard dataframe or spatial unit dataframe with missing or empty geometries.

        :param shp_df: Input GeoDataFrame.
        :type shp_df: geopandas.GeoDataFrame
        :returns: GeoDataFrame with missing geometries removed.
        :rtype: geopandas.GeoDataFrame
        """
        return shp_df[shp_df["geometry"].notnull() & ~shp_df["geometry"].is_empty]

    def _make_geometries_valid(self, shp_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Make all geometries in the GeoDataFrame valid.

        :param shp_df: Input GeoDataFrame.
        :type shp_df: geopandas.GeoDataFrame
        :returns: GeoDataFrame with valid geometries.
        :rtype: geopandas.GeoDataFrame
        """
        shp_df["geometry"] = shp_df["geometry"].apply(make_valid)
        return shp_df

    def _reproject_to_wgs84(self, shp_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        if shp_df.crs != "EPSG:4326":
            shp_df = shp_df.to_crs("EPSG:4326")
        return shp_df

    def _get_best_utm_projection(self, lat, lon):
        """
        Calculate the best UTM projection EPSG code for a given latitude and longitude.

        :param lat: Latitude.
        :type lat: float
        :param lon: Longitude.
        :type lon: float
        :returns: EPSG code string for the best UTM projection.
        :rtype: str
        """
        zone_number = (lon + 180) // 6 + 1
        hemisphere = 326 if lat >= 0 else 327
        epsg_code = hemisphere * 100 + zone_number
        return f"EPSG:{int(epsg_code)}"

    def _add_utm_projection(self, shp_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add a column with the best UTM projection for each geometry in the GeoDataFrame.

        :param ch_shp: Input GeoDataFrame.
        :type ch_shp: geopandas.GeoDataFrame
        :returns: GeoDataFrame with UTM projection column added.
        :rtype: geopandas.GeoDataFrame
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # get lat and lon
            shp_df["centroid_lon"] = shp_df.centroid.x
            shp_df["centroid_lat"] = shp_df.centroid.y
            # get projection for each hazard
            shp_df["utm_projection"] = shp_df.apply(
                lambda row: self._get_best_utm_projection(
                    lat=row["centroid_lat"], lon=row["centroid_lon"]
                ),
                axis=1,
            )
            # select id, geometry, buffer dist, and utm projection
            buffer_cols = [
                col for col in shp_df.columns if col.startswith("buffer_dist")
            ]
            shp_df = shp_df[
                ["ID_hazard"] + buffer_cols + ["geometry", "utm_projection"]
            ]
            return shp_df

    def _add_buffered_geoms(self, shp_df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add a column with buffered geometries for each row, using the best UTM projection.

        :param shp_df: Input GeoDataFrame.
        :type shp_df: geopandas.GeoDataFrame
        :returns: GeoDataFrame with buffered hazard geometry column added.
        :rtype: geopandas.GeoDataFrame
        """

        # Find all buffer_dist columns
        buffer_cols = [col for col in shp_df.columns if col.startswith("buffer_dist")]

        for buffer_col in buffer_cols:
            # Name for the new buffered geometry column
            suffix = buffer_col.replace("buffer_dist", "").strip("_")
            if suffix:
                new_col = f"buffered_hazard_{suffix}"
            else:
                new_col = "buffered_hazard"

            shp_df[new_col] = None  # Initialize column

            for index, row in shp_df.iterrows():
                best_utm = row["utm_projection"]
                hazard_geom = row["geometry"]

                # create geoseries in best projection
                geom_series = gpd.GeoSeries([hazard_geom], crs=shp_df.crs)
                geom_series_utm = geom_series.to_crs(best_utm)

                # buffer distance is in meters
                buffer_dist = row[buffer_col]
                buffered_hazard_geometry = geom_series_utm.buffer(buffer_dist).iloc[0]
                # back to OG
                buffered_hazard_geometry = (
                    gpd.GeoSeries([buffered_hazard_geometry], crs=best_utm)
                    .to_crs(shp_df.crs)
                    .iloc[0]
                )
                # add
                shp_df.at[index, new_col] = buffered_hazard_geometry

        return shp_df

    def _combine_geometries(
        self,
        shp_df: gpd.GeoDataFrame,
    ) -> gpd.GeoDataFrame:
        """
        Combine all geometries in columns starting with 'buffered_hazard' into single geometries, in chunks for efficiency.

        :param shp_df: Input GeoDataFrame.
        :type shp_df: geopandas.GeoDataFrame
        :returns: GeoDataFrame with one row and merged geometry columns for each buffer.
        :rtype: geopandas.GeoDataFrame
        """
        chunk_size = 500
        buffered_cols = [
            col for col in shp_df.columns if col.startswith("buffered_hazard")
        ]
        merged_geoms = {}
        for col in buffered_cols:
            geoms = list(shp_df[col])
            chunks = [
                geoms[i : i + chunk_size] for i in range(0, len(geoms), chunk_size)
            ]
            partial_unions = [unary_union(chunk) for chunk in chunks]
            final_union = unary_union(partial_unions)
            merged_geoms[col] = [final_union]
        merged_geoms["ID_hazard"] = ["merged_geoms"]
        combined_gdf = gpd.GeoDataFrame(merged_geoms, geometry=col, crs=shp_df.crs)
        return combined_gdf

    def _mask_raster_partial_pixel(self, shp_df: gpd.GeoDataFrame, raster_path: str):
        """
        Calculate the sum of raster values (e.g., population) within each geometry using exact_extract.

        :param shp_df: Input GeoDataFrame.
        :type shp_df: geopandas.GeoDataFrame
        :param raster_path: Path to the raster file.
        :type raster_path: str
        :returns: GeoDataFrame with an 'exposed' column containing the sum for each geometry.
        :rtype: geopandas.GeoDataFrame
        """
        print("Finding exposed population")

        # Open the raster file
        with rasterio.open(raster_path) as src:
            # Ensure CRS alignment
            if shp_df.crs != src.crs:
                shp_df = shp_df.to_crs(src.crs)

        geom_cols = [col for col in shp_df.columns if col.startswith("buffered_hazard")]

        for geom_col in geom_cols:
            # Set the current geometry column as the active geometry
            shp_df = shp_df.set_geometry(geom_col, crs=shp_df.crs)
            # Identify valid geometries
            valid_mask = (
                shp_df.geometry.notnull()
                & ~shp_df.geometry.is_empty
                & shp_df.geometry.is_valid
            )
            valid_gdf = shp_df[valid_mask].copy()
            # Initialize result column with zeros
            exposed_col = f"exposed{geom_col.replace('buffered_hazard', '').strip('')}"
            shp_df[exposed_col] = 0.0
            if not valid_gdf.empty:
                # Use exact_extract to calculate population sums for each geometry
                num_exposed = exact_extract(
                    raster_path,
                    valid_gdf,
                    "sum",
                )
                sums = [hazard["properties"]["sum"] for hazard in num_exposed]
                # Assign results back to the valid rows
                shp_df.loc[valid_mask, exposed_col] = sums

        # select just the ID columns and the exposure columns
        id_cols = [col for col in shp_df.columns if col.startswith("ID")]
        exposure_cols = [col for col in shp_df.columns if col.startswith("exposed")]
        out_cols = id_cols + exposure_cols
        shp_df = shp_df[out_cols]

        # final df
        return shp_df

    def _get_unit_hazard_intersections(self, hazards, spatial_units):
        intersections = {}
        for col in [c for c in hazards.columns if c.startswith("buffered_hazard")]:
            # Select only ID_hazard and the current geometry column
            hazards_subset = hazards[["ID_hazard", col]].copy()
            hazards_geom = hazards_subset.set_geometry(col, crs=hazards.crs)
            intersection = gpd.overlay(hazards_geom, spatial_units, how="intersection")
            intersection = self._remove_missing_geometries(intersection)
            intersection = self._make_geometries_valid(intersection)
            intersection = intersection.rename_geometry(col)
            intersection = intersection.set_geometry(col, crs=hazards.crs)

            intersections[col] = intersection
        intersected_dfs = [
            df for df in intersections.values() if df is not None and not df.empty
        ]

        intersected_hazards = functools.reduce(
            lambda left, right: pd.merge(
                left, right, on=["ID_hazard", "ID_spatial_unit"], how="outer"
            ),
            intersected_dfs,
        )
        return intersected_hazards

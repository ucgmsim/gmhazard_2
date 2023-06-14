from pathlib import Path

import numpy as np
from gmhazard_2 import dbs
from gmhazard_2 import distance
from gmhazard_2 import utils
from gmhazard_2 import test_utils


import pytest


class TestRxRy_OQ:
    """Test Rx and Ry calculation by comparing against the results from OQ"""

    @pytest.fixture(scope="session")
    def rupture_db_ffp(self):
        return Path(__file__).parent / "resources" / "source_db.hdf5"

    @pytest.fixture(scope="session")
    def source_definitions_dir(self):
        return Path(__file__).parent / "resources" / "source_definitions"

    @pytest.mark.parametrize(
        "scenario_id",
        [
            23000001,
            23000002,
            23000003,
            23000004,
            23000005,
            23000006,
            23000007,
            23000008,
            23000009,
            23000010,
        ],
    )
    def test_one(
        self, scenario_id: int, rupture_db_ffp: Path, source_definitions_dir: Path
    ):
        # Load data
        with dbs.SourceModelDB(rupture_db_ffp) as db:
            source_set_info = db.get_source_set_info()
            rupture_section_pts_df = db.get_flt_rupture_section_pts()
            rupture_scenarios_df = db.get_fault_rupture_scenarios()

        # Sort out the ids..
        id = source_set_info.index[0]
        id_2 = source_set_info.loc[id].id2
        set_id = source_set_info.loc[id].set_id
        rupture_sections_ffp = (
            source_definitions_dir / id / f"{id_2}-ruptures_sections.xml"
        )

        # Get the segment coords
        segment_coords, segment_section_ids = distance.get_segment_coords(
            rupture_section_pts_df
        )
        segment_nztm_coords = distance.segment_to_nztm(segment_coords)

        # Only interested in this set of sources
        rupture_section_pts_df = rupture_section_pts_df.loc[
            rupture_section_pts_df.set_id == set_id
        ]
        rupture_scenarios_df = rupture_scenarios_df.loc[
            rupture_scenarios_df.set_id == set_id
        ]
        section_ids = rupture_section_pts_df.section_id.unique()

        # Create the line objects for each section
        lines = utils.create_OQ_lines(
            section_ids, rupture_section_pts_df, rupture_sections_ffp, set_id
        )

        # Process each scenario
        (
            tupp,
            uupp,
            scenario_Rx,
            scenario_Ry,
            plons,
            plats,
            cur_segment_coords,
        ) = test_utils.process_rupture_scenario(
            rupture_scenarios_df,
            scenario_id,
            segment_coords,
            segment_nztm_coords,
            segment_section_ids,
            lines,
        )

        # Get the benchmark data
        cur_bench_dir = Path(__file__).parent / "benchmark_data" / str(scenario_id)
        bench_rx = np.load(cur_bench_dir / "Rx_GMHazard.npy")
        bench_ry = np.load(cur_bench_dir / "Ry_GMHazard.npy")

        assert np.allclose(scenario_Rx, bench_rx)
        assert np.allclose(scenario_Ry, bench_ry)


if __name__ == "__main__":
    TestRxRy_OQ().test_one()

import re
import time
from typing import Dict
from pathlib import Path
import xmltodict

import h5py
import pandas as pd
import numpy as np


def parse_rupture_sections(rupture_sections_ffp: Path):
    """
    Parses the rupture sections in the
    file into a dictionary
    """
    with rupture_sections_ffp.open("r") as f:
        doc = xmltodict.parse(f.read())

    doc = doc["nrml"]["geometryModel"]
    sections = {}
    for i, section in enumerate(doc["section"]):
        id = str(section["@id"])
        cur_positions = []
        for line_string in section["kiteSurface"]["profile"]:
            cur_positions.append(
                np.asarray(
                    [
                        float(cur_value)
                        for cur_value in line_string["gml:LineString"][
                            "gml:posList"
                        ].split()
                    ]
                ).reshape(2, 3)
            )

        sections[id] = np.concatenate(cur_positions, axis=0)

    return sections


def create_section_df(sections: Dict[int, np.ndarray]):
    """
    Creates a single dataframe
    containing data points for all
    given sections
    """
    dfs = []
    for i, (id, cur_coords) in enumerate(sections.items()):
        df = pd.DataFrame(data=cur_coords, columns=["lon", "lat", "depth"])
        df["nshm_section_id"] = id
        df["section_ix"] = i
        # df["section_id"] = id


        dfs.append(df)

    return pd.concat(dfs, axis=0)


def get_rupture_scenarios(rupture_scenario_db_ffp: Path, source_set_id_2: str):
    """Retrieve the rupture scenario parameters"""
    with h5py.File(rupture_scenario_db_ffp, "r") as db:
        assert len(db.keys()) == 1

        mags = db[f"{source_set_id_2}/mag"][:]
        rupture_idxs = db[f"{source_set_id_2}/rupture_idxs"][:]
        prob_occur = db[f"{source_set_id_2}/probs_occur"][:, 1]
        rake = db[f"{source_set_id_2}/rake"][:]

    return mags, rupture_idxs, prob_occur, rake


def get_tectonic_type(rupture_ffp: Path):
    """
    Retrieves the tectonic type
    from the rupture xml file
    """
    with rupture_ffp.open("r") as f:
        txt = f.read()

    matches = re.findall(r'tectonicRegion="(.*)"', txt)

    assert len(matches) == 1

    return matches[0]

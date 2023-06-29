import json
import re
from typing import Dict, List
from pathlib import Path
import xmltodict

import h5py
import pandas as pd
import numpy as np


from . import dbs
from . import constants


class SMLogicTreeBranch:
    def __init__(
        self,
        id: int,
        weight: float,
        flt_source_set_ids: List[int],
        ds_source_set_ids: List[int],
    ):
        self.id = id
        self.weight = weight

        self.flt_source_set_ids = flt_source_set_ids
        self.ds_source_set_ids = ds_source_set_ids


class SMLogicTree:
    def __init__(self, branches: Dict[int, SMLogicTreeBranch]):
        self.branches = branches


def parse_nshm_source_lt(source_lt_ffp: Path, source_db_ffp: Path):
    with source_lt_ffp.open("r") as f:
        source_model_data = json.load(f)

    with dbs.SourceModelDB(source_db_ffp) as db:
        source_set_info = db.get_source_set_info()

    branches = {}
    for i, cur_branch in enumerate(source_model_data):
        branches[i] = SMLogicTreeBranch(
            i,
            cur_branch["weight"],
            [
                source_set_info.loc[cur_id, "set_id"]
                for cur_id in cur_branch["input_ids"]
                if cur_id in source_set_info.index.values
            ],
            None,
        )

    return SMLogicTree(branches)


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

    return constants.TECT_TYPE_MAPPING[matches[0]].value

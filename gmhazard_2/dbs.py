import warnings
import datetime
import functools
from typing import Callable
from pathlib import Path
from contextlib import contextmanager

import tables
import numpy as np
import pandas as pd

from . import source_model


def check_open(f: Callable = None, *, writeable: bool = False):
    """Decorator that can be wrapped around methods/properties
    of the BaseDb (and subclasses) that require the database to be open
    See https://realpython.com/primer-on-python-decorators/#decorators-with-arguments
    """

    def decorator(f: Callable):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            instance = args[0]
            if not instance.is_open:
                raise Exception(
                    "The database has not been opened. Please open the "
                    "database before accessing this method/property."
                )
            if writeable and not instance.writeable:
                raise Exception(
                    "This method requires the database "
                    "to have been opened in write mode."
                )
            return f(*args, **kwargs)

        return wrapper

    if f is None:
        return decorator
    else:
        return decorator(f)


class SourceModelDB:
    def __init__(self, db_ffp: Path, writeable: bool = False):
        self.db_ffp = db_ffp
        self._db = None

        self._writeable = writeable

    @property
    def writeable(self):
        return self._writeable

    @check_open(writeable=False)
    def get_source_set_info(self):
        return self._db["source_sets_info"]

    @check_open(writeable=False)
    def get_flt_rupture_section_pts(self):
        """Gets the locations for each rupture section"""
        source_sets_info = self.get_source_set_info()

        section_points = []
        for cur_id, cur_row in source_sets_info.iterrows():
            if cur_row.type != "fault":
                continue

            cur_df = self._db[f"{cur_row.key}/section_points"]
            cur_df["set_id"] = cur_row.set_id
            section_points.append(cur_df)

        section_points_df = pd.concat(section_points, axis=0)

        return section_points_df

    @check_open(writeable=False)
    def get_fault_rupture_scenarios(self):
        source_set_info = self.get_source_set_info()

        rupture_scenarios = []
        for cur_id, cur_row in source_set_info.iterrows():
            if cur_row.type != "fault":
                continue

            cur_df = self._db[f"{cur_row.key}/rupture_scenarios"]
            cur_df["tect_type"] = cur_row.tect_type
            rupture_scenarios.append(cur_df)

        result_df = pd.concat(rupture_scenarios, axis=0)
        result_df["tect_type"] = pd.Categorical(result_df["tect_type"].values)
        return result_df

    @property
    def is_open(self):
        return self._db is not None

    def __enter__(self):
        """Defining __enter__ and __exit__ allows
        the class to be used as a with statement, i.e.

        with IMDB(file) as db:
            pass
        """
        self.open()
        return self

    def __exit__(self, *args) -> None:
        """Defining __enter__ and __exit__ allows
        the class to be used as a with statement, i.e.

        with IMDB(file) as db:
            pass
        """
        self.close()

    def open(self) -> None:
        """Opens the database"""
        mode = "a" if self._writeable else "r"
        self._db = pd.HDFStore(self.db_ffp, mode=mode)

    def close(self) -> None:
        """Close opened database"""
        self._db.close()
        self._db = None

    @contextmanager
    def use_db(self):
        self.open()
        yield self
        self.close()

    @classmethod
    def create(cls, source_definitions_dir: Path, output_ffp: Path):
        if output_ffp.exists():
            print(f"Specified DB already exists. Quitting.")
            return

        with pd.HDFStore(str(output_ffp), "w") as db:
            source_set_dirs = [
                cur_path
                for cur_path in source_definitions_dir.iterdir()
                if cur_path.is_dir()
            ]

            source_set_info = {}
            for i, cur_dir in enumerate(source_set_dirs, start=1):
                print(f"Processing {i}/{len(source_set_dirs)}")
                cur_source_set_id = cur_dir.name

                # Faults
                if cur_dir.name.startswith("SW"):
                    rupture_sections_ffp = list(
                        cur_dir.glob("*-ruptures_sections.xml")
                    )[0]
                    rupture_scenarios_db_ffp = (
                        hdf5_ffps := list(cur_dir.glob("*-ruptures.hdf5"))
                    )[0]
                    ruptures_ffp = list(cur_dir.glob("*-ruptures.xml"))[0]

                    # Sanity check
                    assert len(hdf5_ffps) == 1

                    cur_source_set_id_2 = rupture_sections_ffp.name.replace(
                        "-ruptures_sections.xml", ""
                    )

                    # Get the rupture sections
                    sections = source_model.parse_rupture_sections(rupture_sections_ffp)
                    section_points_df = source_model.create_section_df(sections)
                    section_points_df["section_id"] = (
                        i * int(1e4) + section_points_df.section_ix.values
                    )

                    # Get rupture scenarios parameters
                    (
                        mags,
                        section_ids,
                        prob_occur,
                        rake,
                    ) = source_model.get_rupture_scenarios(
                        rupture_scenarios_db_ffp, cur_source_set_id_2
                    )

                    # Create a nshm section id -> section id lookup
                    _,unique_ind = np.unique(section_points_df.nshm_section_id, return_index=True)
                    section_id_lookup = pd.Series(index=section_points_df.nshm_section_id.iloc[unique_ind].values,
                                                  data=section_points_df.section_id.iloc[unique_ind].values)

                    # Convert rupture indices to section ids
                    scenario_section_ids = [
                        section_id_lookup.loc[cur_id.decode().split()].values
                        for cur_id in section_ids
                    ]

                    # Combine into single df
                    rupture_scenarios_df = pd.DataFrame(
                        data=np.stack([mags, prob_occur, rake], axis=1),
                        columns=["mag", "prob_occur", "rake"],
                    )
                    rupture_scenarios_df["section_ids"] = scenario_section_ids
                    rupture_scenarios_df.index = (
                        i * int(1e6) + rupture_scenarios_df.index.values + 1
                    )
                    rupture_scenarios_df["set_id"] = i

                    # Get the tectonic type
                    tect_type = source_model.get_tectonic_type(ruptures_ffp)

                    cur_key = f"{cur_source_set_id}/{cur_source_set_id_2}"
                    source_set_info[cur_source_set_id] = dict(
                        set_id=i,
                        id2=cur_source_set_id_2,
                        tect_type=tect_type,
                        key=cur_key,
                        type="fault",
                    )

                    # Write
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        db[f"{cur_key}/section_points"] = section_points_df
                        db[f"{cur_key}/rupture_scenarios"] = rupture_scenarios_df
                # DS
                elif cur_dir.name.startswith("Rml"):
                    pass
                else:
                    raise NotImplementedError()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                db["source_sets_info"] = pd.DataFrame.from_dict(source_set_info).T

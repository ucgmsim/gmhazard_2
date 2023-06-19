from pathlib import Path

import warnings
import h5py
import pandas as pd
import numpy as np
import typer

from contextlib import contextmanager
from pygmt_helper import plotting as pygmt_plt
from gmhazard_2 import source_model
from gmhazard_2 import plotting
from gmhazard_2 import dbs

app = typer.Typer()



@app.command("create-source-model-db")
def create_source_model_db(source_definitions_dir: Path, output_ffp: Path):
    dbs.SourceModelDB.create(source_definitions_dir, output_ffp)


if __name__ == "__main__":
    app()

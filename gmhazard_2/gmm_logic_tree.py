from enum import Enum, auto
from pathlib import Path
from typing import Dict, Union, NamedTuple


import re

import pandas as pd
import xmltodict


from . import constants

from empirical.util.classdef import GMM


class GMMLogicTreeType(Enum):
    # Each branch of the logic
    # tree is a GMM
    # i.e. each branch is terminal
    GMM = auto()
    # Each branch of the logic
    # tree is a sub-logic tree
    # for a specific tectonic type
    TEC_TYPE = auto()


class GMMBranch:
    """
    Represents a terminal branch
    for a specific GMM and GMM parameter set
    """

    def __init__(
        self,
        branch_id: str,
        gmm: GMM,
        weight: float,
        sigma_mu_epsilon: float = None,
        gmm_params: Dict[str, float] = None,
    ):
        self.branch_id = branch_id
        self.gmm = gmm
        self.weight = weight

        self.gmm_params = gmm_params
        self.sigma_mu_epsilon = sigma_mu_epsilon


class GMMRunConfig(NamedTuple):
    """Represents a GMM run configuration"""

    gmm: GMM
    gmm_parameters: Dict[str, float]
    sigma_mu_epsilon: float

    @classmethod
    def from_branch(cls, branch: GMMBranch):
        return cls(
            gmm=branch.gmm,
            gmm_parameters=branch.gmm_params,
            sigma_mu_epsilon=branch.sigma_mu_epsilon,
        )

    def __str__(self):
        return f"{self.gmm.name} - GMM parameters: {self.gmm_parameters} " \
               f"- Sigma Mu Epsilon: {self.sigma_mu_epsilon}"

    def __eq__(self, other: "GMMRunConfig"):
        if self.gmm is not other.gmm:
            return False

        if type(self.gmm_parameters) != type(other.gmm_parameters):
            return False

        if len(self.gmm_parameters) != len(other.gmm_parameters):
            return False

        for k, v in self.gmm_parameters.items():
            if k not in other.gmm_parameters:
                return False
            if v != other.gmm_parameters[k]:
                return False

        return True

    def __hash__(self):
        if self.gmm_parameters is None:
            if self.sigma_mu_epsilon is None:
                return hash(self.gmm)
            else:
                return hash((self.gmm, self.sigma_mu_epsilon))
        else:
            keys = sorted(self.gmm_parameters.keys())
            return hash(
                (
                    self.gmm,
                    tuple((k, self.gmm_parameters[k]) for k in keys),
                    self.sigma_mu_epsilon,
                )
            )


class GMMLogicTree:
    """
    Represents a GMM logic tree
    These can be nested
    """

    def __init__(
        self,
        name: str,
        branches: Dict[
            Union[str, constants.TectonicType], Union[GMMBranch, "GMMLogicTree"]
        ],
        type: GMMLogicTreeType,
        tec_type: constants.TectonicType = None,
    ):
        self.name = name
        self.branches = branches
        self.type = type

        self.tec_type = tec_type


def get_tec_type_gmm_run_configs(gmm_lt: GMMLogicTree):
    """Collects the gmm run configs for each tectonic type"""
    if gmm_lt.type is not GMMLogicTreeType.TEC_TYPE:
        raise ValueError("Logic tree is not a Tectonic Type logic tree")

    gmm_run_configs = {}
    for cur_tect_type, cur_sub_tree in gmm_lt.branches.items():
        if cur_sub_tree.type is not GMMLogicTreeType.GMM:
            raise ValueError("Logic tree is not a GMM logic tree")

        # Get the run configurations
        cur_run_configs = [
            GMMRunConfig.from_branch(cur_branch)
            for cur_branch in cur_sub_tree.branches.values()
        ]
        # Sanity check
        assert len(set(cur_run_configs)) == len(cur_run_configs)

        gmm_run_configs[cur_tect_type] = cur_run_configs

    return gmm_run_configs


def parse_nshm_gmm_lt(gmm_lt_ffp: Path):
    """
    Parses the NSHM GMM LT definition xml
    file into a GMMLogicTree object
    """
    TECT_TYPE_MAPPING = {
        "Active Shallow Crust": constants.TectonicType.active_shallow,
        "Subduction Interface": constants.TectonicType.subduction_interface,
        "Subduction Intraslab": constants.TectonicType.subduction_slab,
    }

    GMM_MAPPING = {
        # Active Shallow Crust
        "Stafford2022": GMM.S_22,
        "Atkinson2022Crust": GMM.A_22,
        "AbrahamsonEtAl2014": GMM.ASK_14,
        "BooreEtAl2014": GMM.BSSA_14,
        "CampbellBozorgnia2014": GMM.CB_14,
        "ChiouYoungs2014": GMM.CY_14,
        "Bradley2013": GMM.Br_13,
        # Subduction Interface
        "Atkinson2022SInter": GMM.A_22,
        "AbrahamsonGulerce2020SInter": GMM.AG_20,
        "ParkerEtAl2021SInter": GMM.P_21,
        "KuehnEtAl2020SInter": GMM.K_20,
        # Subduction Slab
        "Atkinson2022SSlab": GMM.A_22,
        "AbrahamsonGulerce2020SSlab": GMM.AG_20,
        "ParkerEtAl2021SSlab": GMM.P_21,
        "KuehnEtAl2020SSlab": GMM.K_20,
    }

    # Parse the xml
    with gmm_lt_ffp.open("r") as f:
        doc = xmltodict.parse(f.read())
    lt = doc["nrml"]["logicTree"]["logicTreeBranchSet"]

    # Create the sub-trees for each tectonic type
    tec_type_lts = {}
    for cur_tec_lt in lt:
        cur_branches = {}
        for cur_branch_xml in cur_tec_lt["logicTreeBranch"]:
            branch_id = cur_branch_xml["@branchID"]
            uncertainty_model_str = cur_branch_xml["uncertaintyModel"]
            split_str = uncertainty_model_str.split("\n")

            weight = float(cur_branch_xml["uncertaintyWeight"])
            gmm = GMM_MAPPING[split_str[0].strip("[]")]

            # Uncertainty details
            gmm_params, sigma_mu_epsilon = {}, None
            matches = re.findall(r"(.+) = (.+)", uncertainty_model_str)
            for cur_match in matches:
                key = cur_match[0].strip()
                if key == "sigma_mu_epsilon":
                    sigma_mu_epsilon = (
                        v if (v := float(cur_match[1].strip())) != 0.0 else None
                    )
                else:
                    gmm_params[key] = cur_match[1].strip('"')

            cur_branches[branch_id] = GMMBranch(
                branch_id,
                gmm,
                weight,
                sigma_mu_epsilon=sigma_mu_epsilon,
                gmm_params=gmm_params if len(gmm_params) > 0 else None,
            )

        cur_tec_type = TECT_TYPE_MAPPING[cur_tec_lt["@applyToTectonicRegionType"]]
        tec_type_lts[cur_tec_type] = GMMLogicTree(
            cur_tec_type.name, cur_branches, GMMLogicTreeType.GMM
        )

    # Complete GMM logic tree
    gmm_lt = GMMLogicTree("nshm_gmm_lt", tec_type_lts, GMMLogicTreeType.TEC_TYPE)

    return gmm_lt

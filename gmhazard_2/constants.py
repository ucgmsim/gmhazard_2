from enum import Enum


class TectonicType(Enum):
    active_shallow = "active_shallow"
    subduction_interface = "subduction_interface"
    subduction_slab = "subduction_slab"

TECT_TYPE_MAPPING = {
    "Active Shallow Crust": TectonicType.active_shallow,
    "Subduction Interface": TectonicType.subduction_interface,
    "Subduction Intraslab": TectonicType.subduction_slab,
}


SA_PERIODS = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.12,
    0.15,
    0.17,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    1.0,
    1.25,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    7.5,
    10.0,
]

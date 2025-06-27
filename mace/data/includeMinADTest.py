from atomic_data import AtomicData
from mace.tools import AtomicNumberTable
import torch


print("Bonjour, Pohl")

class DummyConfig:
    def __init__(self):
        self.atomic_numbers = [1,8]
        self.positions = [[0.0,0.0,0.0],[4.0,0.0,0.0]]
        self.pbc = [False, False, False]
        self.cell = None
        self.head = "Default"

        self.properties = {
            "energy": -76.0,
            "forces": [[0.0,0.1,0.2],[0.1,0.2,0.0]],
            "dipole": [0.0,0.0,0.0],
            "charges": [-0.8,0.8],
            "m1":[0.1,0.2],
            "m2":[0.5,0.6],
            "m3":[0.3,0.1],
            "Veff":[0.5,0.5]
            }
        self.property_weights = {
            "energy": 1.0,
            "forces": 1.0,
            "dipole": 1.0,
            "charges": 1.0,
            "m1": 1.0,
            "m2": 1.0,
            "m3": 1.0,
            "Veff": 1.0
        }

        self.weight = 1.0

config = DummyConfig()
z_table = AtomicNumberTable([1, 8])

data = AtomicData.from_config(config, z_table=z_table, cutoff=5.0)

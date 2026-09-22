"""One tuning file shared by the simulator and the real car.

    tuning = load_tuning("tuning.json")        # missing file -> defaults
    pipeline = tuning.make_pipeline()

The simulator's tuning panel edits these same values and can export
`tuning.json`; copy that file to the Pi and it drives the real ADAS.
"""

import json
import os
from dataclasses import asdict, dataclass, field, fields

from .aeb import AEBConfig, SpeedModel
from .pipeline import AdasPipeline
from .servo import ServoCalibration
from .vehicle_params import VehicleParams


@dataclass
class Tuning:
    aeb: AEBConfig = field(default_factory=AEBConfig)
    speed_model: SpeedModel = field(default_factory=SpeedModel)
    vehicle: VehicleParams = field(default_factory=VehicleParams)
    servo: ServoCalibration = field(default_factory=ServoCalibration)
    lidar_min_range: float = 0.2

    def make_pipeline(self):
        return AdasPipeline(self.vehicle, self.aeb, self.speed_model, self.lidar_min_range)

    def to_dict(self):
        return {"aeb": asdict(self.aeb), "speed_model": asdict(self.speed_model),
                "vehicle": asdict(self.vehicle), "servo": asdict(self.servo),
                "lidar_min_range": self.lidar_min_range}

    @classmethod
    def from_dict(cls, d):
        def build(klass, data):
            names = {f.name for f in fields(klass)}
            return klass(**{k: v for k, v in (data or {}).items() if k in names})
        return cls(aeb=build(AEBConfig, d.get("aeb")), speed_model=build(SpeedModel, d.get("speed_model")),
                   vehicle=build(VehicleParams, d.get("vehicle")), servo=build(ServoCalibration, d.get("servo")),
                   lidar_min_range=d.get("lidar_min_range", 0.2))


def load_tuning(path="tuning.json"):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return Tuning.from_dict(json.load(f))
    return Tuning()


def save_tuning(tuning, path="tuning.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tuning.to_dict(), f, indent=2)

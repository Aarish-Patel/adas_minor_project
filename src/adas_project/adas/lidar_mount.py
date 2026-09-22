"""Real-car LiDAR mounting calibration, measured directly against the sensor (not the
rear axle) so it matches what a single LiDAR scan gives you: raw range + raw angle.

These are DIFFERENT numbers from VehicleParams' front_overhang/rear_overhang, which are
measured from the axles for the simulator's curve geometry. yaw_offset_deg in particular
has no simulator equivalent: it only exists because a real sensor can be mounted rotated
relative to the chassis, which the simulator assumes away.
"""

from dataclasses import asdict, dataclass


@dataclass
class LidarMount:
    yaw_offset_deg: float = 0.0     # raw LiDAR angle that is actually the car's straight-ahead
    front_overhang_m: float = 0.16  # LiDAR to front bumper (measured with a ruler)
    rear_overhang_m: float = 0.17   # LiDAR to rear bumper
    left_overhang_m: float = 0.10   # LiDAR to left side
    right_overhang_m: float = 0.10  # LiDAR to right side
    min_valid_range_m: float = 0.20  # ignore raw readings closer than this (self-hits: wires,
                                     # mount brackets, etc. sitting in the beam path)

    def to_car_angle(self, raw_angle_deg):
        """Raw LiDAR angle (0..360, sensor's own zero) -> car frame (-180..180, 0 = ahead)."""
        a = (raw_angle_deg - self.yaw_offset_deg) % 360
        return a if a <= 180 else a - 360

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        from dataclasses import fields
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})

"""8-D waypoint table -> JAKA Zu3 main chain.

waypoints  schema, strict validation, control-rate interpolation
replay     the executor: IK + gravity feedforward + gripper + rendering
"""

from .waypoints import (Waypoint, WaypointError, Trajectory, Table, interp_rotvec,
                        load_table, parse_table, plan_trajectory, SETTLE_S)

__all__ = ["Waypoint", "WaypointError", "Trajectory", "Table", "interp_rotvec",
           "load_table", "parse_table", "plan_trajectory", "SETTLE_S"]

#!/usr/bin/env python
"""
| File: 15_fixedwing_ardupilot_multi_vehicle.py
| Author: Pegasus Simulator Team
| License: BSD-3-Clause. Copyright (c) 2025, Pegasus Simulator Team. All rights reserved.
| Description: Example demonstrating how to spawn multiple fixed-wing aircraft with an ArduPilot (ArduPlane) backend.
|
| Usage:
|   1. Make sure ArduPilot is installed (see docs/source/features/ardupilot.rst)
|   2. Build ArduPlane SITL: ./waf configure --board sitl && ./waf plane
|   3. Run this script
|
| Notes:
|   - Each aircraft runs its own ArduPilot instance on UDP ports 14550 + (vehicle_id * 10)
|   - Use MAVProxy or a GCS to arm, takeoff, and fly (e.g., FBWA/TAKEOFF/AUTO)
|   - Fixed-wing aircraft need forward airspeed to generate lift
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.ardupilot_mavlink_backend import (
    ArduPilotMavlinkBackend, ArduPilotMavlinkBackendConfig)
from pegasus.simulator.logic.vehicles.fixedwing import FixedWing, FixedWingConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from scipy.spatial.transform import Rotation


class PegasusFixedWingMultiVehicleApp:
    """
    Example application demonstrating multiple fixed-wing aircraft with ArduPilot.
    """

    def __init__(self, num_vehicles: int = 3, spacing: float = 25.0):
        """
        Method that initializes the application and sets up the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Flat terrain is ideal for fixed-wing takeoff and taxi
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])

        # Spawn multiple fixed-wing aircraft with ArduPilot backends
        for vehicle_id in range(num_vehicles):
            self.create_fixedwing_vehicle(vehicle_id, spacing)

        # Reset the simulation environment so that all articulations are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def create_fixedwing_vehicle(self, vehicle_id: int, spacing: float):
        """
        Create a fixed-wing aircraft with ArduPilot backend.

        Args:
            vehicle_id (int): Unique identifier for the vehicle
            spacing (float): Spacing between aircraft along the Y axis
        """

        # Create the fixed-wing configuration
        config_fixedwing = FixedWingConfig()
        config_fixedwing.control_input_mode = "pwm"

        # Configure ArduPilot backend for fixed-wing (ArduPlane)
        backend_config = ArduPilotMavlinkBackendConfig({
            "vehicle_id": vehicle_id,
            "ardupilot_autolaunch": True,
            "ardupilot_dir": self.pg.ardupilot_path,
            "ardupilot_vehicle_model": "plane",
            "num_rotors": 4,
            # Map ArduPilot PWM outputs directly to FixedWing PWM inputs
            "input_offset": [1.0, 1.0, 1.0, 1.0],
            "input_scaling": [1000.0, 1000.0, 1000.0, 1000.0],
            "input_min": 1000,
            "input_max": 2000,
            "zero_position_armed": [0.0, 0.0, 0.0, 0.0],
        })

        config_fixedwing.backends = [
            ArduPilotMavlinkBackend(config=backend_config),
        ]

        # Spawn the fixed-wing aircraft facing +X with a small height for hand-launch
        FixedWing(
            f"/World/fixedwing{vehicle_id}",
            ROBOTS["FixedWing"],
            vehicle_id,
            [0.0, spacing * vehicle_id, 0.5],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_fixedwing,
        )

        mavlink_port = 14550 + vehicle_id * 10
        carb.log_info(
            f"Fixed-wing {vehicle_id} ready (sysid {vehicle_id + 1}, udp {mavlink_port})"
        )

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running() and not self.stop_sim:

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)

        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    """Main entry point for the fixed-wing multi-vehicle example."""

    print("=" * 70)
    print("Fixed-Wing Multi-Vehicle Simulation with ArduPilot")
    print("=" * 70)
    print()
    print("This example spawns multiple fixed-wing aircraft controlled by ArduPlane.")
    print()
    print("After the simulation starts:")
    print("  - Vehicle 0: udp:127.0.0.1:14550 (sysid 1)")
    print("  - Vehicle 1: udp:127.0.0.1:14560 (sysid 2)")
    print("  - Vehicle 2: udp:127.0.0.1:14570 (sysid 3)")
    print()
    print("Example MAVProxy commands (vehicle 0):")
    print("  mavproxy.py --master=udp:127.0.0.1:14550")
    print("  mode FBWA")
    print("  arm throttle")
    print("  rc 3 1700")
    print()
    print("You can switch to TAKEOFF or AUTO modes if you have a mission loaded.")
    print("=" * 70)

    pg_app = PegasusFixedWingMultiVehicleApp()
    pg_app.run()


if __name__ == "__main__":
    main()

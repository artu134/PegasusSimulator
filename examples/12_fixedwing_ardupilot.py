#!/usr/bin/env python
"""
| File: 12_fixedwing_ardupilot.py
| Author: Pegasus Simulator Team
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Example demonstrating how to spawn a fixed-wing aircraft with ArduPilot (ArduPlane) backend.
|
| Usage:
|   1. Make sure ArduPilot is installed (see docs/source/features/ardupilot.rst)
|   2. Run this script
|   3. Once the aircraft spawns and ArduPilot connects, use MAVProxy or a GCS to control it
|
| Notes:
|   - Fixed-wing aircraft need forward velocity to generate lift
|   - The aircraft starts on the ground; use hand-launch or give it initial velocity
|   - ArduPlane uses different flight modes than ArduCopter (FBWA, STABILIZE, AUTO, etc.)
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
# Note: this simulation app must be instantiated right after the SimulationApp import, otherwise the simulator will crash
# as this is the object that will load all the extensions and load the actual simulator.
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

# Import the Pegasus API for simulating aircraft
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.ardupilot_mavlink_backend import (
    ArduPilotMavlinkBackend, ArduPilotMavlinkBackendConfig)
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.fixedwing import FixedWing, FixedWingConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from scipy.spatial.transform import Rotation


class PegasusFixedWingApp:
    """
    Example application demonstrating fixed-wing aircraft simulation with ArduPilot.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA (flat terrain is good for takeoff)
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])

        # Create and spawn a fixed-wing aircraft
        self.create_fixedwing_vehicle(vehicle_id=0)

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

        # Auxiliar variable for the timeline callback example
        self.stop_sim = False

    def create_fixedwing_vehicle(self, vehicle_id: int):
        """
        Create a fixed-wing aircraft with ArduPilot backend.

        Args:
            vehicle_id (int): Unique identifier for the vehicle
        """

        # Create the fixed-wing configuration
        config_fixedwing = FixedWingConfig()
        
        # Configure ArduPilot backend for fixed-wing (ArduPlane)
        # Note: ArduPlane uses different vehicle models than ArduCopter
        backend_config = ArduPilotMavlinkBackendConfig({
            "vehicle_id": vehicle_id,
            "ardupilot_autolaunch": True,
            "ardupilot_dir": self.pg.ardupilot_path,
            # Use ArduPlane SITL model
            # Options include: "plane", "plane-elevon", "plane-vtail", etc.
            "ardupilot_vehicle_model": "plane",
            # For fixed-wing, we have 4 control channels:
            # Channel 0: Aileron (roll control)
            # Channel 1: Elevator (pitch control)
            # Channel 2: Throttle
            # Channel 3: Rudder (yaw control)
            "num_rotors": 4,  # Actually control channels, not rotors
            "input_offset": [0.0, 0.0, 0.0, 0.0],
            "input_scaling": [1.0, 1.0, 1.0, 1.0],
            "input_min": 1000,
            "input_max": 2000,
            "zero_position_armed": [0.0, 0.0, 0.0, 0.0],
        })
        
        # Set up backends
        config_fixedwing.backends = [
            ArduPilotMavlinkBackend(config=backend_config),
            # Optionally add ROS2 backend for sensor publishing
            # ROS2Backend(vehicle_id=vehicle_id, config={
            #     "namespace": "fixedwing",
            #     "pub_sensors": True,
            #     "pub_graphical_sensors": True,
            #     "pub_state": True,
            #     "sub_control": False,
            #     "pub_tf": True,
            # })
        ]
        
        # Spawn the fixed-wing aircraft
        # Start at a reasonable height for hand-launch simulation
        # The aircraft is oriented facing forward (positive X direction)
        FixedWing(
            f"/World/fixedwing{vehicle_id}",
            ROBOTS['FixedWing'],
            vehicle_id,
            # Start position: slightly elevated for easier takeoff
            [0.0, 0.0, 0.5],
            # Initial orientation: facing forward (no rotation)
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_fixedwing,
        )
        
        carb.log_info(f"Fixed-wing aircraft {vehicle_id} spawned successfully")
        carb.log_info("To control the aircraft:")
        carb.log_info("  1. Wait for ArduPilot to connect")
        carb.log_info("  2. Use MAVProxy commands: mode FBWA, arm throttle, rc 3 1600")
        carb.log_info("  3. Or connect a GCS like QGroundControl")

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
    """Main entry point for the fixed-wing example."""
    
    # Print usage information
    print("=" * 60)
    print("Fixed-Wing Aircraft Simulation with ArduPilot")
    print("=" * 60)
    print()
    print("This example spawns a fixed-wing aircraft controlled by ArduPlane.")
    print()
    print("Prerequisites:")
    print("  - ArduPilot installed at ~/ardupilot (or configure path in Pegasus)")
    print("  - ArduPlane compiled: ./waf configure --board sitl && ./waf plane")
    print()
    print("After the simulation starts:")
    print("  1. ArduPlane SITL will auto-launch")
    print("  2. Connect with MAVProxy or a GCS")
    print("  3. Arm and set throttle to fly")
    print()
    print("Example MAVProxy commands:")
    print("  mode FBWA        # Fly-by-wire mode")
    print("  arm throttle     # Arm the aircraft")
    print("  rc 3 1600        # Set throttle to ~60%")
    print()
    print("=" * 60)
    
    # Instantiate the app
    pg_app = PegasusFixedWingApp()

    # Run the application loop
    pg_app.run()


if __name__ == "__main__":
    main()


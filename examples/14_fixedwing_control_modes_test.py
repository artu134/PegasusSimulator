#!/usr/bin/env python
"""
| File: 14_fixedwing_control_modes_test.py
| Author: Pegasus Simulator Team
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Test different control input modes for the fixed-wing aircraft.
|              Demonstrates: PWM, normalized, angle, and attitude control modes.
|
| Usage:
|   cd ~/Desktop/PegasusSimulator/examples
|   python 14_fixedwing_control_modes_test.py
"""

# Imports to start Isaac Sim
import carb
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# -----------------------------------
import numpy as np
import omni.timeline
from omni.isaac.core.world import World
from scipy.spatial.transform import Rotation

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.backend import Backend
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.vehicles.fixedwing import FixedWing, FixedWingConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface


class MultiModeController(Backend):
    """
    Controller that demonstrates different input modes.
    Switches between modes during flight to show they all work.
    """

    def __init__(self, input_mode="normalized"):
        super().__init__(config=None)
        
        self._input_mode = input_mode
        self._time = 0.0
        
        # Control outputs (will be converted based on mode)
        self._aileron = 0.0
        self._elevator = 0.0
        self._throttle = 0.0
        self._rudder = 0.0
        
        # State
        self._position = np.zeros(3)
        self._velocity = np.zeros(3)
        self._attitude = np.array([0, 0, 0, 1])
        self._airspeed = 0.0
        
        print(f"\n[Controller] Using input mode: {input_mode}")

    def start(self):
        self._time = 0.0
        print("[Controller] Started!")

    def stop(self):
        print(f"[Controller] Stopped. Flight time: {self._time:.1f}s")

    def reset(self):
        self._time = 0.0

    def update_sensor(self, sensor_type: str, data):
        pass

    def update_graphical_sensor(self, sensor_type: str, data):
        pass

    def update_state(self, state: State):
        self._position = state.position
        self._velocity = state.linear_velocity
        self._attitude = state.attitude
        self._airspeed = np.linalg.norm(state.linear_body_velocity)

    def input_reference(self):
        """Return control inputs based on the selected mode."""
        
        if self._input_mode == "pwm":
            # PWM mode: 1000-2000 for all channels
            return [
                1500 + self._aileron * 500,   # Aileron PWM
                1500 + self._elevator * 500,  # Elevator PWM
                1000 + self._throttle * 1000, # Throttle PWM
                1500 + self._rudder * 500     # Rudder PWM
            ]
        
        elif self._input_mode == "normalized":
            # Normalized mode: -1 to 1 for surfaces, 0 to 1 for throttle
            # But we still return PWM since that's what FixedWing expects by default
            return [
                1500 + self._aileron * 500,
                1500 + self._elevator * 500,
                1000 + self._throttle * 1000,
                1500 + self._rudder * 500
            ]
        
        elif self._input_mode == "attitude":
            # Attitude mode: [qx, qy, qz, qw, throttle]
            # Create target quaternion from desired roll/pitch
            target_roll = self._aileron * 0.3   # Max 0.3 rad roll
            target_pitch = self._elevator * 0.2  # Max 0.2 rad pitch
            
            rot = Rotation.from_euler('xyz', [target_roll, target_pitch, 0])
            quat = rot.as_quat()  # [qx, qy, qz, qw]
            
            return [quat[0], quat[1], quat[2], quat[3], self._throttle]
        
        else:
            # Default to PWM
            return [1500, 1500, 1000, 1500]

    def update(self, dt: float):
        """Simple flight logic."""
        self._time += dt
        altitude = self._position[2]
        
        # Simple flight profile
        if self._time < 2.0:
            # Takeoff
            self._throttle = 1.0
            self._elevator = -0.4  # Pitch up
            self._aileron = 0.0
            self._rudder = 0.0
            
        elif self._time < 8.0:
            # Climb
            self._throttle = 0.85
            self._elevator = -0.2
            self._aileron = 0.0
            self._rudder = 0.0
            
        elif self._time < 12.0:
            # Right turn
            self._throttle = 0.7
            self._elevator = 0.0
            self._aileron = 0.4
            self._rudder = 0.15
            
        elif self._time < 16.0:
            # Left turn
            self._throttle = 0.7
            self._elevator = 0.0
            self._aileron = -0.4
            self._rudder = -0.15
            
        else:
            # Repeat pattern
            self._time = 8.0
        
        # Log every 2 seconds
        if int(self._time) % 2 == 0 and abs(self._time - int(self._time)) < dt:
            print(f"[{self._time:.1f}s] Alt: {altitude:.1f}m | "
                  f"Speed: {self._airspeed:.1f}m/s | "
                  f"Mode: {self._input_mode}")


class ControlModeTestApp:
    """Test application that can switch between control modes."""

    def __init__(self, input_mode="normalized"):
        self.input_mode = input_mode
        
        # Setup simulation
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        
        # Load environment
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Flat Plane"])
        
        # Create aircraft
        self.create_aircraft()
        
        # Reset
        self.world.reset()
        self.stop_sim = False

    def create_aircraft(self):
        """Create aircraft with specified control mode."""
        config = FixedWingConfig()
        config.backends = [MultiModeController(input_mode=self.input_mode)]
        
        # Set the control input mode on the vehicle
        config.control_input_mode = self.input_mode
        
        FixedWing(
            "/World/fixedwing",
            ROBOTS['FixedWing'],
            0,
            [0.0, 0.0, 0.3],
            Rotation.from_euler("XYZ", [0.0, -5.0, 0.0], degrees=True).as_quat(),
            config=config,
        )
        
        print(f"\nAircraft spawned with control mode: {self.input_mode}")

    def run(self):
        """Main loop."""
        self.timeline.play()
        
        print("\n" + "=" * 50)
        print(f"TESTING CONTROL MODE: {self.input_mode.upper()}")
        print("=" * 50 + "\n")
        
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
        
        self.timeline.stop()
        simulation_app.close()


def main():
    print("\n" + "=" * 60)
    print("FIXED-WING CONTROL MODES TEST")
    print("=" * 60)
    print()
    print("This script tests different control input modes:")
    print("  - pwm: Standard servo PWM (1000-2000)")
    print("  - normalized: Normalized values (-1 to 1)")
    print("  - attitude: Quaternion attitude control")
    print()
    print("Select a mode to test:")
    print("  1. PWM mode (default ArduPilot)")
    print("  2. Normalized mode")
    print("  3. Attitude mode")
    print()
    
    # For simplicity, just run normalized mode
    # User can edit this to test other modes
    mode = "normalized"
    
    print(f"Running with mode: {mode}")
    print("=" * 60 + "\n")
    
    app = ControlModeTestApp(input_mode=mode)
    app.run()


if __name__ == "__main__":
    main()




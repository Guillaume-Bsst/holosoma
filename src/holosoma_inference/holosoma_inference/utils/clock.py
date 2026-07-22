"""
Clock synchronization module for simulation and policy synchronization.

This module provides ZMQ-based clock synchronization between the simulation
and policy inference to resolve timing issues when simulation rates vary.
"""

import time

import zmq
from loguru import logger


class ClockPub:
    """
    Clock publisher that publishes elapsed milliseconds via ZMQ.
    Used by the simulation to broadcast timing information.
    """

    def __init__(self, port=5555):
        """Initialize the clock publisher.

        Args:
            port (int): ZMQ port to publish on (default: 5555)
        """
        self.port = port
        self.context = None
        self.socket = None
        self.start_time = None
        self.enabled = False

    def start(self):
        """Start the clock publisher."""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.port}")
            self.start_time = time.time()
            self.enabled = True
            logger.info(f"Clock publisher started on port {self.port}")
        except Exception as e:
            logger.error(f"Failed to start clock publisher: {e}")
            self.enabled = False

    def restart(self):
        """Restart the clock publisher (reset start time)."""
        if self.enabled:
            self.start_time = time.time()
            logger.info("Clock publisher restarted")

    def publish(self, sim_time):
        """Publish simulation time in milliseconds.

        Args:
            sim_time (float): Current simulation time in seconds
        """
        if not self.enabled or not self.socket:
            return

        try:
            sim_time_ms = int(sim_time * 1000)
            self.socket.send_string(str(sim_time_ms), zmq.NOBLOCK)
        except zmq.Again:
            # Non-blocking send failed, skip this publish
            pass
        except Exception as e:
            logger.warning(f"Clock publish failed: {e}")

    def close(self):
        """Close the clock publisher."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.enabled = False


class ClockSub:
    """
    Clock subscriber that receives elapsed milliseconds via ZMQ.
    Used by the policy to get synchronized timing information.
    """

    def __init__(self, port=5555):
        """Initialize the clock subscriber.

        Args:
            port (int): ZMQ port to subscribe to (default: 5555)
        """
        self.port = port
        self.context = None
        self.socket = None
        self.last_clock = 0
        self._offset = 0

    def start(self):
        """Start the clock subscriber."""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")  # Subscribe to all messages
        self.socket.setsockopt(zmq.RCVTIMEO, 10)  # 10ms timeout
        logger.info(f"Clock subscriber started, connecting to port {self.port}")

    def _drain_messages(self) -> None:
        """Drain all pending clock messages from the socket."""
        if self.socket is None:
            return

        while True:
            try:
                message = self.socket.recv_string(zmq.NOBLOCK)
                self.last_clock = int(message)
            except zmq.Again:  # noqa: PERF203
                break

    def get_clock(self):
        """Get current elapsed milliseconds.

        Returns:
            int: Elapsed milliseconds since simulation start
        """
        # Try to receive the latest clock value (drain all pending messages)
        self._drain_messages()

        self._offset = min(self._offset, self.last_clock)

        adjusted_clock = self.last_clock - self._offset
        return max(adjusted_clock, 0)

    def reset_origin(self) -> None:
        """Reset the clock origin to the latest received timestamp."""
        self._drain_messages()
        self._offset = self.last_clock

    def close(self):
        """Close the clock subscriber."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()


class PoseSub:
    """Object+ref-body pose subscriber (ZMQ SUB) -- run_sim's free box AND robot torso world poses.

    Counterpart of holosoma's PosePub (SimulatorBridge publishes at every physics step when a
    free box exists in the scene). Used by WholeBodyTrackingPolicy --task.live-object-obs to
    compute obj_pos_b/obj_ori_b from the REAL simulated box relative to the REAL simulated torso
    (the Unitree low state has no world root position, so the torso pose must come from the sim).
    Message: 14 floats "bx by bz bqw bqx bqy bqz tx ty tz tqw tqx tqy tqz" (world, wxyz quats).
    """

    def __init__(self, port=5556):
        self.port = port
        self.context = None
        self.socket = None
        self.last_pose = None  # (box_pos(3), box_quat_wxyz(4), ref_pos(3), ref_quat_wxyz(4)) or None

    def start(self):
        """Start the pose subscriber."""
        import numpy as np  # noqa: PLC0415 -- keep module import surface unchanged

        self._np = np
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.SUB)
        self.socket.connect(f"tcp://localhost:{self.port}")
        self.socket.setsockopt(zmq.SUBSCRIBE, b"")
        self.socket.setsockopt(zmq.RCVTIMEO, 10)
        logger.info(f"Object-pose subscriber started, connecting to port {self.port}")

    def get_pose(self):
        """Latest (box_pos(3), box_quat_wxyz(4), ref_pos(3), ref_quat_wxyz(4)), or None."""
        if self.socket is None:
            return self.last_pose
        while True:
            try:
                message = self.socket.recv_string(zmq.NOBLOCK)
            except zmq.Again:
                break
            parts = message.split()
            if len(parts) == 14:
                v = [float(x) for x in parts]
                np = self._np
                self.last_pose = (
                    np.array(v[0:3], dtype=np.float32),
                    np.array(v[3:7], dtype=np.float32),
                    np.array(v[7:10], dtype=np.float32),
                    np.array(v[10:14], dtype=np.float32),
                )
        return self.last_pose

    def close(self):
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()

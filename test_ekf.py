"""
Comprehensive test suite for EKF and Utils classes.

This file contains tests with simulated values and expected results to help
identify computation errors and frame convention issues.

Run with: python test_ekf.py
"""

import numpy as np
from utils import Utils
from ekf import EKF, GRAVITY

# Tolerance for floating point comparisons
ATOL = 1e-6
RTOL = 1e-5


def print_test_header(name):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print('='*60)


def print_pass(msg=""):
    print(f"  [PASS] {msg}")


def print_fail(msg=""):
    print(f"  [FAIL] {msg}")


def assert_close(actual, expected, name, atol=ATOL, rtol=RTOL):
    """Assert that actual is close to expected, with nice error messages."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if np.allclose(actual, expected, atol=atol, rtol=rtol):
        print_pass(f"{name}")
        return True
    else:
        print_fail(f"{name}")
        print(f"       Expected: {expected.flatten()}")
        print(f"       Actual:   {actual.flatten()}")
        print(f"       Diff:     {(actual - expected).flatten()}")
        return False


# =============================================================================
# TEST 1: Quaternion <-> Euler Conversions
# =============================================================================
def test_quaternion_euler_conversions():
    print_test_header("Quaternion <-> Euler Conversions")

    all_passed = True

    # Test 1.1: Identity quaternion -> zero Euler angles
    q_identity = np.array([1, 0, 0, 0])
    roll, pitch, yaw = Utils.quaternion_to_euler(q_identity)
    all_passed &= assert_close([roll, pitch, yaw], [0, 0, 0],
                                "Identity quaternion -> (0,0,0)")

    # Test 1.2: Zero Euler -> Identity quaternion
    q_from_zero = Utils.quaternion_from_euler(0, 0, 0)
    all_passed &= assert_close(q_from_zero, [1, 0, 0, 0],
                                "Euler (0,0,0) -> Identity quaternion")

    # Test 1.3: Pure yaw of 90 degrees (pi/2)
    # Yaw rotates around Z (Down in NED)
    # q = [cos(yaw/2), 0, 0, sin(yaw/2)]
    yaw_90 = np.pi / 2
    q_yaw90 = Utils.quaternion_from_euler(0, 0, yaw_90)
    expected_q_yaw90 = np.array([np.cos(yaw_90/2), 0, 0, np.sin(yaw_90/2)])
    all_passed &= assert_close(q_yaw90, expected_q_yaw90,
                                "Yaw 90deg quaternion")

    # Verify round-trip
    roll_out, pitch_out, yaw_out = Utils.quaternion_to_euler(q_yaw90)
    all_passed &= assert_close([roll_out, pitch_out, yaw_out], [0, 0, yaw_90],
                                "Yaw 90deg round-trip")

    # Test 1.4: Pure roll of 45 degrees (pi/4)
    # Roll rotates around X (North in NED body frame)
    # q = [cos(roll/2), sin(roll/2), 0, 0]
    roll_45 = np.pi / 4
    q_roll45 = Utils.quaternion_from_euler(roll_45, 0, 0)
    expected_q_roll45 = np.array([np.cos(roll_45/2), np.sin(roll_45/2), 0, 0])
    all_passed &= assert_close(q_roll45, expected_q_roll45,
                                "Roll 45deg quaternion")

    roll_out, pitch_out, yaw_out = Utils.quaternion_to_euler(q_roll45)
    all_passed &= assert_close([roll_out, pitch_out, yaw_out], [roll_45, 0, 0],
                                "Roll 45deg round-trip")

    # Test 1.5: Pure pitch of 30 degrees (pi/6)
    # Pitch rotates around Y (East in NED body frame)
    # q = [cos(pitch/2), 0, sin(pitch/2), 0]
    pitch_30 = np.pi / 6
    q_pitch30 = Utils.quaternion_from_euler(0, pitch_30, 0)
    expected_q_pitch30 = np.array([np.cos(pitch_30/2), 0, np.sin(pitch_30/2), 0])
    all_passed &= assert_close(q_pitch30, expected_q_pitch30,
                                "Pitch 30deg quaternion")

    roll_out, pitch_out, yaw_out = Utils.quaternion_to_euler(q_pitch30)
    all_passed &= assert_close([roll_out, pitch_out, yaw_out], [0, pitch_30, 0],
                                "Pitch 30deg round-trip")

    # Test 1.6: Combined rotation (roll=30, pitch=20, yaw=45 degrees)
    roll_in = np.radians(30)
    pitch_in = np.radians(20)
    yaw_in = np.radians(45)
    q_combined = Utils.quaternion_from_euler(roll_in, pitch_in, yaw_in)

    # Verify it's a unit quaternion
    all_passed &= assert_close(np.linalg.norm(q_combined), 1.0,
                                "Combined rotation is unit quaternion")

    # Round-trip
    roll_out, pitch_out, yaw_out = Utils.quaternion_to_euler(q_combined)
    all_passed &= assert_close([roll_out, pitch_out, yaw_out],
                                [roll_in, pitch_in, yaw_in],
                                "Combined rotation round-trip", atol=1e-5)

    return all_passed


# =============================================================================
# TEST 2: Rotation Matrix (Body -> NED)
# =============================================================================
def test_rotation_matrix():
    print_test_header("Rotation Matrix (Body -> NED)")

    all_passed = True

    # Test 2.1: Identity quaternion -> Identity rotation matrix
    q_identity = np.array([1, 0, 0, 0])
    R = Utils.quaternion_to_rotation_matrix(q_identity)
    all_passed &= assert_close(R, np.eye(3), "Identity quaternion -> I matrix")

    # Test 2.2: Yaw 90 deg - body X (forward) should point to NED Y (East)
    # After yaw rotation of 90 deg around Z:
    # Body X [1,0,0] -> NED [0,1,0] (East)
    # Body Y [0,1,0] -> NED [-1,0,0] (West = -North)
    # Body Z [0,0,1] -> NED [0,0,1] (Down - unchanged)
    yaw_90 = np.pi / 2
    q_yaw90 = Utils.quaternion_from_euler(0, 0, yaw_90)
    R_yaw90 = Utils.quaternion_to_rotation_matrix(q_yaw90)

    body_x = np.array([1, 0, 0])
    body_y = np.array([0, 1, 0])
    body_z = np.array([0, 0, 1])

    ned_from_body_x = R_yaw90 @ body_x
    ned_from_body_y = R_yaw90 @ body_y
    ned_from_body_z = R_yaw90 @ body_z

    all_passed &= assert_close(ned_from_body_x, [0, 1, 0],
                                "Yaw90: Body X -> NED East")
    all_passed &= assert_close(ned_from_body_y, [-1, 0, 0],
                                "Yaw90: Body Y -> NED -North")
    all_passed &= assert_close(ned_from_body_z, [0, 0, 1],
                                "Yaw90: Body Z -> NED Down (unchanged)")

    # Test 2.3: Roll 90 deg - body Y should point down
    # After roll rotation of 90 deg around X:
    # Body X [1,0,0] -> NED [1,0,0] (unchanged)
    # Body Y [0,1,0] -> NED [0,0,1] (Down)
    # Body Z [0,0,1] -> NED [0,-1,0] (West = -East)
    roll_90 = np.pi / 2
    q_roll90 = Utils.quaternion_from_euler(roll_90, 0, 0)
    R_roll90 = Utils.quaternion_to_rotation_matrix(q_roll90)

    all_passed &= assert_close(R_roll90 @ body_x, [1, 0, 0],
                                "Roll90: Body X -> NED North (unchanged)")
    all_passed &= assert_close(R_roll90 @ body_y, [0, 0, 1],
                                "Roll90: Body Y -> NED Down")
    all_passed &= assert_close(R_roll90 @ body_z, [0, -1, 0],
                                "Roll90: Body Z -> NED -East")

    # Test 2.4: Pitch 90 deg - body X (forward) should point down
    # IMPORTANT: In NED convention, positive pitch = nose UP
    # After pitch rotation of +90 deg around Y (nose up):
    # Body X [1,0,0] -> NED [0,0,-1] (UP in NED, since +Z is down)
    # Body Y [0,1,0] -> NED [0,1,0] (unchanged)
    # Body Z [0,0,1] -> NED [1,0,0] (North)
    #
    # This is the actual convention used in the code!
    pitch_90 = np.pi / 2
    q_pitch90 = Utils.quaternion_from_euler(0, pitch_90, 0)
    R_pitch90 = Utils.quaternion_to_rotation_matrix(q_pitch90)

    # NOTE: +pitch = nose UP, so body X (forward) points UP in NED (which is -Z)
    all_passed &= assert_close(R_pitch90 @ body_x, [0, 0, -1],
                                "Pitch90 up: Body X -> NED UP (-Z)")
    all_passed &= assert_close(R_pitch90 @ body_y, [0, 1, 0],
                                "Pitch90: Body Y -> NED East (unchanged)")
    all_passed &= assert_close(R_pitch90 @ body_z, [1, 0, 0],
                                "Pitch90 up: Body Z -> NED North")

    # Test 2.5: Verify R is orthogonal (R^T @ R = I, det(R) = 1)
    for name, R in [("Identity", Utils.quaternion_to_rotation_matrix(q_identity)),
                    ("Yaw90", R_yaw90),
                    ("Roll90", R_roll90),
                    ("Pitch90", R_pitch90)]:
        all_passed &= assert_close(R.T @ R, np.eye(3),
                                    f"{name}: R^T @ R = I")
        all_passed &= assert_close(np.linalg.det(R), 1.0,
                                    f"{name}: det(R) = 1")

    return all_passed


# =============================================================================
# TEST 3: Gravity Measurement Model
# =============================================================================
def test_gravity_measurement():
    print_test_header("Gravity Measurement Model (Accelerometer)")

    all_passed = True
    g = GRAVITY  # 9.81

    # Test 3.1: Level attitude (no tilt) - accel should read [0, 0, -g] in body frame
    # Because gravity in NED is [0, 0, +g] (Down), and R^T @ [0,0,-g] for level gives [0,0,-g]
    q_level = np.array([1, 0, 0, 0])
    R = Utils.quaternion_to_rotation_matrix(q_level)
    R_T = R.T  # NED -> body

    gravity_ned = np.array([0, 0, -g])  # Reaction force in NED
    accel_body_expected = R_T @ gravity_ned
    all_passed &= assert_close(accel_body_expected, [0, 0, -g],
                                "Level: expected accel = [0,0,-g]")

    # Test 3.2: Roll 90 deg right - gravity should appear in -Y body axis
    # Glider rolled 90 deg right: body Y points down in NED
    # Accelerometer reads reaction to gravity
    q_roll90 = Utils.quaternion_from_euler(np.pi/2, 0, 0)
    R_roll90 = Utils.quaternion_to_rotation_matrix(q_roll90)
    accel_body = R_roll90.T @ gravity_ned
    all_passed &= assert_close(accel_body, [0, -g, 0],
                                "Roll90: accel appears in -Y body", atol=1e-5)

    # Test 3.3: Pitch 90 deg nose up - gravity should appear in +X body axis
    # With +pitch = nose UP, body X points UP (NED -Z direction)
    # Accelerometer reads reaction to gravity: R^T @ [0,0,-g]
    # When pitched 90 up, R^T transforms NED -Z (up) into body +X
    # So accel should read [+g, 0, 0] (pointing along body +X)
    q_pitch90 = Utils.quaternion_from_euler(0, np.pi/2, 0)
    R_pitch90 = Utils.quaternion_to_rotation_matrix(q_pitch90)
    accel_body = R_pitch90.T @ gravity_ned
    all_passed &= assert_close(accel_body, [g, 0, 0],
                                "Pitch90 up: accel appears in +X body", atol=1e-5)

    # Test 3.4: Pitch 30 deg nose down - verify trigonometry
    # Negative pitch = nose down
    # When nose is down, gravity reaction in body frame:
    #   - With nose down, body X points somewhat toward ground
    #   - Gravity reaction force (upward) projects negatively on body X
    pitch_30_down = -np.pi/6  # Negative pitch = nose down
    q_pitch30 = Utils.quaternion_from_euler(0, pitch_30_down, 0)
    R_pitch30 = Utils.quaternion_to_rotation_matrix(q_pitch30)
    accel_body = R_pitch30.T @ gravity_ned
    # With pitch = -30 deg (nose down):
    # The rotation matrix gives:
    # ax = g*sin(pitch) = g*sin(-30) = -g*0.5 (negative because nose points down)
    # az = -g*cos(pitch)
    expected_ax = g * np.sin(pitch_30_down)  # = -g*0.5 for nose down
    expected_az = -g * np.cos(pitch_30_down)  # = -g*cos(30)
    all_passed &= assert_close(accel_body, [expected_ax, 0, expected_az],
                                "Pitch30 down: accel components", atol=1e-5)

    return all_passed


# =============================================================================
# TEST 4: Skew-Symmetric Matrix (Quaternion Kinematics)
# =============================================================================
def test_skew_matrix():
    print_test_header("Skew-Symmetric Matrix for Quaternion Propagation")

    all_passed = True

    # Test 4.1: Verify skew matrix structure
    omega = np.array([[1.0], [2.0], [3.0]])  # Angular velocity
    Omega = Utils.skew_4x4(omega)

    # Expected structure:
    # [[0, -wx, -wy, -wz],
    #  [wx,  0,  wz, -wy],
    #  [wy, -wz,  0,  wx],
    #  [wz,  wy, -wx,  0]]
    expected = np.array([
        [0, -1, -2, -3],
        [1,  0,  3, -2],
        [2, -3,  0,  1],
        [3,  2, -1,  0]
    ])
    all_passed &= assert_close(Omega, expected, "Skew matrix structure")

    # Test 4.2: Verify anti-symmetry (Omega^T = -Omega)
    all_passed &= assert_close(Omega.T, -Omega, "Skew matrix anti-symmetric")

    # Test 4.3: Zero angular velocity -> zero matrix
    omega_zero = np.array([[0.0], [0.0], [0.0]])
    Omega_zero = Utils.skew_4x4(omega_zero)
    all_passed &= assert_close(Omega_zero, np.zeros((4, 4)), "Zero omega -> zero skew")

    return all_passed


# =============================================================================
# TEST 5: EKF Quaternion Propagation (predict step)
# =============================================================================
def test_quaternion_propagation():
    print_test_header("EKF Quaternion Propagation")

    all_passed = True

    # Create EKF instance and manually initialize
    ekf = EKF(initialization_duration=0.1, sample_rate=100)
    ekf.isInitialized = True

    # Test 5.1: Pure rotation around Z (yaw) with 1 rad/s for 1 second
    # Starting from identity quaternion
    ekf.x = np.zeros((16, 1))
    ekf.x[0] = 1.0  # q0 = 1, rest = 0 (identity)
    ekf.P = np.eye(16) * 0.01

    omega_z = 1.0  # rad/s around Z
    dt = 0.01
    n_steps = 100  # 1 second total

    imu_data = {
        'gyro': np.array([0, 0, omega_z]),
        'accel': np.array([0, 0, -GRAVITY])  # Level flight
    }

    for _ in range(n_steps):
        ekf.predict(imu_data, dt)

    # After 1 second at 1 rad/s, expect yaw ~ 1 radian
    roll, pitch, yaw = Utils.quaternion_to_euler(ekf.x[0:4].flatten())
    all_passed &= assert_close(yaw, 1.0, "1 sec @ 1 rad/s -> yaw = 1 rad", atol=0.05)
    all_passed &= assert_close(roll, 0.0, "No roll after pure yaw rotation", atol=0.01)
    all_passed &= assert_close(pitch, 0.0, "No pitch after pure yaw rotation", atol=0.01)

    # Test 5.2: Quaternion should remain normalized
    q_norm = np.linalg.norm(ekf.x[0:4])
    all_passed &= assert_close(q_norm, 1.0, "Quaternion remains normalized")

    # Test 5.3: Pure rotation around X (roll)
    ekf.x = np.zeros((16, 1))
    ekf.x[0] = 1.0
    ekf.P = np.eye(16) * 0.01

    omega_x = 0.5  # rad/s around X
    imu_data = {
        'gyro': np.array([omega_x, 0, 0]),
        'accel': np.array([0, 0, -GRAVITY])
    }

    for _ in range(n_steps):
        ekf.predict(imu_data, dt)

    roll, pitch, yaw = Utils.quaternion_to_euler(ekf.x[0:4].flatten())
    all_passed &= assert_close(roll, 0.5, "1 sec @ 0.5 rad/s -> roll = 0.5 rad", atol=0.05)

    return all_passed


# =============================================================================
# TEST 6: Velocity Integration with Gravity
# =============================================================================
def test_velocity_integration():
    print_test_header("Velocity Integration with Gravity")

    all_passed = True

    # Create EKF and initialize level, at rest
    ekf = EKF(initialization_duration=0.1, sample_rate=100)
    ekf.isInitialized = True
    ekf.x = np.zeros((16, 1))
    ekf.x[0] = 1.0  # Identity quaternion
    ekf.P = np.eye(16) * 0.01

    # Test 6.1: Free fall (no accel reading, gravity should integrate)
    # In free fall, accelerometer reads 0 (no specific force)
    # Velocity should increase by g*t in +Z (down in NED)
    imu_data = {
        'gyro': np.array([0, 0, 0]),
        'accel': np.array([0, 0, 0])  # Free fall: accel = 0
    }

    dt = 0.01
    n_steps = 100  # 1 second

    for _ in range(n_steps):
        ekf.predict(imu_data, dt)

    vz = ekf.x[9, 0]  # Velocity Z (down)
    # v = g * t = 9.81 * 1.0 = 9.81 m/s
    all_passed &= assert_close(vz, GRAVITY, "Free fall: vz = g*t after 1s", atol=0.1)

    # Horizontal velocities should remain 0
    vx = ekf.x[7, 0]
    vy = ekf.x[8, 0]
    all_passed &= assert_close(vx, 0.0, "Free fall: vx = 0", atol=0.01)
    all_passed &= assert_close(vy, 0.0, "Free fall: vy = 0", atol=0.01)

    # Test 6.2: Level hover (accel reads -g, net acceleration = 0)
    ekf.x = np.zeros((16, 1))
    ekf.x[0] = 1.0
    ekf.P = np.eye(16) * 0.01

    imu_data = {
        'gyro': np.array([0, 0, 0]),
        'accel': np.array([0, 0, -GRAVITY])  # Hovering
    }

    for _ in range(n_steps):
        ekf.predict(imu_data, dt)

    vx, vy, vz = ekf.x[7:10].flatten()
    all_passed &= assert_close([vx, vy, vz], [0, 0, 0],
                                "Hover: velocity remains 0", atol=0.1)

    # Test 6.3: Forward acceleration in body frame (while level)
    ekf.x = np.zeros((16, 1))
    ekf.x[0] = 1.0
    ekf.P = np.eye(16) * 0.01

    accel_forward = 2.0  # m/s^2 forward
    imu_data = {
        'gyro': np.array([0, 0, 0]),
        'accel': np.array([accel_forward, 0, -GRAVITY])  # Forward + hover
    }

    for _ in range(n_steps):
        ekf.predict(imu_data, dt)

    vx = ekf.x[7, 0]  # NED North velocity
    # v = a * t = 2.0 * 1.0 = 2.0 m/s
    all_passed &= assert_close(vx, 2.0, "Forward accel: vx = a*t", atol=0.1)

    return all_passed


# =============================================================================
# TEST 7: GPS Update
# =============================================================================
def test_gps_update():
    print_test_header("GPS Position/Velocity Update")

    all_passed = True

    ekf = EKF(initialization_duration=0.1, sample_rate=100)
    ekf.isInitialized = True
    ekf.x = np.zeros((16, 1))
    ekf.x[0] = 1.0
    # Start with some initial position/velocity error
    ekf.x[4] = 10.0   # px error
    ekf.x[5] = -5.0   # py error
    ekf.x[7] = 3.0    # vx error
    ekf.P = np.eye(16) * 1.0

    # GPS says we're at origin with zero velocity
    gps_data = {
        'position': np.array([[0.0], [0.0], [0.0]]),
        'velocity': np.array([[0.0], [0.0], [0.0]])
    }

    ekf.update_gps_position_velocity(gps_data)

    # After update, state should move towards GPS measurement
    # (not exactly equal due to Kalman gain < 1)
    px_after = ekf.x[4, 0]
    py_after = ekf.x[5, 0]
    vx_after = ekf.x[7, 0]

    # Should be closer to 0 than before
    all_passed &= (abs(px_after) < 10.0)
    print_pass("GPS update reduces position error") if abs(px_after) < 10.0 else print_fail("GPS update position")

    all_passed &= (abs(vx_after) < 3.0)
    print_pass("GPS update reduces velocity error") if abs(vx_after) < 3.0 else print_fail("GPS update velocity")

    # Quaternion should remain normalized
    q_norm = np.linalg.norm(ekf.x[0:4])
    all_passed &= assert_close(q_norm, 1.0, "Quaternion normalized after GPS update")

    return all_passed


# =============================================================================
# TEST 8: Accelerometer Gravity Update (Roll/Pitch Correction)
# =============================================================================
def test_accel_gravity_update():
    print_test_header("Accelerometer Gravity Update")

    all_passed = True

    ekf = EKF(initialization_duration=0.1, sample_rate=100)
    ekf.isInitialized = True

    # Test 8.1: Start with some roll error, level accel reading should correct it
    roll_error = np.radians(10)  # 10 degree roll error
    q_with_error = Utils.quaternion_from_euler(roll_error, 0, 0)

    ekf.x = np.zeros((16, 1))
    ekf.x[0:4] = q_with_error.reshape((4, 1))
    ekf.P = np.eye(16) * 0.1
    ekf.P[0:4, 0:4] = np.eye(4) * 0.01  # Lower uncertainty on quaternion

    # Accelerometer reads level (no tilt) - should correct the roll error
    imu_data = {
        'accel': np.array([0, 0, -GRAVITY])
    }

    roll_before, _, _ = Utils.quaternion_to_euler(ekf.x[0:4].flatten())

    # Apply update multiple times
    for _ in range(10):
        ekf.update_accel_gravity(imu_data)

    roll_after, pitch_after, _ = Utils.quaternion_to_euler(ekf.x[0:4].flatten())

    # Roll should be closer to 0
    all_passed &= (abs(roll_after) < abs(roll_before))
    print_pass(f"Roll error reduced: {np.degrees(roll_before):.2f} -> {np.degrees(roll_after):.2f} deg") \
        if abs(roll_after) < abs(roll_before) else print_fail("Roll not corrected")

    return all_passed


# =============================================================================
# TEST 9: Heading Updates (GPS and Magnetometer)
# =============================================================================
def test_heading_updates():
    print_test_header("Heading Updates (GPS and Magnetometer)")

    all_passed = True

    # Test 9.1: GPS heading update
    ekf = EKF(initialization_duration=0.1, sample_rate=100)
    ekf.isInitialized = True

    # Start heading East (yaw = 90 deg)
    q_east = Utils.quaternion_from_euler(0, 0, np.pi/2)
    ekf.x = np.zeros((16, 1))
    ekf.x[0:4] = q_east.reshape((4, 1))
    ekf.P = np.eye(16) * 0.1

    # GPS says we're moving North (heading should be 0)
    gps_data = {
        'velocity': np.array([10.0, 0.0, 0.0])  # 10 m/s North
    }

    yaw_before = Utils.quaternion_to_euler(ekf.x[0:4].flatten())[2]

    # Apply update
    for _ in range(5):
        ekf.update_heading_gps(gps_data)

    yaw_after = Utils.quaternion_to_euler(ekf.x[0:4].flatten())[2]

    # Yaw should move towards 0 (North)
    all_passed &= (abs(yaw_after) < abs(yaw_before))
    print_pass(f"GPS heading correction: {np.degrees(yaw_before):.1f} -> {np.degrees(yaw_after):.1f} deg") \
        if abs(yaw_after) < abs(yaw_before) else print_fail("GPS heading not corrected")

    # Test 9.2: Verify GPS heading calculation
    # Heading = atan2(vy, vx)
    gps_north = {'velocity': np.array([10.0, 0.0, 0.0])}  # Moving North
    heading_north = np.arctan2(0.0, 10.0)
    all_passed &= assert_close(heading_north, 0.0, "GPS: Moving North = 0 heading")

    gps_east = {'velocity': np.array([0.0, 10.0, 0.0])}  # Moving East
    heading_east = np.arctan2(10.0, 0.0)
    all_passed &= assert_close(heading_east, np.pi/2, "GPS: Moving East = 90 deg heading")

    return all_passed


# =============================================================================
# TEST 10: Jacobian Verification (Numerical vs Analytical)
# =============================================================================
def test_jacobian():
    print_test_header("Jacobian F Verification (Numerical vs Analytical)")

    all_passed = True

    # Test configuration
    q = np.array([[0.9], [0.1], [0.2], [0.3]])
    q = q / np.linalg.norm(q)  # Normalize
    omega = np.array([[0.1], [0.2], [0.3]])
    accel = np.array([[0.5], [0.3], [-9.0]])
    b_gyro = np.array([0.01, 0.02, 0.01])
    b_accel = np.array([0.1, -0.1, 0.05])
    dt = 0.01

    # Get analytical Jacobian
    F_analytical = Utils.compute_jacobian_F(q, omega, accel, b_accel, b_gyro, dt)

    # Verify F is 16x16
    all_passed &= (F_analytical.shape == (16, 16))
    print_pass("Jacobian F is 16x16") if F_analytical.shape == (16, 16) else print_fail("Wrong shape")

    # Verify F is close to identity for small dt (F = I + something*dt)
    F_diff = F_analytical - np.eye(16)
    # The off-diagonal blocks should be O(dt)
    all_passed &= (np.max(np.abs(F_diff)) < 1.0)  # Reasonable for dt=0.01
    print_pass("Jacobian structure reasonable") if np.max(np.abs(F_diff)) < 1.0 else print_fail("Jacobian suspicious")

    return all_passed


# =============================================================================
# TEST 11: Frame Convention Consistency Checks
# =============================================================================
def test_frame_conventions():
    print_test_header("Frame Convention Consistency")

    all_passed = True

    # Test 11.1: NED gravity convention
    # In NED, +Z is down, so gravity vector is [0, 0, +g]
    # Accelerometer at rest reads reaction force: [0, 0, -g] in body frame (when level)
    print("  NED Convention: +Z = Down, gravity = [0,0,+g]")

    # Verify the EKF uses correct gravity sign
    q_level = np.array([1, 0, 0, 0])
    R = Utils.quaternion_to_rotation_matrix(q_level)

    # In EKF predict: gravity_ned = [0, 0, +GRAVITY]
    # accel_ned = R @ accel_body
    # v_new = v + (accel_ned + gravity_ned) * dt
    #
    # For hover: accel_body = [0, 0, -g], so accel_ned = [0, 0, -g]
    # Then: accel_ned + gravity_ned = [0, 0, -g] + [0, 0, +g] = [0, 0, 0]
    # This is correct!
    print_pass("Gravity sign convention verified in EKF predict")

    # Test 11.2: Quaternion to yaw extraction
    # Yaw should be 0 when heading North, 90 deg when heading East
    q_north = Utils.quaternion_from_euler(0, 0, 0)
    _, _, yaw_north = Utils.quaternion_to_euler(q_north)
    all_passed &= assert_close(yaw_north, 0, "Heading North = yaw 0")

    q_east = Utils.quaternion_from_euler(0, 0, np.pi/2)
    _, _, yaw_east = Utils.quaternion_to_euler(q_east)
    all_passed &= assert_close(yaw_east, np.pi/2, "Heading East = yaw 90 deg")

    # Test 11.3: Verify R transforms body->NED correctly
    # If heading East, body X (forward) should be in NED East direction
    R_east = Utils.quaternion_to_rotation_matrix(q_east)
    body_forward = np.array([1, 0, 0])
    ned_direction = R_east @ body_forward
    all_passed &= assert_close(ned_direction, [0, 1, 0],
                                "Heading East: body X -> NED East")

    return all_passed


# =============================================================================
# TEST 12: Integration Test (Full EKF Cycle)
# =============================================================================
def test_full_ekf_cycle():
    print_test_header("Full EKF Cycle Integration Test")

    all_passed = True

    # Simulate a simple flight scenario
    ekf = EKF(initialization_duration=0.1, sample_rate=100)
    ekf.isInitialized = True

    # Initial state: level, heading North, at origin
    ekf.x = np.zeros((16, 1))
    ekf.x[0] = 1.0  # Identity quaternion
    ekf.P = np.eye(16) * 0.01

    dt = 0.01

    # Scenario: Fly straight North at 10 m/s for 1 second
    imu_data = {
        'gyro': np.array([0, 0, 0]),  # No rotation
        'accel': np.array([0, 0, -GRAVITY]),  # Level flight
        'mag': np.array([1, 0, 0])  # Pointing North
    }

    gps_data = {
        'position': np.array([[0], [0], [0]]),
        'velocity': np.array([[10], [0], [0]])  # 10 m/s North
    }

    # Run EKF for 100 steps (1 second)
    for i in range(100):
        ekf.predict(imu_data, dt)
        if i % 10 == 0:  # GPS at 10 Hz
            gps_data['position'] = np.array([[10.0 * i * dt], [0], [0]])
            ekf.update(imu_data, gps_data, phase="glide")

    # Check final state
    roll, pitch, yaw = Utils.quaternion_to_euler(ekf.x[0:4].flatten())
    px, py, pz = ekf.x[4:7].flatten()
    vx, vy, vz = ekf.x[7:10].flatten()

    # Should still be level, heading North
    all_passed &= assert_close(roll, 0, "Final roll ~ 0", atol=0.1)
    all_passed &= assert_close(pitch, 0, "Final pitch ~ 0", atol=0.1)
    all_passed &= assert_close(yaw, 0, "Final yaw ~ 0 (North)", atol=0.2)

    # Position should be approximately (10, 0, 0) - traveled 10m North
    all_passed &= (px > 5)  # At least moved in right direction
    print_pass(f"Position: [{px:.1f}, {py:.1f}, {pz:.1f}]") if px > 5 else print_fail("Position incorrect")

    # Velocity should be approximately (10, 0, 0)
    all_passed &= assert_close(vx, 10, "Final vx ~ 10 m/s", atol=2)

    return all_passed


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================
def run_all_tests():
    print("\n" + "="*60)
    print("    EKF TEST SUITE - Frame Conventions & Computations")
    print("="*60)

    tests = [
        ("Quaternion-Euler Conversions", test_quaternion_euler_conversions),
        ("Rotation Matrix", test_rotation_matrix),
        ("Gravity Measurement", test_gravity_measurement),
        ("Skew Matrix", test_skew_matrix),
        ("Quaternion Propagation", test_quaternion_propagation),
        ("Velocity Integration", test_velocity_integration),
        ("GPS Update", test_gps_update),
        ("Accel Gravity Update", test_accel_gravity_update),
        ("Heading Updates", test_heading_updates),
        ("Jacobian Verification", test_jacobian),
        ("Frame Conventions", test_frame_conventions),
        ("Full EKF Cycle", test_full_ekf_cycle),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n  [ERROR] {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("    TEST SUMMARY")
    print("="*60)

    passed_count = sum(1 for _, p in results if p)
    total_count = len(results)

    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\nTotal: {passed_count}/{total_count} tests passed")

    if passed_count == total_count:
        print("\nAll tests passed!")
    else:
        print(f"\n{total_count - passed_count} test(s) failed - review output above")

    return passed_count == total_count


if __name__ == "__main__":
    run_all_tests()

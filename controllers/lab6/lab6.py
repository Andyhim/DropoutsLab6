import numpy as np
import sys
import copy

try:
    from controller import Supervisor
except ImportError:
    print("ERROR: Must run inside Webots as Supervisor."); sys.exit(1)

global TASK



'''NOTE - COMMENT/UNCOMMENT TO CHANGE TASK
"pick and place" - Place block in bin (original task)
"block stack" - Stack four blocks (extra credit task)
|    |    |    |    |
v    v    v    v    v'''
TASK = "pick and place"
#TASK = "block stack"




# ============================================================
# Robot Interface (DO NOT MODIFY)
# ============================================================
class UR5eInterface:
    MOTOR_NAMES = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                   "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
    SENSOR_NAMES = [n + "_sensor" for n in MOTOR_NAMES]
    HAND_MOTORS_1 = ["finger_1_joint_1", "finger_2_joint_1", "finger_middle_joint_1"]
    HAND_MOTORS_2 = ["finger_1_joint_2", "finger_2_joint_2", "finger_middle_joint_2"]

    def __init__(self, robot):
        self.robot = robot
        self.timestep = int(robot.getBasicTimeStep())
        self.motors, self.sensors = [], []
        for i in range(6):
            m = robot.getDevice(self.MOTOR_NAMES[i])
            s = robot.getDevice(self.SENSOR_NAMES[i])
            s.enable(self.timestep); m.setVelocity(1.0)
            self.motors.append(m); self.sensors.append(s)
        self.hand_motors_1, self.hand_motors_2 = [], []
        for name in self.HAND_MOTORS_1:
            m = robot.getDevice(name)
            if m: self.hand_motors_1.append(m)
        for name in self.HAND_MOTORS_2:
            m = robot.getDevice(name)
            if m: self.hand_motors_2.append(m)
        for _ in range(10): robot.step(self.timestep)

    def get_joint_positions(self):
        return np.array([s.getValue() for s in self.sensors])

    def set_joint_positions(self, q):
        for i in range(6): self.motors[i].setPosition(float(q[i]))

    def set_speed(self, speed):
        for m in self.motors: m.setVelocity(speed)

    def open_gripper(self):
        for m in self.hand_motors_1: m.setPosition(0.05)
        for m in self.hand_motors_2: m.setPosition(0.0)

    def close_gripper(self):
        for m in self.hand_motors_1: m.setPosition(0.4)
        for m in self.hand_motors_2: m.setPosition(0.4)

    def wait(self, seconds):
        for _ in range(int(seconds * 1000 / self.timestep)):
            if self.robot.step(self.timestep) == -1: return

    def move_to(self, q, wait_time=2.0):
        self.set_joint_positions(q); self.wait(wait_time)
        return self.get_joint_positions()


# ============================================================
# Constants (DO NOT MODIFY)
# ============================================================
# UR5e Standard DH Parameters
UR5E_DH_A     = [0.0, -0.425, -0.3922, 0.0, 0.0, 0.0]
UR5E_DH_D     = [0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996]
UR5E_DH_ALPHA = [np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0]
GRIPPER_OFFSET = 0.18  # Robotiq 3F gripper length

# Webots UR5e PROTO has a 180-deg Z rotation between PROTO frame and DH frame
R_PROTO = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
ROBOT_POS = None  # Set at runtime from Supervisor

HOME = [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]
MAX_HEIGHT = 1.1 # Adjust how high up the arm goes to avoid collisions during manipulation (world z-coord)
BLOCK_HEIGHT_OFFSET = 0.025 # Offset for the gripper to grab block in the middle, not the bottom

BLOCK_WORLD = [0.3, -0.15, 0.765]
BIN_WORLD = [0.3, 0.2, 0.745]

GRAY_BLOCK = [0.4, -0.1, 0.765]
GREEN_BLOCK = [0.26, 0.2, 0.765]
BLUE_BLOCK = [0.25, 0, 0.765]
RED_BLOCK = [0.25, -0.2, 0.765]
STACK_ZONE = [0.4, 0.15, 0.74]


# ============================================================
# PROVIDED: Coordinate Transforms (DO NOT MODIFY)
# ============================================================
def _dh(a, d, alpha, theta):
    """Standard DH transformation matrix (one joint)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0,   sa,     ca,    d   ],
        [0,   0,      0,     1   ]])


def world_to_base(pw):
    """Convert world position to DH base frame."""
    return R_PROTO.T @ (np.array(pw) - ROBOT_POS)


def base_to_world(pb):
    """Convert DH base frame position to world."""
    return R_PROTO @ np.array(pb) + ROBOT_POS



'''
All lab 6 solution code written by Andy Himelstein
'''


# ============================================================
# TASK 1: Forward Kinematics (15 pts)
# ============================================================
def forward_kinematics(q):
    """
    Compute the gripper tip position given joint angles.

    Parameters:
        q : array-like, 6 joint angles (radians)

    Returns:
        position : numpy array [x, y, z] in robot base frame
    """
    T = np.eye(4)
    for i in range(len(q)):
        dh = _dh(UR5E_DH_A[i], UR5E_DH_D[i], UR5E_DH_ALPHA[i], q[i])
        T = T @ dh
    
    gripper_offset_local = np.array([0, 0, GRIPPER_OFFSET])
    translation = T[0:3, 3] + T[0:3, 0:3] @ gripper_offset_local

    orientation = T[0:3, 2] # Z-axis rotation column

    return np.concatenate((translation, orientation)) # [x, y, z, ax, ay, az]

# ============================================================
# TASK 2: Numerical Jacobian (15 pts)
# ============================================================
def compute_jacobian(q, delta=1e-5):
    """
    Compute the 3x6 position Jacobian using central finite differences.

    Parameters:
        q     : numpy array (6,) current joint angles
        delta : perturbation size

    Returns:
        J : numpy array (3, 6)
    """
    # TODO: Implement (~8-10 lines)
    J = np.zeros((6, 6))
    for i in range(len(q)):
        delta_ei = np.zeros(6)
        delta_ei[i] = delta
        fk_pos = forward_kinematics(q + delta_ei)
        fk_neg = forward_kinematics(q - delta_ei)
        J_col = (fk_pos - fk_neg) / (2 * delta)
        J[:, i] = J_col
    return J
        


# ============================================================
# TASK 3: Define Waypoints (10 pts)
# ============================================================
def get_waypoints_pnp(block_world, bin_world):
    """
    Define the world-frame positions the gripper should visit.

    Parameters:
        block_world : [x, y, z] of block center in world frame
        bin_world   : [x, y, z] of bin center in world frame

    Returns:
        dict with keys: 'above_block', 'at_block', 'above_bin'
        Each value is a list [x, y, z] in world frame.
    """
    above_block = copy.deepcopy(block_world)
    above_bin = copy.deepcopy(bin_world)
    above_bin[2] = above_block[2] = MAX_HEIGHT

    at_block = copy.deepcopy(block_world)
    at_block[2] = at_block[2] + BLOCK_HEIGHT_OFFSET

    return {"above_block": above_block, "at_block": at_block, "above_bin": above_bin}


def get_waypoints_stack(gray_block, green_block, blue_block, red_block, stack_zone):

    above_gray = copy.deepcopy(gray_block)
    above_green = copy.deepcopy(green_block)
    above_blue = copy.deepcopy(blue_block)
    above_red = copy.deepcopy(red_block)
    above_stack = copy.deepcopy(stack_zone)

    above_gray[2] = above_green[2] = above_blue[2] = above_red[2] = above_stack[2] = MAX_HEIGHT

    at_gray = copy.deepcopy(gray_block)
    at_green = copy.deepcopy(green_block)
    at_blue = copy.deepcopy(blue_block)
    at_red = copy.deepcopy(red_block)

    at_gray[2] = at_gray[2] + BLOCK_HEIGHT_OFFSET
    at_green[2] = at_green[2] + BLOCK_HEIGHT_OFFSET
    at_blue[2] = at_blue[2] + BLOCK_HEIGHT_OFFSET
    at_red[2] = at_red[2] + BLOCK_HEIGHT_OFFSET

    waypoints = {}

    waypoints["above_gray"] = above_gray
    waypoints["above_green"] = above_green
    waypoints["above_blue"] = above_blue
    waypoints["above_red"] = above_red
    waypoints["above_stack"] = above_stack
    waypoints["at_gray"] = at_gray
    waypoints["at_green"] = at_green
    waypoints["at_blue"] = at_blue
    waypoints["at_red"] = at_red

    return waypoints


# ============================================================
# TASK 4: Gradient Descent IK (20 pts)
# ============================================================
def gradient_descent_ik(target_world, q_init, 
                         learning_rate=0.5,
                         max_iterations=1000, 
                         orientation_target=np.array([0,0,-1]),
                         pos_tolerance=0.1, ori_tolerance=0.1):
    """
    Find joint angles that place the gripper at target_world.

    Parameters:
        target_world   : [x, y, z] desired position in world frame
        q_init         : starting joint angles (numpy array of 6)
        learning_rate  : step size (default 0.5)
        max_iterations : max steps (default 1000)
        tolerance      : converge when error < this (meters)

    Returns:
        q_solution : numpy array of 6 joint angles
        converged  : bool
        errors     : list of position error at each iteration
    """
    errors = []
    q_solution = np.zeros([6,])
    converged = False
    q = q_init

    target_base = np.append(world_to_base(target_world), orientation_target)

    for i in range(max_iterations):
        error = target_base - forward_kinematics(q)
        errors.append(error)

        pos_error = error[:3]
        ori_error = error[3:]

        if np.linalg.norm(pos_error) < pos_tolerance and np.linalg.norm(ori_error) < ori_tolerance:
            converged = True
            q_solution = q

        J = compute_jacobian(q)
        q = q + learning_rate * np.transpose(J) @ error

        q[np.where(q > 2 * np.pi)] = 2 * np.pi
        q[np.where(q < -2 * np.pi)] = -2 * np.pi
    
    return q_solution, converged, errors
    


# ============================================================
# TASK 5: Pick and Place Sequence (10 pts)
# ============================================================
def pick_and_place(arm, block_world=BLOCK_WORLD, bin_world=BIN_WORLD):
    """
    Pick up the block and place it in the bin.

    """
    waypoints = get_waypoints_pnp(block_world, bin_world)

    above_block = waypoints["above_block"]
    at_block = waypoints["at_block"]
    above_bin = waypoints["above_bin"]

    # Move to above block
    q_solution, converged, errors = gradient_descent_ik(above_block, HOME)
    arm.move_to(q_solution, wait_time=3)

    # Move to block
    q_solution, converged, errors = gradient_descent_ik(at_block, q_solution)
    arm.move_to(q_solution)

    # Grab block
    arm.close_gripper()
    arm.wait(1.5)

    # Lift block up
    q_solution, converged, errors = gradient_descent_ik(above_block, q_solution)
    arm.move_to(q_solution)

    # Move block to above bin
    q_solution, converged, errors = gradient_descent_ik(above_bin, q_solution)
    arm.move_to(q_solution)

    # Release block
    arm.open_gripper()
    arm.wait(1.5)


# Performs movements for stacking blocks (extra credit task)
def block_stack(arm, gray_block=GRAY_BLOCK, green_block=GREEN_BLOCK, blue_block=BLUE_BLOCK, red_block=RED_BLOCK, stack_zone=STACK_ZONE):
    
    waypoints = get_waypoints_stack(gray_block, green_block, blue_block, red_block, stack_zone)

    pos = grab_and_stack(arm, waypoints["above_gray"], waypoints["at_gray"], waypoints["above_stack"], -0.3, HOME)
    pos = grab_and_stack(arm, waypoints["above_green"], waypoints["at_green"], waypoints["above_stack"], -0.25, pos)
    pos = grab_and_stack(arm, waypoints["above_blue"], waypoints["at_blue"], waypoints["above_stack"], -0.2, pos)
    pos = grab_and_stack(arm, waypoints["above_red"], waypoints["at_red"], waypoints["above_stack"], -0.15, pos)


# Performs movement chain for stacking one block. height_offset makes it so that each block can be placed *just* on top of the other
def grab_and_stack(arm, above_block, at_block, above_stack, height_offset, init_pos):

    # Move to above block
    pos, converged, errors = gradient_descent_ik(above_block, init_pos)
    arm.move_to(pos)

    # Move to block
    pos, converged, errors = gradient_descent_ik(at_block, pos)
    arm.move_to(pos)

    # Grab block
    arm.close_gripper()
    arm.wait(1.5)

    # Move to above block
    pos, converged, errors = gradient_descent_ik(above_block, pos)
    arm.move_to(pos)

    above_stack_offset = (above_stack[0], above_stack[1], above_stack[2] + height_offset)

    # Move to above stack
    pos, converged, errors = gradient_descent_ik(above_stack, pos)
    arm.move_to(pos)

    # Move to above stack (with offset - places block gently)
    pos, converged, errors = gradient_descent_ik(above_stack_offset, pos)
    arm.move_to(pos)

    # Release block
    arm.open_gripper()
    arm.wait(1.5)

    # Move to above stack
    pos, converged, errors = gradient_descent_ik(above_stack, pos)
    arm.move_to(pos)

    # Return last position (for chaining)
    return pos


# ============================================================
# MAIN — DO NOT MODIFY
# ============================================================
def main():

    # Initialize robot with supervisor
    robot = Supervisor()
    node = robot.getSelf()
    arm = UR5eInterface(robot)
    arm.set_speed(0.6) # Set joint speed upon initialization

    # User supervisor to set robot's base position
    global ROBOT_POS
    ROBOT_POS = np.array(node.getPosition())

    # SEE TOP OF SCRIPT FOR CHANGING TASK
    if TASK == "pick and place":
        pick_and_place(arm)
    elif TASK == "block stack":
        block_stack(arm)
    

if __name__ == "__main__":
    main()

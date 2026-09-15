"""
Ambiente Gymnasium per il braccio robotico MeArm simulato in PyBullet.
Ottimizzato per Forza 4 con griglia fisica e collisioni attive.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pybullet as p
    import pybullet_data
except ImportError as e:
    raise ImportError("pybullet non e' installato. Esegui: pip install pybullet") from e

URDF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "urdf", "mearm_real", "mearm_real.urdf"
)

JOINT_NAMES = ["joint_0", "joint_1", "joint_2", "joint_3"] 

WORKSPACE_MAX_RADIUS = 0.227  
WORKSPACE_MAX_HEIGHT = 0.225  
WORKSPACE_MIN_HEIGHT = 0.0    

HOME_POSITION_RAD = {
    "joint_0": -0.1745,   
    "joint_1": -0.5236,   
    "joint_2": 0.6981,    
    "joint_3": -1.5708,   
}

ALL_MIMIC_JOINTS = {
    "joint_2_mimic_1":       ("joint_2", -1.0, -1.57),  
    "arm1_mimic_0":          ("joint_1", -1.0, 0.0),    
    "gripper_holder":        ("joint_2", 1.0, -1.57),   
    "joint_3_mimic":         ("joint_3", -1.0, 0.0),  
    "arm_0_support_mimic":   ("joint_1", 1.0, 1.57),
    "arm_0_connector_mimic": ("joint_1", -1.0, -1.57),
    "joint_2_mimic_0a":      ("joint_2", -1.0, 1.57),
    "joint_2_mimic_0b":      ("joint_1", -1.0, 0.0),
    "gripper_support_mimic": ("joint_2", -1.0, -1.57),
}

class MeArmRealEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, max_steps=200, action_scale=0.08, success_threshold=0.025):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.success_threshold = success_threshold  
        self._step_count = 0

        self.alpha = 0.15
        self.internal_targets = None 

        self._client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)

        self.robot_id = None
        self.joint_indices = []        
        self.joint_limits = []
        self.all_mimics_indices = {}        
        self.target_pos = np.zeros(3)
   
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        obs_dim = 4 + 4 + 3 + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self._load_scene()

    def _load_scene(self):
        p.resetSimulation(physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client) 

        self.robot_id = p.loadURDF(
            URDF_PATH,
            basePosition=[0, 0, 0],
            useFixedBase=True,
            physicsClientId=self._client,
        )

        n_links = p.getNumJoints(self.robot_id, physicsClientId=self._client)
        
        for link_idx in range(-1, n_links):
            mass_val = 0.0 if link_idx == -1 else 2.0
            inertia_val = [0, 0, 0] if link_idx == -1 else [0.01, 0.01, 0.01]
            p.changeDynamics(self.robot_id, link_idx, mass=mass_val, localInertiaDiagonal=inertia_val, physicsClientId=self._client)

        for link_idx in range(-1, n_links): 
            p.setCollisionFilterPair(self.plane_id, self.robot_id, -1, link_idx, enableCollision=0, physicsClientId=self._client)

        for i in range(-1, n_links):
            for j in range(-1, n_links):
                if i != j:
                    p.setCollisionFilterPair(self.robot_id, self.robot_id, i, j, enableCollision=0, physicsClientId=self._client)

        joint_name_to_idx = {p.getJointInfo(self.robot_id, i)[1].decode("utf-8"): i for i in range(n_links)}
        self.joint_indices = [joint_name_to_idx[n] for n in JOINT_NAMES] 
        
        self.joint_limits = []
        for j in self.joint_indices:
            info = p.getJointInfo(self.robot_id, j, physicsClientId=self._client)
            lo, hi = info[8], info[9]
            if lo > hi: lo, hi = -np.pi, np.pi
            self.joint_limits.append((lo, hi))

        for mimic_name, (primary_name, mult, offset) in ALL_MIMIC_JOINTS.items():
            if mimic_name not in joint_name_to_idx: continue
            mimic_j = joint_name_to_idx[mimic_name] 
            primary_pos_in_list = JOINT_NAMES.index(primary_name)
            info = p.getJointInfo(self.robot_id, mimic_j, physicsClientId=self._client)
            m_lo, m_hi = info[8], info[9]
            if m_lo > m_hi: m_lo, m_hi = -np.pi, np.pi
            self.all_mimics_indices[mimic_name] = (mimic_j, primary_pos_in_list, mult, offset, m_lo, m_hi)

        self.ee_link_index = None
        for j in range(n_links):
            if p.getJointInfo(self.robot_id, j)[12].decode("utf-8") == "tool":
                self.ee_link_index = j
                break

        # --- GRIGLIA FORZA 4 (Fisica e Trasparente) ---
        # --- GRIGLIA FORZA 4 (Fisica Adattata per Sfere da 13mm) ---
        GRID_X = 0.14  
        glass_color = [0.1, 0.4, 0.9, 0.2]

        # Base (Leggermente ristretta)
        base_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.01, 0.075, 0.005], physicsClientId=self._client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=base_col, basePosition=[GRID_X, 0, 0.005], physicsClientId=self._client)
        
        # Pareti Anteriore e Posteriore (Avvicinate per lasciare un vuoto interno di soli 14mm)
        front_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.002, 0.075, 0.04], physicsClientId=self._client)
        front_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.075, 0.04], rgbaColor=glass_color, physicsClientId=self._client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=front_col, baseVisualShapeIndex=front_vis, basePosition=[GRID_X - 0.009, 0, 0.04], physicsClientId=self._client)
        
        back_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.002, 0.075, 0.04], physicsClientId=self._client)
        back_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.002, 0.075, 0.04], rgbaColor=glass_color, physicsClientId=self._client)
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=back_col, baseVisualShapeIndex=back_vis, basePosition=[GRID_X + 0.009, 0, 0.04], physicsClientId=self._client)
        
        # 8 Divisori verticali (Ispessiti per lasciare un vuoto interno di soli 13.6mm per colonna)
        for i in range(8):
            y_pos = 0.07 - (i * 0.02)
            div_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.007, 0.0032, 0.04], physicsClientId=self._client)
            div_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.007, 0.0032, 0.04], rgbaColor=[0.1, 0.3, 0.8, 0.4], physicsClientId=self._client)
            p.createMultiBody(baseMass=0, baseCollisionShapeIndex=div_col, baseVisualShapeIndex=div_vis, basePosition=[GRID_X, y_pos, 0.04], physicsClientId=self._client)

        # Caricatore Gettoni
        mag_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.015, height=0.02, physicsClientId=self._client)
        mag_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=0.015, length=0.02, rgbaColor=[0.2, 0.8, 0.2, 0.8], physicsClientId=self._client)
        self.mag_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=mag_col, baseVisualShapeIndex=mag_visual, basePosition=[0.07, 0.07, 0.01], physicsClientId=self._client)
    def _apply_mimic_joints(self, primary_positions):
        for mimic_name, (mimic_j, primary_idx, mult, offset, m_lo, m_hi) in self.all_mimics_indices.items():
            target = mult * primary_positions[primary_idx] + offset
            clipped = float(np.clip(target, m_lo, m_hi))
            p.setJointMotorControl2(
                self.robot_id, mimic_j, p.POSITION_CONTROL,
                targetPosition=clipped, force=10000.0, maxVelocity=1000.0,
                positionGain=1.0, velocityGain=1.0, physicsClientId=self._client,
            )

    def _sync_mimic_kinematics(self, current_positions):
        for mimic_name, (mimic_j, primary_idx, mult, offset, m_lo, m_hi) in self.all_mimics_indices.items():
            target = mult * current_positions[primary_idx] + offset
            clipped = float(np.clip(target, m_lo, m_hi))
            p.resetJointState(self.robot_id, mimic_j, targetValue=clipped, physicsClientId=self._client)

    def _get_joint_states(self):
        states = p.getJointStates(self.robot_id, self.joint_indices, physicsClientId=self._client)
        positions = np.array([s[0] for s in states], dtype=np.float32)
        velocities = np.array([s[1] for s in states], dtype=np.float32)
        return positions, velocities

    def _get_ee_pos(self):
        state = p.getLinkState(self.robot_id, self.ee_link_index, physicsClientId=self._client)
        return np.array(state[0], dtype=np.float32)

    def _get_obs(self): 
        positions, velocities = self._get_joint_states()
        ee_pos = self._get_ee_pos()
        delta_pos = self.target_pos - ee_pos
        return np.concatenate([positions, velocities, ee_pos, self.target_pos, delta_pos]).astype(np.float32)

    def set_manual_target(self, x, y, z):
        azimuth = np.arctan2(y, x)
        azimuth = np.clip(azimuth, -0.7853, 0.7853)
        distanza_xy = np.sqrt(x**2 + y**2)
        distanza_xy = np.clip(distanza_xy, 0.04, WORKSPACE_MAX_RADIUS)
        x = distanza_xy * np.cos(azimuth)
        y = distanza_xy * np.sin(azimuth)
        z = np.clip(z, WORKSPACE_MIN_HEIGHT, WORKSPACE_MAX_HEIGHT)
        self.target_pos = np.array([x, y, z], dtype=np.float32)
        return self._get_obs()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        home = np.array([np.clip(HOME_POSITION_RAD[name], lo, hi) for name, (lo, hi) in zip(JOINT_NAMES, self.joint_limits)], dtype=np.float32)
        self.internal_targets = home.copy()
        
        for j, val in zip(self.joint_indices, home):
            p.resetJointState(self.robot_id, j, targetValue=float(val), physicsClientId=self._client)
            
        self._sync_mimic_kinematics(home)
        self.target_pos = np.array([0.10, 0.12, 0.05], dtype=np.float32) # Target iniziale al caricatore
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        
        # Il PPO genera azioni da -1 a 1. Le moltiplichiamo per action_scale per muovere dolcemente i motori
        for i, (j_idx, act) in enumerate(zip(self.joint_indices, action)):
            lo, hi = self.joint_limits[i]
            
            self.internal_targets[i] += float(act) * self.action_scale
            self.internal_targets[i] = float(np.clip(self.internal_targets[i], lo, hi))
            
            p.setJointMotorControl2(
                self.robot_id, j_idx, p.POSITION_CONTROL, 
                targetPosition=self.internal_targets[i], 
                force=500.0, maxVelocity=5.0, physicsClientId=self._client
            )
            
        # Aggiorniamo la fisica e i tiranti
        self._apply_mimic_joints(self.internal_targets)
        p.stepSimulation(physicsClientId=self._client)
        
        current_positions, _ = self._get_joint_states()
        self._sync_mimic_kinematics(current_positions)
        
        obs = self._get_obs()
        ee_pos = self._get_ee_pos()
        dist = float(np.linalg.norm(self.target_pos - ee_pos))
        
        reward = -dist
        terminated = bool(dist < self.success_threshold)
        truncated = bool(self._step_count >= self.max_steps)
        
        return obs, reward, terminated, truncated, {}
    def render(self):
        pass

    def close(self):
        if p.isConnected(self._client): p.disconnect(self._client)
import os
import sys
import time
import math
import numpy as np
import multiprocessing as mp
import customtkinter as ctk

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.mearm_real_env import MeArmRealEnv


class MeArmConnect4Panel(ctk.CTk):
    def __init__(self, shared_target_col, shared_user_col):
        super().__init__()
        self.title("Forza 4 - MeArm Robot")
        self.geometry("450x500")
        
        self.shared_target_col = shared_target_col
        self.shared_user_col = shared_user_col
        self.grid = [[0 for _ in range(7)] for _ in range(6)]
        self.game_over = False #Flag per fermare il gioco
        
        self.build_ui()

    #costruzione interfaccia
    def build_ui(self):
        ctk.CTkLabel(self, text="MeArm Forza 4", font=ctk.CTkFont(size=24, weight="bold")).pack(pady=(20, 10))
        self.status_label = ctk.CTkLabel(self, text="Seleziona una colonna (Tu sei il Giallo)", text_color="gray")
        self.status_label.pack(pady=(0, 15))

        self.col_w = 50
        self.row_h = 50
        self.canvas = ctk.CTkCanvas(
            self, width=self.col_w*7, height=self.row_h*6, 
            bg="#1f538d", highlightthickness=0
        )
        self.canvas.pack(pady=10)
        
        self.canvas.bind("<Button-1>", self.on_column_click)
        self.draw_board()
    
    
    def draw_board(self):
        self.canvas.delete("all")
        for r in range(6):
            for c in range(7):
                x0 = c * self.col_w + 5
                y0 = r * self.row_h + 5
                x1 = x0 + self.col_w - 10
                y1 = y0 + self.row_h - 10
                
                if self.grid[r][c] == 0: color = "#2b2b2b"
                elif self.grid[r][c] == 1: color = "#e74c3c"
                else: color = "#f1c40f"
                    
                self.canvas.create_oval(x0, y0, x1, y1, fill=color, outline="#153b66", width=2)

    def get_valid_locations(self, grid):
        return [c for c in range(7) if grid[0][c] == 0]

    def get_next_open_row(self, grid, col):
        for r in range(5, -1, -1):
            if grid[r][col] == 0:
                return r
        return None

    #check per vittoria
    def check_winner(self, grid, piece):
        # Orizzontale
        for c in range(7-3):
            for r in range(6):
                if grid[r][c] == piece and grid[r][c+1] == piece and grid[r][c+2] == piece and grid[r][c+3] == piece:
                    return True
        # Verticale
        for c in range(7):
            for r in range(6-3):
                if grid[r][c] == piece and grid[r+1][c] == piece and grid[r+2][c] == piece and grid[r+3][c] == piece:
                    return True
        # Diagonale positiva
        for c in range(7-3):
            for r in range(6-3):
                if grid[r][c] == piece and grid[r+1][c+1] == piece and grid[r+2][c+2] == piece and grid[r+3][c+3] == piece:
                    return True
        # Diagonale negativa
        for c in range(7-3):
            for r in range(3, 6):
                if grid[r][c] == piece and grid[r-1][c+1] == piece and grid[r-2][c+2] == piece and grid[r-3][c+3] == piece:
                    return True
        return False

    def is_terminal_node(self, grid):
        return self.check_winner(grid, 1) or self.check_winner(grid, 2) or len(self.get_valid_locations(grid)) == 0

    def find_winning_move(self, grid, piece):
        for col in self.get_valid_locations(grid):
            row = self.get_next_open_row(grid, col)
            test_grid = [r[:] for r in grid]
            test_grid[row][col] = piece
            if self.check_winner(test_grid, piece):
                return col
        return None

    def evaluate_window(self, window, piece):
        score = 0
        opp_piece = 2 if piece == 1 else 1
        if window.count(piece) == 4: score += 100
        elif window.count(piece) == 3 and window.count(0) == 1: score += 5
        elif window.count(piece) == 2 and window.count(0) == 2: score += 2
        if window.count(opp_piece) == 3 and window.count(0) == 1: score -= 4
        return score

    def score_position(self, grid, piece):
        score = 0
        for r in range(6):
            for c in range(4):
                window = [grid[r][c+i] for i in range(4)]
                score += self.evaluate_window(window, piece)
        for c in range(7):
            for r in range(3):
                window = [grid[r+i][c] for i in range(4)]
                score += self.evaluate_window(window, piece)

        for r in range(3):
            for c in range(4):
                window = [grid[r+i][c+i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        for r in range(3, 6):
            for c in range(4):
                window = [grid[r-i][c+i] for i in range(4)]
                score += self.evaluate_window(window, piece)

        return score

    def minimax(self, grid, depth, maximizingPlayer):
        valid_locations = self.get_valid_locations(grid)
        is_terminal = self.is_terminal_node(grid)
        
        if depth == 0 or is_terminal:
            if is_terminal:
                if self.check_winner(grid, 1): # Vince il Robot
                    return (None, 100000000000000)
                elif self.check_winner(grid, 2): # Vinci tu
                    return (None, -10000000000000)
                else: # Pareggio
                    return (None, 0)
            else:
                return (None, self.score_position(grid, 1))
            
        if maximizingPlayer:
            value = -math.inf
            best_col = np.random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(grid, col)
                b_copy = [r[:] for r in grid]
                b_copy[row][col] = 1
                new_score = self.minimax(b_copy, depth-1, False)[1]
                if new_score > value:
                    value = new_score
                    best_col = col
            return best_col, value
        else:
            value = math.inf
            best_col = np.random.choice(valid_locations)
            for col in valid_locations:
                row = self.get_next_open_row(grid, col)
                b_copy = [r[:] for r in grid]
                b_copy[row][col] = 2
                new_score = self.minimax(b_copy, depth-1, True)[1]
                if new_score < value:
                    value = new_score
                    best_col = col
            return best_col, value

    def on_column_click(self, event):
        if self.game_over: return  # Blocca i clic a partita finita
        if self.shared_target_col.value != -1: return

        col = event.x // self.col_w
        if 0 <= col < 7:
            row = self.get_next_open_row(self.grid, col)
            if row is not None:
                # 1. IL TUO TURNO
                self.grid[row][col] = 2 
                self.draw_board()
                self.shared_user_col.value = col 
                
                if self.check_winner(self.grid, 2):
                    self.status_label.configure(text="🎉 HAI VINTO! 🎉", text_color="#2ecc71")
                    self.game_over = True
                    return
                
                if len(self.get_valid_locations(self.grid)) == 0:
                    self.status_label.configure(text="🤝 PAREGGIO! Griglia piena.", text_color="#f1c40f")
                    self.game_over = True
                    return
                
                self.status_label.configure(text="Il robot sta calcolando la mossa...", text_color="orange")
                self.update()
                
                # 2. IL TURNO DEL ROBOT
                robot_col = self.find_winning_move(self.grid, 1)
                if robot_col is None:
                    robot_col, _ = self.minimax(self.grid, 3, True)
                if robot_col is not None:
                    robot_row = self.get_next_open_row(self.grid, robot_col)
                    self.grid[robot_row][robot_col] = 1
                    self.draw_board()
                    
                    if self.check_winner(self.grid, 1):
                        self.status_label.configure(text="💀 HAI PERSO! Vince il Robot 💀", text_color="#e74c3c")
                        self.game_over = True
                        self.shared_target_col.value = robot_col
                        return

                    if len(self.get_valid_locations(self.grid)) == 0:
                        self.status_label.configure(text="🤝 PAREGGIO! Griglia piena.", text_color="#f1c40f")
                        self.game_over = True
                        self.shared_target_col.value = robot_col
                        return
                    
                    self.status_label.configure(text=f"Il robot muove nella colonna {robot_col+1}", text_color="white")
                    self.shared_target_col.value = robot_col

    def reset_status(self):
        self.status_label.configure(text="Tocca a te! Seleziona una colonna.", text_color="gray")

def run_gui(shared_target_col, shared_user_col):
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("blue")
    app = MeArmConnect4Panel(shared_target_col, shared_user_col)
    
    def check_status():
        # Riporta lo status a "Tocca a te" solo se il gioco non è finito
        if not app.game_over:
            if shared_target_col.value == -1 and "muove" in app.status_label.cget("text"):
                app.reset_status()
        app.after(200, check_status)
        
    check_status()
    app.mainloop()

#processo pybullet
if __name__ == "__main__":
    import pybullet as p
    mp.set_start_method('spawn')

    shared_target_col = mp.Value('i', -1)
    shared_user_col = mp.Value('i', -1)

    gui_process = mp.Process(target=run_gui, args=(shared_target_col, shared_user_col))
    gui_process.start()

    env = MeArmRealEnv(render_mode="human")
    env.reset()
    
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=env._client)

    #usa calibra.py per calibrare angoli braccio
    HOME_ANGLES = [0.620, -0.069, 1.182]
    
    COL_ANGLES = [
        [ 0.422, -0.285, 0.446], # Colonna 1
        [ 0.289, -0.167, 0.364], # Colonna 2

        [ 0.148, -0.147, 0.364], # Colonna 3
        [ 0.013, -0.100, 0.364], # Colonna 4
        [-0.124, -0.108, 0.364], # Colonna 5

        [-0.265, -0.170, 0.364], # Colonna 6
        [-0.389, -0.248, 0.364]  # Colonna 7
    ]
    # ==========================================

    for i in range(3):
        p.resetJointState(env.robot_id, env.joint_indices[i], targetValue=HOME_ANGLES[i], physicsClientId=env._client)
    p.resetJointState(env.robot_id, env.joint_indices[3], targetValue=1.57, physicsClientId=env._client)
    
    current_positions, _ = env._get_joint_states()
    env._sync_mimic_kinematics(current_positions)

    IDLE = 0
    GRAB = 1
    LIFT_ARM = 2
    ROTATE_TO_COL = 3
    DROP = 4
    ROTATE_TO_HOME = 5
    LOWER_ARM = 6
    
    state = IDLE
    gripper_open = True
    
    target_joints = np.array(HOME_ANGLES, dtype=np.float32)
    current_joints = np.array(HOME_ANGLES, dtype=np.float32)
    
    timer = 0
    held_token = None
    active_col = -1 

    GRID_X = 0.14
    TOKEN_RADIUS = 0.0060  # 6 millimetri per sfere perfette

    try:
        while True:
            #sfera umano
            if shared_user_col.value != -1:
                col_y = 0.06 - (shared_user_col.value * 0.02)
                u_vis = p.createVisualShape(p.GEOM_SPHERE, radius=TOKEN_RADIUS, rgbaColor=[0.9, 0.8, 0.1, 1])
                u_col = p.createCollisionShape(p.GEOM_SPHERE, radius=TOKEN_RADIUS)
                
                #Salviamo l'ID della sfera creata
                u_id = p.createMultiBody(
                    baseMass=0.02, 
                    baseCollisionShapeIndex=u_col, baseVisualShapeIndex=u_vis,
                    basePosition=[GRID_X, col_y, 0.15] 
                )
                #fisica anti-rimbalzo e anti-vibrazione
                p.changeDynamics(u_id, -1, restitution=0.0, rollingFriction=0.1, spinningFriction=0.1)
                
                shared_user_col.value = -1

            ee_pos = env._get_ee_pos()

            #MACCHINA A STATI
            if state == IDLE:
                target_joints = np.array(HOME_ANGLES)
                gripper_open = True
                if shared_target_col.value != -1:
                    active_col = shared_target_col.value
                    state = GRAB
                    timer = 20 
                    
            elif state == GRAB:
                gripper_open = False
                timer -= 1
                if timer <= 0:
                    #SFERA ROBOT TRATTENUTA DALLA PINZA(puramente visiva)
                    t_vis = p.createVisualShape(p.GEOM_SPHERE, radius=TOKEN_RADIUS, rgbaColor=[0.9, 0.1, 0.1, 1])
                    held_token = p.createMultiBody(baseMass=0, baseVisualShapeIndex=t_vis, basePosition=[ee_pos[0], ee_pos[1], ee_pos[2]])
                    state = LIFT_ARM
                    
            elif state == LIFT_ARM:
                #Alza braccio tenendo la Base a Home
                target_joints[0] = HOME_ANGLES[0]
                target_joints[1] = COL_ANGLES[active_col][1]
                target_joints[2] = COL_ANGLES[active_col][2]
                
                #Calcoliamo l'errore, sul bersaglio esatto
                if np.linalg.norm(current_joints - target_joints) < 0.015:
                    state = ROTATE_TO_COL
                    
            elif state == ROTATE_TO_COL:
                #Ruota con braccio già alzato
                target_joints[0] = COL_ANGLES[active_col][0]
                target_joints[1] = COL_ANGLES[active_col][1]
                target_joints[2] = COL_ANGLES[active_col][2]
                
                if np.linalg.norm(current_joints - target_joints) < 0.015:
                    state = DROP
                    timer = 20 
                    
            elif state == DROP:
                timer -= 1
                if timer <= 0:
                    gripper_open = True 
                    
                    if held_token is not None:
                        p.removeBody(held_token)
                        held_token = None
                    
                    #GENERA SFERA CADUTA (sfera fisica robot)
                    t_vis = p.createVisualShape(p.GEOM_SPHERE, radius=TOKEN_RADIUS, rgbaColor=[0.9, 0.1, 0.1, 1])
                    t_col = p.createCollisionShape(p.GEOM_SPHERE, radius=TOKEN_RADIUS)
                    
                    # Salviamo l'ID della sfera creata
                    t_id = p.createMultiBody(
                        baseMass=0.02, 
                        baseCollisionShapeIndex=t_col, baseVisualShapeIndex=t_vis,
                        basePosition=[ee_pos[0], ee_pos[1], ee_pos[2] - 0.01]
                    )
                    #fisica anti-rimbalzo e anti-vibrazione
                    p.changeDynamics(t_id, -1, restitution=0.0, rollingFriction=0.1, spinningFriction=0.1)
                    
                    shared_target_col.value = -1 
                    state = ROTATE_TO_HOME

            elif state == ROTATE_TO_HOME:
                #Ruota indietro mantenendo la quota in alto
                target_joints[0] = HOME_ANGLES[0]
                target_joints[1] = COL_ANGLES[active_col][1]
                target_joints[2] = COL_ANGLES[active_col][2]
                
                if np.linalg.norm(current_joints - target_joints) < 0.015:
                    state = LOWER_ARM
                    
            elif state == LOWER_ARM:
                #Abbassa il braccio solo dopo aver ruotato
                target_joints = np.array(HOME_ANGLES)
                
                if np.linalg.norm(current_joints - target_joints) < 0.015:
                    state = IDLE

            #attuazione motori
            current_joints = current_joints * 0.93 + target_joints * 0.07
            
            for i in range(3):
                p.setJointMotorControl2(
                    env.robot_id, env.joint_indices[i], p.POSITION_CONTROL, 
                    targetPosition=float(current_joints[i]), force=1000.0, maxVelocity=5.0, 
                    physicsClientId=env._client
                )
            
            gripper_angle = 1.57 if gripper_open else 0.0 
            p.setJointMotorControl2(
                env.robot_id, env.joint_indices[3], p.POSITION_CONTROL, 
                targetPosition=gripper_angle, force=500.0, maxVelocity=5.0, 
                physicsClientId=env._client
            )
            
            full_joints = [current_joints[0], current_joints[1], current_joints[2], gripper_angle]
            env._apply_mimic_joints(full_joints)
            p.stepSimulation(physicsClientId=env._client)
            
            actual_positions, _ = env._get_joint_states()
            env._sync_mimic_kinematics(actual_positions)
            
            if held_token is not None:
                p.resetBasePositionAndOrientation(held_token, [ee_pos[0], ee_pos[1], ee_pos[2]], [0,0,0,1])

            time.sleep(1.0 / 60.0)
            
    except KeyboardInterrupt:
        pass
    finally:
        gui_process.terminate()
        env.close()

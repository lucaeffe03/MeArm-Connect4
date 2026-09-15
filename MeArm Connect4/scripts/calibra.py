import os
import sys
import time
import pybullet as p

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from envs.mearm_real_env import MeArmRealEnv

def main():
    env = MeArmRealEnv(render_mode="human")
    env.reset()

    
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=env._client)

    #Slider per i motori
    slider_base = p.addUserDebugParameter("1. Base (Rotazione)", env.joint_limits[0][0], env.joint_limits[0][1], 0.620)
    slider_spalla = p.addUserDebugParameter("2. Spalla (Avanti/Indietro)", env.joint_limits[1][0], env.joint_limits[1][1], -0.069)
    slider_gomito = p.addUserDebugParameter("3. Gomito (Su/Giu)", env.joint_limits[2][0], env.joint_limits[2][1], 1.182)
    
    #Cubi olografici azzurri come riferimento
    GRID_X = 0.14
    GRID_Z_DROP = 0.08  #Altezza perfetta di sgancio
    
    for col in range(7):
        col_y = 0.06 - (col * 0.02)
        ghost_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.008, 0.008, 0.008], rgbaColor=[0.0, 1.0, 1.0, 0.4])
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=ghost_vis, basePosition=[GRID_X, col_y, GRID_Z_DROP])

    #blocco rosso sulla pinza per riferimento (da sovrappore con olografico)
    t_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.008, 0.008, 0.008], rgbaColor=[0.9, 0.1, 0.1, 1.0])
    held_token = p.createMultiBody(baseMass=0, baseVisualShapeIndex=t_vis, basePosition=[0,0,0])

    print("\n" + "="*60)
    print(" MODALITA' CALIBRAZIONE MANUALE (Telecamera Libera)")
    print(" Sovrapponi il blocco rosso ai cubi azzurri!")
    print(" (Usa Ctrl+C nel terminale per uscire e salvare)")
    print("="*60 + "\n")

    v_base = 0.620
    v_spalla = -0.069
    v_gomito = 1.182

    try:
        while p.isConnected(env._client):
            try:
                v_base = p.readUserDebugParameter(slider_base)
                v_spalla = p.readUserDebugParameter(slider_spalla)
                v_gomito = p.readUserDebugParameter(slider_gomito)
            except p.error:
                pass

            #Applica i valori ai motori
            p.setJointMotorControl2(env.robot_id, env.joint_indices[0], p.POSITION_CONTROL, v_base)
            p.setJointMotorControl2(env.robot_id, env.joint_indices[1], p.POSITION_CONTROL, v_spalla)
            p.setJointMotorControl2(env.robot_id, env.joint_indices[2], p.POSITION_CONTROL, v_gomito)
            
            env._apply_mimic_joints([v_base, v_spalla, v_gomito, 0.0])
            p.stepSimulation(physicsClientId=env._client)
            
            #Aggiorna la posizione del blocchetto rosso sulla pinza
            ee_pos = env._get_ee_pos()
            p.resetBasePositionAndOrientation(held_token, [ee_pos[0], ee_pos[1], ee_pos[2]], [0,0,0,1])

            #Stampa i valori
            sys.stdout.write(f"\r[ANGOLI] Base: {v_base:.3f} | Spalla: {v_spalla:.3f} | Gomito: {v_gomito:.3f}      ")
            sys.stdout.flush()
            
            time.sleep(1.0 / 60.0)
            
    except KeyboardInterrupt:
        pass 
    finally:
        print("\n\nCalibrazione terminata con successo.")
        if p.isConnected(env._client):
            env.close()

if __name__ == "__main__":
    main()
import time
import numpy as np
import rerun as rr
from ekf import EKF
from imu_api import ImuReader

def main():
    PC_IP = "192.168.1.144"
    
    # Nouvelle API 0.28+
    rec = rr.new_recording(application_id="Glider_INS_Remote")
    
    print(f"📡 Tentative de connexion à {PC_IP}:9876...")
    
    # Connexion TCP vers le viewer sur ton PC
    rr.connect(f"{PC_IP}:9876", recording=rec, flush_timeout_sec=1.0)
    
    # Alternative si connect ne marche pas :
    # rr.send(rec.to_tcp(f"{PC_IP}:9876"))
    
    print("✅ Connecté!")
    
    # Configuration du repère 3D
    rr.log("world", rr.ViewCoordinates.RUB, recording=rec, static=True)
    
    # Initialisation des composants
    imu = ImuReader(port="/dev/ttyS0", baudrate=115200)
    ekf = EKF(initialization_duration=5.0, sample_rate=100)
    
    last_time = time.time()
    
    print("🚀 Démarrage du système...")
    
    with imu:
        while True:
            data = imu.read(timeout=0.1)
            if data is None:
                continue
            
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            accel = np.array([data['ax'], data['ay'], data['az']]).reshape(3, 1)
            gyro = np.array([data['gx'], data['gy'], data['gz']]).reshape(3, 1)
            mag = np.array([data['mx'], data['my'], data['mz']]).reshape(3, 1)
            
            imu_data = {'accel': accel, 'gyro': gyro, 'mag': mag}
            
            if not ekf.isInitialized:
                progress = ekf.compute_initial_state(imu_data)
                if progress is not None:
                    rr.log("debug/calib_progress", rr.Scalar(progress * 100), recording=rec)
                continue
            
            ekf.predict(imu_data, dt)
            ekf.update(imu_data, gps_data=None, phase="glide")
            
            log_to_rerun(ekf, data, rec)

def log_to_rerun(ekf, raw_data, rec):
    """ Centralise l'envoi des données à Rerun """
    
    q = ekf.x[0:4].flatten()
    pos = ekf.x[4:7].flatten()
    vel = ekf.x[7:10].flatten()
    bg = ekf.x[10:13].flatten()
    ba = ekf.x[13:16].flatten()
    
    # Quaternion : Rerun attend [x, y, z, w]
    rr_quat = [q[1], q[2], q[3], q[0]]
    
    # Visualisation 3D
    rr.log(
        "world/glider",
        rr.Transform3D(
            translation=pos,
            rotation=rr.Quaternion(xyzw=rr_quat)
        ),
        recording=rec
    )
    
    rr.log(
        "world/glider/body", 
        rr.Boxes3D(half_sizes=[0.5, 0.2, 0.05], colors=[0, 255, 0]),
        recording=rec
    )
    
    # Télémétrie
    rr.log("telemetry/velocity_norm", rr.Scalar(np.linalg.norm(vel)), recording=rec)
    rr.log("telemetry/altitude", rr.Scalar(pos[2]), recording=rec)
    
    # Biais
    rr.log("debug/bias/gyro_x", rr.Scalar(bg[0]), recording=rec)
    rr.log("debug/bias/gyro_y", rr.Scalar(bg[1]), recording=rec)
    rr.log("debug/bias/gyro_z", rr.Scalar(bg[2]), recording=rec)
    
    # Données brutes
    rr.log("debug/accel_raw_norm", rr.Scalar(np.linalg.norm([raw_data['ax'], raw_data['ay'], raw_data['az']])), recording=rec)

if __name__ == "__main__":
    main()
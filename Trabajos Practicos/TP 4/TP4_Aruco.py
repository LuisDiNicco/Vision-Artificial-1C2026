import cv2
import numpy as np
import pyrender
import trimesh
from trimesh.transformations import rotation_matrix

mesh_trimesh = trimesh.load(
    "vision_artificial.obj",
    force="mesh"
)

size = mesh_trimesh.bounds[1] - mesh_trimesh.bounds[0]
print("size:", size)

print(type(mesh_trimesh))

mesh_trimesh.apply_scale(0.02)

mesh_trimesh.apply_translation(
    -mesh_trimesh.centroid
)

material = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    roughnessFactor=0.8,
    baseColorFactor=[1.0, 0.0, 0.0, 1.0]  # rojo
)

mesh = pyrender.Mesh.from_trimesh(
    mesh_trimesh,
    material=material
)

# ==========================================
# CALIBRACIÓN DE CÁMARA (REEMPLAZAR)
# ==========================================
class CameraCalibration:
    
    def __init__(self, checkerboard_size=(9, 6), square_size=30):

        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.objpoints = []
        self.imgpoints = []
        
    def prepare_object_points(self):
        objp = np.zeros((np.prod(self.checkerboard_size), 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0],
                               0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    def calibrate_from_camera(self, camera_id=0, num_images=20):

        cap = cv2.VideoCapture(camera_id)
        objp = self.prepare_object_points()
        count = 0
        
        print(f"Presiona SPACE para capturar imagen ({num_images} necesarias)")
        print("Presiona ESC para terminar calibración")
        
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            if ret:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                cv2.drawChessboardCorners(frame, self.checkerboard_size, corners, ret)
                cv2.putText(frame, f"Capturas: {count}/{num_images}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE para capturar, ESC para terminar",
                           (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                key = cv2.waitKey(1) & 0xFF
                if key == 32:  
                    self.objpoints.append(objp)
                    self.imgpoints.append(corners)
                    count += 1
                    print(f"Imagen {count} capturada")
                elif key == 27: 
                    break
            else:
                cv2.putText(frame, "Tablero de ajedrez no detectado",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.waitKey(1)
            
            cv2.imshow('Calibration', frame)
        
        cap.release()
        cv2.destroyAllWindows()
        
        if len(self.objpoints) > 0:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
                self.objpoints, self.imgpoints, gray.shape[::-1], None, None
            )
            
            if ret:
                print("Calibración exitosa!")
                print(f"Camera Matrix:\n{camera_matrix}")
                print(f"Dist Coeffs:\n{dist_coeffs}")
                return camera_matrix, dist_coeffs
            else:
                print("Error en la calibración")
                return None, None
        else:
            print("No se capturaron imágenes suficientes")
            return None, None

# ==========================================
# CONFIGURACIÓN ARUCO
# ==========================================

class AdvancedARQRApp:
    
    def __init__(self, camera_matrix=None, dist_coeffs=None):
        aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        aruco_params = cv2.aruco.DetectorParameters()

        self.detector = cv2.aruco.ArucoDetector(
            aruco_dict,
            aruco_params
        )

        # Tamaño real del marcador 
        self.marker_size = 0.05  
        
        if camera_matrix is not None and dist_coeffs is not None:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
        else:
            self.camera_matrix = np.array([
                [1000, 0, 640],
                [0, 1000, 360],
                [0, 0, 1]
            ], dtype=np.float32)
            self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        self.draw_mode = 'cube'
        self.show_info = True
    
    def create_tetrahedron(self, size):

        h = size / 2
        s = size / (2 * np.sqrt(3))

        vertices = np.array([
            [-s, -s, 0],
            [ s, -s, 0],
            [ 0,  s, 0],
            [ 0,  0, h * 2]
        ], dtype=np.float32)

        faces = [
            [0,1,2],
            [0,1,3],
            [1,2,3],
            [2,0,3]
        ]

        return vertices, faces
    
    def create_cube(self, size):

        h = size

        vertices = np.array([

            # Base sobre el marcador (z = 0)
            [-h/2, -h/2,  0],   # 0
            [ h/2, -h/2,  0],   # 1
            [ h/2,  h/2,  0],   # 2
            [-h/2,  h/2,  0],   # 3

            # Parte superior (z = -h)
            [-h/2, -h/2, h],   # 4
            [ h/2, -h/2, h],   # 5
            [ h/2,  h/2, h],   # 6
            [-h/2,  h/2, h],   # 7

        ], dtype=np.float32)

        faces = [

            [0, 1, 2, 3],

            [4, 5, 6, 7],

            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7]

        ]

        return vertices, faces
    
    def draw_ar_objects(self, frame, rvec, tvec):
        
        if self.draw_mode == 'tetrahedron':
            vertices, faces = self.create_tetrahedron(self.marker_size)
            frame = self.draw_custom_object(
                frame, vertices, faces, rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )
        elif self.draw_mode == 'cube':
            vertices, faces = self.create_cube(self.marker_size)
            frame = self.draw_custom_object(
                frame, vertices, faces, rvec, tvec,
                self.camera_matrix, self.dist_coeffs
            )
        elif self.draw_mode == '3D':
            self.draw_3d_object(frame, rvec, tvec)
        elif self.draw_mode == 'text':
            # -----------------------------
            # Pose OpenCV -> matriz 4x4
            # -----------------------------
            R, _ = cv2.Rodrigues(rvec)

            pose_cv = np.eye(4, dtype=np.float32)
            pose_cv[:3, :3] = R
            pose_cv[:3, 3] = tvec.flatten()

            # -----------------------------
            # Conversión OpenCV -> OpenGL
            # -----------------------------

            opencv_to_opengl = np.array([
                [1,  0,  0, 0],
                [0, -1,  0, 0],
                [0,  0, -1, 0],
                [0,  0,  0, 1]
            ], dtype=np.float32)

            pose_gl = opencv_to_opengl @ pose_cv

            # -----------------------------
            # Offset del modelo
            # -----------------------------

            model_transform = np.eye(4, dtype=np.float32)

            # Ajustar según tu modelo
            rot_z = rotation_matrix(
                np.radians(90),
                [1, 0, 0]
            )

            model_transform = rot_z

            model_transform[2,3] = 0.03
            #model_transform[2, 3] = 0.03

            model_pose = pose_gl @ model_transform

            # -----------------------------
            # Agregar modelo
            # -----------------------------

            node = scene.add(
                mesh,
                pose=model_pose
            )

            try:

                color, depth = renderer.render(scene)

                # Convertir RGB -> BGR
                color_bgr = cv2.cvtColor(
                    color,
                    cv2.COLOR_RGB2BGR
                )

                # Máscara de píxeles renderizados
                mask = depth > 0

                # Superponer únicamente donde existe geometría
                frame[mask] = color_bgr[mask]

            finally:

                scene.remove_node(node)
        
        return frame
    
    def draw_custom_object(self,frame, object_vertices, object_faces, rvec, tvec, camera_matrix, dist_coeffs):

        points_2d, _ = cv2.projectPoints(
            object_vertices,
            rvec,
            tvec,
            camera_matrix,
            dist_coeffs
        )
        
        points_2d= np.int32(points_2d).reshape(-1, 2)
        
        colors = [
            (255, 0, 0),      # Azul
            (0, 255, 0),      # Verde
            (0, 0, 255),      # Rojo
            (255, 255, 0),    # Cian
            (255, 0, 255),    # Magenta
            (0, 255, 255)     # Amarillo
        ]
            
        for i, face in enumerate(object_faces):

            color = colors[i % len(colors)]

            face_points = np.array(
                [points_2d[idx] for idx in face],
                dtype=np.int32
            )

            cv2.polylines(
                frame,
                [face_points],
                True,
                color,
                2
            )

        return frame

    def draw_info(self, frame, rvec, tvec):
        if not self.show_info:
            return frame
        
        distance = np.linalg.norm(tvec)
        cv2.putText(frame, f"TX: {tvec[0][0]:.2f}mm", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"TY: {tvec[1][0]:.2f}mm", (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"TZ: {tvec[2][0]:.2f}mm", (10, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"Distancia: {distance:.2f}mm", (10, 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Modo: {self.draw_mode}", (10, 220),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        
        y_offset = frame.shape[0] - 120
        cv2.putText(frame, "Controles:", (10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "C: Cubo | P: Piramide | 3: Objeto | T: Texto", (10, y_offset + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(frame, "I: Info | Q: Salir", (10, y_offset + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame
    
    def draw_3d_object(self, frame, rvec, tvec):

        R, _ = cv2.Rodrigues(rvec)

        pose_cv = np.eye(4)
        pose_cv[:3, :3] = R
        pose_cv[:3, 3] = tvec.flatten()

        print("tvec =", tvec.flatten())
        print(pose_cv)

        opencv_to_opengl = np.array([
            [1,  0,  0, 0],
            [0, -1,  0, 0],
            [0,  0, -1, 0],
            [0,  0,  0, 1]
        ])
        pose_gl = opencv_to_opengl @ pose_cv

        cube_offset = np.eye(4)
        #cube_offset[2, 3] = -0.025
        #cube_offset[0,3] = 0.03
        #cube_offset[1,3] = 0.03
        cube_offset[2,3] = 0.025

        cube_node = scene.add(
            cube_mesh,
            pose=pose_gl @ cube_offset
        )

        color, depth = renderer.render(scene)

        print("depth min:", depth.min())
        print("depth max:", depth.max())
        print("pixels:", np.count_nonzero(depth))

        scene.remove_node(cube_node)

        mask = depth > 0

        frame[mask] = color[:, :, :3][mask]

    def run(self, camera_id=0):

        cap = cv2.VideoCapture(0)

        obj_points = np.array([
        [-self.marker_size/2,  self.marker_size/2, 0],
        [ self.marker_size/2,  self.marker_size/2, 0],
        [ self.marker_size/2, -self.marker_size/2, 0],
        [-self.marker_size/2, -self.marker_size/2, 0]
        ], dtype=np.float32)

        print("\n=== CONTROLES ===")
        print("C: Mostrar cubo")
        print("P: Mostrar piramide")
        print("T: Mostrar Texto")
        print("3: Mostrar objeto 3D")
        print("I: Mostrar/ocultar información")
        print("Q: Salir")

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            corners, ids, rejected = self.detector.detectMarkers(frame)

            if ids is not None:

                cv2.aruco.drawDetectedMarkers(frame, corners, ids)

                for marker_corners, marker_id in zip(corners, ids):

                    img_points = marker_corners.reshape(4, 2).astype(np.float32)
                    print(type(self.camera_matrix))
                    print(self.camera_matrix.shape)
                    print(self.camera_matrix.dtype)
                    print(self.camera_matrix)
                    success, rvec, tvec = cv2.solvePnP(
                        obj_points,
                        img_points,
                        self.camera_matrix,
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )

                    if success:

                        # Dibujar ejes XYZ
                        cv2.drawFrameAxes(
                            frame,
                            self.camera_matrix,
                            self.dist_coeffs,
                            rvec,
                            tvec,
                            self.marker_size * 0.5
                        )

                        center = np.mean(img_points, axis=0).astype(int)

                        cv2.putText(
                            frame,
                            f"ID {marker_id[0]}",
                            tuple(center),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                        # Distancia aproximada al marcador
                        distance = np.linalg.norm(tvec)

                        cv2.putText(
                            frame,
                            f"{distance:.2f} m",
                            (center[0], center[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 0, 0),
                            2
                        )

                        frame = self.draw_ar_objects(
                            frame,
                            rvec,
                            tvec
                        )

                        if self.show_info:
                            frame = self.draw_info(
                                frame,
                                rvec,
                                tvec
                            )

                    else:

                        cv2.putText(
                            frame,
                            "ArUco no detectado",
                            (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0,0,255),
                            2
                        )

            cv2.imshow("ArUco Pose", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('p'):
                self.draw_mode = 'tetrahedron'
            elif key == ord('c'):
                self.draw_mode = 'cube'
            elif key == ord('t'):
                self.draw_mode = 'text'
            elif key == ord('3'):
                self.draw_mode = '3D'
            elif key == ord('i'):
                self.show_info = not self.show_info

        cap.release()
        cv2.destroyAllWindows()

# ==========================================
# VIDEO
# ==========================================

if __name__ == "__main__":
    import sys

    print("=== AR Aruco ===")
    print("1. Calibrar cámara primero")
    print("2. Correr aplicación AR")
    
    choice = input("¿Deseas calibrar la cámara? (s/n): ").lower()
    
    camera_matrix = None
    dist_coeffs = None
    
    if choice == 's':
        calibration = CameraCalibration(checkerboard_size=(9, 6), square_size=30)
        camera_matrix, dist_coeffs = calibration.calibrate_from_camera(num_images=10)

    app = AdvancedARQRApp(camera_matrix, dist_coeffs)

    # ----------------------------------
    # Cámara calibrada
    # ----------------------------------

    fx = app.camera_matrix[0, 0]
    fy = app.camera_matrix[1, 1]
    cx = app.camera_matrix[0, 2]
    cy = app.camera_matrix[1, 2]

    # ----------------------------------
    # Escena
    # ----------------------------------

    scene = pyrender.Scene(
        bg_color=[0, 0, 0, 0],
        ambient_light=[0.2, 0.2, 0.2]
    )

    # Cubo de 5 cm
    cube = trimesh.creation.box(
        extents=(0.05, 0.05, 0.05)
    )

    cube_mesh = pyrender.Mesh.from_trimesh(
        cube,
        smooth=False
    )

    light = pyrender.DirectionalLight(
        color=np.ones(3),
        intensity=3.0
    )

    light_node = scene.add(
        light,
        pose=np.eye(4)
    )

    # Cámara virtual usando intrínsecos reales
    camera = pyrender.IntrinsicsCamera(
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy
    )

    camera_node = scene.add(
        camera,
        pose=np.eye(4)
    )

    # Renderer
    renderer = pyrender.OffscreenRenderer(
        viewport_width=1280,
        viewport_height=720
    )
    app.run()
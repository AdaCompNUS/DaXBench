import os
import time

import cv2
import jax.numpy as jnp
import numpy as np
import pyglet
import pyrender
import trimesh

os.environ['PYOPENGL_PLATFORM'] = 'egl'
pyglet.options['shadow_window'] = False
my_path = os.path.dirname(os.path.abspath(__file__))


class BasicPyRenderer:
    def __init__(self, cam_pose=None, screen_size=(640, 480)):
        # Scene creation
        scene = pyrender.Scene(ambient_light=np.array([0.05, 0.05, 0.05, 1.0]))

        # Wood trimesh
        wood_trimesh = trimesh.load(f'{my_path}/models/wood.obj')
        wood_mesh = pyrender.Mesh.from_trimesh(wood_trimesh)
        ground_size = np.array([1.0, 1.0, 0])
        self.wood_node = scene.add(wood_mesh)
        self.wood_node.scale *= ground_size / (wood_mesh.bounds[1] - wood_mesh.bounds[0])
        self.wood_node.scale[2] = 1.0
        self.wood_node.translation = [0.48 * ground_size[0], 0.46 * ground_size[1], -0.02]

        # Light creation
        direc_l = pyrender.DirectionalLight(color=np.ones(3), intensity=1.0)
        spot_l = pyrender.SpotLight(color=np.ones(3), intensity=10.0,
                                    innerConeAngle=np.pi / 16, outerConeAngle=np.pi / 6)
        point_l = pyrender.PointLight(color=np.ones(3), intensity=10.0)

        # Camera creation
        cam = pyrender.PerspectiveCamera(yfov=(np.pi / 3.0))

        light_pose = self.look_at(np.array([0.9, 0.5, 1.0]), np.array([0.6, 0.5, 0]))
        if cam_pose is None: cam_pose = light_pose

        self.direc_l_node = scene.add(direc_l, pose=light_pose)
        self.spot_l_node = scene.add(spot_l, pose=light_pose)

        self.cam_node = scene.add(cam, pose=cam_pose)
        self.renderer = pyrender.OffscreenRenderer(viewport_width=screen_size[0], viewport_height=screen_size[1])
        self.scene = scene

    @staticmethod
    def look_at(camera_position, camera_target, up_vector=np.array([0, 0, 1])):
        vector = camera_position - camera_target
        vector = vector / np.linalg.norm(vector)

        vector2 = np.cross(up_vector, vector)
        vector2 = vector2 / np.linalg.norm(vector2)

        vector3 = np.cross(vector, vector2)
        view_mat = np.array([
            [vector2[0], vector3[0], vector[0], 0.0],
            [vector2[1], vector3[1], vector[1], 0.0],
            [vector2[2], vector3[2], vector[2], 0.0],
            [-np.dot(vector2, camera_position), -np.dot(vector3, camera_position), -np.dot(vector, camera_position),
             1.0]
        ])

        # pose_mat = np.eye(4)
        # pose_mat[:3, 3] = camera_position
        # pose_mat[:3, :3] = np.linalg.inv(view_mat.T)[:3, :3]
        pose_mat = np.linalg.inv(view_mat.T)
        return pose_mat


class MeshPyRenderer(BasicPyRenderer):
    def __init__(self):
        super().__init__()
        self.obj_node = None
        self.obj_node_ = None
        self.gripper0 = None
        self.gripper1 = None

    def render(self, x_grid, indices, ps0, visualize=True):

        x_grid = x_grid[..., [0, 2, 1]]
        indices_ = indices[:, [0, 2, 1]]
        vertices = x_grid.reshape((-1, 3))

        # front side
        tms = trimesh.Trimesh(vertices=vertices, faces=indices)
        m = pyrender.Mesh.from_trimesh(tms)
        if self.obj_node is not None:
            self.scene.remove_node(self.obj_node)
        self.obj_node = self.scene.add(m)

        # back side
        tms = trimesh.Trimesh(vertices=vertices, faces=indices_)
        m = pyrender.Mesh.from_trimesh(tms)
        if self.obj_node_ is not None:
            self.scene.remove_node(self.obj_node_)
        self.obj_node_ = self.scene.add(m)

        if self.gripper0 is None:
            boxf_trimesh = trimesh.creation.icosphere(radius=0.01, subdivisions=4)
            boxf_face_colors = np.random.uniform(size=boxf_trimesh.faces.shape)
            boxf_trimesh.visual.face_colors = boxf_face_colors
            gripper0 = pyrender.Mesh.from_trimesh(boxf_trimesh, smooth=False)
            self.gripper0 = self.scene.add(gripper0)
        self.gripper0.translation = jnp.array(ps0)[:3][jnp.array((0, 2, 1))]

        color, depth = self.renderer.render(self.scene)
        color = color[:, ::-1, ::-1]
        depth = depth[:, ::-1]

        if visualize:
            cv2.imshow('color', color)
            cv2.imshow('depth', depth)
            cv2.waitKey(10)
        return color, depth


class ParticlePyRenderer(BasicPyRenderer):
    def __init__(self, cam_pose=None):
        super().__init__(cam_pose=cam_pose)
        self.obj_node = None

    def render(self, state, visualize=True, radius=0.008):
        x = np.array(state[0][0])

        # covert x from x,z,y to x,y,z
        x = x[:, [0, 2, 1]]

        sm = trimesh.creation.uv_sphere(radius=radius)
        sm.visual.vertex_colors = [0.5, 0.5, 0.5]
        tfs = np.tile(np.eye(4), (len(x), 1, 1))
        tfs[:, :3, 3] = x
        m = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        if self.obj_node is not None:
            self.scene.remove_node(self.obj_node)
        self.obj_node = self.scene.add(m)

        color, depth = self.renderer.render(self.scene)
        if visualize:
            cv2.imshow('color', color[:, :, ::-1])
            # cv2.imshow('depth', depth)
            cv2.waitKey(10)
        return color, depth


class WaterPyRenderer(BasicPyRenderer):
    def __init__(self, cam_pose=None, screen_size=(640, 480)):
        super().__init__(cam_pose=cam_pose, screen_size=screen_size)
        self.obj_node = None

    def render(self, state, visualize=True):
        x = state.x[0]
        x = np.array(x[:, [0, 2, 1]])

        m = pyrender.Mesh.from_points(x, colors=np.array([100, 100, 249, 125]) / 255.0)

        # sm = trimesh.creation.uv_sphere(radius=0.002)
        # sm.visual.vertex_colors = np.array([100, 100, 249, 125]) / 255.0
        # tfs = np.tile(np.eye(4), (len(x), 1, 1))
        # tfs[:, :3, 3] = x
        # m = pyrender.Mesh.from_trimesh(sm, poses=tfs)

        if self.obj_node is not None:
            self.scene.remove_node(self.obj_node)
        self.obj_node = self.scene.add(m)

        color, depth = self.renderer.render(self.scene)
        if visualize:
            cv2.imshow('color', color[:, :, ::-1])
            cv2.imshow('depth', depth)
            cv2.waitKey(10)
        return color, depth


if __name__ == '__main__':
    # render = BasicPyRenderer()
    # img, _ = render.renderer.render(render.scene)
    # cv2.imshow('img', img[:, :, ::-1])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    width = np.array([0.006, 0.006, 0.5])[None, :]
    init_pos = np.array([0.5, 0.02, 0.5])[None, :]
    x = (np.random.uniform(size=(6000, 3)) * 2 - 1) * (0.5 * width) + init_pos

    render = ParticlePyRenderer()
    for i in range(100):
        render.render((x,))
        time.sleep(1)

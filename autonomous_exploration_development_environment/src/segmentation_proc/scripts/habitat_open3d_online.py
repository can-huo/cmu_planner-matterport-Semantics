#!/usr/bin/python3
"""RGB / depth / semantic renderer for Intel Xe.

Publishes the same topics as CMU habitat_online_v0.2.1.py, but uses Open3D
offscreen rendering instead of habitat-sim (which needs NVIDIA).
"""

import argparse
import math
import os
import struct

import numpy as np
import open3d as o3d
from open3d.visualization.rendering import MaterialRecord, OffscreenRenderer
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as ROS_Image

D3_40_COLORS_RGB = np.array(
    [
        [31, 119, 180], [174, 199, 232], [255, 127, 14], [255, 187, 120],
        [44, 160, 44], [152, 223, 138], [214, 39, 40], [255, 152, 150],
        [148, 103, 189], [197, 176, 213], [140, 86, 75], [196, 156, 148],
        [227, 119, 194], [247, 182, 210], [127, 127, 127], [199, 199, 199],
        [188, 189, 34], [219, 219, 141], [23, 190, 207], [158, 218, 229],
        [57, 59, 121], [82, 84, 163], [107, 110, 207], [156, 158, 222],
        [99, 121, 57], [140, 162, 82], [181, 207, 107], [206, 219, 156],
        [140, 109, 49], [189, 158, 57], [231, 186, 82], [231, 203, 148],
        [132, 60, 57], [173, 73, 74], [214, 97, 107], [231, 150, 156],
        [123, 65, 115], [165, 81, 148], [206, 109, 189], [222, 158, 214],
    ],
    dtype=np.uint8,
)


def quat_to_rot(x, y, z, w):
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def load_semantic_mesh(ply_path):
    with open(ply_path, "rb") as f:
        n_vert = n_face = None
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                n_vert = int(line.split()[-1])
            elif line.startswith("element face"):
                n_face = int(line.split()[-1])
            elif line == "end_header":
                break
        vert_raw = np.fromfile(f, dtype=np.uint8, count=n_vert * 35).reshape(n_vert, 35)
        xyz = vert_raw[:, 0:12].view("<f4").copy()
        face_dt = np.dtype(
            {
                "names": ["n", "i0", "i1", "i2", "mat", "seg", "cat"],
                "formats": ["u1", "<i4", "<i4", "<i4", "<i4", "<i4", "<i4"],
                "offsets": [0, 1, 5, 9, 13, 17, 21],
                "itemsize": 25,
            }
        )
        faces = np.fromfile(f, dtype=face_dt, count=n_face)

    tri = np.stack([faces["i0"], faces["i1"], faces["i2"]], axis=1).astype(np.int32)
    cats = np.maximum(faces["cat"], 0).astype(np.int64)
    palette = D3_40_COLORS_RGB[cats % 40].astype(np.float64) / 255.0
    colors = np.zeros((n_vert, 3), dtype=np.float64)
    for k in range(3):
        colors[tri[:, k]] = palette

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    mesh.triangles = o3d.utility.Vector3iVector(tri)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()
    return mesh


def make_renderer(width, height, background):
    renderer = OffscreenRenderer(width, height)
    renderer.scene.set_background(background)
    renderer.scene.show_skybox(False)
    try:
        renderer.scene.scene.enable_sun_light(True)
        renderer.scene.scene.set_sun_light([0.35, 0.15, -1.0], [1.0, 1.0, 1.0], 90000)
    except Exception:
        pass
    return renderer


class HabitatOpen3D(Node):
    def __init__(self, args):
        super().__init__("habitat_open3d_online")
        self.args = args
        self.width = args.width
        self.height = args.height
        self.hfov = args.hfov
        self.near = args.near
        self.far = args.far
        self.camera_offset_z = args.camera_offset_z
        self.have_odom = False
        self.stamp = self.get_clock().now().to_msg()
        self.pos = np.zeros(3, dtype=np.float64)
        self.quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        self.get_logger().info("Loading RGB mesh: " + args.rgb_mesh)
        model = o3d.io.read_triangle_model(args.rgb_mesh)
        self.rgb_renderer = make_renderer(self.width, self.height, [0.0, 0.0, 0.0, 1.0])
        self.rgb_renderer.scene.add_model("matterport", model)

        self.get_logger().info("Loading semantic mesh: " + args.semantic_mesh)
        sem_mesh = load_semantic_mesh(args.semantic_mesh)
        sem_mat = MaterialRecord()
        sem_mat.shader = "defaultUnlit"
        self.sem_renderer = make_renderer(self.width, self.height, [0.0, 0.0, 0.0, 1.0])
        self.sem_renderer.scene.add_geometry("semantic", sem_mesh, sem_mat)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
        )
        self.create_subscription(Odometry, "/state_estimation", self.on_odom, qos)
        self.color_pub = self.create_publisher(ROS_Image, "/habitat_camera/color/image", 2)
        self.depth_pub = self.create_publisher(ROS_Image, "/habitat_camera/depth/image", 2)
        self.semantic_pub = self.create_publisher(ROS_Image, "/habitat_camera/semantic/image", 2)
        self.create_timer(1.0 / max(args.rate, 1.0), self.on_timer)
        self.get_logger().info(
            "Open3D habitat renderer ready %dx%d @ %.1f Hz" % (self.width, self.height, args.rate)
        )

    def on_odom(self, msg):
        self.stamp = msg.header.stamp
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pos = np.array([p.x, p.y, p.z], dtype=np.float64)
        self.quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        self.have_odom = True

    def set_camera(self, renderer):
        rot = quat_to_rot(*self.quat)
        eye = self.pos + np.array([0.0, 0.0, self.camera_offset_z])
        forward = rot[:, 0]
        up = rot[:, 2]
        center = eye + forward
        cam = renderer.scene.camera
        cam.set_projection(
            self.hfov,
            float(self.width) / float(self.height),
            self.near,
            self.far,
            cam.FovType.Horizontal,
        )
        cam.look_at(center, eye, up)

    def publish_image(self, pub, encoding, array, step_scale):
        msg = ROS_Image()
        msg.header.stamp = self.stamp
        msg.header.frame_id = "habitat_camera"
        msg.height = self.height
        msg.width = self.width
        msg.encoding = encoding
        msg.is_bigendian = False
        msg.step = step_scale * self.width
        msg.data = array.tobytes()
        pub.publish(msg)

    def on_timer(self):
        if not self.have_odom:
            return
        try:
            self.set_camera(self.rgb_renderer)
            rgb = np.asarray(self.rgb_renderer.render_to_image())
            depth = np.asarray(self.rgb_renderer.render_to_depth_image(True))
            self.set_camera(self.sem_renderer)
            sem = np.asarray(self.sem_renderer.render_to_image())
        except Exception as exc:
            self.get_logger().warn("render failed: %s" % exc)
            return

        depth = np.nan_to_num(depth, nan=self.far, posinf=self.far, neginf=0.0)
        depth_u8 = np.clip(depth / 10.0 * 255.0, 0, 255).astype(np.uint8)
        if rgb.ndim == 3 and rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        if sem.ndim == 3 and sem.shape[2] == 4:
            sem = sem[:, :, :3]
        self.publish_image(self.color_pub, "rgb8", np.ascontiguousarray(rgb), 3)
        self.publish_image(self.depth_pub, "mono8", np.ascontiguousarray(depth_u8), 1)
        self.publish_image(self.semantic_pub, "rgb8", np.ascontiguousarray(sem), 3)


def default_paths():
    try:
        from ament_index_python.packages import get_package_share_directory
        mesh = os.path.join(get_package_share_directory("vehicle_simulator"), "mesh", "matterport")
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        mesh = os.path.abspath(os.path.join(here, "..", "..", "vehicle_simulator", "mesh", "matterport"))
    return (
        os.path.join(mesh, "meshes", "matterport.glb"),
        os.path.join(mesh, "preview", "pointcloud.ply"),
    )


def main():
    rgb_default, sem_default = default_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb-mesh", default=rgb_default)
    parser.add_argument("--semantic-mesh", default=sem_default)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--rate", type=float, default=8.0)
    parser.add_argument("--hfov", type=float, default=90.0)
    parser.add_argument("--near", type=float, default=0.1)
    parser.add_argument("--far", type=float, default=100.0)
    parser.add_argument("--camera-offset-z", type=float, default=0.0)
    args, _unknown = parser.parse_known_args()

    rclpy.init()
    node = HabitatOpen3D(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

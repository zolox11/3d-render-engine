import numpy as np
from pyrr import Matrix44

from objects import Scene, GLBObject, DirectionalLight, PointLight, Plane
from loader import load_model
from camera import Camera


def init_scene(ctx, program, config):
    scene = Scene(ctx)

    # ================= OBJECTS =================

    ironman = GLBObject(load_model, ctx, program, "models/iron_man.glb")
    ironman.set_position(-2.2, 0, 1.2)
    ironman.set_scale(0.001, 0.001, 0.001)
    ironman.set_rotation(0, np.pi / 6, 0)
    ironman.set_roughness(0.4)
    ironman.set_metallic(1.0)
    ironman.set_ao(1.0)
    scene.add(ironman)

    # ================= FLOOR =================

    floor = Plane(ctx, program, size=config.FLOOR_SIZE, color=(0.2, 0.2, 0.2))
    floor.set_position(0, -0.5, 0)
    floor.set_roughness(0.9)
    floor.set_metallic(0.0)
    floor.set_ao(1.0)
    scene.add(floor)

    # ================= LIGHTING =================

    sun = DirectionalLight(
        direction=(0.3, -1.0, 0.2),
        color=(1.0, 0.95, 0.85),
        intensity=2.2
    )
    scene.add_light(sun)

    rim_light = PointLight(
        position=(5.0, 3.0, 2.0),
        color=(0.6, 0.8, 1.0),
        intensity=1.2
    )
    scene.add_light(rim_light)

    fill_light = PointLight(
        position=(-5.0, 2.0, -2.0),
        color=(1.0, 0.7, 0.6),
        intensity=0.7
    )
    scene.add_light(fill_light)

    # ================= CAMERA =================

    camera = Camera(position=config.CAMERA_START_POS)

    # ================= PROJECTION =================

    aspect = config.WINDOW_WIDTH / config.WINDOW_HEIGHT
    proj = Matrix44.perspective_projection(
        config.CAMERA_FOV,
        aspect,
        config.CAMERA_NEAR,
        config.CAMERA_FAR
    )

    return scene, camera, proj
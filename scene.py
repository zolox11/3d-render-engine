import numpy as np
from pyrr import Matrix44, Vector3

from objects import Scene, GLBObject, DirectionalLight, PointLight, Cube, Plane
from loader import load_model
from camera import Camera



def init_scene(ctx, program, config):
    scene = Scene(ctx)

    # ================= OBJECTS =================

    cube = Cube(ctx, program, color=(0.8, 0.3, 0.3))
    cube.set_position(0, 0.5, 0)
    cube.set_scale(1, 1, 1)
    cube.set_roughness(0.5)
    cube.set_metallic(0.0)
    cube.set_ao(1.0)
    scene.add(cube)

    spiderman = GLBObject(load_model,ctx, program, "models/spiderman.glb")
    spiderman.set_position(2, 0, 0)
    spiderman.set_scale(1.5, 1.5, 1.5)
    spiderman.set_rotation(-np.pi / 2, 0, 0)
    scene.add(spiderman)

    # ================= FLOOR =================

    floor = Plane(ctx, program, size=config.FLOOR_SIZE, color=(0.2, 0.2, 0.5))
    floor.set_position(0, -0.5, 0)
    floor.set_roughness(0.9)
    floor.set_metallic(0.0)
    floor.set_ao(1.0)
    floor.physics_enabled = False
    floor.gravity_enabled = False
    floor.mass = 0.0
    floor.collidable = True
    floor.collider_half_size = Vector3((config.FLOOR_SIZE * 0.5, 0.1, config.FLOOR_SIZE * 0.5))
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
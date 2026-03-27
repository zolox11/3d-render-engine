import numpy as np
from pyrr import Matrix44, Vector3

from objects import Scene, GLBObject, DirectionalLight, PointLight, Cube, Plane, TerrainObject
from loader import load_model
from camera import Camera
import terrain

terrain_toggle = False



def init_scene(ctx, program, config):
    scene = Scene(ctx)

    if terrain_toggle == True:
        terrain = TerrainObject(ctx,program,"height1.png",size=300,height_scale=40,resolution=1.5,texture_tiling=500.0)
        terrain.set_roughness(1)
        terrain.set_metallic(0)
        terrain.set_ao(0.4)
        scene.add(terrain)
    else:
        spiderman = GLBObject(load_model,ctx, program, "models/spiderman.glb")
        spiderman.set_position(0, -0.5, 0)
        spiderman.set_rotation(-np.pi / 2, 0, 0)
        spiderman.set_scale(1.25,1.25,1.25)
        spiderman.set_roughness(1)
        spiderman.set_metallic(0)
        spiderman.set_ao(1)
        scene.add(spiderman)

        tree = GLBObject(load_model,ctx, program, "models/tree.glb")
        tree.set_position(5, -0.5, 5)
        tree.set_rotation(0, 0, 0)
        tree.set_scale(0.05,0.05,0.05)
        tree.set_roughness(1)
        tree.set_metallic(0)
        tree.set_ao(1)
        tree.collidable = True
        tree.collider_half_size = Vector3((1.5, 3.0, 1.5))
        scene.add(tree)


        floor = Plane(ctx,program,size=config.FLOOR_SIZE,texture_path="textures/grass.png", tiling=1.0)
        floor.set_position(0,-0.5,0)
        floor.set_roughness(1)
        floor.set_metallic(0)
        floor.set_ao(0.4)
        floor.collidable = True
        floor.collider_half_size = Vector3((config.FLOOR_SIZE * 0.5, 0.5, config.FLOOR_SIZE * 0.5))
        scene.add(floor)



    # ================= OBJECTS =================
   
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
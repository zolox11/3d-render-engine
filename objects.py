import numpy as np
from pyrr import Matrix44, Vector3
import moderngl

# -------------------------
# Material System
# -------------------------
class Material:
    """Material properties for PBR rendering"""
    def __init__(self, roughness=0.5, metallic=0.0, emissive=(0.0, 0.0, 0.0), ao=1.0):
        self.roughness = max(0.0, min(1.0, roughness))  # 0.0 = mirror, 1.0 = diffuse
        self.metallic = max(0.0, min(1.0, metallic))    # 0.0 = dielectric, 1.0 = metal
        self.emissive = Vector3(emissive)                # RGB emissive color
        self.ao = max(0.0, min(1.0, ao))                 # Ambient occlusion factor

    def set_roughness(self, value):
        self.roughness = max(0.0, min(1.0, value))

    def set_metallic(self, value):
        self.metallic = max(0.0, min(1.0, value))

    def set_emissive(self, r, g, b):
        self.emissive = Vector3([r, g, b])

    def set_ao(self, value):
        self.ao = max(0.0, min(1.0, value))


# -------------------------
# Transform System
# -------------------------
class Transform:
    def __init__(self, position=(0, 0, 0), rotation=(0, 0, 0), scale=(1, 1, 1)):
        self.position = Vector3(position)
        self.rotation = Vector3(rotation)  # radians
        self.scale = Vector3(scale)

    def get_matrix(self):
        t = Matrix44.from_translation(self.position)
        rx = Matrix44.from_x_rotation(self.rotation.x)
        ry = Matrix44.from_y_rotation(self.rotation.y)
        rz = Matrix44.from_z_rotation(self.rotation.z)
        s = Matrix44.from_scale(self.scale)
        # Standard TRS order
        return t * rz * ry * rx * s

    def get_normal_matrix(self):
        """Return 3x3 inverse transpose matrix for normal transformation"""
        mat = self.get_matrix()
        mat3x3 = np.array(mat[:3, :3], dtype='f4')
        return np.linalg.inv(mat3x3).T.flatten()


# -------------------------
# Base Object
# -------------------------
class Object3D:
    def __init__(self, model=None):
        self.model = model
        self.transform = Transform()
        self.material = Material()
        self.cast_shadow = True
        self.receive_shadow = True

        # Physics properties
        self.physics_enabled = False
        self.gravity_enabled = False
        self.gravity_strength = 1.0
        self.collidable = True
        self.mass = 0.0  # mass=0 means static body
        self.velocity = Vector3((0.0, 0.0, 0.0))
        self.on_ground = False
        self.linear_damping = 4.0
        self.ground_timer = 0.0
        self.ground_time_buffer = 0.1  # keep on_ground true for a short window after last touch

        # Collider AABB half extents in local space
        self.collider_half_size = Vector3((0.5, 0.5, 0.5))

    def set_position(self, x, y, z):
        self.transform.position = Vector3([x, y, z])

    def set_rotation(self, x, y, z):
        self.transform.rotation = Vector3([x, y, z])

    def set_scale(self, x, y, z):
        self.transform.scale = Vector3([x, y, z])

    def set_roughness(self, value):
        self.material.set_roughness(value)

    def set_metallic(self, value):
        self.material.set_metallic(value)

    def set_emissive(self, r, g, b):
        self.material.set_emissive(r, g, b)

    def set_ao(self, value):
        self.material.set_ao(value)

    def render(self, program, proj, view):
        if self.model:
            # For GLB model, handle per-instance transforms in Model.render
            self.model.render(program, proj, view, root_transform=self.transform.get_matrix())
            return

        model_matrix = self.transform.get_matrix()
        mvp = proj * view * model_matrix

        if "mvp" in program:
            program["mvp"].write(mvp.astype("f4").tobytes())
        if "model" in program:
            program["model"].write(model_matrix.astype("f4").tobytes())


# -------------------------
# GLB Object
# -------------------------
class GLBObject(Object3D):
    def __init__(self, loader_func, ctx, program, path):
        model = loader_func(ctx, program, path)
        super().__init__(model)
        self.cast_shadow = True
        self.receive_shadow = True
        self.physics_enabled = False
        self.gravity_enabled = False
        self.mass = 0.0
        self.collidable = True
        # default approximate GLB collider, can be refined by user
        self.collider_half_size = Vector3((1.0, 1.0, 1.0))

    # ---- Direct access shortcuts ----
    @property
    def position(self):
        return self.transform.position

    @position.setter
    def position(self, value):
        self.transform.position = value

    @property
    def rotation(self):
        return self.transform.rotation

    @rotation.setter
    def rotation(self, value):
        self.transform.rotation = value

    @property
    def scale(self):
        return self.transform.scale

    @scale.setter
    def scale(self, value):
        self.transform.scale = value

    def set_position_vec(self, vec):
        self.transform.position = vec

    def set_rotation_vec(self, vec):
        self.transform.rotation = vec

    def set_scale_vec(self, vec):
        self.transform.scale = vec

    def render(self, program, proj, view):
        # GLB uses texture-based PBR
        if "u_use_color" in program:
            program["u_use_color"].value = False

        # Apply per-object material properties
        if "u_roughness_override" in program:
            program["u_roughness_override"].value = self.material.roughness
        if "u_metallic_override" in program:
            program["u_metallic_override"].value = self.material.metallic
        if "u_emissive_override" in program:
            program["u_emissive_override"].value = tuple(self.material.emissive)
        if "u_ao_override" in program:
            program["u_ao_override"].value = self.material.ao

        super().render(program, proj, view)


# -------------------------
# Cube Primitive
# -------------------------
class Cube(Object3D):
    def __init__(self, ctx, program, color=(1.0, 1.0, 1.0)):
        super().__init__(None)
        self.ctx = ctx
        self.program = program
        self.color = np.array(color, dtype='f4')

        # 1x1 color texture for compatibility
        color_bytes = (np.array(list(color) + [1.0]) * 255).astype('u1').tobytes()
        self.texture = ctx.texture((1, 1), 4, color_bytes)

        # Positions (3f), Normals (3f), UVs (2f), Tangents (3f)
        vertices = np.array([
            # Back face (Z = -1)
            -1, -1, -1,  0,  0, -1,  0, 0,  1, 0, 0,
             1, -1, -1,  0,  0, -1,  1, 0,  1, 0, 0,
             1,  1, -1,  0,  0, -1,  1, 1,  1, 0, 0,
            -1,  1, -1,  0,  0, -1,  0, 1,  1, 0, 0,
            # Front face (Z = +1)
            -1, -1,  1,  0,  0,  1,  0, 0, -1, 0, 0,
             1, -1,  1,  0,  0,  1,  1, 0, -1, 0, 0,
             1,  1,  1,  0,  0,  1,  1, 1, -1, 0, 0,
            -1,  1,  1,  0,  0,  1,  0, 1, -1, 0, 0,
            # Left face (X = -1)
            -1, -1, -1, -1,  0,  0,  0, 0,  0, 0, -1,
            -1, -1,  1, -1,  0,  0,  1, 0,  0, 0, -1,
            -1,  1,  1, -1,  0,  0,  1, 1,  0, 0, -1,
            -1,  1, -1, -1,  0,  0,  0, 1,  0, 0, -1,
            # Right face (X = +1)
             1, -1, -1,  1,  0,  0,  0, 0,  0, 0,  1,
             1, -1,  1,  1,  0,  0,  1, 0,  0, 0,  1,
             1,  1,  1,  1,  0,  0,  1, 1,  0, 0,  1,
             1,  1, -1,  1,  0,  0,  0, 1,  0, 0,  1,
            # Top face (Y = +1)
            -1,  1, -1,  0,  1,  0,  0, 0,  1, 0, 0,
             1,  1, -1,  0,  1,  0,  1, 0,  1, 0, 0,
             1,  1,  1,  0,  1,  0,  1, 1,  1, 0, 0,
            -1,  1,  1,  0,  1,  0,  0, 1,  1, 0, 0,
            # Bottom face (Y = -1)
            -1, -1, -1,  0, -1,  0,  0, 0,  1, 0, 0,
             1, -1, -1,  0, -1,  0,  1, 0,  1, 0, 0,
             1, -1,  1,  0, -1,  0,  1, 1,  1, 0, 0,
            -1, -1,  1,  0, -1,  0,  0, 1,  1, 0, 0,
        ], dtype='f4')

        indices = np.array([
            0, 1, 2, 2, 3, 0,
            4, 6, 5, 4, 7, 6,
            8, 9, 10, 10, 11, 8,
            12, 14, 13, 12, 15, 14,
            16, 17, 18, 18, 19, 16,
            20, 22, 21, 20, 23, 22
        ], dtype='i4')

        vbo = ctx.buffer(vertices.tobytes())
        ibo = ctx.buffer(indices.tobytes())
        self.vao = ctx.vertex_array(
            program,
            [
                (vbo, "3f 3f 2f 3f", "in_position", "in_normal", "in_uv", "in_tangent")
            ],
            ibo
        )

        # Physics defaults for cube collider
        self.physics_enabled = False
        self.gravity_enabled = False
        self.mass = 0.0
        self.collidable = True
        self.collider_half_size = Vector3((1.0, 1.0, 1.0))

    def render(self, program, proj, view):
        self.texture.use(location=0)
        
        if "u_use_color" in program:
            program["u_use_color"].value = True
        if "u_color" in program:
            program["u_color"].value = tuple(self.color)

        # Apply per-object material properties
        if "u_roughness_override" in program:
            program["u_roughness_override"].value = self.material.roughness
        if "u_metallic_override" in program:
            program["u_metallic_override"].value = self.material.metallic
        if "u_emissive_override" in program:
            program["u_emissive_override"].value = tuple(self.material.emissive)
        if "u_ao_override" in program:
            program["u_ao_override"].value = self.material.ao

        model_matrix = self.transform.get_matrix()
        mvp = proj * view * model_matrix

        if "mvp" in program:
            program["mvp"].write(mvp.astype("f4").tobytes())
        if "model" in program:
            program["model"].write(model_matrix.astype("f4").tobytes())

        # Render cube without face culling
        self.ctx.disable(moderngl.CULL_FACE)
        self.vao.render()
        self.ctx.enable(moderngl.CULL_FACE)


# -------------------------
# Plane Primitive
# -------------------------
class Plane(Object3D):
    def __init__(self, ctx, program, size=100.0, color=(0.2, 0.2, 0.2)):
        super().__init__(None)
        self.ctx = ctx
        self.program = program
        self.color = np.array(color, dtype='f4')

        # Create a solid color texture for the plane
        color_bytes = (np.array(list(color) + [1.0]) * 255).astype('u1').tobytes()
        self.texture = ctx.texture((1, 1), 4, color_bytes)
        
        # Create dummy metallic/roughness texture (1x1, RGBA)
        dummy_mr = np.array([[0, 200, 255, 255]], dtype='u1')
        self.mr_texture = ctx.texture((1, 1), 4, dummy_mr.tobytes())
        
        # Create dummy normal map (neutral blue - 1x1, RGBA)
        dummy_normal = np.array([[128, 128, 255, 255]], dtype='u1')
        self.normal_texture = ctx.texture((1, 1), 4, dummy_normal.tobytes())
        
        # Create dummy emissive (black - 1x1, RGBA)
        dummy_emissive = np.array([[0, 0, 0, 255]], dtype='u1')
        self.emissive_texture = ctx.texture((1, 1), 4, dummy_emissive.tobytes())

        # Plane vertices
        res = size / 2.0
        vertices = np.array([
            -size, 0, -size,  0, 1, 0,  0,   res,  1, 0, 0,
             size, 0, -size,  0, 1, 0,  res, res,  1, 0, 0,
             size, 0,  size,  0, 1, 0,  res, 0,    1, 0, 0,
            -size, 0,  size,  0, 1, 0,  0,   0,    1, 0, 0,
        ], dtype='f4')

        indices = np.array([
            0, 2, 1, 0, 3, 2
        ], dtype='i4')

        vbo = ctx.buffer(vertices.tobytes())
        ibo = ctx.buffer(indices.tobytes())

        self.vao = ctx.vertex_array(
            program,
            [(vbo, "3f 3f 2f 3f", "in_position", "in_normal", "in_uv", "in_tangent")],
            ibo
        )

        # Physics defaults for plane collider
        self.physics_enabled = False
        self.gravity_enabled = False
        self.mass = 0.0
        self.collidable = True
        self.collider_half_size = Vector3((size, 0.05, size))

    def render(self, program, proj, view):
        # Bind all textures to correct slots
        self.texture.use(location=0)  # Albedo
        self.mr_texture.use(location=1)  # Metallic/Roughness
        self.normal_texture.use(location=2)  # Normal
        self.emissive_texture.use(location=3)  # Emissive
        
        if "u_use_color" in program:
            program["u_use_color"].value = True
        if "u_color" in program:
            program["u_color"].value = tuple(self.color)

        # Apply per-object material properties
        if "u_roughness_override" in program:
            program["u_roughness_override"].value = self.material.roughness
        if "u_metallic_override" in program:
            program["u_metallic_override"].value = self.material.metallic
        if "u_emissive_override" in program:
            program["u_emissive_override"].value = tuple(self.material.emissive)
        if "u_ao_override" in program:
            program["u_ao_override"].value = self.material.ao

        model_matrix = self.transform.get_matrix()
        mvp = proj * view * model_matrix

        if "mvp" in program:
            program["mvp"].write(mvp.astype("f4").tobytes())
        if "model" in program:
            program["model"].write(model_matrix.astype("f4").tobytes())

        self.vao.render()


class SimpleModel:
    def __init__(self, vao):
        self.vao = vao

    def render(self, program=None):
        self.vao.render()


# -------------------------
# Lighting System
# -------------------------
class Light:
    def apply(self, program, index=0):
        pass


class DirectionalLight(Light):
    def __init__(self, direction=(1, -1, 1), color=(1, 1, 1), intensity=1.5):
        # Create direction vector and normalize it
        dir_vec = Vector3(direction)
        length = (dir_vec.x**2 + dir_vec.y**2 + dir_vec.z**2) ** 0.5
        if length > 0:
            self.direction = Vector3([dir_vec.x/length, dir_vec.y/length, dir_vec.z/length])
        else:
            self.direction = Vector3([0, -1, 0])  # Default down
        
        self.color = Vector3(color)
        self.intensity = intensity
        self.shadow_distance = 100.0
        self.shadow_resolution = 2048
    
    def get_light_space_matrix(self, center_pos=(0, 0, 0)):
        """Generate view-projection matrix for shadow mapping"""
        # Orthographic projection for directional light (sun)
        extent = self.shadow_distance / 2.0
        proj = Matrix44.orthogonal_projection(-extent, extent, -extent, extent, 0.1, self.shadow_distance * 2)
        
        # Light position far away in the direction of the light
        light_pos = Vector3(center_pos) - (self.direction * self.shadow_distance)
        view = Matrix44.look_at(
            eye=light_pos,
            target=center_pos,
            up=(0, 1, 0)
        )
        return proj * view

    def apply(self, program, index=0):
        if index < 8:  # MAX_LIGHTS limit
            program[f"lights[{index}].type"].value = 0  
            program[f"lights[{index}].direction"].value = tuple(self.direction)
            program[f"lights[{index}].position"].value = (0.0, 0.0, 0.0)
            program[f"lights[{index}].color"].value = tuple(self.color)
            program[f"lights[{index}].intensity"].value = self.intensity


class PointLight(Light):
    def __init__(self, position=(0, 5, 0), color=(1, 1, 1), intensity=2.0, radius=50.0):
        self.position = Vector3(position)
        self.color = Vector3(color)
        self.intensity = intensity
        self.radius = radius  # Light attenuation distance

    def apply(self, program, index=0):
        if index < 8:  # MAX_LIGHTS limit
            program[f"lights[{index}].type"].value = 1  
            program[f"lights[{index}].position"].value = tuple(self.position)
            program[f"lights[{index}].direction"].value = (0.0, 0.0, 0.0)
            program[f"lights[{index}].color"].value = tuple(self.color)
            program[f"lights[{index}].intensity"].value = self.intensity

    def set_position(self, x, y, z):
        """Update light position"""
        self.position = Vector3([x, y, z])


class SpotLight(Light):
    def __init__(self, position=(0, 10, 0), direction=(0, -1, 0), color=(1, 1, 1), 
                 intensity=2.0, angle=30.0, fade_angle=40.0):
        self.position = Vector3(position)
        dir_vec = Vector3(direction)
        length = (dir_vec.x**2 + dir_vec.y**2 + dir_vec.z**2) ** 0.5
        if length > 0:
            self.direction = Vector3([dir_vec.x/length, dir_vec.y/length, dir_vec.z/length])
        else:
            self.direction = Vector3([0, -1, 0])
        
        self.color = Vector3(color)
        self.intensity = intensity
        self.angle = angle
        self.fade_angle = fade_angle

    def apply(self, program, index=0):
        if index < 8:
            program[f"lights[{index}].type"].value = 1
            program[f"lights[{index}].position"].value = tuple(self.position)
            program[f"lights[{index}].direction"].value = tuple(self.direction)
            program[f"lights[{index}].color"].value = tuple(self.color)
            program[f"lights[{index}].intensity"].value = self.intensity


# -------------------------
# Scene Manager
# -------------------------
class Scene:
    def __init__(self, ctx):
        self.ctx = ctx
        self.objects = []
        self.dir_lights = []
        self.point_lights = []

    def add(self, obj):
        self.objects.append(obj)

    def add_light(self, light):
        if isinstance(light, DirectionalLight):
            self.dir_lights.append(light)
        elif isinstance(light, (PointLight, SpotLight)):
            self.point_lights.append(light)

    def render(self, program, proj, view, camera_pos):
        """Render the scene with lighting"""
        if "cam_pos" in program:
            program["cam_pos"].value = tuple(camera_pos)

        all_lights = self.dir_lights + self.point_lights

        if "light_count" in program:
            program["light_count"].value = min(len(all_lights), 8)

        for i, light in enumerate(all_lights):
            if i < 8:  # MAX_LIGHTS limit
                light.apply(program, i)

        # Set light space matrix for the first directional light
        if self.dir_lights:
            light_space_matrix = self.dir_lights[0].get_light_space_matrix((0, 0, 0))
            if "lightSpaceMatrix" in program:
                program["lightSpaceMatrix"].write(light_space_matrix.astype("f4").tobytes())

        # Render all objects
        for obj in self.objects:
            obj.render(program, proj, view)
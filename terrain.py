import numpy as np
from PIL import Image
from pyrr import Vector3, Matrix44
from texture import TerrainTexturer

class Terrain:
    def __init__(
        self,
        ctx,
        program,
        heightmap_path,
        size=100.0,
        height_scale=20.0,
        resolution=1.0,
        texture_tiling=200
    ):
        self.ctx = ctx
        self.program = program

        self.size = size
        self.height_scale = height_scale
        self.resolution = float(resolution)
        self.texture_tiling = texture_tiling

        # 1. Heightmap Data
        self.height_data = self.load_heightmap(heightmap_path)

        # 2. AAA Texture System
        # This handles Grass(5), Rock(6), Sand(7), Snow(8), Detail(9), Noise(10)
        self.texturer = TerrainTexturer(ctx, program)
        self.texturer.setup_terrain()

        # 3. Mesh Generation
        self.vertices, self.indices = self.generate_mesh()

        self.vbo = ctx.buffer(self.vertices.tobytes())
        self.ibo = ctx.buffer(self.indices.tobytes())

        # Vertex Layout: pos(3f), normal(3f), uv(2f), tangent(3f)
        self.vao = ctx.vertex_array(
            program,
            [(self.vbo, "3f 3f 2f 3f", "in_position", "in_normal", "in_uv", "in_tangent")],
            self.ibo
        )

        self.model_matrix = Matrix44.identity()

    def load_heightmap(self, path):
        img = Image.open(path).convert("L")
        return np.asarray(img).astype("f4") / 255.0

    def sample_height(self, fx, fz):
        h, w = self.height_data.shape
        fx = np.clip(fx, 0, w - 1)
        fz = np.clip(fz, 0, h - 1)

        x0, z0 = int(np.floor(fx)), int(np.floor(fz))
        x1, z1 = min(x0 + 1, w - 1), min(z0 + 1, h - 1)

        sx, sz = fx - x0, fz - z0

        h00 = self.height_data[z0, x0]
        h10 = self.height_data[z0, x1]
        h01 = self.height_data[z1, x0]
        h11 = self.height_data[z1, x1]

        h0 = h00 * (1 - sx) + h10 * sx
        h1 = h01 * (1 - sx) + h11 * sx
        return (h0 * (1 - sz) + h1 * sz) * self.height_scale

    def get_normal(self, x, z):
        eps = 1.0
        hL = self.sample_height(x - eps, z)
        hR = self.sample_height(x + eps, z)
        hD = self.sample_height(x, z - eps)
        hU = self.sample_height(x, z + eps)

        normal = Vector3([hL - hR, 2.0, hD - hU])
        return normal / np.linalg.norm(normal)

    def generate_mesh(self):
        h, w = self.height_data.shape
        step = max(0.5, self.resolution)

        vertices = []
        indices = []
        
        # Proper grid indexing for variable resolution
        z_range = np.arange(0, h, step)
        x_range = np.arange(0, w, step)
        
        for z in z_range:
            for x in x_range:
                y = self.sample_height(x, z)
                xpos = (x / (w - 1) - 0.5) * self.size
                zpos = (z / (h - 1) - 0.5) * self.size
                normal = self.get_normal(x, z)
                uv = ((x / w) * self.texture_tiling, (z / h) * self.texture_tiling)
                
                # tangent fallback (TBN will handle the rest in shader)
                vertices.extend([
                    xpos, y, zpos,
                    normal.x, normal.y, normal.z,
                    uv[0], uv[1],
                    1.0, 0.0, 0.0 
                ])

        num_x = len(x_range)
        num_z = len(z_range)
        for z in range(num_z - 1):
            for x in range(num_x - 1):
                i0 = z * num_x + x
                i1 = i0 + 1
                i2 = (z + 1) * num_x + x
                i3 = i2 + 1
                indices.extend([i0, i2, i1, i1, i2, i3])

        return np.array(vertices, dtype="f4"), np.array(indices, dtype="i4")

    def get_height_at(self, world_x, world_z):
        x = (world_x / self.size + 0.5) * (self.height_data.shape[1] - 1)
        z = (world_z / self.size + 0.5) * (self.height_data.shape[0] - 1)
        return self.sample_height(x, z)

    def render(self, program, proj, view):
        mvp = proj * view * self.model_matrix

        if "mvp" in program:
            program["mvp"].write(mvp.astype("f4").tobytes())
        if "model" in program:
            program["model"].write(self.model_matrix.astype("f4").tobytes())

        # -------------------------
        # BIND AAA TEXTURE SYSTEM
        # -------------------------
        # This asserts Slot 5=Grass, 6=Rock, etc. every frame.
        self.texturer.bind()
        self.texturer.set_params()

        # Flags
        if "u_use_color" in program:
            program["u_use_color"].value = False

        self.vao.render()

    def update_lod(self, camera_pos):
        dist = np.linalg.norm(np.array(camera_pos))
        prev_res = self.resolution

        if dist > 200: self.resolution = 4.0
        elif dist > 100: self.resolution = 2.0
        else: self.resolution = 1.0

        if abs(prev_res - self.resolution) > 0.1:
            self.vertices, self.indices = self.generate_mesh()
            
            # Re-upload to existing buffers
            self.vbo.release() # Release old to prevent memory leak
            self.ibo.release()
            self.vbo = self.ctx.buffer(self.vertices.tobytes())
            self.ibo = self.ctx.buffer(self.indices.tobytes())
            
            # Re-link VAO with new buffers
            self.vao = self.ctx.vertex_array(
                self.program,
                [(self.vbo, "3f 3f 2f 3f", "in_position", "in_normal", "in_uv", "in_tangent")],
                self.ibo
            )
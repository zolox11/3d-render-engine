import numpy as np
from gltflib import GLTF, ComponentType
from PIL import Image
import io
import base64
import moderngl
from pyrr import Matrix44, Vector3

# -------------------------
# Utility
# -------------------------
def get_numpy_dtype(component_type):
    return {
        ComponentType.BYTE: np.int8,
        ComponentType.UNSIGNED_BYTE: np.uint8,
        ComponentType.SHORT: np.int16,
        ComponentType.UNSIGNED_SHORT: np.uint16,
        ComponentType.UNSIGNED_INT: np.uint32,
        ComponentType.FLOAT: np.float32,
    }[component_type]

def num_components(accessor_type):
    return {
        "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
        "MAT2": 4, "MAT3": 9, "MAT4": 16,
    }[accessor_type]

# -------------------------
# Node Transform
# -------------------------
def get_node_matrix(node):
    if node.matrix:
        return np.array(node.matrix, dtype=np.float32).reshape(4, 4)
    t = np.array(node.translation if node.translation else [0, 0, 0], dtype=np.float32)
    r = np.array(node.rotation if node.rotation else [0, 0, 0, 1], dtype=np.float32)
    s = np.array(node.scale if node.scale else [1, 1, 1], dtype=np.float32)
    x, y, z, w = r
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    rot = np.array([
        [1 - 2*(yy + zz), 2*(xy - wz),     2*(xz + wy),     0],
        [2*(xy + wz),     1 - 2*(xx + zz), 2*(yz - wx),     0],
        [2*(xz - wy),     2*(yz + wx),     1 - 2*(xx + yy), 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)
    scale = np.diag([s[0], s[1], s[2], 1.0])
    trans = np.eye(4, dtype=np.float32)
    trans[:3, 3] = t
    return trans @ rot @ scale

# -------------------------
# Model Container Classes
# -------------------------
class MeshInstance:
    """Stores a specific placement of a mesh in the world."""
    def __init__(self, vao, textures, transform):
        self.vao = vao
        self.textures = textures
        self.transform = transform

class Model:
    """Handles rendering of all mesh instances with uniform safety."""
    def __init__(self, ctx, instances):
        self.instances = instances
        # 1x1 default textures to prevent 'black' parts when maps are missing
        self.white_tex = ctx.texture((1, 1), 4, np.array([255, 255, 255, 255], dtype='u1').tobytes())
        self.black_tex = ctx.texture((1, 1), 4, np.array([0, 0, 0, 255], dtype='u1').tobytes())
        self.flat_normal = ctx.texture((1, 1), 4, np.array([127, 127, 255, 255], dtype='u1').tobytes())

    def render(self, program, proj=None, view=None, root_transform=None):
        if root_transform is None:
            root_matrix = Matrix44.identity()
        else:
            root_matrix = Matrix44(root_transform)

        for inst in self.instances:
            # Apply root object transform (allows GLBObject set_position()/rotation()/scale)
            model_matrix = root_matrix * Matrix44(inst.transform)

            if "mvp" in program and proj is not None and view is not None:
                mvp = proj * view * model_matrix
                program["mvp"].write(mvp.astype("f4").tobytes())

            # Safer uniform writing for simple shaders (like shadow shaders)
            if "model" in program:
                program["model"].write(model_matrix.astype("f4").tobytes())

            # 1. Albedo / Base Color
            if "texture0" in program:
                t_base = inst.textures.get("base") or self.white_tex
                t_base.use(location=0)

            # 2. Metallic / Roughness
            if "texture1" in program:
                t_mr = inst.textures.get("mr")
                if t_mr:
                    t_mr.use(location=1)
                    if "u_use_mr_map" in program: program["u_use_mr_map"].value = True
                else:
                    self.black_tex.use(location=1)
                    if "u_use_mr_map" in program: program["u_use_mr_map"].value = False

            # 3. Normal Map
            if "texture2" in program:
                t_norm = inst.textures.get("normal")
                if t_norm:
                    t_norm.use(location=2)
                    if "u_use_normal_map" in program: program["u_use_normal_map"].value = True
                else:
                    self.flat_normal.use(location=2)
                    if "u_use_normal_map" in program: program["u_use_normal_map"].value = False

            # 4. Emissive Map
            if "texture3" in program:
                t_em = inst.textures.get("emissive") or self.black_tex
                t_em.use(location=3)

            inst.vao.render()

# -------------------------
# Helper Functions
# -------------------------
def read_accessor(gltf, accessor_index):
    accessor = gltf.model.accessors[accessor_index]
    if accessor.bufferView is None:
        return np.zeros((accessor.count, num_components(accessor.type)), dtype=np.float32)
    
    buffer_view = gltf.model.bufferViews[accessor.bufferView]
    buffer = gltf.resources[buffer_view.buffer]
    if hasattr(buffer, "load"): buffer.load()

    dtype = get_numpy_dtype(accessor.componentType)
    comp_count = num_components(accessor.type)
    start = (buffer_view.byteOffset or 0) + (accessor.byteOffset or 0)
    count = accessor.count * comp_count

    arr = np.frombuffer(buffer.data, dtype=dtype, count=count, offset=start)
    arr = arr.reshape((accessor.count, comp_count))

    if accessor.normalized:
        arr = arr.astype(np.float32)
        if dtype == np.uint8: arr /= 255.0
        elif dtype == np.uint16: arr /= 65535.0
    return arr.astype("f4")

def get_image_bytes(gltf, image):
    if image.bufferView is not None:
        bv = gltf.model.bufferViews[image.bufferView]
        res = gltf.resources[bv.buffer]
        if hasattr(res, "load"): res.load()
        return res.data[bv.byteOffset : bv.byteOffset + bv.byteLength]
    if image.uri:
        if image.uri.startswith("data:"):
            return base64.b64decode(image.uri.split(",")[1])
        res = gltf.get_resource(image.uri)
        if hasattr(res, "load"): res.load()
        return res.data
    return None

def load_texture(ctx, image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
    tex = ctx.texture(img.size, 4, img.tobytes())
    tex.build_mipmaps()
    tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
    return tex

# -------------------------
# Main Loader Logic
# -------------------------
def load_model(ctx, program, path):
    gltf = GLTF.load(path, load_file_resources=True)
    mesh_cache = {}

    for mesh_index, mesh in enumerate(gltf.model.meshes):
        mesh_cache[mesh_index] = []
        for primitive in mesh.primitives:
            if primitive.attributes.POSITION is None: continue

            pos = read_accessor(gltf, primitive.attributes.POSITION)
            norm = read_accessor(gltf, primitive.attributes.NORMAL) if primitive.attributes.NORMAL is not None else np.zeros_like(pos)
            uv = read_accessor(gltf, primitive.attributes.TEXCOORD_0) if primitive.attributes.TEXCOORD_0 is not None else np.zeros((pos.shape[0], 2), dtype="f4")
            
            if primitive.attributes.TANGENT is not None:
                tan_raw = read_accessor(gltf, primitive.attributes.TANGENT)
                tan = tan_raw[:, :3]
            else:
                tan = np.zeros_like(pos)

            if primitive.indices is not None:
                indices = read_accessor(gltf, primitive.indices).flatten().astype("i4")
            else:
                indices = np.arange(len(pos), dtype="i4")

            v_data = np.hstack([pos, norm, uv, tan]).astype("f4")
            vbo = ctx.buffer(v_data.tobytes())
            ibo = ctx.buffer(indices.tobytes())
            
            vao = ctx.vertex_array(
                program, 
                [(vbo, "3f 3f 2f 3f", "in_position", "in_normal", "in_uv", "in_tangent")], 
                ibo,
                skip_errors=True
            )

            texs = {"base": None, "mr": None, "normal": None, "emissive": None}
            if primitive.material is not None:
                mat = gltf.model.materials[primitive.material]
                if mat.pbrMetallicRoughness:
                    pbr = mat.pbrMetallicRoughness
                    if pbr.baseColorTexture:
                        data = get_image_bytes(gltf, gltf.model.images[pbr.baseColorTexture.index])
                        if data: texs["base"] = load_texture(ctx, data)
                    if pbr.metallicRoughnessTexture:
                        data = get_image_bytes(gltf, gltf.model.images[pbr.metallicRoughnessTexture.index])
                        if data: texs["mr"] = load_texture(ctx, data)
                if mat.normalTexture:
                    data = get_image_bytes(gltf, gltf.model.images[mat.normalTexture.index])
                    if data: texs["normal"] = load_texture(ctx, data)
                if mat.emissiveTexture:
                    data = get_image_bytes(gltf, gltf.model.images[mat.emissiveTexture.index])
                    if data: texs["emissive"] = load_texture(ctx, data)

            mesh_cache[mesh_index].append({"vao": vao, "textures": texs})

    instances = []
    scene_idx = gltf.model.scene if gltf.model.scene is not None else 0
    scene = gltf.model.scenes[scene_idx]

    def walk(node_idx, p_mat):
        node = gltf.model.nodes[node_idx]
        w_mat = p_mat @ get_node_matrix(node)
        if node.mesh is not None:
            for m in mesh_cache[node.mesh]:
                # MeshInstance is now defined above this function
                instances.append(MeshInstance(m["vao"], m["textures"], w_mat))
        if node.children:
            for c in node.children: 
                walk(c, w_mat)

    for root in scene.nodes: 
        walk(root, np.eye(4, dtype="f4"))

    return Model(ctx, instances)
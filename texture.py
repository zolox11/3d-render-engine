import numpy as np
from PIL import Image
import moderngl

class TerrainTexturer:
    def __init__(self, ctx, program):
        self.ctx = ctx
        self.program = program

        self.textures = {}
        self.defaults = {}

        # Terrain tuning parameters (AAA style controls)
        self.params = {
            "grass_min_height": 0.2,
            "grass_max_height": 15.0,
            "rock_slope_start": 0.35,
            "rock_slope_end": 0.9,
            "snow_height": 25.0,

            "noise_strength": 0.25,
            "detail_strength": 0.35,

            "macro_tiling": 0.05,
            "detail_tiling": 10.0,

            "blend_sharpness": 5.0,

            # NEW FIX PARAMETERS
            "height_falloff": 2.0,
            "slope_power": 2.0,
            "texture_gamma": 2.2,
            
            # THE NEON FIX: Multiplier for Grass (R, G, B)
            # Lowering Green (0.7) and Blue (0.5) makes it look like real grass
            "grass_adj": (0.8, 0.7, 0.5) 
        }

        self._create_defaults()

    # -------------------------
    # DEFAULT SAFE TEXTURES
    # -------------------------
    def _create_defaults(self):
        def make(color):
            data = np.array(color, dtype=np.uint8)
            tex = self.ctx.texture((1, 1), 4, data.tobytes())
            tex.build_mipmaps()
            tex.repeat_x = True
            tex.repeat_y = True
            return tex

        self.defaults["white"] = make([255, 255, 255, 255])
        self.defaults["gray"] = make([128, 128, 128, 255])
        self.defaults["black"] = make([0, 0, 0, 255])

    # -------------------------
    # LOAD TEXTURE
    # -------------------------
    def load(self, name, path, fallback="gray"):
        try:
            img = Image.open(path).convert("RGBA")
            tex = self.ctx.texture(img.size, 4, img.tobytes())
            # LINEAR_MIPMAP_LINEAR prevents the neon flickering at distances
            tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
            tex.repeat_x = True
            tex.repeat_y = True
            tex.build_mipmaps()

            self.textures[name] = tex
            print(f"✅ Loaded {name}: {path}")

        except Exception as e:
            print(f"⚠️ Failed {name}, using fallback: {e}")
            self.textures[name] = self.defaults[fallback]

    # -------------------------
    # SETUP TERRAIN LAYERS
    # -------------------------
    def setup_terrain(self):
        # Ensure these paths match your folder structure exactly
        self.load("grass", "textures/grass.png")
        self.load("rock", "textures/rock.png")
        self.load("sand", "textures/sand.png")
        self.load("snow", "textures/snow.png")
        self.load("detail", "textures/detail.png")
        self.load("noise", "textures/noise.png")

    # -------------------------
    # BIND TEXTURES
    # -------------------------
    def bind(self):
        # Texture Units 5-10 reserved for terrain
        mapping = {
            "grass": 5,
            "rock": 6,
            "sand": 7,
            "snow": 8,
            "detail": 9,
            "noise": 10,
        }

        for name, slot in mapping.items():
            tex = self.textures.get(name, self.defaults["gray"])
            tex.use(location=slot)

        # Tell the shader which unit corresponds to which sampler
        if "tex_grass" in self.program:
            self.program["tex_grass"].value = 5
        if "tex_rock" in self.program:
            self.program["tex_rock"].value = 6
        if "tex_sand" in self.program:
            self.program["tex_sand"].value = 7
        if "tex_snow" in self.program:
            self.program["tex_snow"].value = 8
        if "detailTex" in self.program:
            self.program["detailTex"].value = 9
        if "noiseTex" in self.program:
            self.program["noiseTex"].value = 10

    # -------------------------
    # PARAM CONTROL (AAA TUNING)
    # -------------------------
    def set_params(self, **kwargs):
        """Update any parameter in self.params and re-upload."""
        for key, value in kwargs.items():
            if key in self.params:
                self.params[key] = value
        
        self._upload_uniforms()

    # -------------------------
    # UPLOAD TO SHADER
    # -------------------------
    def _upload_uniforms(self):
        p = self.params

        # Color Adjustment for Grass (The "Neon Fix")
        if "u_grass_adj" in self.program:
            self.program["u_grass_adj"].value = p["grass_adj"]

        # Heights & Slopes
        if "u_grass_min_height" in self.program:
            self.program["u_grass_min_height"].value = p["grass_min_height"]
        if "u_grass_max_height" in self.program:
            self.program["u_grass_max_height"].value = p["grass_max_height"]
        if "u_rock_slope_start" in self.program:
            self.program["u_rock_slope_start"].value = p["rock_slope_start"]
        if "u_rock_slope_end" in self.program:
            self.program["u_rock_slope_end"].value = p["rock_slope_end"]
        if "u_snow_height" in self.program:
            self.program["u_snow_height"].value = p["snow_height"]

        # Effect Strengths
        if "u_noise_strength" in self.program:
            self.program["u_noise_strength"].value = p["noise_strength"]
        if "u_detail_strength" in self.program:
            self.program["u_detail_strength"].value = p["detail_strength"]
        if "u_macro_tiling" in self.program:
            self.program["u_macro_tiling"].value = p["macro_tiling"]
        if "u_detail_tiling" in self.program:
            self.program["u_detail_tiling"].value = p["detail_tiling"]
        if "u_blend_sharpness" in self.program:
            self.program["u_blend_sharpness"].value = p["blend_sharpness"]

        # Math Fixes
        if "u_height_falloff" in self.program:
            self.program["u_height_falloff"].value = p["height_falloff"]
        if "u_slope_power" in self.program:
            self.program["u_slope_power"].value = p["slope_power"]
        if "u_texture_gamma" in self.program:
            self.program["u_texture_gamma"].value = p["texture_gamma"]

    # -------------------------
    # TRIPLANAR SUPPORT FLAG
    # -------------------------
    def enable_triplanar(self, enabled=True):
        if "u_use_triplanar" in self.program:
            self.program["u_use_triplanar"].value = 1 if enabled else 0

    def get_debug_params(self):
        return self.params.copy()
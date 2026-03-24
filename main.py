import pygame
from pygame.locals import DOUBLEBUF, OPENGL
import moderngl
import numpy as np
from pyrr import Matrix44, Vector3
import sys

from camera import Camera
from loader import load_model
from objects import Scene, GLBObject, DirectionalLight, PointLight, Cube, Plane
from scene import init_scene

# =====================================================
# CONFIGURATION
# =====================================================
class Config:
    """Centralized configuration management"""
    WINDOW_WIDTH = 1280
    WINDOW_HEIGHT = 720
    WINDOW_TITLE = "AAA Renderer"
    FPS_TARGET = 360
    
    # OpenGL Settings
    GL_MAJOR_VERSION = 3
    GL_MINOR_VERSION = 3
    
    # Rendering
    SHADOW_MAP_SIZE = 2048
    USE_VSYNC = True
    ENABLE_GRID = True
    
    # Camera
    CAMERA_FOV = 45.0
    CAMERA_NEAR = 0.1
    CAMERA_FAR = 1000.0
    CAMERA_START_POS = (0.0, 1.5, 5.0)
    CAMERA_SPEED = 10.0
    CAMERA_SENSITIVITY = 0.003
    
    # Scene
    SCENE_AMBIENT = (0.03, 0.03, 0.03)
    FLOOR_SIZE = 400
    
    # Debug
    DEBUG_MODE = False
    SHOW_PERFORMANCE = True


# =====================================================
# UTILITY CLASSES
# =====================================================
class PerformanceMonitor:
    """Simple FPS and timing monitor"""
    def __init__(self):
        self.frame_times = []
        self.frame_count = 0
        self.start_time = pygame.time.get_ticks()

    def update(self, dt):
        self.frame_times.append(dt)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        self.frame_count += 1

    def get_fps(self):
        if self.frame_times:
            avg_time = sum(self.frame_times) / len(self.frame_times)
            return 1.0 / avg_time if avg_time > 0 else 0
        return 0

    def get_frame_time(self):
        if self.frame_times:
            return self.frame_times[-1] * 1000  # ms
        return 0


class Grid:
    """Optimized grid renderer"""
    def __init__(self, ctx, program, size=20, divisions=20):
        self.program = program
        self.size = size
        self.divisions = divisions
        
        # Generate grid vertices
        vertices = []
        half = size / 2.0
        step = size / divisions
        
        # Vertical lines
        for i in range(divisions + 1):
            z = -half + i * step
            vertices += [-half, 0, z, half, 0, z]
        
        # Horizontal lines
        for i in range(divisions + 1):
            x = -half + i * step
            vertices += [x, 0, -half, x, 0, half]
        
        self.vertex_count = len(vertices) // 3
        self.vbo = ctx.buffer(np.array(vertices, dtype="f4").tobytes())
        self.vao = ctx.simple_vertex_array(program, self.vbo, "in_position")

    def render(self, mvp):
        """Render the grid"""
        self.program["mvp"].write(mvp.astype("f4").tobytes())
        self.vao.render(mode=moderngl.LINES)


# =====================================================
# RENDER SYSTEM (MODULAR ARCHITECTURE)
# =====================================================
class Renderer:
    """Main rendering orchestrator"""
    def __init__(self, ctx, width, height, config):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.config = config
        
        # Render passes (modular pipeline)
        self.shadow_pass = None
        self.forward_pass = None
        self.grid_pass = None
        
        self.setup_context()

    def setup_context(self):
        """Configure OpenGL context"""
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.CULL_FACE)

    def setup_passes(self, program, grid_program, shadow_program):
        """Initialize all rendering passes"""
        self.shadow_pass = ShadowPass(
            self.ctx,
            shadow_program,
            self.config.SHADOW_MAP_SIZE
        )
        
        self.forward_pass = ForwardPass(
            self.ctx,
            program,
            self.shadow_pass
        )
        
        self.grid_pass = GridPass(
            self.ctx,
            grid_program,
            self.config.FLOOR_SIZE,
            200
        )

    def begin_frame(self):
        """Prepare for rendering"""
        self.ctx.clear(0.15, 0.15, 0.15, 1.0)

    def render_frame(self, scene, camera, proj):
        """Execute complete render pipeline"""
        self.begin_frame()
        
        # Stage 1: Shadow mapping
        if scene.dir_lights:
            self.shadow_pass.render(scene)
        
        # Stage 2: Forward rendering to screen
        self.ctx.screen.use()
        self.forward_pass.render(scene, camera, proj)
        
        # Stage 3: Grid overlay
        if self.config.ENABLE_GRID:
            mvp = proj * camera.get_view_matrix()
            self.grid_pass.render(mvp)

    def cleanup(self):
        """Release GPU resources"""
        if self.shadow_pass:
            self.shadow_pass.cleanup()
        if self.forward_pass:
            self.forward_pass.cleanup()

class ShadowPass:
    """Dedicated shadow map rendering pass"""
    def __init__(self, ctx, shadow_program, shadow_size):
        self.ctx = ctx
        self.shadow_program = shadow_program
        self.shadow_size = shadow_size
        
        # Create shadow depth texture
        self.depth_texture = ctx.depth_texture((shadow_size, shadow_size))
        self.depth_texture.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.depth_texture.compare_func = '<'
        self.depth_texture.repeat_x = False
        self.depth_texture.repeat_y = False
        
        # Create framebuffer
        self.fbo = ctx.framebuffer(depth_attachment=self.depth_texture)

    def render(self, scene):
        """Render scene to shadow map"""
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.shadow_size, self.shadow_size)
        self.fbo.clear(depth=1.0)

        # Disable culling to prevent artifacts in shadow pass
        self.ctx.disable(moderngl.CULL_FACE)

        if scene.dir_lights:
            light_space_matrix = scene.dir_lights[0].get_light_space_matrix((0, 0, 0))

            for obj in scene.objects:
                if getattr(obj, 'cast_shadow', False):
                    try:
                        obj.render(self.shadow_program, light_space_matrix, Matrix44.identity())
                    except Exception as e:
                        print(f"⚠️  Shadow render failed for {obj.__class__.__name__}: {e}")

        self.ctx.enable(moderngl.CULL_FACE)
        # Restore viewport to screen dimensions (main render will rebind screen anyway)
        self.ctx.viewport = (0, 0, self.ctx.screen.width, self.ctx.screen.height)

    def bind(self, location=4):
        """Bind shadow map texture to shader"""
        self.depth_texture.use(location=location)

    def cleanup(self):
        """Release resources"""
        self.depth_texture.release()
        self.fbo.release()

class ForwardPass:
    """Forward rendering pass with lighting"""
    def __init__(self, ctx, program, shadow_pass):
        self.ctx = ctx
        self.program = program
        self.shadow_pass = shadow_pass
        
        self.setup_uniforms()

    def setup_uniforms(self):
        """Configure texture unit bindings"""
        if 'texture0' in self.program:
            self.program['texture0'].value = 0
        if 'texture1' in self.program:
            self.program['texture1'].value = 1
        if 'texture2' in self.program:
            self.program['texture2'].value = 2
        if 'texture3' in self.program:
            self.program['texture3'].value = 3
        if 'shadowMap' in self.program:
            self.program['shadowMap'].value = 4

        # Set default values for material override uniforms (-1 means use texture/default)
        if 'u_roughness_override' in self.program:
            self.program['u_roughness_override'].value = -1.0
        if 'u_metallic_override' in self.program:
            self.program['u_metallic_override'].value = -1.0
        if 'u_emissive_override' in self.program:
            self.program['u_emissive_override'].value = (0.0, 0.0, 0.0)
        if 'u_ao_override' in self.program:
            self.program['u_ao_override'].value = -1.0

    def render(self, scene, camera, proj):
        """Render scene with lighting"""
        view = camera.get_view_matrix()
        
        # Set camera position
        if "cam_pos" in self.program:
            self.program["cam_pos"].value = tuple(camera.position)
        
        # Setup lights
        all_lights = scene.dir_lights + scene.point_lights
        if "light_count" in self.program:
            self.program["light_count"].value = min(len(all_lights), 8)
        
        for i, light in enumerate(all_lights):
            if i < 8:
                light.apply(self.program, i)
        
        # Set light space matrix
        if scene.dir_lights:
            light_space_matrix = scene.dir_lights[0].get_light_space_matrix((0, 0, 0))
            if "lightSpaceMatrix" in self.program:
                self.program["lightSpaceMatrix"].write(light_space_matrix.astype("f4").tobytes())
        
        # Bind shadow map
        self.shadow_pass.bind(location=4)
        
        # Render all objects
        for obj in scene.objects:
            obj.render(self.program, proj, view)

    def cleanup(self):
        """Release resources"""
        pass


class GridPass:
    """Grid overlay rendering"""
    def __init__(self, ctx, program, size, divisions):
        self.grid = Grid(ctx, program, size, divisions)

    def render(self, mvp):
        """Render grid"""
        self.grid.render(mvp)


# =====================================================
# APPLICATION
# =====================================================
class Application:
    """Main application class"""
    def __init__(self, config):
        self.config = config
        self.running = True
        self.clock = pygame.time.Clock()
        self.perf_monitor = PerformanceMonitor()
        
        self.init_pygame()
        self.init_opengl()
        self.init_scene()

    def init_pygame(self):
        """Initialize pygame window"""
        pygame.init()
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, self.config.GL_MAJOR_VERSION)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, self.config.GL_MINOR_VERSION)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)
        
        self.display = pygame.display.set_mode(
            (self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT),
            DOUBLEBUF | OPENGL
        )
        pygame.display.set_caption(self.config.WINDOW_TITLE)
        
        # Input handling
        pygame.event.set_grab(True)
        pygame.mouse.set_visible(False)

    def init_opengl(self):
        """Initialize ModernGL context and shaders"""
        self.ctx = moderngl.create_context()
        
        # Load shaders
        with open('shaders/vertex.txt', 'r') as f:
            vertex_shader = f.read()
        with open('shaders/fragment.txt', 'r') as f:
            fragment_shader = f.read()
        
        with open('shaders/grid_vertex.glsl', 'r') as f:
            grid_vs = f.read()
        with open('shaders/grid_fragment.glsl', 'r') as f:
            grid_fs = f.read()
        
        with open('shaders/shadow_vertex.glsl', 'r') as f:
            shadow_vs = f.read()
        with open('shaders/shadow_fragment.glsl', 'r') as f:
            shadow_fs = f.read()
        
        # Compile programs
        self.main_program = self.ctx.program(vertex_shader=vertex_shader, fragment_shader=fragment_shader)
        self.grid_program = self.ctx.program(vertex_shader=grid_vs, fragment_shader=grid_fs)
        self.shadow_program = self.ctx.program(vertex_shader=shadow_vs, fragment_shader=shadow_fs)
        
        # Initialize renderer
        self.renderer = Renderer(self.ctx, self.config.WINDOW_WIDTH, self.config.WINDOW_HEIGHT, self.config)
        self.renderer.setup_passes(self.main_program, self.grid_program, self.shadow_program)
    
    def init_scene(self):
        self.scene, self.camera, self.proj = init_scene(
            self.ctx,
            self.main_program,
            self.config
        )

    def handle_events(self):
        """Process input events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.running = False
                elif event.key == pygame.K_F1 and self.config.SHOW_PERFORMANCE:
                    pass  # Could toggle perf display here

    def update(self, dt):
        """Update game logic"""
        # Handle input
        keys = pygame.key.get_pressed()
        self.camera.process_keyboard(keys, dt)
        self.camera.process_mouse(*pygame.mouse.get_rel())

    def render(self):
        """Render frame"""
        self.renderer.render_frame(self.scene, self.camera, self.proj)

    def display_performance(self):
        """Optional: Display performance metrics"""
        if self.config.SHOW_PERFORMANCE:
            fps = self.perf_monitor.get_fps()
            frame_time = self.perf_monitor.get_frame_time()
            # Could render text overlay here
            # print(f"FPS: {fps:.1f} | Frame: {frame_time:.2f}ms")

    def run(self):
        """Main application loop"""
        print("🎮 Application Started")
        print(f"Resolution: {self.config.WINDOW_WIDTH}x{self.config.WINDOW_HEIGHT}")
        print(f"Shadow Map Size: {self.config.SHADOW_MAP_SIZE}x{self.config.SHADOW_MAP_SIZE}")
        print(f"Target FPS: {self.config.FPS_TARGET}")
        
        while self.running:
            dt = self.clock.tick(self.config.FPS_TARGET) / 1000.0
            self.perf_monitor.update(dt)
            
            self.handle_events()
            self.update(dt)
            self.render()
            self.display_performance()
            
            pygame.display.flip()
        
        self.cleanup()

    def cleanup(self):
        """Clean up resources"""
        print("🛑 Shutting down...")
        self.renderer.cleanup()
        pygame.quit()
        sys.exit()


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    try:
        app = Application(Config)
        app.run()
    except Exception as e:
        print(f"❌ Fatal Error: {e}")
        import traceback
        traceback.print_exc()
        pygame.quit()
        sys.exit(1)
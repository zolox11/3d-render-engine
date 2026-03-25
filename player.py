import pygame
from pyrr import Vector3
from objects import Object3D


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def _lerp(a, b, t):
    return a + (b - a) * t


class PlayerController:
    def __init__(self, camera, physics_engine, scene, start_position=(0.0, 20.0, 5.0)):
        self.camera = camera
        self.physics_engine = physics_engine

        # -------------------------
        # Player settings
        # -------------------------
        self.height = 2
        self.radius = 0.35

        self.walk_speed = 5.5
        self.run_speed = 9.0
        self.jump_speed = 5
        self.air_control = 0.3
        self.smooth_accel = 12.0
        self.mouse_sensitivity = 0.16

        # -------------------------
        # Physics body
        # -------------------------
        self.body = Object3D()
        self.body.transform.position = Vector3(start_position)
        self.body.collider_half_size = Vector3((self.radius, self.height * 0.5, self.radius))
        self.body.physics_enabled = True
        self.body.gravity_enabled = True
        self.body.gravity_strength = 1.0
        self.body.collidable = True
        self.body.mass = 3
        self.body.linear_damping = 6

        # 🔥 FIX: grounding buffer
        

        # Camera height
        self.eye_height = 1.6

        # Smoothed movement
        self.smoothed_input = Vector3((0.0, 0.0, 0.0))

        # Prevent initial mouse jump
        pygame.mouse.get_rel()

        # -------------------------
        # 🔥 FIX: Register scene ONCE
        # -------------------------
        self.physics_engine.register_body(self.body)
        self.scene = scene
        self.body.velocity.y = self.jump_speed

    # -------------------------
    # INPUT
    # -------------------------
    def process_input(self, dt):
        keys = pygame.key.get_pressed()

        # Joystick (DS4Windows / PS4 controller)
        joystick = None
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()

        # Mouse look (keyboard/mouse fallback)
        dx, dy = pygame.mouse.get_rel()

        # Right stick look (override/add if controller present)
        if joystick:
            # Right stick axes (commonly 2 = right x, 3 = right y)
            rx = joystick.get_axis(2)
            ry = joystick.get_axis(3)

            # Deadzone
            deadzone = 0.1
            if abs(rx) < deadzone:
                rx = 0.0
            if abs(ry) < deadzone:
                ry = 0.0

            dx += rx * 10.0  # scale sensitivity
            dy += ry * 10.0

        # Apply mouse + right stick look
        self.camera.yaw += dx * self.mouse_sensitivity
        self.camera.pitch -= dy * self.mouse_sensitivity
        self.camera.pitch = _clamp(self.camera.pitch, -89.0, 89.0)
        self.camera._update_vectors()

        # Movement input (keyboard + left stick)
        move = Vector3((0.0, 0.0, 0.0))

        # Keyboard movement
        if keys[pygame.K_w]:
            move += self.camera.front
        if keys[pygame.K_s]:
            move -= self.camera.front
        if keys[pygame.K_a]:
            move -= self.camera.right
        if keys[pygame.K_d]:
            move += self.camera.right

        # Controller left stick
        if joystick:
            lx = joystick.get_axis(0)
            ly = joystick.get_axis(1)

            deadzone = 0.1
            if abs(lx) < deadzone:
                lx = 0.0
            if abs(ly) < deadzone:
                ly = 0.0

            move += self.camera.right * lx
            move -= self.camera.front * ly

        move.y = 0.0

        if move.length > 0.0:
            move = move.normalized

        # Sprint: keyboard (LShift) OR R2 trigger
        sprint = keys[pygame.K_LSHIFT]

        if joystick:
            # R2 is often axis 5 (ranges from -1 to 1 depending on driver)
            r2 = joystick.get_axis(5)
            if r2 > 0.5:
                sprint = True

        target_speed = self.run_speed if sprint else self.walk_speed
        target_velocity = move * target_speed

        lerp_t = _clamp(dt * self.smooth_accel, 0.0, 1.0)

        self.smoothed_input = Vector3((
            _lerp(self.smoothed_input.x, target_velocity.x, lerp_t),
            0.0,
            _lerp(self.smoothed_input.z, target_velocity.z, lerp_t)
        ))

        on_ground = self.body.on_ground

        # Apply horizontal movement
        air_factor = self.air_control if not on_ground else 1.0

        self.body.velocity.x = _lerp(
            self.body.velocity.x,
            self.smoothed_input.x * air_factor,
            lerp_t
        )

        self.body.velocity.z = _lerp(
            self.body.velocity.z,
            self.smoothed_input.z * air_factor,
            lerp_t
        )

        # Jump: Space OR X button (button 0 on DS4 via DS4Windows)
        jump_pressed = keys[pygame.K_SPACE]

        if joystick:
            if joystick.get_numbuttons() > 0:
                if joystick.get_button(0):  # X button
                    jump_pressed = True

        if jump_pressed and self.body.on_ground:
            self.body.velocity.y = self.jump_speed
    
    def update_camera(self):
        self.camera.position = Vector3((
            self.body.transform.position.x,
            self.body.transform.position.y + self.eye_height,
            self.body.transform.position.z
        ))

    # -------------------------
    # TELEPORT
    # -------------------------
    def teleport(self, position):
        self.body.transform.position = Vector3(position)
        self.body.velocity = Vector3((0.0, 0.0, 0.0))
        self.body.ground_timer = 0.1
        self.body.on_ground = True

        self.camera.position = Vector3((
            position[0],
            position[1] + self.eye_height,
            position[2]
        ))
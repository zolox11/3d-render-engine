import pygame
from pyrr import Vector3
from objects import Object3D
import numpy as np


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
        self.height = 1.5
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

        # -------------------------
        # Orbit camera settings
        # -------------------------
        self.orbit_mode = False
        self.orbit_distance = 20.0
        self.orbit_theta = 0.0  # horizontal rotation
        self.orbit_phi = 30.0   # vertical rotation
        self.orbit_center = Vector3((0.0, 0.0, 0.0))
        self.orbit_sensitivity = 0.3
        self.orbit_pan_speed = 10.0  # speed to move orbit center

        # Camera height
        self.eye_height = 1.6

        # Smoothed movement
        self.smoothed_input = Vector3((0.0, 0.0, 0.0))

        # Prevent initial mouse jump
        pygame.mouse.get_rel()

        # Register scene once
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

        # Mouse look
        dx, dy = pygame.mouse.get_rel()

        # Right stick look (controller)
        if joystick:
            rx = joystick.get_axis(2)
            ry = joystick.get_axis(3)
            deadzone = 0.1
            if abs(rx) < deadzone:
                rx = 0.0
            if abs(ry) < deadzone:
                ry = 0.0
            dx += rx * 10.0
            dy += ry * 10.0

        # -------------------------
        # ORBIT CAMERA INPUT
        # -------------------------
        if self.orbit_mode == True:
            # Mouse rotates orbit camera
            self.orbit_theta += dx * self.orbit_sensitivity
            self.orbit_phi -= dy * self.orbit_sensitivity
            self.orbit_phi = _clamp(self.orbit_phi, 5.0, 85.0)

            # WASD / left stick moves orbit center
            move = Vector3((0.0, 0.0, 0.0))
            # Horizontal movement relative to camera
            if keys[pygame.K_w]:
                move += self.camera.front
            if keys[pygame.K_s]:
                move -= self.camera.front
            if keys[pygame.K_a]:
                move -= self.camera.right
            if keys[pygame.K_d]:
                move += self.camera.right
            # Vertical movement
            if keys[pygame.K_q]:  # Q = down
                move -= Vector3((0.0, 1.0, 0.0))
            if keys[pygame.K_e]:  # E = up
                move += Vector3((0.0, 1.0, 0.0))

            # Controller left stick
            if joystick:
                lx = joystick.get_axis(0)
                ly = joystick.get_axis(1)
                deadzone = 0.1
                if abs(lx) >= deadzone:
                    move += self.camera.right * lx
                if abs(ly) >= deadzone:
                    move -= self.camera.front * ly

            if move.length > 0:
                move = move.normalized
                self.orbit_center += move * self.orbit_pan_speed * dt

            # Update camera position
            self.update_orbit_camera(dt)
            return  # skip normal player movement

        # -------------------------
        # NORMAL FIRST-PERSON INPUT
        # -------------------------
        self.camera.yaw += dx * self.mouse_sensitivity
        self.camera.pitch -= dy * self.mouse_sensitivity
        self.camera.pitch = _clamp(self.camera.pitch, -89.0, 89.0)
        self.camera._update_vectors()

        # Movement input
        move = Vector3((0.0, 0.0, 0.0))
        if keys[pygame.K_w]:
            move += self.camera.front
        if keys[pygame.K_s]:
            move -= self.camera.front
        if keys[pygame.K_a]:
            move -= self.camera.right
        if keys[pygame.K_d]:
            move += self.camera.right

        if joystick:
            lx = joystick.get_axis(0)
            ly = joystick.get_axis(1)
            deadzone = 0.1
            if abs(lx) >= deadzone:
                move += self.camera.right * lx
            if abs(ly) >= deadzone:
                move -= self.camera.front * ly

        move.y = 0.0
        if move.length > 0:
            move = move.normalized

        # Sprint
        sprint = keys[pygame.K_LSHIFT]
        if joystick:
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

        # Jump
        jump_pressed = keys[pygame.K_SPACE]
        if joystick:
            if joystick.get_numbuttons() > 0:
                if joystick.get_button(0):
                    jump_pressed = True
        if jump_pressed and self.body.on_ground:
            self.body.velocity.y = self.jump_speed

    # -------------------------
    # ORBIT CAMERA FUNCTIONS
    # -------------------------
    def toggle_orbit(self, enable=None):
        if enable is None:
            self.orbit_mode = not self.orbit_mode
        else:
            self.orbit_mode = enable

    def update_orbit_camera(self, dt=None):
        phi_rad = np.radians(self.orbit_phi)
        theta_rad = np.radians(self.orbit_theta)

        x = self.orbit_center.x + self.orbit_distance * np.sin(phi_rad) * np.cos(theta_rad)
        y = self.orbit_center.y + self.orbit_distance * np.cos(phi_rad)
        z = self.orbit_center.z + self.orbit_distance * np.sin(phi_rad) * np.sin(theta_rad)

        self.camera.position = Vector3((x, y, z))

        front = self.orbit_center - self.camera.position
        if front.length > 0:
            self.camera.front = front.normalized

        self.camera.right = self.camera.front.cross(Vector3((0.0, 1.0, 0.0))).normalized
        self.camera.up = self.camera.right.cross(self.camera.front).normalized

    # -------------------------
    # UPDATE CAMERA
    # -------------------------
    def update_camera(self, dt=None):
        if self.orbit_mode:
            self.update_orbit_camera(dt)
        else:
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
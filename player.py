import pygame
from pyrr import Vector3
from objects import Object3D
import numpy as np


def _clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def _lerp(a, b, t):
    return a + (b - a) * t


def _smooth_damp(current, target, current_velocity, smooth_time, dt):
    """
    Simple SmoothDamp approximation (Unity-style feel).
    """
    smooth_time = max(0.0001, smooth_time)
    omega = 2.0 / smooth_time

    x = omega * dt
    exp = 1.0 / (1.0 + x + 0.48 * x * x + 0.235 * x * x * x)

    change = current - target
    temp = (current_velocity + omega * change) * dt
    new_velocity = (current_velocity - omega * temp) * exp
    new_value = target + (change + temp) * exp

    return new_value, new_velocity


class PlayerController:
    def __init__(self, camera, physics_engine, scene, start_position=(0.0, 20.0, 5.0)):
        self.camera = camera
        self.physics_engine = physics_engine
        self.scene = scene

        # -------------------------
        # Player settings (AAA-style tuning)
        # -------------------------
        self.height = 1.8
        self.radius = 0.35

        self.walk_speed = 4.5
        self.run_speed = 8.0
        self.acceleration = 18.0
        self.air_control = 0.35
        self.jump_speed = 4.5
        self.gravity_scale = 1.0

        self.mouse_sensitivity = 0.45
        self.look_smooth_time = 0.04
        self.move_smooth_time = 0.08

        self.max_fall_speed = 60.0

        # -------------------------
        # Physics body
        # -------------------------
        self.body = Object3D()
        self.body.transform.position = Vector3(start_position)

        # Capsule approximation
        self.body.collider_half_size = Vector3((self.radius, self.height * 0.5, self.radius))

        self.body.physics_enabled = True
        self.body.gravity_enabled = True
        self.body.gravity_strength = 1.0
        self.body.collidable = True
        self.body.mass = 3
        self.body.linear_damping = 8.0

        # Velocity state
        self.body.velocity = Vector3((0.0, 0.0, 0.0))

        # -------------------------
        # Camera state
        # -------------------------
        self.pitch = 0.0
        self.yaw = 0.0

        self.current_look_velocity = Vector3((0.0, 0.0, 0.0))
        self.current_move_velocity = Vector3((0.0, 0.0, 0.0))

        # -------------------------
        # Grounding
        # -------------------------
        self.coyote_time = 0.1
        self.jump_buffer_time = 0.1
        self.ground_timer = 0.0
        self.jump_buffer = 0.0

        # -------------------------
        # Orbit camera
        # -------------------------
        self.orbit_mode = False
        self.orbit_distance = 20.0
        self.orbit_theta = 0.0
        self.orbit_phi = 30.0
        self.orbit_center = Vector3((0.0, 0.0, 0.0))
        self.orbit_sensitivity = 0.3
        self.orbit_pan_speed = 10.0

        # Eye height
        self.eye_height = 1.6

        # Init mouse
        pygame.mouse.get_rel()

        # Register physics
        self.physics_engine.register_body(self.body)

    # -------------------------
    # INPUT
    # -------------------------
    def process_input(self, dt):
        keys = pygame.key.get_pressed()

        joystick = None
        if pygame.joystick.get_count() > 0:
            joystick = pygame.joystick.Joystick(0)
            joystick.init()

        dx, dy = pygame.mouse.get_rel()

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
        # ORBIT MODE
        # -------------------------
        if self.orbit_mode:
            self.orbit_theta += dx * self.orbit_sensitivity
            self.orbit_phi -= dy * self.orbit_sensitivity
            self.orbit_phi = _clamp(self.orbit_phi, 5.0, 85.0)

            move = Vector3((0.0, 0.0, 0.0))

            if keys[pygame.K_w]:
                move += self.camera.front
            if keys[pygame.K_s]:
                move -= self.camera.front
            if keys[pygame.K_a]:
                move -= self.camera.right
            if keys[pygame.K_d]:
                move += self.camera.right

            if keys[pygame.K_q]:
                move -= Vector3((0.0, 1.0, 0.0))
            if keys[pygame.K_e]:
                move += Vector3((0.0, 1.0, 0.0))

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

            self.update_orbit_camera(dt)
            return

        # -------------------------
        # FIRST PERSON LOOK
        # -------------------------
        target_yaw = self.yaw + dx * self.mouse_sensitivity
        target_pitch = self.pitch - dy * self.mouse_sensitivity
        target_pitch = _clamp(target_pitch, -89.0, 89.0)

        self.yaw, self.current_look_velocity.x = _smooth_damp(self.yaw, target_yaw, self.current_look_velocity.x, self.look_smooth_time, dt)
        self.pitch, self.current_look_velocity.y = _smooth_damp(self.pitch, target_pitch, self.current_look_velocity.y, self.look_smooth_time, dt)

        self.camera.yaw = self.yaw
        self.camera.pitch = self.pitch
        self.camera._update_vectors()

        # -------------------------
        # MOVEMENT INPUT
        # -------------------------
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

        sprint = keys[pygame.K_LSHIFT]
        if joystick:
            r2 = joystick.get_axis(5)
            if r2 > 0.5:
                sprint = True

        target_speed = self.run_speed if sprint else self.walk_speed
        target_velocity = move * target_speed

        # Smooth movement (Unity-like feel)
        self.current_move_velocity.x, _ = _smooth_damp(
            self.current_move_velocity.x, target_velocity.x, 0.0, self.move_smooth_time, dt
        )
        self.current_move_velocity.z, _ = _smooth_damp(
            self.current_move_velocity.z, target_velocity.z, 0.0, self.move_smooth_time, dt
        )

        on_ground = self.body.on_ground
        control_factor = 1.0 if on_ground else self.air_control

        self.body.velocity.x = self.current_move_velocity.x * control_factor
        self.body.velocity.z = self.current_move_velocity.z * control_factor

        # -------------------------
        # JUMP BUFFERING
        # -------------------------
        jump_pressed = keys[pygame.K_SPACE]
        if joystick and joystick.get_numbuttons() > 0:
            if joystick.get_button(0):
                jump_pressed = True

        if jump_pressed:
            self.jump_buffer = self.jump_buffer_time

        self.jump_buffer = max(0.0, self.jump_buffer - dt)

        if (self.body.on_ground or self.ground_timer > 0.0) and self.jump_buffer > 0.0:
            self.body.velocity.y = self.jump_speed
            self.jump_buffer = 0.0
            self.ground_timer = 0.0
            self.body.on_ground = False

    # -------------------------
    # ORBIT CAMERA
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
    # CAMERA UPDATE
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
        self.jump_buffer = 0.0

        self.camera.position = Vector3((
            position[0],
            position[1] + self.eye_height,
            position[2]
        ))
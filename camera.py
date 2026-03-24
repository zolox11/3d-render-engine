import numpy as np
from pyrr import Matrix44, Vector3
import pygame
import math

class Camera:
    def __init__(self, position=(0.0, 0.0, 5.0), up=(0.0, 1.0, 0.0)):
        self.position = Vector3(position)
        self.world_up = Vector3(up)

        self.yaw = -90.0
        self.pitch = 0.0

        # These will be populated by _update_vectors
        self.front = Vector3((0.0, 0.0, -1.0))
        self.right = Vector3()
        self.up = Vector3()

        self.movement_speed = 5.0
        self.rotation_speed = 90.0 
        self.mouse_sensitivity = 0.1

        self._update_vectors()

        # Joystick initialization
        if not pygame.joystick.get_init():
            pygame.joystick.init()
        
        self.joystick = None
        if pygame.joystick.get_count() > 0:
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()

    def _update_vectors(self):
        # Calculate the new Front vector
        yaw_rad = math.radians(self.yaw)
        pitch_rad = math.radians(self.pitch)

        front = Vector3([
            math.cos(yaw_rad) * math.cos(pitch_rad),
            math.sin(pitch_rad),
            math.sin(yaw_rad) * math.cos(pitch_rad)
        ])
        
        self.front = front.normalized
        # Re-calculate Right and Up vectors
        self.right = self.front.cross(self.world_up).normalized
        self.up = self.right.cross(self.front).normalized

    def process_keyboard(self, keys, dt):
        velocity = self.movement_speed * dt
        if keys[pygame.K_w]: self.position += self.front * velocity
        if keys[pygame.K_s]: self.position -= self.front * velocity
        if keys[pygame.K_a]: self.position -= self.right * velocity
        if keys[pygame.K_d]: self.position += self.right * velocity

    def process_mouse(self, dx, dy):
        self.yaw += dx * self.mouse_sensitivity
        self.pitch -= dy * self.mouse_sensitivity # dy is usually inverted in screen space

        # Constrain pitch to avoid screen flip
        self.pitch = max(-89.0, min(89.0, self.pitch))
        self._update_vectors()

    def process_gamepad(self, dt):
        if not self.joystick:
            return

        # Note: pygame.event.pump() should usually be called in your main loop,
        # but having it here ensures the joystick state is fresh.
        pygame.event.pump()
        
        move_vel = self.movement_speed * dt
        rot_vel = self.rotation_speed * dt
        has_changed = False

        # --- MOVEMENT (D-PAD) ---
        if self.joystick.get_numhats() > 0:
            dx, dy = self.joystick.get_hat(0)
            if dx != 0 or dy != 0:
                self.position += self.front * dy * move_vel
                self.position += self.right * dx * move_vel

        # --- VERTICAL (L1 / R1) ---
        # Note: Button indices vary by controller; 4/5 are typical for L1/R1
        if self.joystick.get_button(4): 
            self.position -= self.world_up * move_vel
        if self.joystick.get_button(5): 
            self.position += self.world_up * move_vel

        # --- ROTATION (Face Buttons) ---
        if self.joystick.get_button(3): # Triangle/Y
            self.pitch += rot_vel
            has_changed = True
        if self.joystick.get_button(0): # Cross/A
            self.pitch -= rot_vel
            has_changed = True
        if self.joystick.get_button(2): # Square/X
            self.yaw -= rot_vel
            has_changed = True
        if self.joystick.get_button(1): # Circle/B
            self.yaw += rot_vel
            has_changed = True

        # Only update vectors if rotation actually happened
        if has_changed:
            self.pitch = max(-89.0, min(89.0, self.pitch))
            self._update_vectors()

    def get_view_matrix(self):
        return Matrix44.look_at(self.position, self.position + self.front, self.up)
import math
from pyrr import Vector3


class AABB:
    def __init__(self, min_point: Vector3, max_point: Vector3):
        self.min = Vector3(min_point)
        self.max = Vector3(max_point)

    def intersects(self, other: "AABB") -> bool:
        return (
            self.max.x >= other.min.x and self.min.x <= other.max.x and
            self.max.y >= other.min.y and self.min.y <= other.max.y and
            self.max.z >= other.min.z and self.min.z <= other.max.z
        )

    def penetration_vector(self, other: "AABB"):
        if not self.intersects(other):
            return None

        dx = min(self.max.x, other.max.x) - max(self.min.x, other.min.x)
        dy = min(self.max.y, other.max.y) - max(self.min.y, other.min.y)
        dz = min(self.max.z, other.max.z) - max(self.min.z, other.min.z)

        rel = self.get_center() - other.get_center()

        # Return smallest penetration axis as a vector
        if dy <= dx and dy <= dz:
            sign = 1.0 if rel.y >= 0 else -1.0
            return Vector3((0.0, dy * sign, 0.0))

        if dx <= dz:
            sign = 1.0 if rel.x >= 0 else -1.0
            return Vector3((dx * sign, 0.0, 0.0))

        sign = 1.0 if rel.z >= 0 else -1.0
        return Vector3((0.0, 0.0, dz * sign))

    def get_center(self) -> Vector3:
        return (self.min + self.max) / 2.0


class PhysicsEngine:
    def __init__(self, gravity=Vector3((0.0, -9.81, 0.0))):
        self.gravity = Vector3(gravity)
        self.dynamic_bodies = []
        self.static_bodies = []

    def reset(self):
        self.dynamic_bodies.clear()
        self.static_bodies.clear()

    def register_body(self, obj):
        if obj is None or not hasattr(obj, "collidable") or not obj.collidable:
            return

        if hasattr(obj, "physics_enabled") and obj.physics_enabled and obj.mass > 0.0:
            self.dynamic_bodies.append(obj)
        else:
            self.static_bodies.append(obj)

    def load_scene(self, scene):
        self.reset()
        for obj in getattr(scene, "objects", []):
            self.register_body(obj)

    @staticmethod
    def get_aabb(obj):
        # Use collider half-size and object transform position. Rotation is not used in AABB because it is more expensive.
        center = Vector3(obj.transform.position)
        half = Vector3(getattr(obj, "collider_half_size", Vector3((0.5, 0.5, 0.5))))
        scale = Vector3(obj.transform.scale)
        half = Vector3((abs(half.x * scale.x), abs(half.y * scale.y), abs(half.z * scale.z)))

        # avoid zero-size colliders for very small scales
        min_half = 0.1
        half = Vector3((max(half.x, min_half), max(half.y, min_half), max(half.z, min_half)))

        return AABB(center - half, center + half)

    def resolve_collision(self, dynamic_obj, static_obj):
        if not static_obj.collidable:
            return False

        dyn_aabb = self.get_aabb(dynamic_obj)
        stat_aabb = self.get_aabb(static_obj)

        if not dyn_aabb.intersects(stat_aabb):
            return False

        dx = min(dyn_aabb.max.x, stat_aabb.max.x) - max(dyn_aabb.min.x, stat_aabb.min.x)
        dy = min(dyn_aabb.max.y, stat_aabb.max.y) - max(dyn_aabb.min.y, stat_aabb.min.y)
        dz = min(dyn_aabb.max.z, stat_aabb.max.z) - max(dyn_aabb.min.z, stat_aabb.min.z)

        # Debug: keep logs of interactions
        print(f"[PHYSICS] COLLIDE: dyn={dyn_aabb.min}-{dyn_aabb.max} stat={stat_aabb.min}-{stat_aabb.max} dx={dx:.3f} dy={dy:.3f} dz={dz:.3f}")

        # Prefer Y axis for thin static colliders (plane-like behavior) and ground contacts
        stat_half = Vector3((stat_aabb.max.x - stat_aabb.min.x,
                             stat_aabb.max.y - stat_aabb.min.y,
                             stat_aabb.max.z - stat_aabb.min.z)) / 2.0
        is_thin_y = stat_half.y * 3.0 < max(stat_half.x, stat_half.z)

        grounded = False
        if is_thin_y or dy <= min(dx, dz):
            rel_y = dyn_aabb.get_center().y - stat_aabb.get_center().y
            direction = 1.0 if rel_y >= 0 else -1.0

            # Prevent slipping through thin planes due zero-penetration small errors
            push = max(dy, 1e-4)
            penetration = Vector3((0.0, push * direction, 0.0))

            # Mark grounded only when dynamic object is above static and in contact range
            touching = dy > 0.0 or abs(dyn_aabb.max.y - stat_aabb.min.y) <= 0.05
            grounded = (direction > 0 and touching)

            print(f"[PHYSICS] Y PENT: dx={dx:.6f} dy={dy:.6f} dz={dz:.6f} push={push:.6f} direction={direction} touching={touching} grounded={grounded}")
        elif dx <= dz:
            rel_x = dyn_aabb.get_center().x - stat_aabb.get_center().x
            penetration = Vector3((dx * (1.0 if rel_x >= 0 else -1.0), 0.0, 0.0))
            print(f"[PHYSICS] X PENT: penetration={penetration}")
        else:
            rel_z = dyn_aabb.get_center().z - stat_aabb.get_center().z
            penetration = Vector3((0.0, 0.0, dz * (1.0 if rel_z >= 0 else -1.0)))
            print(f"[PHYSICS] Z PENT: penetration={penetration}")

        dynamic_obj.transform.position += penetration

        # Cancel velocity components along penetration axis
        if abs(penetration.x) > 0.0:
            dynamic_obj.velocity.x = 0.0
        if abs(penetration.y) > 0.0:
            dynamic_obj.velocity.y = 0.0
        if abs(penetration.z) > 0.0:
            dynamic_obj.velocity.z = 0.0

        return grounded

    def step(self, dt):
        for body in list(self.dynamic_bodies):
            if not getattr(body, "physics_enabled", False) or not getattr(body, "collidable", False):
                continue

            # Reset ground flags for this frame
            body.on_ground = False
            body.ground_timer = max(0.0, getattr(body, "ground_timer", 0.0) - dt)

            # Apply gravity only when airborne
            if not body.on_ground and getattr(body, "gravity_enabled", False):
                gravity_scale = getattr(body, "gravity_strength", 1.0)
                body.velocity += self.gravity * dt * gravity_scale

            # Perform damping for smoother motion (horizontal only)
            damping = getattr(body, "linear_damping", 10.0)
            body.velocity.x *= max(0.0, 1.0 - damping * dt)
            body.velocity.z *= max(0.0, 1.0 - damping * dt)

            # Optional vertical air resistance; default is 0 (no extra slowdown)
            if not getattr(body, "on_ground", False):
                air_resistance = getattr(body, "vertical_air_resistance", 0.0)
                if air_resistance > 0.0:
                    body.velocity.y *= max(0.0, 1.0 - air_resistance * dt)

            # Terminal velocity
            terminal_speed = getattr(body, "terminal_velocity", 50.0)
            if body.velocity.y < -terminal_speed:
                body.velocity.y = -terminal_speed

            # Move
            body.transform.position += body.velocity * dt

            # Collisions
            grounded = False
            for static_obj in self.static_bodies + [o for o in self.dynamic_bodies if o is not body]:
                if self.resolve_collision(body, static_obj):
                    grounded = True

            # 🔥 Correct grounding buffer
            if grounded:
                body.ground_timer = 0.1
            else:
                body.ground_timer = max(0.0, body.ground_timer - dt)

            body.on_ground = body.ground_timer > 0.0

            # Stop vertical velocity if grounded
            if body.on_ground and body.velocity.y < 0.0:
                body.velocity.y = 0.0

            # Clamp world fall limit
            if body.transform.position.y < -1000:
                body.transform.position.y = -1000
                body.velocity.y = 0


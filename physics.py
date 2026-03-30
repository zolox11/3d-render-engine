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

    def get_center(self) -> Vector3:
        return (self.min + self.max) / 2.0

    def size(self) -> Vector3:
        return self.max - self.min

    def expand(self, amount: float):
        return AABB(self.min - Vector3((amount, amount, amount)),
                    self.max + Vector3((amount, amount, amount)))


class PhysicsEngine:
    def __init__(self, gravity=Vector3((0.0, -9.81, 0.0))):
        self.gravity = Vector3(gravity)

        self.dynamic_bodies = []
        self.static_bodies = []

        # Simulation tuning
        self.max_substeps = 4
        self.fixed_timestep = 1.0 / 60.0

    def reset(self):
        self.dynamic_bodies.clear()
        self.static_bodies.clear()

    def register_body(self, obj):
        if obj is None or not hasattr(obj, "collidable") or not obj.collidable:
            return

        if getattr(obj, "physics_enabled", False) and obj.mass > 0.0:
            if obj not in self.dynamic_bodies:
                self.dynamic_bodies.append(obj)
        else:
            if obj not in self.static_bodies:
                self.static_bodies.append(obj)

    def load_scene(self, scene):
        self.reset()
        for obj in getattr(scene, "objects", []):
            self.register_body(obj)

    # -------------------------
    # AABB GENERATION
    # -------------------------
    @staticmethod
    def get_aabb(obj):
        center = Vector3(obj.transform.position)

        half = Vector3(getattr(obj, "collider_half_size", Vector3((0.5, 0.5, 0.5))))
        scale = Vector3(obj.transform.scale)

        half = Vector3((
            abs(half.x * scale.x),
            abs(half.y * scale.y),
            abs(half.z * scale.z)
        ))

        # Prevent degenerate colliders
        min_half = 0.05
        half = Vector3((
            max(half.x, min_half),
            max(half.y, min_half),
            max(half.z, min_half)
        ))

        return AABB(center - half, center + half)

    # -------------------------
    # COLLISION RESOLUTION
    # -------------------------
    def resolve_collision(self, dynamic_obj, static_obj):
        if not static_obj.collidable:
            return False

        dyn_aabb = self.get_aabb(dynamic_obj)
        stat_aabb = self.get_aabb(static_obj)

        if not dyn_aabb.intersects(stat_aabb):
            return False

        # Penetration depths
        dx = min(dyn_aabb.max.x, stat_aabb.max.x) - max(dyn_aabb.min.x, stat_aabb.min.x)
        dy = min(dyn_aabb.max.y, stat_aabb.max.y) - max(dyn_aabb.min.y, stat_aabb.min.y)
        dz = min(dyn_aabb.max.z, stat_aabb.max.z) - max(dyn_aabb.min.z, stat_aabb.min.z)

        rel = dyn_aabb.get_center() - stat_aabb.get_center()

        abs_dx, abs_dy, abs_dz = abs(dx), abs(dy), abs(dz)

        penetration = Vector3((0.0, 0.0, 0.0))
        grounded = False

        # Resolve along smallest axis (stable separation)
        if abs_dy <= abs_dx and abs_dy <= abs_dz:
            sign = 1.0 if rel.y >= 0 else -1.0
            penetration = Vector3((0.0, abs_dy * sign, 0.0))

            # Only consider grounded if pushing upward out of surface
            if sign > 0:
                grounded = True

        elif abs_dx <= abs_dz:
            sign = 1.0 if rel.x >= 0 else -1.0
            penetration = Vector3((abs_dx * sign, 0.0, 0.0))
        else:
            sign = 1.0 if rel.z >= 0 else -1.0
            penetration = Vector3((0.0, 0.0, abs_dz * sign))

        # Apply positional correction
        dynamic_obj.transform.position += penetration

        # Velocity correction (project out collision axis)
        if abs(penetration.x) > 0.0:
            dynamic_obj.velocity.x = 0.0
        if abs(penetration.y) > 0.0:
            dynamic_obj.velocity.y = 0.0
        if abs(penetration.z) > 0.0:
            dynamic_obj.velocity.z = 0.0

        return grounded

    # -------------------------
    # GRAVITY + INTEGRATION
    # -------------------------
    def apply_gravity(self, body, dt):
        if not getattr(body, "gravity_enabled", False):
            return

        gravity_strength = getattr(body, "gravity_strength", 1.0)
        body.velocity += self.gravity * gravity_strength * dt

    def clamp_velocity(self, body):
        terminal_velocity = getattr(body, "terminal_velocity", 60.0)

        if body.velocity.y < -terminal_velocity:
            body.velocity.y = -terminal_velocity

    def apply_damping(self, body, dt):
        damping = getattr(body, "linear_damping", 5.0)

        if damping > 0.0:
            factor = max(0.0, 1.0 - damping * dt)
            body.velocity.x *= factor
            body.velocity.z *= factor

        # Optional vertical damping
        vd = getattr(body, "vertical_damping", 0.0)
        if vd > 0.0:
            vf = max(0.0, 1.0 - vd * dt)
            body.velocity.y *= vf

    def integrate(self, body, dt):
        body.transform.position += body.velocity * dt

    # -------------------------
    # SOLVER LOOP
    # -------------------------
    def solve_collisions(self, body):
        grounded = False

        # Static collisions
        for static_obj in self.static_bodies:
            if self.resolve_collision(body, static_obj):
                grounded = True

        # Dynamic collisions
        for other in self.dynamic_bodies:
            if other is body:
                continue
            self.resolve_collision(body, other)

        return grounded

    # -------------------------
    # STEP
    # -------------------------
    def step(self, dt):
        if dt <= 0.0:
            return

        # Substepping for stability (important for high speeds)
        substeps = max(1, min(self.max_substeps, int(dt / self.fixed_timestep) + 1))
        sub_dt = dt / substeps

        for _ in range(substeps):
            for body in list(self.dynamic_bodies):
                if not getattr(body, "physics_enabled", False) or not getattr(body, "collidable", False):
                    continue

                # Ground state reset
                body.on_ground = False

                # Coyote timer
                body.ground_timer = max(0.0, getattr(body, "ground_timer", 0.0) - sub_dt)

                # Gravity
                self.apply_gravity(body, sub_dt)

                # Damping
                self.apply_damping(body, sub_dt)

                # Clamp velocity
                self.clamp_velocity(body)

                # Integrate motion
                self.integrate(body, sub_dt)

                # Collision solving (iterative)
                grounded = False
                for _ in range(3):
                    if self.solve_collisions(body):
                        grounded = True

                # Ground handling
                if grounded:
                    body.ground_timer = 0.1
                    body.on_ground = True
                else:
                    body.on_ground = body.ground_timer > 0.0

                # Snap downward velocity when grounded
                if body.on_ground and body.velocity.y < 0.0:
                    body.velocity.y = 0.0

                # Prevent falling out of world
                if body.transform.position.y < -1000.0:
                    body.transform.position.y = -1000.0
                    body.velocity.y = 0.0
                    body.on_ground = True
                    body.ground_timer = 0.1
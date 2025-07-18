import numpy as np
import pygame
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
import json
from dataclasses import dataclass
import time
from numba import jit, prange, float64

# Dataclass for Prolate Spheroid Object
@dataclass
class ProlateSpheroid:
    mass: float
    velocity: np.ndarray  # [vx, vy, vz]
    density: float
    dimensions: tuple  # (a, b, c) where a=b (equatorial), c (polar) for prolate
    position: np.ndarray  # [x, y, z]
    magnetic_field: float  # Strength
    magnetic_axis: np.ndarray  # Direction vector
    angular_velocity: float  # Rotation speed
    rotation_axis: np.ndarray  # Axis of rotation

    def update_position(self, dt):
        # Update position based on velocity and time step
        self.position += self.velocity * dt

    def handle_collision(self, other):
        # Dynamic collision: Break into up to 10 fragments based on mass and velocity
        relative_velocity = np.linalg.norm(self.velocity - other.velocity)
        impact_energy = 0.5 * (self.mass * other.mass) * (relative_velocity ** 2)
        threshold = 1e9  # Increased threshold to reduce collisions
        if impact_energy > threshold:
            fragments = min(10, int(impact_energy / threshold))
            fragment_mass = self.mass / fragments
            # Simplified: Reduce mass of original objects, create new fragments
            self.mass -= fragment_mass * (fragments - 1)
            other.mass -= fragment_mass * (fragments - 1)
            return [ProlateSpheroid(
                mass=fragment_mass,
                velocity=self.velocity + np.random.randn(3) * 0.1,
                density=self.density,
                dimensions=(self.dimensions[0] / 2, self.dimensions[1] / 2, self.dimensions[2] / 2),
                position=self.position + np.random.randn(3) * 0.1,
                magnetic_field=self.magnetic_field,
                magnetic_axis=self.magnetic_axis,
                angular_velocity=self.angular_velocity,
                rotation_axis=self.rotation_axis
            ) for _ in range(fragments - 2)]
        return []

# Numba-accelerated gravity calculation
@jit(nopython=True)  # Keeping parallel=False for stability; can re-enable later
def apply_gravity_numba(positions, velocities, masses, dt, G=6.67430e-11):
    n = len(masses)
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                if r > 0:
                    force = G * (masses[i] * masses[j]) / (r * r)
                    direction_x = dx / r
                    direction_y = dy / r
                    direction_z = dz / r
                    acceleration = force / masses[i]
                    velocities[i, 0] += direction_x * acceleration * dt
                    velocities[i, 1] += direction_y * acceleration * dt
                    velocities[i, 2] += direction_z * acceleration * dt

# Numba-accelerated magnetic force calculation
@jit(nopython=True)  # Keeping parallel=False for stability; can re-enable later
def apply_magnetic_force_numba(positions, velocities, magnetic_fields, masses, dt):
    n = len(masses)
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                r = np.sqrt(dx*dx + dy*dy + dz*dz)
                if r > 0:
                    force = magnetic_fields[i] * magnetic_fields[j] / (r * r)
                    direction_x = dx / r
                    direction_y = dy / r
                    direction_z = dz / r
                    velocities[i, 0] += direction_x * force * dt / masses[i]
                    velocities[i, 1] += direction_y * force * dt / masses[i]
                    velocities[i, 2] += direction_z * force * dt / masses[i]

# OpenGL setup for 3D rendering
def setup_opengl():
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    # Set light position (move closer to the objects)
    glLightfv(GL_LIGHT0, GL_POSITION, [0, 0, 2, 1])  # Position at (0, 0, 2) with w=1 (positional light)
    # Set material properties
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [1.0, 0.5, 1.0, 1.0])  # White diffuse color
    glMaterialfv(GL_FRONT, GL_SPECULAR, [1.0, 1.0, 1.0, 1.0])  # White specular color
    glMaterialf(GL_FRONT, GL_SHININESS, 50.0)  # Shininess
    gluPerspective(45, 1600/1200, 0.1, 500.0)
    glTranslatef(0.0, 0.0, -2)  # Move camera closer (from -5 to -2)

def draw_prolate_spheroid(obj):
    # Simplified rendering: Use a scaled sphere to approximate prolate spheroid
    a, b, c = obj.dimensions  # a = b for prolate
    glPushMatrix()
    glTranslatef(*obj.position)
    glScalef(a, b, c)  # Stretch sphere into prolate shape
    quad = gluNewQuadric()
    gluSphere(quad, 1.0, 20, 20)  # Increased radius to 1.0 for better visibility
    gluDeleteQuadric(quad)
    glPopMatrix()

# Main Simulation Class
class ProlateSimulation:
    def __init__(self):
        pygame.init()
        self.display = (1600, 1200)
        self.screen = pygame.display.set_mode(self.display, pygame.OPENGL | pygame.DOUBLEBUF)
        setup_opengl()
        self.objects = []
        self.paused = False
        self.time_scale = 1.0  # Speed up/down with arrows
        self.menu_open = False
        self.load_simulation("objects.sim")  # Load default state

    def add_object(self, mass, velocity, density, dimensions, position, magnetic_field, magnetic_axis, angular_velocity, rotation_axis):
        obj = ProlateSpheroid(mass, np.array(velocity, dtype=np.float64), density, dimensions, np.array(position, dtype=np.float64),
                              magnetic_field, np.array(magnetic_axis, dtype=np.float64), angular_velocity, np.array(rotation_axis, dtype=np.float64))
        self.objects.append(obj)

    def update(self, dt):
        if not self.paused and len(self.objects) > 0:  # Check for non-empty object list
            try:
                # Prepare arrays for Numba
                positions = np.atleast_2d([obj.position for obj in self.objects]).astype(np.float64)
                velocities = np.atleast_2d([obj.velocity for obj in self.objects]).astype(np.float64)
                masses = np.array([obj.mass for obj in self.objects], dtype=np.float64)
                magnetic_fields = np.array([obj.magnetic_field for obj in self.objects], dtype=np.float64)

                # Apply gravity and magnetic forces using Numba
                apply_gravity_numba(positions, velocities, masses, dt * self.time_scale)
                apply_magnetic_force_numba(positions, velocities, magnetic_fields, masses, dt * self.time_scale)

                # Update object velocities and positions
                for i, obj in enumerate(self.objects):
                    obj.velocity = velocities[i]
                    obj.update_position(dt * self.time_scale)

                # Handle collisions
                new_objects = []
                for i, obj in enumerate(self.objects):
                    for j, other in enumerate(self.objects[i+1:], start=i+1):
                        if np.linalg.norm(obj.position - other.position) < (obj.dimensions[0] + other.dimensions[0]):
                            new_objects.extend(obj.handle_collision(other))
                self.objects.extend(new_objects)
            except Exception as e:
                print(f"Error in update: {e}")
                raise

    def render(self):
        try:
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            for obj in self.objects:
                draw_prolate_spheroid(obj)
            pygame.display.flip()
        except Exception as e:
            print(f"Error in render: {e}")
            raise

    def menu(self):
        self.paused = True
        print("Menu: Add new object (simplified input) or type 'quit' to exit")
        user_input = input("Enter 'quit' to exit, or press Enter to add an object: ")
        if user_input.lower() == 'quit':
            self.save_simulation()
            raise SystemExit("Simulation exited from menu")
        mass = float(input("Mass: "))
        velocity = np.array([float(x) for x in input("Velocity (vx,vy,vz): ").split(",")])
        density = float(input("Density: "))
        dimensions = tuple([float(x) for x in input("Dimensions (a,b,c): ").split(",")])
        position = np.array([float(x) for x in input("Position (x,y,z): ").split(",")])
        magnetic_field = float(input("Magnetic Field Strength: "))
        magnetic_axis = np.array([float(x) for x in input("Magnetic Axis (x,y,z): ").split(",")])
        angular_velocity = float(input("Angular Velocity: "))
        rotation_axis = np.array([float(x) for x in input("Rotation Axis (x,y,z): ").split(",")])
        self.add_object(mass, velocity, density, dimensions, position, magnetic_field, magnetic_axis, angular_velocity, rotation_axis)
        self.paused = False
        self.menu_open = False

    def save_simulation(self, filename="objects.sim"):
        data = {
            "objects": [
                {
                    "mass": obj.mass,
                    "velocity": obj.velocity.tolist(),
                    "density": obj.density,
                    "dimensions": obj.dimensions,
                    "position": obj.position.tolist(),
                    "magnetic_field": obj.magnetic_field,
                    "magnetic_axis": obj.magnetic_axis.tolist(),
                    "angular_velocity": obj.angular_velocity,
                    "rotation_axis": obj.rotation_axis.tolist()
                } for obj in self.objects
            ]
        }
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Simulation saved to {filename}")
        except IOError as e:
            print(f"Error saving simulation to {filename}: {e}")

    def load_simulation(self, filename="objects.sim"):
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                self.objects = []
                for obj_data in data["objects"]:
                    self.add_object(
                        obj_data["mass"],
                        np.array(obj_data["velocity"], dtype=np.float64),
                        obj_data["density"],
                        tuple(obj_data["dimensions"]),
                        np.array(obj_data["position"], dtype=np.float64),
                        obj_data["magnetic_field"],
                        np.array(obj_data["magnetic_axis"], dtype=np.float64),
                        obj_data["angular_velocity"],
                        np.array(obj_data["rotation_axis"], dtype=np.float64)
                    )
            print(f"Loaded {len(self.objects)} objects from {filename}")
        except (FileNotFoundError, json.JSONDecodeError):
            # If the file doesn't exist or contains invalid JSON, start with an empty simulation
            self.objects = []
            print(f"Could not load {filename} (file not found or invalid JSON). Starting with empty simulation.")

    def run(self):
        clock = pygame.time.Clock()
        running = True
        try:
            while running:
                dt = clock.tick(60) / 1000.0  # Delta time in seconds
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_m:
                            self.menu_open = True
                            self.menu()
                        elif event.key == pygame.K_p:
                            self.paused = not self.paused
                        elif event.key == pygame.K_s:
                            self.save_simulation()
                            running = False
                        elif event.key == pygame.K_n:
                            self.objects = []  # New simulation
                        elif event.key == pygame.K_UP:
                            self.time_scale *= 1.1  # Speed up
                        elif event.key == pygame.K_DOWN:
                            self.time_scale /= 1.1  # Slow down
                        elif event.key == pygame.K_h:
                            print("Controls:\nM: Open menu\nP: Pause\nS: Save and stop\nN: New simulation\nUP/DOWN: Speed up/down\nH: Help")

                self.update(dt)
                self.render()
        finally:
            pygame.quit()  # Ensure Pygame cleanup happens even if an exception occurs

if __name__ == "__main__":
    print("Create Class")
    sim = ProlateSimulation()
    print("Create a default object")
    # Add a test object to start with
    # sim.add_object(
    #     mass=1e5,
    #     velocity=[0, 0, 0],
    #     density=1000,
    #     dimensions=(0.2, 0.2, 0.5),  # Prolate spheroid (a=b, c>a)
    #     position=[0, 0, 0],
    #     magnetic_field=0.1,
    #     magnetic_axis=[0, 0, 1],
    #     angular_velocity=0.1,
    #     rotation_axis=[0, 0, 1]
    # )
    sim.save_simulation()
    print("Run Simulation")
    sim.run()
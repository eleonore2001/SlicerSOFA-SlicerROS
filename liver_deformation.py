import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import vtk
import slicer

try:
  import slicersofa
except ModuleNotFoundError:
  import sys
  sys.path.append(".")
  import slicersofa

import importlib
importlib.reload(slicersofa)
importlib.reload(slicersofa.util)
importlib.reload(slicersofa.meshes)
simulating = True
probeTarget = [0,0.0,0.0]


import Sofa
import Sofa.Core
import Sofa.Simulation

from stlib3.scene import MainHeader


PYVISTA = True
if PYVISTA:
    try:
        import pyvista as pv
        plotter = pv.Plotter()
    except ModuleNotFoundError:
        PYVISTA = False


#How this works : a stl file of the tip of the robot interacts directly with the liver (via the SOFA extension). To make this tip follow the movement of the robot, I applied to it the transformation wrist to tip (via the ROS extension). This way the
#"SOFA" tip interacts with the liver following the movement of my Phantom Omni tip.
# Create a root node
root = Sofa.Core.Node("root")

#Imports to enable the use of the fixedconstraint thing
MainHeader(root, plugins=[
        "Sofa.Component.IO.Mesh",
        "Sofa.Component.LinearSolver.Direct",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.Mapping.Linear",
        "Sofa.Component.Mass",
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.Setting",
        "Sofa.Component.SolidMechanics.FEM.Elastic",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Topology.Container.Dynamic",
        "Sofa.Component.Visual",
        "Sofa.GL.Component.Rendering3D",
        "Sofa.Component.AnimationLoop",
        "Sofa.Component.Collision.Detection.Algorithm",
        "Sofa.Component.Collision.Detection.Intersection",
        "Sofa.Component.Collision.Geometry",
        "Sofa.Component.Collision.Response.Contact",
        "Sofa.Component.Constraint.Lagrangian.Solver",
        "Sofa.Component.Constraint.Lagrangian.Correction",
        "Sofa.Component.LinearSystem",
        "Sofa.Component.MechanicalLoad",
        "MultiThreading",
        "Sofa.Component.SolidMechanics.Spring",
        "Sofa.Component.Constraint.Lagrangian.Model",
        "Sofa.Component.Mapping.NonLinear",
        "Sofa.Component.Topology.Container.Constant",
        "Sofa.Component.Topology.Mapping",
        "Sofa.Component.Engine.Select",
        "Sofa.Component.Constraint.Projective",
        "Sofa.Component.Topology.Container.Grid",

    ])

def transformation_matrix_to_rigid3d(matrix):
    # Extract translation
    tx, ty, tz = matrix[0, 3], matrix[1, 3], matrix[2, 3]

    # Extract rotation matrix
    m = matrix[:3, :3]
    tr = m[0, 0] + m[1, 1] + m[2, 2]

    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m[2, 1] - m[1, 2]) / S
        qy = (m[0, 2] - m[2, 0]) / S
        qz = (m[1, 0] - m[0, 1]) / S
    elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
        S = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        qw = (m[2, 1] - m[1, 2]) / S
        qx = 0.25 * S
        qy = (m[0, 1] + m[1, 0]) / S
        qz = (m[0, 2] + m[2, 0]) / S
    elif m[1, 1] > m[2, 2]:
        S = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        qw = (m[0, 2] - m[2, 0]) / S
        qx = (m[0, 1] + m[1, 0]) / S
        qy = 0.25 * S
        qz = (m[1, 2] + m[2, 1]) / S
    else:
        S = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        qw = (m[1, 0] - m[0, 1]) / S
        qx = (m[0, 2] + m[2, 0]) / S
        qy = (m[1, 2] + m[2, 1]) / S
        qz = 0.25 * S

    # Return in Rigid3d order: [x, y, z, qx, qy, qz, qw]
    return [tx, ty, tz, qx, qy, qz, qw]

def rotation_matrix_x(angle_rad):
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([
        [1, 0,  0, 0],
        [0, c, -s, 0],
        [0, s,  c, 0],
        [0, 0,  0, 1]
    ])


# Create the scene

#slicer.mrmlScene.Clear()
slicer.app.processEvents()
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

mesh_dir = Path("meshes")
liver_mesh_file = mesh_dir / "liver2.msh"
instrument_surface_mesh_file = mesh_dir/ "tip.stl"

# Simulation Hyperparameters
dt = 0.01
#collision_detection_method = "MinProximityIntersection"  # Which algorithm to use for collision detection
#collision_detection_method = "LocalMinDistance"  # Which algorithm to use for collision detection
collision_detection_method = "DiscreteIntersection"  # Which algorithm to use for collision detection
#collision_detection_method = "NewProximityIntersection"  # Which algorithm to use for collision detection
alarm_distance = 10.0  # This will tell the collision detection algorithm to start checking for actual collisions
contact_distance = 0.8  # This is the distance at which the collision detection algorithm will consider two objects to be in contact

liver_mass = 300.0  # g
liver_youngs_modulus = 2.5 * 1000.0  # 15 kPa -> 15 * 1000 in mm/s/g
liver_poisson_ratio = 0.45

instrument_mass = 500.0  # g
instrument_pose = probeTarget + [0.0, 0.0, 0.0, 1.0]  # x, y, z, qx, qy, qz, qw


slicersofa.util.initPlugins(root)

# The simulation scene
root.gravity = [0.0, 0.0, -9.81 * 10.0]
root.dt = dt

root.addObject("FreeMotionAnimationLoop")  # One of several simulation loops.
root.addObject("VisualStyle", displayFlags=["showForceFields", "showBehaviorModels", "showCollisionModels"])

root.addObject("CollisionPipeline")  # This object will be used to manage the collision detection
root.addObject("ParallelBruteForceBroadPhase")  # The broad phase checks for overlaps in bounding boxes
root.addObject("ParallelBVHNarrowPhase")  # And the narrow phase checks for collisions
#root.addObject(collision_detection_method, alarmDistance=alarm_distance, contactDistance=contact_distance)  # Using this algorithm for collision detection
root.addObject(collision_detection_method)  # Using this algorithm for collision detection

root.addObject("CollisionResponse", response="FrictionContactConstraint")  # This object will be used to manage the collision response
root.addObject("GenericConstraintSolver")  # And this object will be used to solve the constraints resulting from collisions etc.

scene_node = root.addChild("scene")  # The scene node will contain all the objects in the scene

################
#### Liver ####
################
liver_node = scene_node.addChild("liver")
# Load the liver mesh, scale it from m to mm, and move it up by 100 mm
liver_node.addObject("MeshGmshLoader", filename=str(liver_mesh_file), scale=50.0, translation=[-100.0,100.0, 0.0])
liver_node.addObject("TetrahedronSetTopologyContainer", src=liver_node.MeshGmshLoader.getLinkPath())  # This is the container for the tetrahedra
liver_node.addObject("TetrahedronSetTopologyModifier")  # And this loads some algorithms for modifying the topology. Not always necessary, but good to have.
liver_node.addObject("EulerImplicitSolver")  # This is the ODE solver (for the time)
liver_node.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")  # And the linear solver (for the space).
liver_node.addObject("MechanicalObject", name="MechanicalObject")
# This components holds the positions, velocities, and forces of the vertices
liver_node.addObject("TetrahedralCorotationalFEMForceField", youngModulus=liver_youngs_modulus, poissonRatio=liver_poisson_ratio)  # This is the FEM algorithm that will calculate the forces
liver_node.addObject("UniformMass", totalMass=liver_mass)  # This will give all the vertices the same mass, summing up to the total mass
liver_node.addObject("LinearSolverConstraintCorrection")  # This will compute the corrections to the forces to satisfy the constraints
liver_node.addObject('BoxROI', name='boxROI', box="-200 50 -48 0 200 -12", drawBoxes="1")
liver_node.addObject('FixedConstraint' ,indices='@boxROI.indices')
liver_collision_node = liver_node.addChild("collision")
liver_collision_node.addObject("TriangleSetTopologyContainer")  # Another topology container, this time for the triangles for collision
liver_collision_node.addObject("TriangleSetTopologyModifier")  # And the modifier for the triangles
liver_collision_node.addObject("Tetra2TriangleTopologicalMapping")  # This will map the tetrahedra from the parent node to the triangles in this node
liver_collision_node.addObject("PointCollisionModel")  # This will create the collision model based on the points stored in the TriangleSetTopologyContainer
liver_collision_node.addObject("LineCollisionModel")  # This will create the collision model based on the points stored in the TriangleSetTopologyContainer
liver_collision_node.addObject("TriangleCollisionModel")  # And for the triangles

##################
### Instrument ###
##################
# Rigid objects are handled by one vertex that is store in a Rigid3d object, so 7 values (x, y, z, qx, qy, qz, qw) for position and orientation
# If we want to control the position of the object, we can either set velocities and forces in the MechanicalObject, or we use a motion target
# and add some springs, to "fake" a force control. If we set the position directly, collision checking would no longer work, because the object
# "teleports" through the scene.
instrument_node = scene_node.addChild("instrument")
instrument_node.addObject("EulerImplicitSolver")
instrument_node.addObject("CGLinearSolver")

instrument_motion_target_node = instrument_node.addChild("motion_target")
instrument_motion_target_node.addObject("MechanicalObject", template="Rigid3d", position=instrument_pose,scale=800.0)

instrument_node.addObject("MechanicalObject", template="Rigid3d", position=instrument_pose)
instrument_node.addObject("UniformMass", totalMass=instrument_mass)
instrument_node.addObject("RestShapeSpringsForceField", external_rest_shape=instrument_motion_target_node.getLinkPath(), stiffness=1e32, angularStiffness=1e32)
instrument_node.addObject("UncoupledConstraintCorrection")  # <- different to the deformable objects, where the points are not uncoupled


instrument_collision_node = instrument_node.addChild("collision")
#instrument_collision_node.addObject("MeshSTLLoader", filename=str(instrument_surface_mesh_file), scale=30.0, translation=[0.0, 0.0, 200.0])
instrument_collision_node.addObject("MeshSTLLoader", filename=str(instrument_surface_mesh_file), scale=800.0)
#nstrument_collision_node.addObject("VisualTransform", rotation=[1.0, 0.0,3.14])  # Rotation around X-axis (180 degrees in radians)
instrument_collision_node.addObject("TriangleSetTopologyContainer", src=instrument_collision_node.MeshSTLLoader.getLinkPath())
instrument_collision_node.addObject("MechanicalObject", template="Vec3d")
# there are quite a few points in this model. Simulation might slow down. You can just add a different obj file.
#instrument_collision_node.addObject("PointCollisionModel")
instrument_collision_node.addObject("PointCollisionModel")
instrument_collision_node.addObject("LineCollisionModel")
instrument_collision_node.addObject("TriangleCollisionModel")
instrument_collision_node.addObject("RigidMapping")

# Initialize the simulation
Sofa.Simulation.init(root)

# Get the liver vertex positions
liver_positions = root.scene.liver.MechanicalObject.position.array()
liver_positions = root.scene.liver.MechanicalObject.position.array()
instrument_positions = root.scene.instrument.collision.MechanicalObject.position.array()

# Similarly you can access the topology of the object
liver_triangles = root.scene.liver.collision.TriangleSetTopologyContainer.triangles.array()
# however, the values of TriangleSetTopologyContainer.position will not change, because the simulated positions are stored in the MechanicalObject
instrument_triangles = root.scene.instrument.collision.TriangleSetTopologyContainer.triangles.array()

if PYVISTA:
    for position_array, triangle_array, color, name in zip([liver_positions, instrument_positions], [liver_triangles, instrument_triangles], ["red","green"], ['liver','instrument']):
        faces = np.zeros((triangle_array.shape[0], 4), dtype=np.uint64)
        faces[:, 1:] = triangle_array
        faces[:, 0] = 3
        mesh = pv.PolyData(position_array, faces)
        plotter.add_mesh(mesh, color=color, opacity=0.5, show_edges=True, lighting=True)

        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        modelNode.SetName(name)
        modelNode.CreateDefaultDisplayNodes()
        modelNode.SetAndObservePolyData(mesh)
    instrumentModel = slicer.util.getNode('instrument')
    # instrumentModel.GetDisplayNode().SetOpacity(0.0)

def timeStep():
    Sofa.Simulation.animate(root, root.dt.value)
    with root.scene.instrument.motion_target.MechanicalObject.position.writeable() as target_poses:
        trans_node = slicer.util.getNode('ros2:tf2lookup:wristTotip')
        transformation_matrix_SOFA= slicer.util.arrayFromTransformMatrix(trans_node, toWorld = True)
        SOFA_to_SLICER = [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0] ]
        correction_rotation = rotation_matrix_x(np.pi / 2)
        transformation_matrix_SLICER = transformation_matrix_SOFA @ SOFA_to_SLICER @ correction_rotation
        target_poses[0, 0:7] = transformation_matrix_to_rigid3d(transformation_matrix_SLICER)

    if PYVISTA:
        # the valuess in the numpy array are update, but pyvista does copy, not reference the data
        for i, position_array in enumerate([liver_positions, instrument_positions]):
            plotter.meshes[i].points = position_array
   
    if simulating:
        qt.QTimer.singleShot(10, timeStep)


timeStep()


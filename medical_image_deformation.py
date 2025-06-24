# -*- coding: utf-8 -*-
"""Basic scene using Cosserat in SofaPython3.
Cosserat Needle Simulation with Deformable Volume Interaction

This module implements a SOFA-based simulation of a Cosserat needle interacting
with a deformable FEM volume, integrated with 3D Slicer for visualization.

Dependencies:
- SOFA Framework
- 3D Slicer
- SlicerSOFA extension
- PyVista (optional, for enhanced visualization)
- NumPy
- VTK

Author: [Eleonore Germond]
License: [MIT]

Based on the work done with SofaPython. See POEMapping.py
"""
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
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
import copy
simulating = True


import Sofa
import Sofa.Core
import Sofa.Simulation
import sys
sys.path.append('/home/slicer/SlicerSOFA/Release/SofaCosserat/examples/python3')
sys.path.append('//home/slicer/SlicerSOFA-pieper/Experiments/liver-torus-probe')

PYVISTA = True
if PYVISTA:
    try:
        import pyvista as pv
        plotter = pv.Plotter()
    except ModuleNotFoundError:
        PYVISTA = False
import vtk
from vtk.util.numpy_support import numpy_to_vtk

from cosserat.needle.needleController import Animation
from cosserat.needle.params import NeedleParameters, GeometryParams, PhysicsParams, FemParams, ContactParams
from cosserat.usefulFunctions import pluginList
from cosserat.cosseratObject import Cosserat

from deformation_manager import DeformationManager


class FEMVolumeCreator():
    def __init__(self):
        self.minPoint2 = None
        self.maxPoint2 = None

    def defineBoundaries(self):
        # Start the FEM cube definition
        roiNode = slicer.util.getNode("R")  
        center = [0.0, 0.0, 0.0]
        roiNode.GetCenter(center)

        xyz = [0,0,0]
        radius = [0,0,0]
        roiNode.GetXYZ(xyz)
        roiNode.GetRadiusXYZ(radius)

        minPoint = [center[i] - radius[i] for i in range(3)]
        maxPoint = [center[i] + radius[i] for i in range(3)]
        self.minPoint2= " ".join(f"{v:.2f}" for v in minPoint)
        self.maxPoint2= " ".join(f"{v:.2f}" for v in maxPoint)

    def createFemCubeWithParams(self,parentNode, geometry):
        FemNode = parentNode.addChild("FemNode")

        gelVolume = FemNode.addChild("gelVolume")
        gelVolume.addObject("RegularGridTopology", name="HexaTop", n=geometry.mesh, min= self.minPoint2,max = self.maxPoint2)
        cont = gelVolume.addObject("TetrahedronSetTopologyContainer", name="TetraContainer", position="@HexaTop.position")
        gelVolume.addObject("TetrahedronSetTopologyModifier", name="Modifier")
        gelVolume.addObject("Hexa2TetraTopologicalMapping", input="@HexaTop", output="@TetraContainer", swapping="false")

        GelSurface = FemNode.addChild("GelSurface")
        GelSurface.addObject("TriangleSetTopologyContainer", name="triangleContainer",
                            position="@../gelVolume/HexaTop.position")
        GelSurface.addObject("TriangleSetTopologyModifier", name="Modifier")
        GelSurface.addObject("Tetra2TriangleTopologicalMapping", input="@../gelVolume/TetraContainer",
                            output="@triangleContainer", flipNormals="false")

        gelNode = FemNode.addChild("gelNode")
        gelNode.addObject("EulerImplicitSolver", rayleighMass=geometry.rayleigh, rayleighStiffness=geometry.rayleigh)
        gelNode.addObject('SparseLDLSolver', name='precond', template="CompressedRowSparseMatrixMat3x3d")
        gelNode.addObject('TetrahedronSetTopologyContainer', src="@../gelVolume/TetraContainer", name='container')
        gelNode.addObject('MechanicalObject', name='tetras', template='Vec3d')
        gelNode.addObject('TetrahedronFEMForceField', template='Vec3d', name='FEM', method='large',
                        poissonRatio=geometry.poissonRatio, youngModulus=geometry.youngModulus)
        gelNode.addObject('BoxROI', name='ROI1', box=geometry.box, drawBoxes='true')
        gelNode.addObject('RestShapeSpringsForceField', points='@ROI1.indices', stiffness='1e12')

        surfaceNode = gelNode.addChild("surfaceNode")
        surfaceNode.addObject('TriangleSetTopologyContainer', name="surfContainer",
                            src="@../../GelSurface/triangleContainer")
        surfaceNode.addObject('MechanicalObject', name='msSurface')
        surfaceNode.addObject('TriangleCollisionModel', name='surface')
        surfaceNode.addObject('LineCollisionModel', name='line')
        surfaceNode.addObject('BarycentricMapping')
        gelNode.addObject('LinearSolverConstraintCorrection')

        return FemNode

    def addDependencies(self,cubeNode):
        gelNode = cubeNode.getChild('gelNode')
        constraintPointsNode = gelNode.addChild('constraintPoints')
        constraintPointsNode.addObject("PointSetTopologyContainer", name="constraintPtsContainer", listening="1")
        constraintPointsNode.addObject("PointSetTopologyModifier", name="constraintPtsModifier", listening="1")
        constraintPointsNode.addObject("MechanicalObject", template="Vec3d", showObject=True, showIndices=True,
                                    name="constraintPointsMo", position=[], showObjectScale=0, listening="1")
        constraintPointsNode.addObject('PointsManager', name="pointsManager", listening="1",
                                    beamPath="/solverNode/needle/rigidBase/cosseratInSofaFrameNode/slidingPoint"
                                                "/slidingPointMO")

        constraintPointsNode.addObject('BarycentricMapping', useRestPosition="false", listening="1")
    

        # @info : This is the constraint point that will be used to compute the distance between the needle and the volume
        conttactL = rootNode.addObject('ContactListener', name="contactListener",
                                        collisionModel1=cubeNode.gelNode.surfaceNode.surface.getLinkPath(),
                                        collisionModel2=needleCollisionModel.collisionStats.getLinkPath())

        # These stats will represents the distance between the contraint point in the volume and
        # their projection on the needle
        # It 's also important to say that the x direction is not taken into account
        distanceStatsNode = slidingPoint.addChild('distanceStatsNode')
        constraintPointsNode.addChild(distanceStatsNode)
        constraintPoinMo = distanceStatsNode.addObject('MechanicalObject', name="distanceStats", template="Vec3d",
                                                        position=[], listening="1", showObject="1", showObjectScale="0.1")
        inputVolumeMo = constraintPointsNode.constraintPointsMo.getLinkPath()
        inputNeedleMo = slidingPoint.slidingPointMO.getLinkPath()
        outputDistanceMo = distanceStatsNode.distanceStats.getLinkPath()

        # ---------------------------------------------------
        # @info: Start controller node
        rootNode.addObject(Animation(needle, conttactL, generic,
                            constraintPointsNode, rootNode, constraintPoinMo))

        distanceStatsNode.addObject(
            'CosseratNeedleSlidingConstraint', name="computeDistanceComponent")
        distanceStatsNode.addObject('DifferenceMultiMapping', name="pointsMulti", input1=inputVolumeMo, lastPointIsFixed=0,
                                    input2=inputNeedleMo, output=outputDistanceMo, direction="@../../FramesMO.position")
 
class Visualization():
    def __init__(self):
        self.plotter = pv.Plotter()
        self.needle_points = rootNode.solverNode.needle.rigidBase.cosseratInSofaFrameNode.needleCollision.beamContainer.edges.array()
        self.cube_points = rootNode.FemNode.gelNode.surfaceNode.surfContainer.triangles.array()

    def visualization_initial_needle(self):
        needle_positions_quat = rootNode.solverNode.needle.rigidBase.cosseratInSofaFrameNode.FramesMO.position.array()  #the position and orientation 
        needle_positions = needle_positions_quat[:, :3]  #just the position

        faces = np.hstack([[2, a, b] for a, b in self.needle_points]).astype(np.uint64)
        mesh = pv.PolyData()
        mesh.points = needle_positions
        mesh.lines = faces
        self.plotter.add_mesh(mesh, color="red", line_width=5, render_lines_as_tubes=True)

        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        modelNode.SetName("needle")
        modelNode.CreateDefaultDisplayNodes()
        modelNode.SetAndObservePolyData(mesh)

    def visualization_initial_cube(self):
        cube_positions = rootNode.FemNode.gelNode.surfaceNode.msSurface.position.array()
        faces = np.zeros((self.cube_points.shape[0], 4), dtype=np.uint64)
        faces[:, 1:] = self.cube_points
        faces[:, 0] = 3
        mesh = pv.PolyData(cube_positions, faces)
        self.plotter.add_mesh(mesh, color="red", opacity=0.5, show_edges=True, lighting=True)

        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        modelNode.SetName("cube")
        modelNode.CreateDefaultDisplayNodes()
        modelNode.SetAndObservePolyData(mesh)
        #instrumentModel = slicer.util.getNode('needle') 
        #instrumentModel.GetDisplayNode().SetOpacity(0.0)
    
    def update_visualization(self):
        needle_positions_quat=rootNode.solverNode.needle.rigidBase.cosseratInSofaFrameNode.FramesMO.position.array()
        needle_positions_write = needle_positions_quat[:, :3]   #here you access the positions of the points of the needle
        needle_positions = copy.deepcopy(needle_positions_write)

        cube_positions = rootNode.FemNode.gelNode.surfaceNode.msSurface.position.array()
        for i in range(16):
            needle_positions[i][1]+= 0.01  # if not, you can not see the needle (too thin)

        updated_position_arrays = [cube_positions, needle_positions]
        triangle_arrays = [self.cube_points, self.needle_points]

        for i, (position_array, triangle_array) in enumerate(zip(updated_position_arrays, triangle_arrays)):
            if i == 0:  # cube: triangles
                faces = np.zeros((triangle_array.shape[0], 4), dtype=np.uint64)
                faces[:, 1:] = triangle_array
                faces[:, 0] = 3
            else:  # needle: lines
                faces = np.hstack([[2, a, b] for a, b in triangle_array]).astype(np.uint64)

            new_mesh = pv.PolyData()
            new_mesh.points = position_array.copy()
            if i == 0:
                new_mesh.faces = faces
            else:
                new_mesh.lines = faces
            self.plotter.meshes[i].deep_copy(new_mesh)

class SOFASceneCreation():
    def __init__(self):
        self.RootNode = Sofa.Core.Node("root")
        self.solverNode = None
        self.generic = None

    def rootNode_creation(self):
        self.RootNode.addObject('RequiredPlugin', name='plugins', pluginName=[pluginList])
        plugin_list_2 = [
            "MultiThreading",  # Needed to use components [ParallelBVHNarrowPhase,ParallelBruteForceBroadPhase]
            "Sofa.Component.Constraint.Projective",  # Needed to use components [FixedProjectiveConstraint]
            "Sofa.Component.LinearSystem",
            "Sofa.Component.Constraint.Lagrangian.Model",
            "Sofa.Component.Topology.Container.Constant",
        ]
        for plugin in plugin_list_2:
            self.RootNode.addObject("RequiredPlugin", name=plugin)

        self.RootNode.addObject('VisualStyle', displayFlags='showVisualModels showBehaviorModels hideCollisionModels '
                                                        'hideBoundingCollisionModels hideForceFields '
                                                        'hideInteractionForceFields hideWireframe showMechanicalMappings')

        slicersofa.util.initPlugins(self.RootNode)

        self.RootNode.addObject('CollisionPipeline')
        self.RootNode.addObject("DefaultVisualManagerLoop")
        self.RootNode.addObject('RuleBasedContactManager',
                            responseParams='mu=0.1', response='FrictionContactConstraint')
        self.RootNode.addObject('BruteForceBroadPhase')
        self.RootNode.addObject('BVHNarrowPhase')
        self.RootNode.addObject('LocalMinDistance', name="Proximity", alarmDistance=0.5,
                            contactDistance=ContactParams.contactDistance,
                            coneFactor=ContactParams.coneFactor, angleCone=0.1)

        self.RootNode.addObject('FreeMotionAnimationLoop')
        self.generic = self.RootNode.addObject('GenericConstraintSolver', tolerance="1e-20",
                                        maxIterations="500", computeConstraintForces=1, printLog="0")

        gravity = [0, 0, 0]
        dt = 0.1
        self.RootNode.gravity.value = gravity
        self.RootNode.dt = dt

    def solverNode_creation(self):
        self.solverNode = self.RootNode.addChild('solverNode')
        self.solverNode.addObject('EulerImplicitSolver',
                                rayleighStiffness=PhysicsParams.rayleighStiffness)
        self.solverNode.addObject('SparseLDLSolver', name='solver', template="CompressedRowSparseMatrixd")
        self.solverNode.addObject('GenericConstraintCorrection')

def addConstraintPoint(parentNode, beamPath):
    constraintPointsNode = parentNode.addChild('constraintPoints')
    constraintPointsNode.addObject("PointSetTopologyContainer", name="constraintPtsContainer", listening="1")
    constraintPointsNode.addObject("PointSetTopologyModifier", name="constraintPtsModifier", listening="1")
    constraintPointsNode.addObject("MechanicalObject", template="Vec3d", showObject=True, showIndices=True,
                                   name="constraintPointsMo", position=[], showObjectScale=0, listening="1")

    constraintPointsNode.addObject('PointsManager', name="pointsManager", listening="1",
                                   beamPath="/solverNode/needle/rigidBase/cosseratInSofaFrameNode/slidingPoint"
                                            "/slidingPointMO")

    constraintPointsNode.addObject('BarycentricMapping', useRestPosition="false", listening="1")
    return constraintPointsNode

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

def imageDeformation_optimized(deformation_array, positions_array):
    global deformation_manager
    
    # Initialiser si nécessaire
    if not hasattr(imageDeformation_optimized, 'deformation_manager'):
        imageDeformation_optimized.deformation_manager = DeformationManager()
        imageDeformation_optimized.deformation_manager.initialize_deformation_system(
            deformation_array, positions_array)
    else:
        # Mise à jour rapide
        imageDeformation_optimized.deformation_manager.update_deformation(
            deformation_array, positions_array)


#Create the scene 
scene_creator = SOFASceneCreation() 

#Add the root node and the solver node
scene_creator.rootNode_creation()
rootNode = scene_creator.RootNode
generic = scene_creator.generic
scene_creator.solverNode_creation()
solverNode = scene_creator.solverNode

params = NeedleParameters()
nbFrames = GeometryParams.nbFrames
needleGeometryConfig = {'init_pos': [0., 0., 0.], 'tot_length': GeometryParams.totalLength,
                        'nbSectionS': GeometryParams.nbSections, 'nbFramesF': nbFrames,
                        'buildCollisionModel': 1, 'beamMass': PhysicsParams.mass}

slicer.app.processEvents()
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)

needle = Cosserat(parent=solverNode, cosseratGeometry=needleGeometryConfig, radius=GeometryParams.radius,
    name="needle", youngModulus=PhysicsParams.youngModulus, poissonRatio=PhysicsParams.poissonRatio,
    rayleighStiffness=PhysicsParams.rayleighStiffness)
needleCollisionModel = needle.addPointCollisionModel("needleCollision")
slidingPoint = needle.addSlidingPoints()

#Create the FEM cube based on the ROI boundaries
fem_creator = FEMVolumeCreator()
fem_creator.defineBoundaries()
cubeNode = fem_creator.createFemCubeWithParams(rootNode, FemParams)
fem_creator.addDependencies(cubeNode)

slicer.app.processEvents()
Sofa.Simulation.init(rootNode)

#Create the visualization of the needle and the cube
visualization_creator = Visualization()
visualization_creator.visualization_initial_needle()
visualization_creator.visualization_initial_cube()
  

def timeStep():
    Sofa.Simulation.animate(rootNode, rootNode.dt.value)
    with rootNode.solverNode.needle.rigidBase.RigidBaseMO.position.writeable() as posA:
        trans_node = slicer.util.getNode('ros2:tf2lookup:wristTotip')
        transformation_matrix_SOFA= slicer.util.arrayFromTransformMatrix(trans_node, toWorld = True)
        SOFA_to_SLICER = [
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0] ]
        correction_rotation = rotation_matrix_x(np.pi / 2)
        transformation_matrix_SLICER = transformation_matrix_SOFA @ SOFA_to_SLICER @ correction_rotation
        posA[0, 0:7] = transformation_matrix_to_rigid3d(transformation_matrix_SLICER)[0:7]

    visualization_creator.update_visualization()        
  
    slicer.app.processEvents() 
    point_position = rootNode.FemNode.gelNode.tetras.position.array().copy() 
    point_rest_position = rootNode.FemNode.gelNode.tetras.rest_position.array().copy()
    deformation_field = point_position - point_rest_position
    # imageDeformation_optimized(deformation_field, point_position)
    if simulating:
        qt.QTimer.singleShot(100, timeStep)
    
timeStep()
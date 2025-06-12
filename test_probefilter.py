# -*- coding: utf-8 -*-
"""Basic scene using Cosserat in SofaPython3.

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
import time 

import Sofa
import Sofa.Core
import Sofa.Simulation
import sys
sys.path.append('/home/slicer/SlicerSOFA/Release/SofaCosserat/examples/python3')

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

class DeformationManager:
    """Gestionnaire optimisé pour les déformations d'image en temps réel."""
    
    def __init__(self):
        self.transform_node = None
        self.grid_array = None
        self.probe_filter = None
        self.source_data = None
        self.vectors = None
        self.points = None
        self.initialized = False
        self.inProgress = False
    
 
    def initialize_deformation_system(self, initial_deformation_array, initial_positions_array):
        """Initialise le système de déformation une seule fois."""
        
        roi_markup_node = slicer.util.getNode("R")
        # target_node = slicer.util.getNode("liver")
        target_node = slicer.util.getNode("LiverSegmentation000-label")
        if target_node.IsA("vtkMRMLLabelMapVolumeNode"):
            print("Application de la déformation à un LabelMapVolume")
        elif target_node.IsA("vtkMRMLModelNode"):
            print("Application de la déformation à un Model")

        
        print("Initialisation du système de déformation...")
        
        # 1. Créer la structure de données source (réutilisable)
        self.points = vtk.vtkPoints()
        self.vectors = vtk.vtkFloatArray()
        self.vectors.SetNumberOfComponents(3)
        self.vectors.SetName("Displacement")
        
        num_points = initial_deformation_array.shape[0]
        
        # Initialiser avec les données initiales
        for i in range(num_points):
            x, y, z = initial_positions_array[i]
            self.points.InsertNextPoint(x, y, z)
            dx, dy, dz = initial_deformation_array[i]
            self.vectors.InsertNextTuple([dx, dy, dz])
        
        # Créer l'unstructured grid source (réutilisable)
        self.source_data = vtk.vtkPolyData()
        self.source_data.SetPoints(self.points)
        self.source_data.GetPointData().SetVectors(self.vectors)
        self.source_data.GetPointData().SetActiveVectors("Displacement")

        # 2. Créer la grille de destination (fixe)
        roi_bounds = [0] * 6
        roi_markup_node.GetRASBounds(roi_bounds)
        
        roi_size = [
            roi_bounds[1] - roi_bounds[0],
            roi_bounds[3] - roi_bounds[2], 
            roi_bounds[5] - roi_bounds[4]
        ]
        min_size = min(roi_size)
        grid_spacing = [min_size / 20.0] * 3
        grid_dims = [
            int(roi_size[0] / grid_spacing[0]) + 1,
            int(roi_size[1] / grid_spacing[1]) + 1,
            int(roi_size[2] / grid_spacing[2]) + 1
        ]
        
        bounds = [0, 0, 0, 0, 0, 0]
        self.source_data.GetBounds(bounds)
        print(f"BORNES des points source: {bounds}")
        print(f"BORNES du ROI: {roi_bounds}")
        # Créer la grille de probe (fixe)
        probe_grid = vtk.vtkImageData()
        probe_grid.SetDimensions(grid_dims)
        probe_grid.SetOrigin(roi_bounds[0], roi_bounds[2], roi_bounds[4])
        probe_grid.SetSpacing(grid_spacing)
        probe_grid.AllocateScalars(vtk.VTK_DOUBLE, 1)
        
        # 3. Créer le ProbeFilter (réutilisable)
        self.interpolator = vtk.vtkPointInterpolator()
        self.interpolator.SetInputData(probe_grid)
        self.interpolator.SetSourceData(self.source_data)
        
        # Utiliser un kernel gaussien avec un rayon adaptatif
        gaussian_kernel = vtk.vtkGaussianKernel()
        gaussian_kernel.SetSharpness(2.0)  # Ajustez la netteté
        gaussian_kernel.SetRadius(10.0)    # Rayon d'influence
        
        self.interpolator.SetKernel(gaussian_kernel)
        self.interpolator.SetLocator(vtk.vtkStaticPointLocator())  # Accélération
        self.interpolator.Update()

        print("Source bounds:", self.source_data.GetBounds())
        print("Probe grid bounds:",self.interpolator.GetInput().GetBounds()) 
        
        # 4. Récupérer les données interpolées
        probed_grid = self.interpolator.GetOutput()
        probe_vtk_array = probed_grid.GetPointData().GetArray("Displacement")
        if probe_vtk_array is None:
            print("Erreur: Pas de données de déplacement dans la grille interpolée")
            return None
        
        # Convertir en array NumPy
        probe_array = vtk.util.numpy_support.vtk_to_numpy(probe_vtk_array)
        print(np.max(probe_array))

        # IMPORTANT: VTK utilise l'ordre (Z,Y,X) alors que nous définissons (X,Y,Z)
        probe_array_shape = (grid_dims[2], grid_dims[1], grid_dims[0], 3)  # (Z,Y,X,3)
        probe_array = probe_array.reshape(probe_array_shape)
                
        # 4. Créer le nœud de transformation (unique)
        self.transform_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLGridTransformNode")
        self.transform_node.SetName("OptimizedDeformation")
        
        # Créer la grille de déplacement avec les bonnes dimensions
        displacement_grid = vtk.vtkImageData()
        displacement_grid.SetDimensions(grid_dims)
        displacement_grid.SetOrigin(roi_bounds[0], roi_bounds[2], roi_bounds[4])
        displacement_grid.SetSpacing(grid_spacing)
        displacement_grid.AllocateScalars(vtk.VTK_DOUBLE, 3)
        
        # Configurer la transformation
        grid_transform = vtk.vtkGridTransform()
        grid_transform.SetDisplacementGridData(displacement_grid)
        grid_transform.SetInterpolationModeToLinear()
        self.transform_node.SetAndObserveTransformFromParent(grid_transform)
        
        # Récupérer l'array de transformation (référence persistante)
        self.grid_array = slicer.util.arrayFromGridTransform(self.transform_node)
        
        # 5. Appliquer à l'objet cible
        target_node.SetAndObserveTransformNodeID(self.transform_node.GetID())
        transform_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTransformNode")

        target_node.HardenTransform()
        
        print(f"Système initialisé !")
        self.initialized = True
        # Première mise à jour
        self.update_deformation(initial_deformation_array, initial_positions_array)
        
    def update_deformation(self, deformation_array, positions_array):
        """Met à jour rapidement la déformation avec de nouvelles données."""
        
        if not self.initialized:
            print("Erreur: Système non initialisé. Appelez initialize_deformation_system() d'abord.")
            return
        
        # 1. Mettre à jour uniquement les vecteurs dans la structure existante
        num_points = deformation_array.shape[0]
        
        for i in range(num_points):
            x, y, z = positions_array[i]
            self.points.SetPoint(i, x, y, z)
            dx, dy, dz = deformation_array[i]
            self.vectors.SetTuple3(i, dx, dy, dz)
        
        # Marquer les données comme modifiées
        self.vectors.Modified()
        self.points.Modified()
        self.source_data.Modified()
       
        # print(f"Max déplacement dans vector_array: {np.max(self.vectors)}")
        # 2. Re-interpoler (rapide car structures déjà créées)
        self.interpolator.Update()    
        
        # 3. Récupérer et appliquer les nouvelles données : it feels like the output of the probe filter is always nul but the data in entry is correct !
        # probed_grid = self.probe_filter.GetOutput()
        probed_grid = self.interpolator.GetOutput()  #probed_grid est un vtk image data
        
        # print(f"Premier tuple dans probed_grid: {probed_grid.GetPointData().GetVectors().GetTuple3(0)}")
        # print(f"tuple random dans probed_grid: {probed_grid.GetPointData().GetVectors().GetTuple3(745)}")
        probe_vtk_array = probed_grid.GetPointData().GetArray("Displacement")
        print(f"Max déplacement dans grid_array: {np.max(probe_vtk_array)}")
        # print(f"Shape de grid_array: {np.shape(probe_vtk_array)}")
        
        if probe_vtk_array is not None:
            probe_array = vtk.util.numpy_support.vtk_to_numpy(probe_vtk_array)
            
            # Reshape selon les conventions VTK (Z,Y,X,3)
            grid_dims = self.grid_array.shape[:3]  # Récupérer les dimensions actuelles
            probe_array_shape = (grid_dims[0], grid_dims[1], grid_dims[2], 3)
            probe_array = probe_array.reshape(probe_array_shape)            
            # Appliquer directement à la grille existante
            self.grid_array[:] = -1.0 * probe_array
            
            slicer.util.arrayFromGridTransformModified(self.transform_node)
            slicer.app.processEvents()

def addConstraintPoint(parentNode, beamPath):
    constraintPointsNode = parentNode.addChild('constraintPoints')
    constraintPointsNode.addObject("PointSetTopologyContainer", name="constraintPtsContainer", listening="1")
    constraintPointsNode.addObject("PointSetTopologyModifier", name="constraintPtsModifier", listening="1")
    constraintPointsNode.addObject("MechanicalObject", template="Vec3d", showObject=True, showIndices=True,
                                   name="constraintPointsMo", position=[], showObjectScale=0, listening="1")

    # print(f' ====> The beamTip tip is : {dir(beamPath)}')
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

def createFemCubeWithParams(parentNode, geometry,minVol,maxVol):
    FemNode = parentNode.addChild("FemNode")

    gelVolume = FemNode.addChild("gelVolume")
    gelVolume.addObject("RegularGridTopology", name="HexaTop", n=geometry.mesh, min= minVol,max = maxVol)
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
    visu = surfaceNode.addChild("visu")

    visu.addObject("OglModel", name="Visual", src="@../surfContainer",  color="0.0 0.1 0.9 0.40" )
    visu.addObject("BarycentricMapping", input="@..", output="@Visual")

    gelNode.addObject('LinearSolverConstraintCorrection')
    print("all good")

    return FemNode

#those functions enable the creation of a cube at the updated ROI position (makes Slicer crashes for the moment)
def removeCubeAndDependencies(rootNode):
    try:
        FemNode = rootNode.getChild('FemNode')
        gelNode = cubeNode.getChild('gelNode')
        surfaceNode = gelNode.getChild("surfaceNode")
    except:
        print("[INFO] Aucun cubeNode à supprimer.")
        return

    # --- Supprimer les objets de distanceStatsNode ---
    distanceStatsNode = slidingPoint.getChild("distanceStatsNode")
    if distanceStatsNode:
    # Supprimer les objets un par un après les avoir récupérés
        computeDistanceComponent = distanceStatsNode.getObject("computeDistanceComponent")
    if computeDistanceComponent:
        distanceStatsNode.removeObject(computeDistanceComponent)

    pointsMulti = distanceStatsNode.getObject("pointsMulti")
    if pointsMulti:
        distanceStatsNode.removeObject(pointsMulti)

    distanceStats = distanceStatsNode.getObject("distanceStats")
    if distanceStats:
        distanceStatsNode.removeObject(distanceStats)
        print("bien supp")


  
    slidingPoint.removeChild(distanceStatsNode)

    # --- Supprimer le ContactListener ---
    contactListener = rootNode.getObject("contactListener")
    if contactListener:
        rootNode.removeObject(contactListener)

    # --- Supprimer constraintPointsNode et tous ses objets automatiquement ---
    gelNode = cubeNode.getChild("gelNode")
    constraintPointsNode = gelNode.getChild("constraintPoints")
    if constraintPointsNode:
        gelNode.removeChild(constraintPointsNode)
        print("bien supp 2")

    # --- Supprimer le contrôleur (Animation) ---
    for obj in rootNode.objects:
        if obj.__class__.__name__ == "Animation":  # ou remplace par nom explicite si disponible
            rootNode.removeObject(obj)
    print("bien supp 3")
    print("bien supp 4")
    gelNode.removeChild("surfaceNode")
    print("bien supp 3")
    # FemNode.removeChild("gelVolume")
    # FemNode.removeChild("GelSurface")
    FemNode.removeChild("gelNode")
    print("bien supp 4")
    rootNode.removeChild("FemNode")
    print("tout bien supp")

def mapGelCubetoROI():
    roiNode = slicer.util.getNode("R")  
   # Récupération du centre et de la taille
    center = [0.0, 0.0, 0.0]
    roiNode.GetCenter(center)

    xyz = [0,0,0]
    radius = [0,0,0]
    roiNode.GetXYZ(xyz)
    roiNode.GetRadiusXYZ(radius)


    # Calcul des points min et max (en coordonnées RAS)
    minPoint = [center[i] - radius[i] for i in range(3)]
    maxPoint = [center[i] + radius[i] for i in range(3)]

    last_radius= radius

    minPoint2= " ".join(f"{v:.2f}" for v in minPoint)
    maxPoint2= " ".join(f"{v:.2f}" for v in maxPoint)
    removeCubeAndDependencies(rootNode)
    newcubeNode = createFemCubeWithParams(rootNode,FemParams,minPoint2,maxPoint2)
    addDependencies(newcubeNode)
    print("job done ! cube has been replaced")


#the  mapGelCubetoROI function is completed but there is still a problem afterwards
def addDependencies(cubeNode):
    gelNode = cubeNode.getChild('gelNode')
    # constraintPointNode = addConstraintPoint(
    # gelNode, slidingPoint.getLinkPath())
    constraintPointsNode = gelNode.addChild('constraintPoints')
    constraintPointsNode.addObject("PointSetTopologyContainer", name="constraintPtsContainer", listening="1")
    constraintPointsNode.addObject("PointSetTopologyModifier", name="constraintPtsModifier", listening="1")
    constraintPointsNode.addObject("MechanicalObject", template="Vec3d", showObject=True, showIndices=True,
                                   name="constraintPointsMo", position=[], showObjectScale=0, listening="1")
    constraintPointsNode.addObject('PointsManager', name="pointsManager", listening="1",
                                   beamPath="/solverNode/needle/rigidBase/cosseratInSofaFrameNode/slidingPoint"
                                            "/slidingPointMO")

    constraintPointsNode.addObject('BarycentricMapping', useRestPosition="false", listening="1")
   
    print("the depencies have been added 2")

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
    print("the depencies have been added 3")

    # ---------------------------------------------------
    # @info: Start controller node
    rootNode.addObject(Animation(needle, conttactL, generic,
                        constraintPointsNode, rootNode, constraintPoinMo))

    distanceStatsNode.addObject(
        'CosseratNeedleSlidingConstraint', name="computeDistanceComponent")
    distanceStatsNode.addObject('DifferenceMultiMapping', name="pointsMulti", input1=inputVolumeMo, lastPointIsFixed=0,
                                input2=inputNeedleMo, output=outputDistanceMo, direction="@../../FramesMO.position")
    print("the depencies have been added 4")

def addConstraintPoint(parentNode, beamPath):
    constraintPointsNode = parentNode.addChild('constraintPoints')
    constraintPointsNode.addObject("PointSetTopologyContainer", name="constraintPtsContainer", listening="1")
    constraintPointsNode.addObject("PointSetTopologyModifier", name="constraintPtsModifier", listening="1")
    constraintPointsNode.addObject("MechanicalObject", template="Vec3d", showObject=True, showIndices=True,
                                   name="constraintPointsMo", position=[], showObjectScale=0, listening="1")

    # print(f' ====> The beamTip tip is : {dir(beamPath)}')
    constraintPointsNode.addObject('PointsManager', name="pointsManager", listening="1",
                                   beamPath="/solverNode/needle/rigidBase/cosseratInSofaFrameNode/slidingPoint"
                                            "/slidingPointMO")

    constraintPointsNode.addObject('BarycentricMapping', useRestPosition="false", listening="1")
    return constraintPointsNode


def imageDeformation_optimized(deformation_array, positions_array):
    """Version optimisée qui utilise le gestionnaire de déformation."""
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



rootNode = Sofa.Core.Node("root")


params = NeedleParameters()


nbFrames = GeometryParams.nbFrames
needleGeometryConfig = {'init_pos': [0., 0., 0.], 'tot_length': GeometryParams.totalLength,
                        'nbSectionS': GeometryParams.nbSections, 'nbFramesF': nbFrames,
                        'buildCollisionModel': 1, 'beamMass': PhysicsParams.mass}


dt = 0.1
slicer.app.processEvents()
slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutOneUp3DView)


rootNode.addObject('RequiredPlugin', name='plugins', pluginName=[pluginList])

plugin_list = [
    "MultiThreading",  # Needed to use components [ParallelBVHNarrowPhase,ParallelBruteForceBroadPhase]
    "Sofa.Component.AnimationLoop",  # Needed to use components [FreeMotionAnimationLoop]
    "Sofa.Component.Collision.Detection.Algorithm",  # Needed to use components [CollisionPipeline]
    "Sofa.Component.Collision.Detection.Intersection",  # Needed to use components [MinProximityIntersection]
    "Sofa.Component.Collision.Geometry",  # Needed to use components [TriangleCollisionModel]
    "Sofa.Component.Collision.Response.Contact",  # Needed to use components [CollisionResponse]
    "Sofa.Component.Constraint.Lagrangian.Correction",  # Needed to use components [LinearSolverConstraintCorrection]
    "Sofa.Component.Constraint.Lagrangian.Solver",  # Needed to use components [GenericConstraintSolver]
    "Sofa.Component.IO.Mesh",  # Needed to use components [MeshGmshLoader]
    "Sofa.Component.LinearSolver.Direct",  # Needed to use components [SparseLDLSolver]
    "Sofa.Component.Mass",  # Needed to use components [UniformMass]
    "Sofa.Component.MechanicalLoad",  # Needed to use components [PlaneForceField]
    "Sofa.Component.ODESolver.Backward",  # Needed to use components [EulerImplicitSolver]
    "Sofa.Component.SolidMechanics.FEM.Elastic",  # Needed to use components [TetrahedralCorotationalFEMForceField]
    "Sofa.Component.StateContainer",  # Needed to use components [MechanicalObject]
    "Sofa.Component.Topology.Container.Dynamic",  # Needed to use components [HexahedronSetTopologyContainer,HexahedronSetTopologyModifier,TetrahedronSetTopology
    "Sofa.Component.Topology.Mapping",  # Needed to use components [Tetra2TriangleTopologicalMapping]
    "Sofa.Component.Visual",  # Needed to use components [VisualStyle]
    "Sofa.Component.Constraint.Projective",  # Needed to use components [FixedProjectiveConstraint]
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
    "Sofa.Component.Topology.Container.Grid"
]
for plugin in plugin_list:
    rootNode.addObject("RequiredPlugin", name=plugin)



rootNode.addObject('VisualStyle', displayFlags='showVisualModels showBehaviorModels hideCollisionModels '
                                                'hideBoundingCollisionModels hideForceFields '
                                                'hideInteractionForceFields hideWireframe showMechanicalMappings')

slicersofa.util.initPlugins(rootNode)

rootNode.addObject('CollisionPipeline')
rootNode.addObject("DefaultVisualManagerLoop")
rootNode.addObject('RuleBasedContactManager',
                    responseParams='mu=0.1', response='FrictionContactConstraint')
rootNode.addObject('BruteForceBroadPhase')
rootNode.addObject('BVHNarrowPhase')
# rootNode.addObject('LocalMinDistance', alarmDistance=1.0, contactDistance=0.01)
rootNode.addObject('LocalMinDistance', name="Proximity", alarmDistance=0.5,
                    contactDistance=ContactParams.contactDistance,
                    coneFactor=ContactParams.coneFactor, angleCone=0.1)

rootNode.addObject('FreeMotionAnimationLoop')
generic = rootNode.addObject('GenericConstraintSolver', tolerance="1e-20",
                                maxIterations="500", computeConstraintForces=1, printLog="0")

gravity = [0, 0, 0]
rootNode.gravity.value = gravity
rootNode.dt = dt
rootNode.addObject('BackgroundSetting', color='0 0.168627 0.211765')
# rootNode.addObject('OglSceneFrame', style="Arrows", alignment="TopRight")
# ###############
# New adds to use the sliding Actuator
###############
solverNode = rootNode.addChild('solverNode')
solverNode.addObject('EulerImplicitSolver',
                        rayleighStiffness=PhysicsParams.rayleighStiffness)
solverNode.addObject('SparseLDLSolver', name='solver', template="CompressedRowSparseMatrixd")
solverNode.addObject('GenericConstraintCorrection')

needle = Cosserat(parent=solverNode, cosseratGeometry=needleGeometryConfig, radius=GeometryParams.radius,
    name="needle", youngModulus=PhysicsParams.youngModulus, poissonRatio=PhysicsParams.poissonRatio,
    rayleighStiffness=PhysicsParams.rayleighStiffness)
needleCollisionModel = needle.addPointCollisionModel("needleCollision")

# These state is mapped on the needle and used to compute the distance between the needle and the
# FEM constraint points
slidingPoint = needle.addSlidingPoints()

# -----------------
# Start the volume definition
# -----------------
roiNode = slicer.util.getNode("R")  
   # Récupération du centre et de la taille
center = [0.0, 0.0, 0.0]
roiNode.GetCenter(center)

xyz = [0,0,0]
radius = [0,0,0]
roiNode.GetXYZ(xyz)
roiNode.GetRadiusXYZ(radius)


# Calcul des points min et max (en coordonnées RAS)
minPoint = [center[i] - radius[i] for i in range(3)]
maxPoint = [center[i] + radius[i] for i in range(3)]

last_radius= radius

minPoint2= " ".join(f"{v:.2f}" for v in minPoint)
maxPoint2= " ".join(f"{v:.2f}" for v in maxPoint)


cubeNode = createFemCubeWithParams(rootNode, FemParams,minPoint2,maxPoint2)
addDependencies(cubeNode)
# FEM constraint points


slicer.app.processEvents()
Sofa.Simulation.init(rootNode)

needle_positions_quat = rootNode.solverNode.needle.rigidBase.cosseratInSofaFrameNode.FramesMO.position.array()
# needle_positions = needle_positions_quat[:, :3].copy()
needle_positions = needle_positions_quat[:, :3]

# Similarly you can access the topology of the object
needle_points = rootNode.solverNode.needle.rigidBase.cosseratInSofaFrameNode.needleCollision.beamContainer.edges.array()

# print(needle_points)
cube_positions = rootNode.FemNode.gelNode.surfaceNode.msSurface.position.array()
cube_points = rootNode.FemNode.gelNode.surfaceNode.surfContainer.triangles.array()
# root.scene.liver.collision.TriangleSetTopologyContainer.triangles.array()
if PYVISTA:
    for position_array, triangle_array, color, name in zip([needle_positions], [needle_points], ["red"], ['needle']):
        faces = np.hstack([[2, a, b] for a, b in triangle_array]).astype(np.uint64)
        mesh = pv.PolyData()
        mesh.points = position_array
        mesh.lines = faces
        plotter = pv.Plotter()
        plotter.add_mesh(mesh, color="red", line_width=5, render_lines_as_tubes=True)
        
        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        modelNode.SetName(name)
        modelNode.CreateDefaultDisplayNodes()
        modelNode.SetAndObservePolyData(mesh)

if PYVISTA:
    for position_array, triangle_array, color, name in zip([cube_positions], [cube_points], ["red"], ['cube']):
        faces = np.zeros((triangle_array.shape[0], 4), dtype=np.uint64)
        faces[:, 1:] = triangle_array
        faces[:, 0] = 3
        mesh = pv.PolyData(position_array, faces)
        plotter.add_mesh(mesh, color=color, opacity=0.5, show_edges=True, lighting=True)

        modelNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLModelNode")
        modelNode.SetName(name)
        modelNode.CreateDefaultDisplayNodes()
        modelNode.SetAndObservePolyData(mesh)
    # instrumentModel = slicer.util.getNode('needle')
    # instrumentModel.GetDisplayNode().SetOpacity(0.0)

    needle_positions_quat=rootNode.solverNode.needle.rigidBase.cosseratInSofaFrameNode.FramesMO.position.array()
    needle_positions_write = needle_positions_quat[:, :3]
    needle_positions = copy.deepcopy(needle_positions_write)
    for i in range(16):
        needle_positions[i][1]+= needle_positions[i][0]*0.01
    

#Initialisation du pipeline de deformation d'image
point_position = rootNode.FemNode.gelNode.tetras.position.array().copy() 
point_rest_position = rootNode.FemNode.gelNode.tetras.rest_position.array().copy()
initial_deformation_array = point_position - point_rest_position



def timeStep():
    Sofa.Simulation.animate(rootNode, rootNode.dt.value)
    radius = [0,0,0]
    roiNode.GetRadiusXYZ(radius)
    if radius != last_radius:
        print("changement de ROI ")
        time.sleep(0.5)
        mapGelCubetoROI()


    radius = last_radius
    with rootNode.solverNode.needle.rigidBase.RigidBaseMO.rest_position.writeable() as posA:
        trans_node = slicer.util.getNode('ros2:tf2lookup:wristTotip')
        transformation_matrix_SOFA= slicer.util.arrayFromTransformMatrix(trans_node, toWorld = True)
        SOFA_to_SLICER = [
            [-1.0, 0.0, 0.0, -100.0],
            [0.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0] ]
        correction_rotation = rotation_matrix_x(np.pi / 2)
        transformation_matrix_SLICER = transformation_matrix_SOFA @ SOFA_to_SLICER @ correction_rotation
        posA[0, 0:7] = transformation_matrix_to_rigid3d(transformation_matrix_SLICER)[0:7]
    
    needle_positions_quat=rootNode.solverNode.needle.rigidBase.cosseratInSofaFrameNode.FramesMO.position.array()
    needle_positions_write = needle_positions_quat[:, :3]
    needle_positions = copy.deepcopy(needle_positions_write)

    cube_positions = rootNode.FemNode.gelNode.surfaceNode.msSurface.position.array()
    cube_points = rootNode.FemNode.gelNode.surfaceNode.surfContainer.triangles.array()
    needle_points = rootNode.solverNode.needle.rigidBase.cosseratInSofaFrameNode.needleCollision.beamContainer.edges.array()

    for i in range(16):
        needle_positions[i][1]+= 0.01
    if PYVISTA:
        # Rebuild and deep copy new meshes with updated points
        updated_position_arrays = [cube_positions, needle_positions]
        triangle_arrays = [cube_points, needle_points]

        for i, (position_array, triangle_array) in enumerate(zip(updated_position_arrays, triangle_arrays)):
            # Create a fresh PolyData
            if i == 0:  # cube: triangles
                faces = np.zeros((triangle_array.shape[0], 4), dtype=np.uint64)
                faces[:, 1:] = triangle_array
                faces[:, 0] = 3
            else:  # needle: lines
                faces = np.hstack([[2, a, b] for a, b in triangle_array]).astype(np.uint64)

            new_mesh = pv.PolyData()
            new_mesh.points = position_array.copy()  # ensure it's a new buffer
            if i == 0:
                new_mesh.faces = faces
            else:
                new_mesh.lines = faces

            # Update the plotter mesh in-place to avoid memory corruption
            plotter.meshes[i].deep_copy(new_mesh)
        
    slicer.app.processEvents() 
    point_position = rootNode.FemNode.gelNode.tetras.position.array().copy() 
    point_rest_position = rootNode.FemNode.gelNode.tetras.rest_position.array().copy()
    deformation_field = point_position - point_rest_position
    # imageDeformationv2(deformation_field, point_position)
    # imageDeformation_optimized(deformation_field, point_position)
    if simulating:
        qt.QTimer.singleShot(100, timeStep)
    



timeStep()

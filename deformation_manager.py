import slicer
import vtk 


class DeformationManager:
    
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
        #Initialise the deformation system, is to be called just once 
        
        roi_markup_node = slicer.util.getNode("R")
        target_node = slicer.util.getNode("liver")
        
        print("Initialising...")
        
        self.points = vtk.vtkPoints()
        self.vectors = vtk.vtkFloatArray()
        self.vectors.SetNumberOfComponents(3)
        self.vectors.SetName("Displacement")
        
        num_points = initial_deformation_array.shape[0]
        
        #Fill the vtk type data
        for i in range(num_points):
            x, y, z = initial_positions_array[i]
            self.points.InsertNextPoint(x, y, z)
            dx, dy, dz = initial_deformation_array[i]
            self.vectors.InsertNextTuple([dx, dy, dz])
        
        #Create the PolyData
        self.source_data = vtk.vtkPolyData()
        self.source_data.SetPoints(self.points)
        self.source_data.GetPointData().SetVectors(self.vectors)
        self.source_data.GetPointData().SetActiveVectors("Displacement")

        # Create the probe grid according to the ROI dimension
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
        

        probe_grid = vtk.vtkImageData()
        probe_grid.SetDimensions(grid_dims)
        probe_grid.SetOrigin(roi_bounds[0], roi_bounds[2], roi_bounds[4])
        probe_grid.SetSpacing(grid_spacing)
        probe_grid.AllocateScalars(vtk.VTK_DOUBLE, 1)
        
        # Create the Point Interpolator 
        self.interpolator = vtk.vtkPointInterpolator()
        self.interpolator.SetInputData(probe_grid)
        self.interpolator.SetSourceData(self.source_data)
        
        #Point Interpolator parameters
        gaussian_kernel = vtk.vtkGaussianKernel()
        gaussian_kernel.SetSharpness(2.0)  
        gaussian_kernel.SetRadius(10.0)    
        
        self.interpolator.SetKernel(gaussian_kernel)
        self.interpolator.SetLocator(vtk.vtkStaticPointLocator())  
        self.interpolator.Update()

        print("Source bounds:", self.source_data.GetBounds())
        print("Probe grid bounds:",self.interpolator.GetInput().GetBounds()) 
        
        # Work with the output of Point Interpolator
        probed_grid = self.interpolator.GetOutput()
        probe_vtk_array = probed_grid.GetPointData().GetArray("Displacement")
        if probe_vtk_array is None:
            print("Erreur: Pas de données de déplacement dans la grille interpolée")
            return None
        
        probe_array = vtk.util.numpy_support.vtk_to_numpy(probe_vtk_array)

        # IMPORTANT: VTK use (Z,Y,X) but we use  (X,Y,Z)
        probe_array_shape = (grid_dims[2], grid_dims[1], grid_dims[0], 3)  # (Z,Y,X,3)
        probe_array = probe_array.reshape(probe_array_shape)
                
        # Create the transform node
        self.transform_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLGridTransformNode")
        self.transform_node.SetName("OptimizedDeformation")
        
        # Créer the displacement grid with good dimensions
        displacement_grid = vtk.vtkImageData()
        displacement_grid.SetDimensions(grid_dims)
        displacement_grid.SetOrigin(roi_bounds[0], roi_bounds[2], roi_bounds[4])
        displacement_grid.SetSpacing(grid_spacing)
        displacement_grid.AllocateScalars(vtk.VTK_DOUBLE, 3)
        
        grid_transform = vtk.vtkGridTransform()
        grid_transform.SetDisplacementGridData(displacement_grid)
        grid_transform.SetInterpolationModeToLinear()
        self.transform_node.SetAndObserveTransformFromParent(grid_transform)
        self.grid_array = slicer.util.arrayFromGridTransform(self.transform_node)
        
        target_node.SetAndObserveTransformNodeID(self.transform_node.GetID())
        
        print(f"The system has been initialized !")
        self.initialized = True
        self.update_deformation(initial_deformation_array, initial_positions_array)
        
    def update_deformation(self, deformation_array, positions_array):
        
        if not self.initialized:
            print("Error : you first have to initialize the system")
            return
        
        num_points = deformation_array.shape[0]
        
        for i in range(num_points):
            x, y, z = positions_array[i]
            self.points.SetPoint(i, x, y, z)
            dx, dy, dz = deformation_array[i]
            self.vectors.SetTuple3(i, dx, dy, dz)
        
        self.vectors.Modified()
        self.points.Modified()
        self.source_data.Modified()
 
        self.interpolator.Update()    
        probed_grid = self.interpolator.GetOutput() 
        probe_vtk_array = probed_grid.GetPointData().GetArray("Displacement")
        if probe_vtk_array is not None:
            probe_array = vtk.util.numpy_support.vtk_to_numpy(probe_vtk_array)
            
            grid_dims = self.grid_array.shape[:3]  
            probe_array_shape = (grid_dims[0], grid_dims[1], grid_dims[2], 3)
            probe_array = probe_array.reshape(probe_array_shape)            
            self.grid_array[:] = -1.0 * probe_array
            
            slicer.util.arrayFromGridTransformModified(self.transform_node)
            slicer.app.processEvents()

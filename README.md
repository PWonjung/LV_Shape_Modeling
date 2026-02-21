This is official code for "LV-Net: Anatomy-aware lateral ventricle shape modeling"


# Generate target point cloud
Run ```extract_target_pcd.ipynb``` for the brain subregion mask folder (registered to MNI space) including the classes of LV, hippocampus, Thalamus, Caudate, and Opposite LV produced by synthseg for left lateral ventricle modeling.

c.f. For right lateral ventricle, flip the image to the x-axis.

c.f. You can see the template mesh in ```template_mesh_viewer.ipynb```


# Optimization
Run ```run_optimization.ipynb``` for the LV optimization example.
1. Scaling the template mesh
2. Deformable optimization (coarse(1000 iter) --> fine(+4000 iteration))






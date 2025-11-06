# Shearing Box Modification

Done:
  * Boundary Conditions (TBC)
  * Forcing
  * Background Shear Flow
  * Update Time Within Step Function

ToDo:
  * Kida Vortex: compute from stream line
  * Poisson Solver: runtime not optimal, check whether it enforces incompressibility by computing div v
  * Determine Timestep Size (less important)

ToCheck:
  * bc issue potentially arising from finite difference schemes? incorrect interaction with ghost cells
"""
Example 3: Array-of-Structs (AoS) with NumPy Structured Dtypes

Demonstrates how to use structured types for storing multiple values
per grid point. Common in scientific simulations (fluid dynamics, particles).
"""

import numpy as np
import cnda

def main():
    print("=" * 60)
    print("Example 3: Array-of-Structs (AoS)")
    print("=" * 60)
    
    # Define structured dtype for fluid cell
    print("\n1. Define structured dtype (matches C++ struct)")
    cell_dtype = np.dtype([
        ('u', '<f4'),      # velocity x (float32)
        ('v', '<f4'),      # velocity y (float32)
        ('flag', '<i4')    # status flag (int32)
    ], align=True)
    
    print(f"Fields: {cell_dtype.names}")
    print(f"Total size: {cell_dtype.itemsize} bytes")
    
    # Verify field offsets
    print("\nField layout:")
    for name in cell_dtype.names:
        offset = cell_dtype.fields[name][1]
        print(f"{name}: offset {offset} bytes")
    
    # Create structured array (64×64 fluid grid)
    print("\n2. Create 64×64 fluid simulation grid")
    grid = np.zeros((64, 64), dtype=cell_dtype, order='C')
    print(f"Shape: {grid.shape}")
    print(f"C-contiguous: {grid.flags['C_CONTIGUOUS']}")
    
    # Initialize boundary conditions
    print("\n3. Set boundary conditions")
    grid['flag'][0, :] = -1    # Bottom wall
    grid['flag'][-1, :] = -1   # Top wall
    grid['flag'][:, 0] = -1    # Left wall
    grid['flag'][:, -1] = -1   # Right wall
    
    # Initialize fluid region
    grid['u'][1:-1, 1:-1] = 1.0    # Uniform flow
    grid['v'][1:-1, 1:-1] = 0.0
    grid['flag'][1:-1, 1:-1] = 1   # Fluid cells
    
    boundary_count = np.sum(grid['flag'] == -1)
    fluid_count = np.sum(grid['flag'] == 1)
    print(f"Boundary cells: {boundary_count}")
    print(f"Fluid cells: {fluid_count}")
    
    # Convert to CNDA (zero-copy if struct layout matches)
    print("\n4. Convert to CNDA")
    try:
        cnda_grid = cnda.from_numpy(grid, copy=False)
        print("Zero-copy successful!")
        print("→ Python and C++ share same memory")
    except (TypeError, ValueError) as e:
        print(f"Zero-copy failed: {e}")
        print("→ Using explicit copy instead")
        cnda_grid = cnda.from_numpy(grid, copy=True)
    
    # Export back to NumPy
    result = cnda_grid.to_numpy(copy=False)
    
    # Modify via result (simulating C++ computation)
    print("\n5. Simulate one advection step (Python)")
    result['u'][2:-2, 2:-2] += 0.1 * (
        result['u'][3:-1, 2:-2] - result['u'][1:-3, 2:-2]
    )
    
    # Statistics
    print("\n6. Flow statistics:")
    fluid_mask = result['flag'] == 1
    print(f"   Mean u velocity: {result['u'][fluid_mask].mean():.6f}")
    print(f"   Max u velocity: {result['u'][fluid_mask].max():.6f}")
    print(f"   Mean v velocity: {result['v'][fluid_mask].mean():.6f}")
    
    # Visualize velocity field
    print("\n7. Visualizing velocity field...")
    import matplotlib.pyplot as plt
    
    u_velocity = result['u']
    v_velocity = result['v']
    speed = np.sqrt(u_velocity**2 + v_velocity**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Velocity magnitude
    im0 = axes[0].imshow(speed, cmap='viridis', origin='lower')
    axes[0].set_title('Velocity Magnitude')
    plt.colorbar(im0, ax=axes[0])
    
    # Velocity field (quiver)
    step = 4
    X, Y = np.meshgrid(np.arange(0, 64, step), np.arange(0, 64, step))
    axes[1].quiver(X, Y, u_velocity[::step, ::step], v_velocity[::step, ::step])
    axes[1].set_title('Velocity Field')
    axes[1].set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig('aos_velocity_field.png', dpi=150, bbox_inches='tight')
    print("Saved to: aos_velocity_field.png")
    
    print("\n AoS allows storing multiple values per grid point efficiently!")
    print("Perfect for simulations where each cell has multiple properties.")

if __name__ == "__main__":
    main()

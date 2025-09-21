# python3 plot_heatmap.py phys_field_2D.txt --save-svg
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plot_heatmap_from_file(filename, save_svg=False):
    """
    Plot heatmap from field data file
    """
    try:
        # Read data from file
        data = []
        with open(filename, 'r') as file:
            for line in file:
                # Skip comments and empty lines
                if line.strip() and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        data.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
        if not data:
            print("File contains no data")
            return
        
        # Convert to numpy array
        data = np.array(data)
        x = data[:, 0]
        y = data[:, 1]
        values = data[:, 2]
        
        # Determine grid dimensions
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        nx = len(unique_x)
        ny = len(unique_y)
        
        # Create 2D array for heatmap
        heatmap_data = np.zeros((ny, nx))
        
        # Fill array with values
        for i, (xi, yi, value) in enumerate(data):
            x_idx = np.where(unique_x == xi)[0][0]
            y_idx = np.where(unique_y == yi)[0][0]
            heatmap_data[y_idx, x_idx] = value
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        im = ax.imshow(heatmap_data, 
                      extent=[x.min(), x.max(), y.min(), y.max()],
                      origin='lower',
                      cmap='viridis',
                      aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Field value', fontsize=12)
        
        # Plot settings
        ax.set_xlabel('X coordinate', fontsize=12)
        ax.set_ylabel('Y coordinate', fontsize=12)
        ax.set_title('Physical Field Heatmap', fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save SVG if requested
        if save_svg:
            svg_filename = os.path.splitext(filename)[0] + '_heatmap.svg'
            plt.savefig(svg_filename, format='svg', dpi=300, bbox_inches='tight')
            print(f"SVG saved as: {svg_filename}")
        
        plt.show()
        
        print(f"Heatmap created successfully")
        print(f"Grid size: {nx} x {ny}")
        print(f"Value range: {values.min():.3f} - {values.max():.3f}")
        
    except FileNotFoundError:
        print(f"File {filename} not found")
    except Exception as e:
        print(f"Error processing file: {e}")

# Alternative version using pcolormesh (more accurate display)
def plot_heatmap_pcolormesh(filename, save_svg=False):
    """
    Plot heatmap using pcolormesh
    """
    try:
        # Read data from file
        data = []
        with open(filename, 'r') as file:
            for line in file:
                if line.strip() and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 3:
                        data.append([float(parts[0]), float(parts[1]), float(parts[2])])
        
        if not data:
            print("File contains no data")
            return
        
        data = np.array(data)
        x = data[:, 0]
        y = data[:, 1]
        values = data[:, 2]
        
        # Create coordinate grid
        unique_x = np.unique(x)
        unique_y = np.unique(y)
        
        X, Y = np.meshgrid(unique_x, unique_y)
        
        # Reshape values to 2D array
        Z = values.reshape(len(unique_y), len(unique_x))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap with pcolormesh
        im = ax.pcolormesh(X, Y, Z, 
                          cmap='viridis', 
                          shading='auto')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Field value', fontsize=12)
        
        # Settings
        ax.set_xlabel('X coordinate', fontsize=12)
        ax.set_ylabel('Y coordinate', fontsize=12)
        ax.set_title('Heatmap (pcolormesh)', fontsize=14)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save SVG if requested
        if save_svg:
            svg_filename = os.path.splitext(filename)[0] + '_pcolormesh.svg'
            plt.savefig(svg_filename, format='svg', dpi=300, bbox_inches='tight')
            print(f"SVG saved as: {svg_filename}")
        
        plt.show()
        
    except Exception as e:
        print(f"Error: {e}")

# Function to read metadata from file
def read_metadata(filename):
    """
    Read metadata from file header
    """
    metadata = {}
    try:
        with open(filename, 'r') as file:
            for line in file:
                if line.startswith('#'):
                    if 'Grid dimensions' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            dims = parts[1].strip().split('x')
                            metadata['Nx'] = int(dims[0].strip())
                            metadata['Ny'] = int(dims[1].strip())
                    elif 'Grid spacing' in line:
                        parts = line.split(':')
                        if len(parts) > 1:
                            spacing = parts[1].split(',')
                            metadata['hx'] = float(spacing[0].split('=')[1].strip())
                            metadata['hy'] = float(spacing[1].split('=')[1].strip())
                else:
                    break
        return metadata
    except:
        return {}

def print_usage():
    """Print usage information"""
    print("Usage: python plot_heatmap.py <filename> [--save-svg]")
    print("  filename: path to the data file (e.g., phys_field_2D.txt)")
    print("  --save-svg: optional flag to save plot as SVG file")
    print("\nExamples:")
    print("  python plot_heatmap.py phys_field_2D.txt")
    print("  python plot_heatmap.py data/my_field_data.txt --save-svg")
    print("  python plot_heatmap.py /path/to/your/file.txt --save-svg")

# Main program
if __name__ == "__main__":
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Error: Please specify the input file")
        print_usage()
        sys.exit(1)
    
    filename = sys.argv[1]
    save_svg = False
    
    # Check for optional flags
    if len(sys.argv) > 2:
        for arg in sys.argv[2:]:
            if arg == '--save-svg':
                save_svg = True
            else:
                print(f"Warning: Unknown argument '{arg}'")
    
    # Check if file exists
    if not os.path.isfile(filename):
        print(f"Error: File '{filename}' does not exist")
        print_usage()
        sys.exit(1)
    
    print(f"Processing file: {filename}")
    if save_svg:
        print("SVG export: enabled")
    
    # Read metadata
    metadata = read_metadata(filename)
    if metadata:
        print("File metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    else:
        print("No metadata found in file header")
    
    # Plot heatmap
    plot_heatmap_from_file(filename, save_svg)
    
    # Alternative version (uncomment if needed)
    # plot_heatmap_pcolormesh(filename, save_svg)
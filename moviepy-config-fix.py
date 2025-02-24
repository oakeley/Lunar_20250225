#!/usr/bin/env python3
"""
This script fixes MoviePy's ImageMagick configuration.
Run it after installing ImageMagick to ensure MoviePy can find the executable.
"""
import os
import sys
import subprocess
from pathlib import Path

def find_imagemagick():
    """Find the ImageMagick executable on the system"""
    # Check common paths for convert or magick
    possible_commands = ["convert", "magick"]
    
    # Try to find the command in PATH
    for cmd in possible_commands:
        try:
            result = subprocess.run(["which", cmd], capture_output=True, text=True)
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip()
        except FileNotFoundError:
            pass
    
    # On Windows, check common installation directories
    if sys.platform == "win32":
        common_paths = [
            "C:\\Program Files\\ImageMagick-7.X\\magick.exe",
            "C:\\Program Files\\ImageMagick-7.X\\convert.exe",
            "C:\\Program Files (x86)\\ImageMagick-7.X\\magick.exe",
            "C:\\Program Files (x86)\\ImageMagick-7.X\\convert.exe",
        ]
        
        for i in range(9, 0, -1):  # Check from newest to oldest version
            for base_path in common_paths:
                path = base_path.replace("7.X", f"7.{i}")
                if os.path.exists(path):
                    return path
    
    return None

def update_moviepy_config():
    """Update the MoviePy configuration to point to the correct ImageMagick executable"""
    try:
        import moviepy
        moviepy_dir = Path(moviepy.__file__).parent
        config_file = moviepy_dir / "config_defaults.py"
        
        # Check if config file exists
        if not config_file.exists():
            print(f"Config file not found at {config_file}")
            return False
        
        # Find ImageMagick
        imagemagick_path = find_imagemagick()
        if not imagemagick_path:
            print("ImageMagick executable not found. Please install ImageMagick first.")
            return False
        
        print(f"Found ImageMagick at: {imagemagick_path}")
        
        # Read the current config
        with open(config_file, 'r') as f:
            config_content = f.read()
        
        # Create backup
        with open(f"{config_file}.backup", 'w') as f:
            f.write(config_content)
        
        # Update the config
        if "IMAGEMAGICK_BINARY" in config_content:
            # Replace the line with the new path
            lines = config_content.split('\n')
            for i, line in enumerate(lines):
                if "IMAGEMAGICK_BINARY" in line and "=" in line:
                    lines[i] = f'IMAGEMAGICK_BINARY = "{imagemagick_path}"'
            
            new_content = '\n'.join(lines)
        else:
            # Add the line to the end
            new_content = config_content + f'\n\nIMAGEMAGICK_BINARY = "{imagemagick_path}"\n'
        
        # Write the updated config
        with open(config_file, 'w') as f:
            f.write(new_content)
        
        print(f"Updated MoviePy configuration in {config_file}")
        return True
    
    except ImportError:
        print("MoviePy is not installed. Please install it first.")
        return False
    except Exception as e:
        print(f"Error updating config: {e}")
        return False

def verify_installation():
    """Verify that ImageMagick is properly installed and configured"""
    try:
        # Try running a simple test with MoviePy's TextClip
        from moviepy.editor import TextClip
        clip = TextClip("Test", fontsize=30, color='white')
        print("✅ MoviePy TextClip created successfully!")
        print("ImageMagick is correctly configured.")
        return True
    except Exception as e:
        print(f"❌ Error creating TextClip: {e}")
        print("ImageMagick may not be correctly configured.")
        return False

def main():
    print("MoviePy Configuration Fix Utility")
    print("=================================")
    
    # Check if MoviePy is installed
    try:
        import moviepy
        print(f"MoviePy version: {moviepy.__version__}")
    except ImportError:
        print("MoviePy is not installed. Please install it first:")
        print("  pip install moviepy")
        return
    
    # Update config
    if update_moviepy_config():
        print("\nTesting configuration...")
        verify_installation()
    
    print("\nIf you continue to have issues, you may need to install ImageMagick manually.")

if __name__ == "__main__":
    main()

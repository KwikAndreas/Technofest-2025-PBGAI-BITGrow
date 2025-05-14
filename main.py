import os
import sys
import subprocess

def check_dependencies():
    """Check if all required packages are installed"""
    try:
        import tensorflow as tf
        import numpy as np
        import cv2
        print("All dependencies installed!")
        return True
    except ImportError as e:
        print(f"Missing dependency: {e}")
        return False

def show_menu():
    """Display the main menu"""
    print("\n===== Waste Classification System =====")
    print("1. Train model")
    print("2. Convert model to ONNX")
    print("3. Track waste with camera")
    print("4. Install dependencies")
    print("5. Exit")
    return input("Select option (1-5): ")

def install_dependencies():
    """Install required dependencies"""
    print("Installing dependencies...")
    packages = [
        "tensorflow",
        "numpy",
        "opencv-python",
        "onnx",
        "tf2onnx"
    ]
    
    for package in packages:
        print(f"Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package])
    
    print("Dependencies installed! Please restart the application.")

def main():
    if not os.path.exists("model"):
        os.makedirs("model")
        
    while True:
        choice = show_menu()
        
        if choice == "1":
            print("Running training script...")
            subprocess.run([sys.executable, "train.py"])
        elif choice == "2":
            print("Converting model to ONNX...")
            subprocess.run([sys.executable, "convert_model.py"])
        elif choice == "3":
            print("Starting waste tracking...")
            subprocess.run([sys.executable, "track_waste.py"])
        elif choice == "4":
            install_dependencies()
        elif choice == "5":
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    print("Checking dependencies...")
    if check_dependencies():
        main()
    else:
        print("Would you like to install missing dependencies? (y/n)")
        if input().lower() == 'y':
            install_dependencies()

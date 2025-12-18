import subprocess
import os
import sys


def check_venv():
    """
    Checks if venv is installed.
    Returns True if venv is installed, False otherwise.
    """
    try:
        import venv
        print("Venv is installed, attempting to create virtual environment...")
        return True
    except ImportError:
        print("Venv is not installed. Installing...")
        return False



def install_venv():
    """
    Installs python3-venv if not already installed.
    Returns True if installation is successful, False otherwise.
    """
    try:
        subprocess.run(['sudo', 'apt-get', 'install', 'python3-venv'], check=True)
        print ("Venv installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error while installing python3-venv: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    return False


def create_virtual_environment():
    """
    Creates a virtual environment in the root directory.
    """
    subprocess.run(["python3", "-m", "venv", ".venv"])
    print ("Venv created")


def main():
    if not check_venv():
        if install_venv():
            return True
    
    if not ".venv" in os.listdir():
        create_virtual_environment()
    return False


if __name__ == "__main__":
    main()
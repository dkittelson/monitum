"""
Google Earth Engine Setup & Authentication
Run this once to set up GEE access for forest loss extraction
"""

import sys

def check_and_install_gee():
    """Check if GEE packages are installed, install if needed"""
    
    print("=" * 70)
    print("GOOGLE EARTH ENGINE SETUP")
    print("=" * 70)
    print()
    
    # Check for earthengine-api
    try:
        import ee
        print("✅ earthengine-api is installed")
    except ImportError:
        print("❌ earthengine-api not found")
        print("   Installing earthengine-api...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "earthengine-api"])
        import ee
        print("✅ earthengine-api installed successfully")
    
    # Check for geemap
    try:
        import geemap
        print("✅ geemap is installed")
    except ImportError:
        print("❌ geemap not found")
        print("   Installing geemap...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "geemap"])
        print("✅ geemap installed successfully")
    
    print()
    print("-" * 70)
    print("AUTHENTICATION REQUIRED")
    print("-" * 70)
    print()
    print("Next step: Authenticate with Google Earth Engine")
    print()
    print("1. Run: ee.Authenticate()")
    print("2. This opens a browser to authorize your Google account")
    print("3. Sign in with a Gmail account")
    print("4. Copy the authorization code")
    print("5. Paste it back in the terminal")
    print()
    print("After authentication, run: ee.Initialize()")
    print()
    
    # Try to authenticate
    try:
        print("Attempting authentication...")
        ee.Authenticate()
        print("✅ Authentication successful!")
        
        print("\nInitializing Earth Engine...")
        ee.Initialize()
        print("✅ Earth Engine initialized!")
        
        # Test access
        print("\nTesting access to Hansen Forest Loss dataset...")
        hansen = ee.Image('UMD/hansen/global_forest_change_2023_v1_11')
        print("✅ Successfully connected to Hansen Global Forest Change dataset!")
        
        return True
        
    except Exception as e:
        print(f"\n⚠️  Authentication needs to be done manually")
        print(f"   Error: {e}")
        print("\nRun these commands in Python:")
        print("  import ee")
        print("  ee.Authenticate()")
        print("  ee.Initialize()")
        return False

if __name__ == "__main__":
    check_and_install_gee()

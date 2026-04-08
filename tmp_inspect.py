import sys
try:
    import openenv
    print("openenv.__all__:", getattr(openenv, "__all__", dir(openenv)))
    
    import openenv.core
    print("\nopenenv.core:", dir(openenv.core))
except Exception as e:
    print(f"Error: {e}")

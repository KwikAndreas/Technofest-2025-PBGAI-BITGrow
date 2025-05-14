import cv2
import time

def test_camera():
    # Try different backends
    backends = [
        (cv2.CAP_ANY, "Default"),
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_V4L2, "Video4Linux2"),
    ]
    
    for backend, name in backends:
        print(f"\nTrying camera with {name} backend...")
        cap = cv2.VideoCapture(0, backend)
        
        if not cap.isOpened():
            print(f"Failed to open camera with {name} backend")
            continue
        
        print(f"Success! Camera opened with {name} backend")
        
        # Get camera properties
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera properties: {width}x{height} at {fps} FPS")
        
        # Try to read 10 frames
        for i in range(10):
            start_time = time.time()
            ret, frame = cap.read()
            elapsed = time.time() - start_time
            
            if not ret:
                print(f"Failed to read frame {i}")
                break
                
            print(f"Frame {i}: Read in {elapsed:.4f} seconds")
            
            # Display the frame
            cv2.imshow(f"Camera Test ({name})", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        choice = input("Did the camera work properly? (y/n): ")
        if choice.lower() == 'y':
            return backend
    
    return None

if __name__ == "__main__":
    print("Testing camera access...")
    working_backend = test_camera()
    
    if working_backend is not None:
        print(f"\nCamera test successful with backend: {working_backend}")
        print("Use this backend in the track_waste.py script by changing:")
        print(f"cap = cv2.VideoCapture(CAMERA_INDEX, {working_backend})")
    else:
        print("\nAll camera tests failed. Possible issues:")
        print("1. Camera is being used by another application")
        print("2. Camera drivers need to be updated")
        print("3. Camera hardware issues")
        print("4. No camera is connected to the computer")
        
    print("\nPress any key to exit...")
    input()

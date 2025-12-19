"""
Quick script to create sample test data if you don't have CIFAR-10 images
This creates random colored images for testing purposes
"""
import os
from PIL import Image
import numpy as np

def create_sample_images():
    """Create sample test images for testing the GAN"""
    
    # Create directories
    test_dir = "cifar-10-images-master/test"
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("Creating sample test images...")
    print(f"Directory: {test_dir}")
    
    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Create 10 sample images per class
        for i in range(10):
            # Create a random 32x32 RGB image
            img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # Save image
            img_path = os.path.join(class_dir, f'{i}_{class_name}.png')
            img.save(img_path)
        
        print(f"  ✓ Created 10 images in {class_name}/")
    
    print(f"\n✓ Created {len(classes) * 10} sample test images")
    print(f"  Location: {test_dir}")
    print("\nNow you can run: python stylegan.py --mode test")

if __name__ == "__main__":
    create_sample_images()

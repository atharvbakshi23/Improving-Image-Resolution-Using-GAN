"""
Quick script to create sample training data if you don't have CIFAR-10 images
This creates random colored images for training purposes
"""
import os
from PIL import Image
import numpy as np

def create_sample_images():
    """Create sample training images for training the GAN"""
    
    # Create directories
    train_dir = "cifar-10-images-master/train"
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    print("Creating sample training images...")
    print(f"Directory: {train_dir}")
    
    for class_name in classes:
        class_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        
        # Create 100 sample images per class (1000 total)
        for i in range(100):
            # Create a random 32x32 RGB image
            img_array = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            
            # Save image
            img_path = os.path.join(class_dir, f'{i}_{class_name}.png')
            img.save(img_path)
        
        print(f"  ✓ Created 100 images in {class_name}/")
    
    print(f"\n✓ Created {len(classes) * 100} sample training images")
    print(f"  Location: {train_dir}")
    print("\nNow you can run: python stylegan.py --mode train")

if __name__ == "__main__":
    create_sample_images()

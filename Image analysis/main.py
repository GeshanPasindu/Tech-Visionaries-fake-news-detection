import os

print("🚀 Running Preprocessing...")
os.system("python scripts/preprocess.py")

# print("🚀 Running Feature Extraction...")
# os.system("python scripts/feature_extract.py")

print("🚀 Running model training...")
os.system("python scripts/train_vgg16.py")

print("🚀 Running model testing...")
os.system("python scripts/test_vgg16.py")

print("✅ All steps completed successfully!")


# import os
# from scripts.preprocess import process_dataset
# from scripts.feature_extract import extract_features

# def main():
#     print("Starting Image Processing Pipeline...\n")
    
#     # Step 1: Preprocess the dataset
#     print("Step 1: Preprocessing images...")
#     process_dataset()
#     print("Preprocessing completed!\n")
    
#     # Step 2: Extract features from the preprocessed dataset
#     print("Step 2: Extracting features...")
#     extract_features("Authentic")
#     extract_features("Tampered")
#     print("Feature extraction completed!\n")
    
#     print("✅ All steps completed successfully!")

# if __name__ == "__main__":
#     main()


# import subprocess

# def run_script(script_path):
#     try:
#         print(f"\n🚀 Running {script_path}...")
#         subprocess.run(["python", script_path], check=True)
#         print(f"✅ Finished {script_path}")
#     except subprocess.CalledProcessError as e:
#         print(f"❌ Error while running {script_path}: {e}")
#         exit(1)

# if __name__ == "__main__":
#     run_script("scripts/preprocess.py")
#     run_script("scripts/feature_extract.py")
#     run_script("scripts/train_vgg16.py")
#     run_script("scripts/test_vgg16.py")
#     print("\n🎉 All steps completed successfully!")

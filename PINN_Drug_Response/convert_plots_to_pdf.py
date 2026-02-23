import os
from PIL import Image
import glob

def convert_pngs_to_pdf(directory):
    # Find all fit plots
    png_files = glob.glob(os.path.join(directory, "fit_*.png"))
    
    if not png_files:
        print(f"No PNG files found in {directory}")
        return

    print(f"Found {len(png_files)} plots to convert...")
    
    for png_path in png_files:
        # Define output path
        pdf_path = png_path.replace(".png", ".pdf")
        
        try:
            # Open image and convert to RGB (required for PDF saving)
            img = Image.open(png_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Save as PDF
            img.save(pdf_path, "PDF", resolution=300.0)
            print(f"Successfully converted: {os.path.basename(png_path)} -> {os.path.basename(pdf_path)}")
        except Exception as e:
            print(f"Failed to convert {png_path}: {e}")

    # Also create a combined PDF for convenience
    combined_pdf_path = os.path.join(directory, "all_model_fits_combined.pdf")
    try:
        images = []
        for png_path in sorted(png_files):
            img = Image.open(png_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            images.append(img)
        
        if images:
            images[0].save(combined_pdf_path, save_all=True, append_images=images[1:])
            print(f"\nCreated combined PDF: {combined_pdf_path}")
    except Exception as e:
        print(f"Failed to create combined PDF: {e}")

if __name__ == "__main__":
    TARGET_DIR = "results/nature_submission"
    convert_pngs_to_pdf(TARGET_DIR)

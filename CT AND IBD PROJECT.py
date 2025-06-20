import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import pydicom as pyd

class DataCleaning:
    def _init_(self, file_path):
        self.file_path = file_path
        self.df = None
        self.cleaned_data = None

    def load_data(self):
        """Load data from the specified CSV file."""
        self.df = pd.read_csv(self.file_path)

    def remove_duplicates(self):
        """Remove duplicate rows."""
        self.cleaned_data = self.df.drop_duplicates()

    def clean_data(self):
        """Clean data by removing slashes and dropping rows with missing values."""
        self.cleaned_data = self.cleaned_data.applymap(
            lambda x: x.replace("/", "") if isinstance(x, str) else x
        ).dropna()

    def standardize_file_paths(self):
        """Standardize file paths to lowercase."""
        self.cleaned_data['dcm'] = self.cleaned_data['dcm'].str.lower()
        self.cleaned_data['jpg'] = self.cleaned_data['jpg'].str.lower()

    def validate_file_extensions(self):
        """Identify rows with invalid file extensions."""
        valid_dcm = self.cleaned_data['dcm'].str.endswith('.dcm')
        valid_jpg = self.cleaned_data['jpg'].str.endswith('.jpg')
        return self.cleaned_data[(valid_dcm & valid_jpg) == False]

    def check_duplicate_files(self):
        """Check for duplicate file paths in DCM and JPG columns."""
        duplicate_dcm = self.cleaned_data[self.cleaned_data.duplicated(subset='dcm', keep=False)]
        duplicate_jpg = self.cleaned_data[self.cleaned_data.duplicated(subset='jpg', keep=False)]
        return duplicate_dcm, duplicate_jpg

    def validate_type_column(self):
        """Validate the 'type' column for known entries."""
        known_types = ['aneurysm', 'tumor', 'cancer']
        return self.cleaned_data[self.cleaned_data['type'].isin(known_types) == False]

    def generate_report(self):
        """Generate a summary report of the cleaning process."""
        invalid_rows = self.validate_file_extensions()
        duplicate_dcm, duplicate_jpg = self.check_duplicate_files()
        invalid_types = self.validate_type_column()

        report = {
            "Initial Rows": len(self.df),
            "Cleaned Rows": len(self.cleaned_data),
            "Duplicates Removed": len(self.df) - len(self.cleaned_data),
            "Invalid File Paths": len(invalid_rows),
            "Duplicate DCM Files": len(duplicate_dcm),
            "Duplicate JPG Files": len(duplicate_jpg),
            "Invalid Type Entries": len(invalid_types),
        }

        for key, value in report.items():
            print(f"{key}: {value}")

    def run(self):
        """Run all steps of the data cleaning process."""
        self.load_data()
        self.remove_duplicates()
        self.clean_data()
        self.standardize_file_paths()
        self.generate_report()


class ImageCollage:
    def _init_(self, image_paths):
        self.image_paths = image_paths

    def create_collage(self):
        images = [cv2.resize(cv2.imread(path), (250, 250)) for path in self.image_paths]
        row1 = np.hstack(images[:3])
        row2 = np.hstack(images[3:])
        collage = np.vstack([row1, row2])

        cv2.imshow('Collage', collage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class ImageNormalization:
    def _init_(self, image_path):
        self.image_path = image_path

    def normalize_jpg(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        normalized_image_0_1 = image / 255.0
        normalized_image_minus1_1 = (image / 127.5) - 1

        cv2.imshow('Normalized (0-1)', normalized_image_0_1)
        cv2.imshow('Normalized (-1 to 1)', normalized_image_minus1_1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class DICOMProcessing:
    def _init_(self, dicom_path):
        self.dicom_path = dicom_path

    def normalize_dicom(self):
        image_dcm = pyd.dcmread(self.dicom_path)
        pixel_array = image_dcm.pixel_array
        intercept = image_dcm.RescaleIntercept
        slope = image_dcm.RescaleSlope

        hu_image = pixel_array * slope + intercept
        normalized_hu_image = (hu_image - np.min(hu_image)) / (np.max(hu_image) - np.min(hu_image))

        plt.imshow(normalized_hu_image, cmap='gray')
        plt.colorbar()
        plt.title('Normalized HU Image')
        plt.show()

class ImageInpainting:
    def _init_(self, image_path, mask_path):
        self.image_path = image_path
        self.mask_path = mask_path

    def restore_image(self):
        image = cv2.imread(self.image_path)
        mask = cv2.imread(self.mask_path, cv2.IMREAD_GRAYSCALE)

        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

        cv2.imshow('Inpainted Image', inpainted_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class ContourDetection:
    def _init_(self, image_path):
        self.image_path = image_path

    def detect_contours(self):
        image = cv2.imread(self.image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contoured_image = image.copy()
        cv2.drawContours(contoured_image, contours, -1, (0, 255, 0), 3)

        cv2.imshow('Original Image', image)
        cv2.imshow('Contours', contoured_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class HeatmapGenerator:
    def _init_(self, image_path):
        self.image_path = image_path

    def create_heatmap(self):
        image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))

        plt.imshow(normalized_image, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Pixel Intensity')
        plt.title('Heatmap of Pixel Intensities')
        plt.axis('off')
        plt.show()

class OverlayCreator:
    def _init_(self, ct_image_path, prediction_mask_path):
        self.ct_image_path = ct_image_path
        self.prediction_mask_path = prediction_mask_path

    def overlay_ct_mask(self):
        ct_image = cv2.imread(self.ct_image_path, cv2.IMREAD_GRAYSCALE)
        prediction_mask = cv2.imread(self.prediction_mask_path, cv2.IMREAD_GRAYSCALE)
        prediction_mask = cv2.resize(prediction_mask, (ct_image.shape[1], ct_image.shape[0]))

        ct_image_color = cv2.cvtColor(ct_image, cv2.COLOR_GRAY2BGR)
        prediction_colored = cv2.applyColorMap(np.uint8(prediction_mask), cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(ct_image_color, 0.6, prediction_colored, 0.4, 0)

        cv2.imshow('Overlay', overlay)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

class ImageEditor:
    def _init_(self, base_path):
        self.base_path = base_path

    def process_images(self, start_index=0, end_index=1):
        for i in range(start_index, end_index):
            image_path = os.path.join(self.base_path, f'{i}.jpg')
            try:
                image = Image.open(image_path)
                image.show()

                image_array = np.array(image)
                resized_arr = cv2.resize(image_array, (500, 500))
                image_resized = Image.fromarray(resized_arr)
                image_resized.show()

                print(f"Image {i}: Original shape = {image_array.shape}, "
                      f"Resized shape = {resized_arr.shape}, File Name = {i}.jpg")
            except Exception as e:
                print(f"Error processing image {i}: {e}")

if _name_ == "_main_":
    file_path = r"C:\\Users\\abhi2\\OneDrive\\Documents\\BIOLOGICAL DATA\\ct_brain.csv"
    data_cleaner = DataCleaning(file_path)
    data_cleaner.run()

    image_paths = [
        r"C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\tumor\0.jpg",
        r"C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\tumor\1.jpg",
        r"C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\tumor\2.jpg",
        r"C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\tumor\3.jpg",
        r"C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\tumor\4.jpg",
        r"C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\tumor\5.jpg",
    ]
    collage_creator = ImageCollage(image_paths)
    collage_creator.create_collage()

    image_normalizer = ImageNormalization(r"C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\cancer\4.jpg")
    image_normalizer.normalize_jpg()

    dicom_processor = DICOMProcessing(r"C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\cancerdcm\4.dcm")
    dicom_processor.normalize_dicom()

    inpainter = ImageInpainting(
        r"C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\tumor\20.jpg",
        r"C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\tumor\21.jpg"
    )
    inpainter.restore_image()

    contour_detector = ContourDetection(r"C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\tumor\16.jpg")
    contour_detector.detect_contours()

    heatmap_gen = HeatmapGenerator(r"C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\cancer\0.jpg")
    heatmap_gen.create_heatmap()

    overlay_creator = OverlayCreator(
        r"C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\cancer\0.jpg",
        r"C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\cancer\10.jpg"
    )
    overlay_creator.overlay_ct_mask()

    editor = ImageEditor(r'C:\Users\abhi2\OneDrive\Documents\BIOLOGICAL DATA\files\aneurysm')
    editor.process_images(start_index=0, end_index=1)
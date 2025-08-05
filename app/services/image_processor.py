from PIL import Image
import cv2
import numpy as np
from fastapi import UploadFile
import io

class ImageProcessor:
    @staticmethod
    async def process_image(file: UploadFile):
        try:
            # Read image data
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))

            # Convert to OpenCV format for processing
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # TODO: Image processing logic here
            # Example: Convert to grayscale
            processed_image = cv2.resize(cv_image, (800, 600))

            # Convert back to PIL and return
            processed_PIL = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

            # Save to bytes
            img_byte_arr = io.BytesIO()
            processed_PIL.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)

            return img_byte_arr.getvalue()
        except Exception as e:
            raise ValueError(f"Image processing failed: {str(e)}")
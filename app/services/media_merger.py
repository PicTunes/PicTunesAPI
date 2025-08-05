import ffmpeg
import tempfile
import os
from fastapi import UploadFile

class MediaMerger:
    @staticmethod
    async def merge_media_audio(image_file: UploadFile, audio_file: UploadFile, duration: int = 15):
        """Merge image with audio to create a video"""
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_image:
                image_content = await image_file.read()
                temp_image.write(image_content)
                temp_image_path = temp_image.name

            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_audio:
                audio_content = await audio_file.read()
                temp_audio.write(audio_content)
                temp_audio_path = temp_audio.name
            
            # Output video path
            output_path = tempfile.mktemp(suffix='.mp4')

            # Use ffmpeg-python to merge image and audio
            input_image = ffmpeg.input(temp_image_path, loop=1, t=duration)
            input_audio = ffmpeg.input(temp_audio_path)

            output = ffmpeg.output(
                input_image.video,
                input_audio.audio,
                output_path,
                vcodec='libx264',
                acodec='aac',
                shortest=None
            )

            ffmpeg.run(output, quiet=True, overwrite_output=True)

            # Read result 
            with open(output_path, 'rb') as f:
                result_bytes = f.read()

            # Cleanup
            os.unlink(temp_image_path)
            os.unlink(temp_audio_path)
            os.unlink(output_path)

            return result_bytes
        except Exception as e:
            raise RuntimeError(f"Merging media failed: {str(e)}")
import ffmpeg
from PIL import Image
from pillow_heif import register_heif_opener
import pillow_heif
import os

register_heif_opener()

def media_merger(img, aud, out):
    # read image file
    try:
        temp_img_path = None
        if img.endswith('.heic') or img.endswith('.HEIC'):
            # convert heic to jpg
            print(f"Converting {img} to jpg")
            heif_file = pillow_heif.read_heif(img)
            temp_img = Image.frombytes(
                heif_file.mode,
                heif_file.size,
                heif_file.data.tobytes(),
                "raw",
            )
            temp_img.save(f"{img[:-5]}.jpg")
            temp_img_path = img
            img = f"{img[:-5]}.jpg"
            
        print(f"img: {img}")
        img_content = ffmpeg.input(f"{img}",loop=1)
        aud_content = ffmpeg.input(f"{aud}")
        print(img_content)
        print(aud_content)
        probe = ffmpeg.probe(aud)
        duration = float(probe['streams'][0]['duration'])
        (
            ffmpeg
            .output(
                img_content,
                aud_content,
                out,
                vcodec='libx264',
                acodec='aac',
                pix_fmt='yuv420p',
                shortest=None,
                t=duration
            )
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        print(f"Successfully created video: {out}")

        if temp_img_path:
            os.remove(img)
        return out
        
    except ffmpeg.Error as e:
        # Now stderr will be available
        stderr_output = e.stderr.decode('utf8') if e.stderr else "No stderr output"
        print(f"FFmpeg error: {stderr_output}")
        if temp_img_path:
            os.remove(img)
        raise Exception(f"FFmpeg failed: {stderr_output}")
        
    except Exception as e:
        print(f"Error in media_merger: {e}")
        if temp_img_path:
            os.remove(img)
        raise
import ffmpeg
import os

class MediaMergerClass:
    pass

def media_merger(img, aud):
    # read image file
    img_content = img.file.read()
    with open(f"uploads/{img.filename}", "wb") as f:
        f.write(img_content)

    return
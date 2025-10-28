import ffmpeg
import os

def media_merger(img, aud, out):
    # read image file
    img_content = ffmpeg.input(f"{img}",loop=1)
    aud_content = ffmpeg.input(f"{aud}")
    print(img_content)
    print(aud_content)
    try:
        print(f"processing/{out}")
        img_content.output(
            aud_content,
            f"{out}",
            vcodec='libx264',
            shortest=None,
            **{'c:a': 'aac'}
        ).run(overwrite_output=True)
    except ffmpeg.Error as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
    except Exception as e:
        print(f"error: {e}")
    
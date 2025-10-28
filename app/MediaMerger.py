import ffmpeg
import os

def media_merger(img, aud, out):
    # read image file
    try:
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
        return out
        
    except ffmpeg.Error as e:
        # Now stderr will be available
        stderr_output = e.stderr.decode('utf8') if e.stderr else "No stderr output"
        print(f"FFmpeg error: {stderr_output}")
        raise Exception(f"FFmpeg failed: {stderr_output}")
        
    except Exception as e:
        print(f"Error in media_merger: {e}")
        raise
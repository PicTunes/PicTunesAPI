import ffmpeg
import os

def media_merger(img, aud, out):
    # read image file
    img_content = ffmpeg.input(f"{img}",loop=1)
    aud_content = ffmpeg.input(f"{aud}")
    print(img_content)
    print(aud_content)
    try:
        img_content.output(
            aud_content,
            "processing/output.mp4",
            vcodec='libx264',
            shortest=None,
            **{'c:a': 'aac'}
        ).run(overwrite_output=True)
    except ffmpeg.Error as e:
        print(f"ffmpeg error: {e.stderr.decode()}")
    except Exception as e:
        print(f"error: {e}")
    # .run(overwrite_output=True)


# media_merger(
#     img="/Users/taxihuang/Desktop/DEsktop/general/gradProj/PicTunesAPI/uploads/IMG_4709.JPG",
#     aud="/Users/taxihuang/Desktop/DEsktop/DTM/samples/how to make uk hardcore/KSHMR Exhaust 05.wav"
# )

# '/Users/taxihuang/Desktop/DEsktop/DTM/samples/how to make uk hardcore/KSHMR Exhaust 05.wav'
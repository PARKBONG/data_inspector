from loader.rgb import RGBLoader


def rgb_image_source(loader: RGBLoader, timestamp: float):
    frame = loader.find_nearest(timestamp)
    if not frame:
        return None
    encoded = RGBLoader.encode_image(frame.path)
    return f"data:image/jpeg;base64,{encoded}"

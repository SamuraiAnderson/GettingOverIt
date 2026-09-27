"""Single Unity-screen to image conversion and body-local projection."""
from __future__ import annotations

import numpy as np


def unity_screen_to_image_xy(screen_x: float, screen_y: float, pixel_height: float) -> np.ndarray:
    """Unity screen origin is bottom-left; image row origin is top-left."""
    return np.array([float(screen_x), float(pixel_height) - float(screen_y)], dtype=np.float64)


def image_xy_to_unity_screen(image_x: float, image_y: float, pixel_height: float) -> np.ndarray:
    return np.array([float(image_x), float(pixel_height) - float(image_y)], dtype=np.float64)


def body_transform_point(
    local_xy,
    position_xy,
    rotation_z_deg: float,
    scale_xy,
) -> np.ndarray:
    """Match Unity `Transform.TransformPoint` for a Z rotation and XY scale."""
    lx, ly = (float(local_xy[0]), float(local_xy[1]))
    sx, sy = (float(scale_xy[0]), float(scale_xy[1]))
    rad = np.deg2rad(float(rotation_z_deg))
    c, s = np.cos(rad), np.sin(rad)
    x, y = sx * lx, sy * ly
    return np.array([
        float(position_xy[0]) + c * x - s * y,
        float(position_xy[1]) + s * x + c * y,
    ], dtype=np.float64)


def project_orthographic(
    world_xy,
    camera_xy,
    orthographic_size: float,
    pixel_width: float,
    pixel_height: float,
) -> np.ndarray:
    """Project world XY through a Z-aligned orthographic camera to image pixels."""
    width_m = 2.0 * float(orthographic_size) * (float(pixel_width) / float(pixel_height))
    height_m = 2.0 * float(orthographic_size)
    nx = (float(world_xy[0]) - float(camera_xy[0])) / width_m + 0.5
    ny = (float(world_xy[1]) - float(camera_xy[1])) / height_m + 0.5
    screen_x = nx * float(pixel_width)
    screen_y = ny * float(pixel_height)
    return unity_screen_to_image_xy(screen_x, screen_y, pixel_height)


def metres_per_pixel(orthographic_size: float, pixel_height: float) -> float:
    return 2.0 * float(orthographic_size) / float(pixel_height)


def unity_screen_axes_to_image(agent: dict, pixel_height: float) -> dict:
    """Convert Unity bottom-left screen samples to image-top-left pixels."""
    body = unity_screen_to_image_xy(agent["unityScreenBodyX"], agent["unityScreenBodyY"], pixel_height)
    axis_x = unity_screen_to_image_xy(agent["unityScreenAxisXX"], agent["unityScreenAxisXY"], pixel_height)
    axis_y = unity_screen_to_image_xy(agent["unityScreenAxisYX"], agent["unityScreenAxisYY"], pixel_height)
    return {
        "unity_body_img": body,
        "unity_axis_x_img": axis_x,
        "unity_axis_y_img": axis_y,
    }


def project_unity_local(local_xy, unity_body_img, unity_axis_x_img, unity_axis_y_img) -> np.ndarray:
    """camera.project(body.TransformPoint(b)) using Unity-recorded screen axes."""
    origin = np.asarray(unity_body_img, dtype=np.float64)
    axis_x = np.asarray(unity_axis_x_img, dtype=np.float64) - origin
    axis_y = np.asarray(unity_axis_y_img, dtype=np.float64) - origin
    lx, ly = float(local_xy[0]), float(local_xy[1])
    return origin + axis_x * lx + axis_y * ly

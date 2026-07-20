from base64 import b64decode
from binascii import Error as Base64Error
from numbers import Real


def bad(message, **details):
    payload = {"error": message}
    payload.update(details)
    return payload


def image_payload(payload):
    image = payload.get("image")
    if not isinstance(image, dict):
        return bad("image is required"), None, None

    shape = image.get("shape")
    if not is_rgb_shape(shape):
        return bad("image.shape must be [height, width, 3]"), None, None

    data = image.get("data")
    if not isinstance(data, str):
        return bad("image.data must be base64"), None, None

    try:
        image_bytes = b64decode(data, validate=True)
    except (Base64Error, ValueError):
        return bad("image.data must be base64"), None, None

    height, width, _ = shape
    expected = height * width * 3
    if len(image_bytes) != expected:
        return bad("image.data length does not match image.shape", expected=expected, actual=len(image_bytes)), None, None
    return None, image_bytes, shape


def prompt_payload(payload):
    points = payload.get("points", [])
    labels = payload.get("labels", [])
    box = payload.get("box")
    text = payload.get("text")
    error, mask = mask_prompt(payload.get("mask"))
    if error:
        return error, None, None, None, None, None
    if not valid_prompts(points, labels):
        return bad("points and labels must be equal length lists of [x, y] and 0/1 labels"), None, None, None, None, None
    if box is not None and not is_box(box):
        return bad("box must be [x0, y0, x1, y1]"), None, None, None, None, None
    if text is not None and (not isinstance(text, str) or not text.strip()):
        return bad("text must be a non-empty string"), None, None, None, None, None
    if not points and box is None and mask is None and text is None:
        return bad("points/labels, box, mask, or text prompt is required"), None, None, None, None, None
    return None, points, labels, box, mask, text


def mask_prompt(mask):
    if mask is None:
        return None, None
    if not isinstance(mask, dict):
        return bad("mask must be an object"), None
    shape = mask.get("shape")
    if not is_mask_shape(shape):
        return bad("mask.shape must be [height, width]"), None
    data = mask.get("data")
    if not isinstance(data, str):
        return bad("mask.data must be base64"), None
    try:
        mask_bytes = b64decode(data, validate=True)
    except (Base64Error, ValueError):
        return bad("mask.data must be base64"), None
    expected = shape[0] * shape[1]
    if len(mask_bytes) != expected:
        return bad("mask.data length does not match mask.shape", expected=expected, actual=len(mask_bytes)), None
    return None, {"shape": shape, "data": mask_bytes}


def is_rgb_shape(shape):
    return (
        isinstance(shape, list)
        and len(shape) == 3
        and all(is_int(value) for value in shape)
        and shape[0] > 0
        and shape[1] > 0
        and shape[2] == 3
    )


def is_mask_shape(shape):
    return (
        isinstance(shape, list)
        and len(shape) == 2
        and all(is_int(value) for value in shape)
        and shape[0] > 0
        and shape[1] > 0
    )


def valid_prompts(points, labels):
    return (
        isinstance(points, list)
        and isinstance(labels, list)
        and len(points) == len(labels)
        and all(is_point(point) for point in points)
        and all(is_label(label) for label in labels)
    )


def is_point(point):
    return isinstance(point, list) and len(point) == 2 and all(is_number(value) for value in point)


def is_box(box):
    return isinstance(box, list) and len(box) == 4 and all(is_number(value) for value in box) and box[0] < box[2] and box[1] < box[3]


def is_number(value):
    return isinstance(value, Real) and not isinstance(value, bool)


def is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def is_label(value):
    return is_int(value) and value in (0, 1)

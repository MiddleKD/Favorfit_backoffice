import base64, json
from io import BytesIO
from PIL import Image

import cv2
import numpy as np

class TooMuchRequestQueueError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)
        
    def __str__(self):
        return f"TooMuchRequestQueueError: {self.message}"

class DupledRequestKeyError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)
        
    def __str__(self):
        return f"DupledRequestKeyError: {self.message}"

class RequestKeyDoesNotExistError(Exception):
    def __init__(self, message=None):
        self.message = message
        super().__init__(self.message)
        
    def __str__(self):
        return f"RequestKeyDoesNotExistError: {self.message}"
    
def respond(err, res):
    respond_msg = {'statusCode': 502 if err is not None else 200, 'body': json.dumps(res)}
    return respond_msg

def bs64_to_pil(img_bs64):
    if not img_bs64.startswith("/"):
        img_bs64 = img_bs64.split(",", 1)[1]
    img_data = base64.b64decode(img_bs64)
    img_pil = Image.open(BytesIO(img_data))
    return img_pil

def pil_to_bs64(img_pil):
    buffered = BytesIO()
    img_pil.save(buffered, format="jpeg")
    img_bytes = buffered.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    return img_base64

def img_box_crop(img_pil, box):
    if isinstance(box, list):
        x1, y1, x2, y2 = box
    else:
        x1 = box["x1"]; y1 = box["y1"]
        x2 = box["x2"]; y2 = box["y2"]
    
    return img_pil.crop((x1, y1, x2, y2))

def padding_mask_img(img_pil, mask_pil, box):
    if box == None:
        return mask_pil
    
    if isinstance(box, list):
        x1, y1, x2, y2 = box
    else:
        x1 = box["x1"]; y1 = box["y1"]
        x2 = box["x2"]; y2 = box["y2"]
    
    black_img = Image.new("RGB", img_pil.size)
    black_img.paste(mask_pil, (x1, y1, x2, y2))
    return black_img

def load_instance_from_json(json_like):
    args = json_like["body"]
    if isinstance(args, str):
        args = json_like.loads(args)
    return args

def center_crop_and_resize(input_image, target_size=(512, 512)):
    image = input_image
    width, height = image.size
    left = (width - min(width, height)) // 2
    top = (height - min(width, height)) // 2
    right = (width + min(width, height)) // 2
    bottom = (height + min(width, height)) // 2
    image = image.crop((left, top, right, bottom))

    image = image.resize(target_size)

    return image

def composing_output(img1, img2, mask):
    img1 = np.array(img1)
    mask = np.array(mask)
    img2 = np.array(img2)
    
    composed_output = np.array(img1) * (1-mask/255) + np.array(img2) * (mask/255)
    return Image.fromarray(composed_output.astype(np.uint8))

def make_canny_condition(image, min=100, max=200):
    image = np.array(image)
    image = cv2.Canny(image, min, max)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image

def make_outpaint_condition(image, mask):
    image = np.array(image)
    mask = np.array(mask)
    black_image = np.zeros_like(image)

    composed_output = np.array(black_image) * (1-mask/255) + np.array(image) * (mask/255)
    return Image.fromarray(composed_output.astype(np.uint8))

def make_noise_disk(H, W, C, F):
    noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F: F + H, F: F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise

def make_shuffle_condition(image, h=None, w=None, f=None):
    img = np.array(image)
    H, W, C = img.shape
    if h is None:
        h = H
    if w is None:
        w = W
    if f is None:
        f = 256
    x = make_noise_disk(h, w, 1, f) * float(W - 1)
    y = make_noise_disk(h, w, 1, f) * float(H - 1)
    flow = np.concatenate([x, y], axis=2).astype(np.float32)
    return Image.fromarray(cv2.remap(img, flow, None, cv2.INTER_LINEAR))
    
def resize_store_ratio(image, min_side=512):

    width, height = image.size

    if width < height:
        new_width = min_side
        new_height = int((height / width) * min_side)
    else:
        new_width = int((width / height) * min_side)
        new_height = min_side

    resized_image = image.resize((new_width, new_height))

    return resized_image

def create_init_noise(image_size):
    w, h = image_size
    image_array = np.random.rand(h, w, 3) * 255
    image = Image.fromarray(image_array.astype("uint8")).convert("RGB")
    return image

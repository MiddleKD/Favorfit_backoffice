from Favorfit_diffusion.inference import call_tokenizer, call_diffusion_model, call_controlnet_model, make_multi_controlnet_model,\
    text_to_image_controlnet, inpainting_controlnet
from Favorfit_remove_bg.inference import call_model as model_remove_bg, inference as inference_remove_bg
from Favorfit_recommend_template.inference import inference as inference_recommend_color
from Favorfit_image_to_text.blip_image_to_text import load_interrogator as model_blip, inference as inference_blip
from Favorfit_image_to_text.clip_image_to_text import load_interrogator as model_clip, inference as inference_clip
from Favorfit_super_resolution.inference import call_model as model_super_resolution, inference as inference_super_resolution
from Favorfit_color_enhancement.inference import inference as inference_color_enhancement
from postprocess_remove_bg.rmbg_postprocess import MaskPostProcessor
from utils import *

import os

import torch
from flask import Flask, request

root_model_dir = None
root_model_diffusion_dir = None

model_storage = {
    "tokenizer":None, 
    "diffusion_models":{"clip":None,
                        "encoder":None, 
                        "decoder":None, 
                        "diffusion":None,},
}

app = Flask(__name__)


def prepare_diffusion_models():

    model_storage["tokenizer"] = call_tokenizer()
    model_storage["diffusion_models"] = call_diffusion_model(
        diffusion_state_dict_path=os.path.join(root_model_diffusion_dir, "favorfit_base.pth"),
        lora_state_dict_path=os.path.join(root_model_diffusion_dir, "lora/favorfit_lora.pth"),
    )
    model_storage["controlnet_shuffle"] = call_controlnet_model(os.path.join(root_model_diffusion_dir, "controlnet/shuffle.pth"))
    model_storage["controlnet_canny"] = call_controlnet_model(os.path.join(root_model_diffusion_dir, "controlnet/canny.pth"))
    model_storage["controlnet_outpaint"] = call_controlnet_model(os.path.join(root_model_diffusion_dir, "controlnet/outpaint_v2.pth"))
    model_storage["controlnet_depth"] = call_controlnet_model(os.path.join(root_model_diffusion_dir, "controlnet/depth.pth"))

def remove_bg(img_pil):
    if model_storage.get("remove_bg") is None:
        model = model_remove_bg(os.path.join(root_model_dir, "remove_bg/remove_bg.pth"), device="cuda")
    else:
        model = model_storage["remove_bg"].to("cuda")
    mask_pil = inference_remove_bg(img_pil, model)
    model_storage["remove_bg"] = model.to("cpu")
    return mask_pil

def mask_post_process(mask_pil):
    model = MaskPostProcessor()
    mask_pil = model(mask_pil)
    return mask_pil

def recommend_colors(img_pil, mask_pil=None):
    if mask_pil==None:
        mask_pil = remove_bg(img_pil, post_process=True)
    result_dict = inference_recommend_color(img_pil=img_pil, mask_pil=mask_pil)
    return result_dict

def color_enhancement(img_pil, gamma=0.75, factor=1.7):
    result = inference_color_enhancement(img_pil, gamma, factor)
    return result

def text_to_image_blip(img_pil):
    if model_storage.get("blip") is None:
        model = model_blip(os.path.join(root_model_dir, "image_to_text/blip/blip_large.pth"), device="cuda")
    else:
        model = model_storage["blip"].to("cuda")

    caption = inference_blip(img_pil, model)
    model_storage["blip"] = model.to("cpu")
    return caption

def text_to_image_clip(img_pil):
    if model_storage.get("clip") is None:
        model = model_clip(os.path.join(root_model_dir, "image_to_text/clip"), device="cuda")
    else:
        model = model_storage["clip"].to("cuda")

    caption = inference_clip(img_pil, model)
    model_storage["clip"] = model.to("cpu")
    return caption

def super_resolution(img_pil):
    if model_storage.get("super_resolution") is None:
        model = model_super_resolution(os.path.join(root_model_dir, "super_resolution/super_resolution_x4.pth"), device="cuda")
    else:
        model = model_storage["super_resolution"].to("cuda")

    image_pil = inference_super_resolution(img_pil, model)
    model_storage["super_resolution"] = model.to("cpu")
    return image_pil



@app.route('/')
def call_main():
    return 'Hello, This is Favorfit Back Office backend'


@app.route('/utils/remove_bg/', methods=["POST"])
def remove_bg_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)
    
    mask_pil = remove_bg(img_pil=img_pil)
    torch.cuda.empty_cache()

    if args.get("post_proces") == True:
        mask_pil = mask_post_process(mask_pil=mask_pil)

    result = {"image_b64": "data:application/octet-stream;base64," + pil_to_bs64(mask_pil)}
    return respond(None, result)


@app.route('/utils/remove_bg/postprocess/', methods=["POST"])
def mask_post_process_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    mask_bs64 = args["mask_b64"]
    mask_pil = bs64_to_pil(mask_bs64)

    mask_pil = mask_post_process(mask_pil=mask_pil)

    result = {"image_b64": "data:application/octet-stream;base64," + pil_to_bs64(mask_pil)}
    return respond(None, result)


@app.route('/utils/recommend_colors/', methods=["POST"])
def recommend_colors_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)

    if args.get("mask_b64") is not None:
        mask_bs64 = args["mask_b64"]
        mask_pil = bs64_to_pil(mask_bs64)
    else:
        mask_pil=None

    result = recommend_colors(img_pil=img_pil, mask_pil=mask_pil)
    return respond(None, result)


@app.route('/utils/color_enhancement/', methods=["POST"])
def color_enhancement_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)

    if args.get("gamma") is not None:
        gamma = args["gamma"]
    else:
        gamma = None
    
    if args.get("factor") is not None:
        factor = args["factor"]
    else:
        factor = None

    image_pil = color_enhancement(img_pil=img_pil, gamma=gamma, factor=factor)
    result = {"image_b64": "data:application/octet-stream;base64," + pil_to_bs64(image_pil)}
    return respond(None, result)


@app.route('/utils/text_to_image/blip/', methods=["POST"])
def text_to_image_blip_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)

    result = text_to_image_blip(img_pil=img_pil)
    torch.cuda.empty_cache()

    return respond(None, {"caption":result})


@app.route('/utils/text_to_image/clip/', methods=["POST"])
def text_to_image_clip_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)

    result = text_to_image_clip(img_pil=img_pil)
    torch.cuda.empty_cache()

    return respond(None, {"caption":result})


@app.route('/utils/super_resolution/', methods=["POST"])
def super_resolution_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)

    image_pil = super_resolution(img_pil=img_pil)
    torch.cuda.empty_cache()
    
    result = {"image_b64": "data:application/octet-stream;base64," + pil_to_bs64(image_pil)}
    return respond(None, result)


@app.route('/diffusion/outpaint/', methods=["POST"])
def outpaint_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)

    mask_pil = mask_post_process(remove_bg(img_pil))
    control_pil = make_outpaint_condition(img_pil, mask_pil)
    caption = text_to_image_blip(img_pil)

    model_storage["diffusion_models"].update(model_storage["controlnet_outpaint"])

    output_pils = text_to_image_controlnet(
        control_image=control_pil,
        prompt=f"professional photography, natural shadow, {caption}, realistic, high resolution, 8k",
        uncond_prompt="low quality, worst quality, wrinkled, deformed, distorted, jpeg artifacts,nsfw, paintings, sketches, text, watermark, username, spikey",
        num_per_image=1,
        lora_scale=0.7,
        controlnet_scale=1.0,
        models=model_storage["diffusion_models"],
        seeds=-1,
        device="cuda",
        tokenizer=model_storage["tokenizer"]
    )
    
    model_storage["diffusion_models"].pop("controlnet")
    model_storage["diffusion_models"].pop("controlnet_embedding")
    torch.cuda.empty_cache()

    result = {"image_b64_list": ["data:application/octet-stream;base64," + pil_to_bs64(cur) for cur in output_pils]}
    return respond(None, result)


@app.route('/diffusion/composition/', methods=["POST"])
def composition_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)

    mask_bs64 = args["mask_b64"]
    mask_pil = bs64_to_pil(mask_bs64)

    control_pil = make_canny_condition(img_pil)
    caption = text_to_image_clip(img_pil)

    model_storage["diffusion_models"].update(model_storage["controlnet_canny"])

    output_pils = inpainting_controlnet(
        input_image=img_pil,
        mask_image=mask_pil,
        control_image=control_pil,
        prompt=f"professional photography, natural shadow, {caption}, realistic, high resolution, 8k",
        uncond_prompt="low quality, worst quality, wrinkled, deformed, distorted, jpeg artifacts,nsfw, paintings, sketches, text, watermark, username, spikey",
        num_per_image=1,
        strength=0.6,
        lora_scale=0.7,
        controlnet_scale=1.0,
        models=model_storage["diffusion_models"],
        seeds=-1,
        device="cuda",
        tokenizer=model_storage["tokenizer"]
    )

    model_storage["diffusion_models"].pop("controlnet")
    model_storage["diffusion_models"].pop("controlnet_embedding")
    torch.cuda.empty_cache()

    result = {"image_b64_list": ["data:application/octet-stream;base64," + pil_to_bs64(cur) for cur in output_pils]}
    return respond(None, result)


@app.route('/diffusion/augmentation/style/', methods=["POST"])
def augmentation_base_style():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_base_bs64 = args["image_b64_base"]
    img_base_pil = bs64_to_pil(img_base_bs64)

    img_style_bs64 = args["image_b64_style"]
    img_style_pil = bs64_to_pil(img_style_bs64)

    control_canny_pil = make_canny_condition(img_base_pil)
    control_shuffle_pil = make_shuffle_condition(img_style_pil)
    caption = text_to_image_blip(img_base_pil)

    multi_control_model = make_multi_controlnet_model(
        [model_storage["controlnet_shuffle"], model_storage["controlnet_canny"]]
    )
    model_storage["diffusion_models"].update(multi_control_model)

    output_pils = text_to_image_controlnet(
        control_image=[control_shuffle_pil, control_canny_pil],
        prompt=f"professional photography, natural shadow, {caption}, realistic, high resolution, 8k",
        uncond_prompt="low quality, worst quality, wrinkled, deformed, distorted, jpeg artifacts,nsfw, paintings, sketches, text, watermark, username, spikey",
        num_per_image=1,
        lora_scale=0.7,
        models=model_storage["diffusion_models"],
        seeds=-1,
        device="cuda",
        tokenizer=model_storage["tokenizer"]
    )

    model_storage["diffusion_models"].pop("controlnet")
    model_storage["diffusion_models"].pop("controlnet_embedding")
    torch.cuda.empty_cache()

    result = {"image_b64_list": ["data:application/octet-stream;base64," + pil_to_bs64(cur) for cur in output_pils]}
    return respond(None, result) 


@app.route('/diffusion/augmentation/text/', methods=["POST"])
def augmentation_base_text():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)

    color = args["color"]
    concept = args["concept"]

    control_pil = make_canny_condition(img_pil)

    model_storage["diffusion_models"].update(model_storage["controlnet_canny"])

    output_pils = text_to_image_controlnet(
        control_image=control_pil,
        prompt=f"{{FavorfitStyle}}, {color} theme, {concept},table, pastel color, masterpiece, best quality, no human, artwork, still life, clean, Comfortable natural light, minimalist, modernist, 8K",
        uncond_prompt="shiny, light reflection, low quality, worst quality, deformed, distorted, jpeg artifacts, paintings, nsfw, sketches, text, watermark, username, signature, brand name, icon, spikey",
        num_per_image=1,
        lora_scale=0.7,
        models=model_storage["diffusion_models"],
        seeds=-1,
        device="cuda",
        tokenizer=model_storage["tokenizer"]
    )

    model_storage["diffusion_models"].pop("controlnet")
    model_storage["diffusion_models"].pop("controlnet_embedding")
    torch.cuda.empty_cache()

    result = {"image_b64_list": ["data:application/octet-stream;base64," + pil_to_bs64(cur) for cur in output_pils]}
    return respond(None, result)


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Favorfit Back Office backend")
    parser.add_argument(
        "--root_model_path",
        type=str,
        default="/home/mlfavorfit/lib/favorfit/kjg/0_model_weights",
    )
    parser.add_argument(
        "--root_model_diffusion_path",
        type=str,
        default="/home/mlfavorfit/lib/favorfit/kjg/0_model_weights/diffusion/FavorfitArchitecture",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    root_model_dir = args.root_model_path
    root_model_diffusion_dir = args.root_model_diffusion_path
    
    prepare_diffusion_models()

    app.run(host=args.host, debug=args.debug, port=args.port)

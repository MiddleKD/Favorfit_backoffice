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

root_model_dir = None
root_model_diffusion_dir = None

model_storage = {
    "tokenizer":None, 
    "diffusion_models":{"clip":None,
                        "encoder":None, 
                        "decoder":None, 
                        "diffusion":None,},
    "diffusion_models_inpaint":{"clip":None,
                        "encoder":None, 
                        "decoder":None, 
                        "diffusion":None,},
}

def set_root_model_path(root_model_dir_path, root_model_diffusion_dir_path):
    global root_model_dir, root_model_diffusion_dir
    root_model_dir = root_model_dir_path
    root_model_diffusion_dir = root_model_diffusion_dir_path

def prepare_ai_models():

    model_storage["tokenizer"] = call_tokenizer()
    model_storage["diffusion_models"] = call_diffusion_model(
        diffusion_state_dict_path=os.path.join(root_model_diffusion_dir, "favorfit_base.pth"),
        lora_state_dict_path=os.path.join(root_model_diffusion_dir, "lora/favorfit_lora.pth"),
    )
    model_storage["diffusion_models_inpaint"] = call_diffusion_model(
        diffusion_state_dict_path=os.path.join(root_model_diffusion_dir, "favorfit_inpaint.pth"),
        lora_state_dict_path=os.path.join(root_model_diffusion_dir, "lora/favorfit_lora.pth"),
    )
    model_storage["controlnet_shuffle"] = call_controlnet_model(os.path.join(root_model_diffusion_dir, "controlnet/shuffle.pth"))
    model_storage["controlnet_canny"] = call_controlnet_model(os.path.join(root_model_diffusion_dir, "controlnet/canny.pth"))
    model_storage["controlnet_outpaint"] = call_controlnet_model(os.path.join(root_model_diffusion_dir, "controlnet/outpaint_v2.pth"))
    model_storage["controlnet_depth"] = call_controlnet_model(os.path.join(root_model_diffusion_dir, "controlnet/depth.pth"))

    model_storage["remove_bg"] = model_remove_bg(os.path.join(root_model_dir, "remove_bg/remove_bg.pth"), device="cuda")
    model_storage["blip"] = model_blip(os.path.join(root_model_dir, "image_to_text/blip/blip_large.pth"), device="cuda")
    model_storage["clip"] = model_clip(os.path.join(root_model_dir, "image_to_text/clip"), device="cuda")
    model_storage["super_resolution"] = model_super_resolution(os.path.join(root_model_dir, "super_resolution/super_resolution_x4.pth"), device="cuda")


def remove_bg(img_pil, post_process=True, return_dict=False):
    torch.cuda.empty_cache()
    model = model_storage["remove_bg"].to("cuda")
    mask_pil = inference_remove_bg(img_pil, model)
    model_storage["remove_bg"] = model.to("cpu")

    if post_process == True:
        mask_pil = mask_post_process(mask_pil)

    if return_dict == True:
        return {"type":"remove_bg", "image_b64": "data:application/octet-stream;base64," + pil_to_bs64(mask_pil)}
    else:
        return mask_pil

def mask_post_process(mask_pil, return_dict=False):
    model = MaskPostProcessor()
    mask_pil = model(mask_pil)
    if return_dict == True:
        return {"type":"mask_post_process", "image_b64": "data:application/octet-stream;base64," + pil_to_bs64(mask_pil)}
    else:
        return mask_pil

def recommend_colors(img_pil, mask_pil=None, return_dict=False):
    if mask_pil==None:
        mask_pil = remove_bg(img_pil)
    result_dict = inference_recommend_color(img_pil=img_pil, mask_pil=mask_pil)
    if return_dict == True:
        return {"type":"recommend_colors", **result_dict}
    else:
        return result_dict

def color_enhancement(img_pil, gamma=0.75, factor=1.7, return_dict=False):
    image_pil = inference_color_enhancement(img_pil, gamma, factor)
    if return_dict == True:
        return {"type":"color_enhancement", "image_b64": "data:application/octet-stream;base64," + pil_to_bs64(image_pil)}
    else:
        return image_pil

def text_to_image_blip(img_pil, return_dict=False):
    torch.cuda.empty_cache()
    model = model_storage["blip"].to("cuda")

    caption = inference_blip(img_pil, model)
    model_storage["blip"] = model.to("cpu")

    if return_dict == True:
        return {"type":"blip", "caption": caption}
    else:
        return caption

def text_to_image_clip(img_pil, return_dict=False):
    torch.cuda.empty_cache()
    model = model_storage["clip"].to("cuda")

    caption = inference_clip(img_pil, model)
    model_storage["clip"] = model.to("cpu")
    
    if return_dict == True:
        return {"type":"clip", "caption": caption}
    else:
        return caption

def super_resolution(img_pil, return_dict=False):
    torch.cuda.empty_cache()
    model = model_storage["super_resolution"].to("cuda")

    image_pil = inference_super_resolution(img_pil, model)
    model_storage["super_resolution"] = model.to("cpu")
    
    if return_dict == True:
        return {"type":"super_resolution", "image_b64": "data:application/octet-stream;base64," + pil_to_bs64(image_pil)}
    else:
        return image_pil


def outpaint(img_pil, mask_pil, num_per_image, return_dict=False):
    torch.cuda.empty_cache()
    img_pil = resize_store_ratio(img_pil)
    mask_pil = resize_store_ratio(mask_pil)
    control_pil = make_outpaint_condition(img_pil, mask_pil)
    caption = text_to_image_blip(img_pil)
    
    model_storage["diffusion_models"].update(model_storage["controlnet_outpaint"])

    output_pils = text_to_image_controlnet(
        control_image=control_pil,
        prompt=f"professional photography, natural shadow, {caption}, realistic, high resolution, 8k",
        uncond_prompt="low quality, worst quality, wrinkled, deformed, distorted, jpeg artifacts,nsfw, paintings, sketches, text, watermark, username, spikey",
        num_per_image=num_per_image,
        lora_scale=0.7,
        controlnet_scale=1.0,
        models=model_storage["diffusion_models"],
        seeds=-1,
        device="cuda",
        tokenizer=model_storage["tokenizer"]
    )
    
    model_storage["diffusion_models"].pop("controlnet")
    model_storage["diffusion_models"].pop("controlnet_embedding")

    if return_dict == True:
        return {"image_b64_list": ["data:application/octet-stream;base64," + pil_to_bs64(cur) for cur in output_pils]}
    else:
        return output_pils

def composition(img_pil, mask_pil, num_per_image, return_dict=False):
    torch.cuda.empty_cache()
    img_pil = resize_store_ratio(img_pil)
    mask_pil = resize_store_ratio(mask_pil)
    control_pil = make_canny_condition(img_pil)
    caption = text_to_image_clip(img_pil)

    model_storage["diffusion_models_inpaint"].update(model_storage["controlnet_canny"])

    output_pils = inpainting_controlnet(
        input_image=img_pil,
        mask_image=mask_pil,
        control_image=control_pil,
        prompt=f"professional photography, natural shadow, {caption}, realistic, high resolution, 8k",
        uncond_prompt="low quality, worst quality, wrinkled, deformed, distorted, jpeg artifacts,nsfw, paintings, sketches, text, watermark, username, spikey",
        num_per_image=num_per_image,
        strength=0.6,
        lora_scale=0.7,
        controlnet_scale=1.0,
        models=model_storage["diffusion_models_inpaint"],
        seeds=-1,
        device="cuda",
        tokenizer=model_storage["tokenizer"]
    )

    model_storage["diffusion_models_inpaint"].pop("controlnet")
    model_storage["diffusion_models_inpaint"].pop("controlnet_embedding")

    if return_dict == True:
        return {"image_b64_list": ["data:application/octet-stream;base64," + pil_to_bs64(cur) for cur in output_pils]}
    else:
        return output_pils

def augmentation_base_style(img_base_pil, img_style_pil, num_per_image, return_dict=False):
    torch.cuda.empty_cache()
    img_base_pil = resize_store_ratio(img_base_pil)
    img_style_pil = resize_store_ratio(img_style_pil)
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
        num_per_image=num_per_image,
        lora_scale=0.7,
        models=model_storage["diffusion_models"],
        seeds=-1,
        device="cuda",
        tokenizer=model_storage["tokenizer"]
    )

    model_storage["diffusion_models"].pop("controlnet")
    model_storage["diffusion_models"].pop("controlnet_embedding")

    if return_dict == True:
        return {"image_b64_list": ["data:application/octet-stream;base64," + pil_to_bs64(cur) for cur in output_pils]}
    else:
        return output_pils

def augmentation_base_text(img_pil, color, concept, num_per_image, return_dict=False):
    torch.cuda.empty_cache()
    img_pil = resize_store_ratio(img_pil)
    control_pil = make_canny_condition(img_pil)

    model_storage["diffusion_models"].update(model_storage["controlnet_canny"])

    output_pils = text_to_image_controlnet(
        control_image=control_pil,
        prompt=f"{{FavorfitStyle}}, {color} theme, {concept},table, pastel color, masterpiece, best quality, no human, artwork, still life, clean, Comfortable natural light, minimalist, modernist, 8K",
        uncond_prompt="shiny, light reflection, low quality, worst quality, deformed, distorted, jpeg artifacts, paintings, nsfw, sketches, text, watermark, username, signature, brand name, icon, spikey",
        num_per_image=num_per_image,
        lora_scale=0.7,
        models=model_storage["diffusion_models"],
        seeds=-1,
        device="cuda",
        tokenizer=model_storage["tokenizer"]
    )

    model_storage["diffusion_models"].pop("controlnet")
    model_storage["diffusion_models"].pop("controlnet_embedding")

    if return_dict == True:
        return {"image_b64_list": ["data:application/octet-stream;base64," + pil_to_bs64(cur) for cur in output_pils]}
    else:
        return output_pils

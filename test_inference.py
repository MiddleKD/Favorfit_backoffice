from Favorfit_diffusion.inference import call_tokenizer, call_diffusion_model, call_controlnet_model, make_multi_controlnet_model,\
    text_to_image_controlnet, inpainting_controlnet
from Favorfit_remove_bg.inference import call_model as model_remove_bg, inference as inference_remove_bg
from Favorfit_recommend_template.inference import inference as inference_recommend_color
from Favorfit_image_to_text.blip_image_to_text import load_interrogator as model_blip, inference as inference_blip
from Favorfit_image_to_text.clip_image_to_text import load_interrogator as model_clip, inference as inference_clip
from Favorfit_super_resolution.inference import call_model as model_super_resolution, inference as inference_super_resolution
from postprocess_remove_bg.rmbg_postprocess import MaskPostProcessor

import os
from PIL import Image
from time import time
import torch

model_storage = {
    "tokenizer":None, 
    "diffusion_models":{"clip":None,
                        "encoder":None, 
                        "decoder":None, 
                        "diffusion":None,},
}


# Outpaint
def outpaint():
    model_storage["diffusion_models"].update(model_storage["controlnet_outpaint"])

    control_image = Image.open("./inference_test_image/object.png").convert("RGB")
    caption = 'a close up shot of a green beverage in a glass'

    result = text_to_image_controlnet(
        control_image=control_image,
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

    return result


# Composition
def composition():
    model_storage["diffusion_models"].update(model_storage["controlnet_canny"])

    input_image = Image.open("./inference_test_image/object_with_template.png").convert("RGB")
    control_image = Image.open("./inference_test_image/object_with_template_canny.png").convert("RGB")
    mask_image = Image.open("./inference_test_image/template_mask.png").convert("L")
    caption = 'blue beverage with ice and lime in glass'

    result = inpainting_controlnet(
        input_image=input_image,
        mask_image=mask_image,
        control_image=control_image,
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

    return result


# Augmentation base style
def augmentation_base_style():
    multi_control_model = make_multi_controlnet_model(
        [model_storage["controlnet_shuffle"], model_storage["controlnet_canny"]]
    )
    model_storage["diffusion_models"].update(multi_control_model)

    control_shuffle_image = Image.open("./inference_test_image/shuffle.png").convert("RGB")
    control_canny_image = Image.open("./inference_test_image/canny_template.png").convert("RGB")
    caption = 'two empty wooden plates, one of which is holding dried flowers and another has a'

    result = text_to_image_controlnet(
        control_image=[control_shuffle_image, control_canny_image],
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

    return result


# Augmentation base text
def augmentation_base_text(theme="", concept=""):
    model_storage["diffusion_models"].update(model_storage["controlnet_depth"])

    control_image = Image.open("./inference_test_image/depth.png").convert("RGB")

    result = text_to_image_controlnet(
        control_image=control_image,
        prompt=f"{{FavorfitStyle}}, {theme} theme, {concept},table, pastel color, masterpiece, best quality, no human, artwork, still life, clean, Comfortable natural light, minimalist, modernist, 8K",
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

    return result


def prepare_diffusion_models(root_model_dir):

    model_storage["tokenizer"] = call_tokenizer()
    model_storage["diffusion_models"] = call_diffusion_model(
        diffusion_state_dict_path=os.path.join(root_model_dir, "favorfit_base.pth"),
        lora_state_dict_path=os.path.join(root_model_dir, "lora/favorfit_lora.pth"),
    )
    model_storage["controlnet_shuffle"] = call_controlnet_model(os.path.join(root_model_dir, "controlnet/shuffle.pth"))
    model_storage["controlnet_canny"] = call_controlnet_model(os.path.join(root_model_dir, "controlnet/canny.pth"))
    model_storage["controlnet_outpaint"] = call_controlnet_model(os.path.join(root_model_dir, "controlnet/outpaint_v2.pth"))
    model_storage["controlnet_depth"] = call_controlnet_model(os.path.join(root_model_dir, "controlnet/depth.pth"))

def test_diffusion_model(root_model_dir):
    torch.cuda.empty_cache()
    
    if model_storage["diffusion_models"]["diffusion"] is None:
        print("Prepare diffusion models:")
        start_time = time()
        prepare_diffusion_models(root_model_dir)
        print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")
    else:
        start_time = time()
        print("Load diffusion models:")
        print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")

    print("Run outpaint:")
    start_time = time()
    outpaint()[0].show()
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")

    print("Run compostion:")
    start_time = time()
    composition()[0].show()
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")

    print("Run augmentation base style:")
    start_time = time()
    augmentation_base_style()[0].show()
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")

    print("Run augmentation base text:")
    start_time = time()
    augmentation_base_text(theme="red", concept="marble")[0].show()
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")

    torch.cuda.empty_cache()
    # prompt_concept_blocks = ["fabric", "marble", "geometry", "wood", "stone"]
    # prompt_color_blocks = ["red theme", "pink theme", "orange theme", "yellow theme", "brown theme", "turquoise theme", "green theme", "blue theme", "purple theme", "black theme", "white theme"]


def test_util_model(img_path, root_dir):
    torch.cuda.empty_cache()

    # Open image file
    img_pil = Image.open(img_path).resize([500, 600])


    # remove bg
    start_time = time()
    if model_storage.get("remove_bg") is None:
        print("remove_bg init")
        model = model_remove_bg(os.path.join(root_dir, "remove_bg/remove_bg.pth"), device="cuda")
    else:
        print("remove_bg load")
        model = model_storage["remove_bg"].to("cuda")
    mask_pil = inference_remove_bg(img_pil, model)
    model_storage["remove_bg"] = model.to("cpu")
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")


    # post precess mask
    model = MaskPostProcessor()
    mask_pil = model(mask_pil)

    
    # recommend color
    start_time = time()
    recommend_colors = inference_recommend_color(img_pil=img_pil, mask_pil=mask_pil)
    print("Recommend colors: ", recommend_colors)
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")


    # text to image blip
    start_time = time()
    if model_storage.get("blip") is None:
        print("BLIP init")
        model = model_blip(os.path.join(root_dir, "image_to_text/blip/blip_large.pth"), device="cuda")
    else:
        print("BLIP load")
        model = model_storage["blip"].to("cuda")
    caption = inference_blip(img_pil, model)
    model_storage["blip"] = model.to("cpu")
    print("BLIP caption: ", caption)
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")


    # text to image clip
    start_time = time()
    if model_storage.get("clip") is None:
        print("CLIP init")
        model = model_clip(os.path.join(root_dir, "image_to_text/clip"), device="cuda")
    else:
        print("CLIP load")
        model = model_storage["clip"].to("cuda")
    caption = inference_clip(img_pil, model)
    model_storage["clip"] = model.to("cpu")
    print("CLIP caption: ", caption)
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")


    # super resolution
    start_time = time()
    if model_storage.get("super_resolution") is None:
        print("SR init")
        model = model_super_resolution(os.path.join(root_dir, "super_resolution/super_resolution_x4.pth"), device="cuda")
    else:
        print("SR load")
        model = model_storage["super_resolution"].to("cuda")
    high_resolution_img = inference_super_resolution(img_pil, model)
    model_storage["super_resolution"] = model.to("cpu")
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    img_path = "./inference_test_image/template.png"

    root_model_dir = "/home/mlfavorfit/lib/favorfit/kjg/0_model_weights"
    root_model_diffusion_dir = "/home/mlfavorfit/lib/favorfit/kjg/0_model_weights/diffusion/FavorfitArchitecture"

    test_util_model(img_path, root_model_dir)
    test_diffusion_model(root_model_diffusion_dir)

    test_util_model(img_path, root_model_dir)
    test_diffusion_model(root_model_diffusion_dir)

    import time
    time.sleep(10)

    print("Test Success!")

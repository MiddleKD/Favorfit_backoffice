from model_api import *
from time import time

def test_util_model():
    torch.cuda.empty_cache()

    # Open image file
    img_pil = Image.open("./inference_test_image/template.png").resize([512, 512])


    # remove bg
    start_time = time()
    mask_pil = remove_bg(img_pil=img_pil, post_process=False)
    print("Remove BG: ", mask_pil)
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")


    # post precess mask
    mask_pil = mask_post_process(mask_pil=mask_pil)

    
    # recommend color
    start_time = time()
    recommend_colors_result = recommend_colors(img_pil=img_pil, mask_pil=mask_pil)
    print("Recommend colors: ", recommend_colors_result)
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")


    # color enhancement
    start_time = time()
    color_enhancement_result = color_enhancement(img_pil=img_pil)
    print("Color enhancement: ", color_enhancement_result)
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")


    # text to image blip
    start_time = time()
    caption = text_to_image_blip(img_pil=img_pil)
    print("BLIP caption: ", caption)
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")


    # text to image clip
    start_time = time()
    caption = text_to_image_clip(img_pil=img_pil)
    print("CLIP caption: ", caption)
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")


    # super resolution
    start_time = time()
    super_resolution_result = super_resolution(img_pil=img_pil)
    print("Super resolution: ", super_resolution_result.size)
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")

    torch.cuda.empty_cache()


def test_diffusion_model():
    torch.cuda.empty_cache()
    
    print("Run outpaint:")
    start_time = time()
    img_pil = Image.open("./inference_test_image/object.png").convert("RGB")
    mask_pil = Image.open("./inference_test_image/mask.png").convert("RGB")
    outpaint(img_pil=img_pil, mask_pil=mask_pil, num_per_image=1)[0].show()
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")

    print("Run compostion:")
    start_time = time()
    img_pil = Image.open("./inference_test_image/object_with_template.png").convert("RGB")
    mask_pil = Image.open("./inference_test_image/template_mask.png").convert("RGB")
    composition(img_pil=img_pil, mask_pil=mask_pil, num_per_image=1)[0].show()
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")

    print("Run augmentation base style:")
    start_time = time()
    img_base_pil = Image.open("./inference_test_image/template.png").convert("RGB")
    img_style_pil = Image.open("./inference_test_image/shuffle.jpg").convert("RGB")
    augmentation_base_style(img_base_pil=img_base_pil, img_style_pil=img_style_pil, num_per_image=1)[0].show()
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")

    print("Run augmentation base text:")
    start_time = time()
    img_pil = Image.open("./inference_test_image/template.png").convert("RGB")
    augmentation_base_text(img_pil=img_pil, color="red", concept="marble", num_per_image=1)[0].show()
    print(f"\ttakes time: {time()-start_time:.2f}", end="\n\n")

    torch.cuda.empty_cache()


if __name__ == "__main__":
    root_model_dir = "/home/mlfavorfit/lib/favorfit/kjg/0_model_weights"
    root_model_diffusion_dir = "/home/mlfavorfit/lib/favorfit/kjg/0_model_weights/diffusion/FavorfitArchitecture"

    set_root_model_path(root_model_dir_path=root_model_dir, root_model_diffusion_dir_path=root_model_diffusion_dir)
    prepare_ai_models()

    test_util_model()
    test_diffusion_model()

    test_util_model()
    test_diffusion_model()

    import time
    time.sleep(10)

    print("Test Success!")

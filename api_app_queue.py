from flask import Flask, request, jsonify
from queue import Queue, Empty
import threading
import atexit

from model_api import *

app = Flask(__name__)
request_queue = Queue()
response_queue = Queue()
lock = threading.Lock()

def queue_process():
    while True:
        data = request_queue.get()

        if data is None:
            break

        process_function = data.get("process_function")
        result = process_function(**data.get("params"))
        result["request_id"] = data.get("request_id")

        with lock:
            response_queue.put(result)

        request_queue.task_done()

def enqueue_request(data):
    with lock:
        request_queue.put(data)

# 백그라운드 스레드에서 요청 처리
background_thread = threading.Thread(target=queue_process)
background_thread.start()

# Flask 앱이 완전히 종료될 때 처리할 함수 등록
def cleanup():
    request_queue.put(None)
    background_thread.join()

# Flask 앱이 완전히 종료될 때 cleanup 함수를 호출하도록 설정
atexit.register(cleanup)


@app.route('/')
def call_main():
    return 'Hello, This is Favorfit Back Office backend'


@app.route('/get_result/', methods=['GET'])
def get_result():
    try:
        result_data = response_queue.get(block=False)
        return respond(None, result_data)
    except Empty:
        return respond(None, {"state":"Empty resoponse queue", "request_qsize":request_queue.qsize()})


@app.route('/utils/remove_bg/', methods=["POST"])
def remove_bg_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)
    
    enqueue_request({"process_function":remove_bg, 
                     "request_id":args.get("request_id",None),
                     "params":{
                         "img_pil":img_pil,
                         "post_process":args.get("post_process",False),
                         "return_dict":True,
                        }})
    
    return respond(None, {"state":"queued", "type":"remove_bg", "request_qsize":request_queue.qsize()})


@app.route('/utils/remove_bg/postprocess/', methods=["POST"])
def mask_post_process_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    mask_bs64 = args["mask_b64"]
    mask_pil = bs64_to_pil(mask_bs64)

    result_dict = mask_post_process(mask_pil=mask_pil, return_dict=True)
    return respond(None, result_dict)


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

    enqueue_request({"process_function":recommend_colors, 
                     "request_id":args.get("request_id",None),
                     "params":{
                         "img_pil":img_pil,
                         "mask_pil":mask_pil,
                         "return_dict":True,
                         }})

    return respond(None, {"state":"queued", "type":"recommend_colors", "request_qsize":request_queue.qsize()})


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
    
    enqueue_request({"process_function":color_enhancement, 
                     "request_id":args.get("request_id",None),
                     "params":{
                         "img_pil":img_pil,
                         "gamma":gamma,
                         "factor":factor,
                         "return_dict":True,
                         }})
    
    return respond(None, {"state":"queued", "type":"color_enhancement", "request_qsize":request_queue.qsize()})


@app.route('/utils/text_to_image/blip/', methods=["POST"])
def text_to_image_blip_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)

    enqueue_request({"process_function":text_to_image_blip, 
                     "request_id":args.get("request_id",None),
                     "params":{
                         "img_pil":img_pil,
                         "return_dict":True,
                         }})

    return respond(None, {"state":"queued", "type":"blip", "request_qsize":request_queue.qsize()})


@app.route('/utils/text_to_image/clip/', methods=["POST"])
def text_to_image_clip_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)

    enqueue_request({"process_function":text_to_image_clip, 
                     "request_id":args.get("request_id",None),
                     "params":{
                         "img_pil":img_pil,
                         "return_dict":True,
                         }})

    return respond(None, {"state":"queued", "type":"clip", "request_qsize":request_queue.qsize()})


@app.route('/utils/super_resolution/', methods=["POST"])
def super_resolution_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)

    enqueue_request({"process_function":super_resolution, 
                     "request_id":args.get("request_id",None),
                     "params":{
                         "img_pil":img_pil,
                         "return_dict":True,
                         }})
    
    return respond(None, {"state":"queued", "type":"super_resolution", "request_qsize":request_queue.qsize()})

    
@app.route('/diffusion/outpaint/', methods=["POST"])
def outpaint_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)

    enqueue_request({"process_function":outpaint, 
                     "request_id":args.get("request_id",None),
                     "params":{
                         "img_pil":img_pil,
                         "num_per_image":args.get("num_per_image", 1),
                         "return_dict":True,
                         }})
    
    return respond(None, {"state":"queued", "type":"outpaint", "request_qsize":request_queue.qsize()})


@app.route('/diffusion/composition/', methods=["POST"])
def composition_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)

    mask_bs64 = args["mask_b64"]
    mask_pil = bs64_to_pil(mask_bs64)

    enqueue_request({"process_function":composition, 
                     "request_id":args.get("request_id",None),
                     "params":{
                         "img_pil":img_pil,
                         "mask_pil":mask_pil,
                         "num_per_image":args.get("num_per_image", 1),
                         "return_dict":True,
                         }})

    return respond(None, {"state":"queued", "type":"composition", "request_qsize":request_queue.qsize()})

    
@app.route('/diffusion/augmentation/style/', methods=["POST"])
def augmentation_base_style_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_base_bs64 = args["image_b64_base"]
    img_base_pil = bs64_to_pil(img_base_bs64)

    img_style_bs64 = args["image_b64_style"]
    img_style_pil = bs64_to_pil(img_style_bs64)

    enqueue_request({"process_function":augmentation_base_style, 
                     "request_id":args.get("request_id",None),
                     "params":{
                         "img_base_pil":img_base_pil,
                         "img_style_pil":img_style_pil,
                         "num_per_image":args.get("num_per_image", 1),
                         "return_dict":True,
                         }})

    return respond(None, {"state":"queued", "type":"augmentation_style", "request_qsize":request_queue.qsize()})


@app.route('/diffusion/augmentation/text/', methods=["POST"])
def augmentation_base_text_api():
    data = request.get_json()
    args = load_instance_from_json(data)

    img_bs64 = args["image_b64"]
    img_pil = bs64_to_pil(img_bs64)

    color = args["color"]
    concept = args["concept"]

    enqueue_request({"process_function":augmentation_base_text, 
                     "request_id":args.get("request_id",None),
                     "params":{
                         "img_pil":img_pil,
                         "color":color,
                         "concept":concept,
                         "num_per_image":args.get("num_per_image", 1),
                         "return_dict":True,
                         }})

    return respond(None, {"state":"queued", "type":"augmentation_text", "request_qsize":request_queue.qsize()})


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
    
    set_root_model_path(root_model_dir_path=args.root_model_path, root_model_diffusion_dir_path=args.root_model_diffusion_path)
    prepare_diffusion_models()

    app.run(host=args.host, debug=args.debug, port=args.port)

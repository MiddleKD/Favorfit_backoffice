from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from queue import Queue
import threading
import atexit

from model_api import *
from utils import TooMuchRequestQueueError, DupledRequestKeyError, RequestKeyDoesNotExistError

# app = Flask(__name__)
request_queue = Queue()
response_dict = {}
lock = threading.Lock()

def enqueue_request(data):
    with lock:
        request_queue.put(data)

request_queue_max_len = None
def check_request_queue_length(max_len=None):
    if max_len is None:
        max_len = response_dict_max_len
    if request_queue.qsize() > max_len:
        raise TooMuchRequestQueueError("try again in a few minutes")

response_dict_max_len = None
def cleanup_response_dict(max_len=None):
    if max_len is None:
        max_len = response_dict_max_len
    if len([True for cur in response_dict if response_dict[cur] is not None]) > max_len:
        print(f"WARNINIG: Response dict length is over max_len({max_len}), response dict will be cleaned up!")
        with lock:
            response_dict.clear()
    
def queue_process():
    while True:
        data = request_queue.get()

        if data is None:
            break
        
        try:
            request_id = data.get("request_id")
            process_function = data.get("process_function")
            result = process_function(**data.get("params"))
            result["request_id"] = request_id
        except Exception as e:
            result = {"state": "error " + str(e)}
        
        cleanup_response_dict(max_len=response_dict_max_len)

        with lock:
            response_dict[data.get("request_id")] = result

        request_queue.task_done()

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


@app.route('/get_result/', methods=['POST'])
def get_result():
    try:
        data = request.get_json()
        args = load_instance_from_json(data)
        result_data = response_dict.get(args["request_id"])
        
        if result_data is None: raise RequestKeyDoesNotExistError("request id dose not exist")

        response_dict.pop(args["request_id"])

        return respond(result_data.get("error"), result_data)
    except Exception as e:
        return respond(e, {"state": "error " + str(e), "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})


@app.route('/utils/remove_bg/', methods=["POST"])
def remove_bg_api():
    try:
        data = request.get_json()
        args = load_instance_from_json(data)

        img_bs64 = args["image_b64"]
        img_pil = bs64_to_pil(img_bs64)
        
        result_json = remove_bg(img_pil=img_pil, 
                                post_process=args.get("post_process",False),
                                box=args.get("box",None),
                                return_dict=True)
    except Exception as e:
        return respond(e, {"state": "error " + str(e), "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})
    return respond(None, result_json)


@app.route('/utils/remove_bg/postprocess/', methods=["POST"])
def mask_post_process_api():
    try:
        data = request.get_json()
        args = load_instance_from_json(data)

        mask_bs64 = args["mask_b64"]
        mask_pil = bs64_to_pil(mask_bs64)

        result_dict = mask_post_process(mask_pil=mask_pil, 
                                        return_dict=True)
    except Exception as e:
        return respond(e, {"state": "error " + str(e), "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})
    return respond(None, result_dict)


@app.route('/utils/recommend_colors/', methods=["POST"])
def recommend_colors_api():
    try:
        data = request.get_json()
        args = load_instance_from_json(data)

        img_bs64 = args["image_b64"]
        img_pil = bs64_to_pil(img_bs64)

        if args.get("mask_b64") is not None:
            mask_bs64 = args["mask_b64"]
            mask_pil = bs64_to_pil(mask_bs64)
        else:
            mask_pil=None

        result_dict = recommend_colors(img_pil=img_pil,
                                    mask_pil=mask_pil,
                                    return_dict=True)
    except Exception as e:
        return respond(e, {"state": "error " + str(e), "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})
    return respond(None, result_dict)


@app.route('/utils/color_enhancement/', methods=["POST"])
def color_enhancement_api():
    try:
        data = request.get_json()
        args = load_instance_from_json(data)

        img_bs64 = args["image_b64"]
        img_pil = bs64_to_pil(img_bs64)

        if args.get("gamma") is not None:
            gamma = args["gamma"]
        else:
            gamma = 0.75
        
        if args.get("factor") is not None:
            factor = args["factor"]
        else:
            factor = 1.7
        
        result_dict = color_enhancement(img_pil=img_pil,
                                        gamma=gamma,
                                        factor=factor,
                                        return_dict=True)
    except Exception as e:
        return respond(e, {"state": "error " + str(e), "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})
    return respond(None, result_dict)


@app.route('/utils/text_to_image/blip/', methods=["POST"])
def text_to_image_blip_api():
    try:
        data = request.get_json()
        args = load_instance_from_json(data)

        img_bs64 = args["image_b64"]
        img_pil = bs64_to_pil(img_bs64)

        result_dict = text_to_image_blip(img_pil=img_pil,
                                        return_dict=True)
    except Exception as e:
        return respond(e, {"state": "error " + str(e), "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})
    return respond(None, result_dict)


@app.route('/utils/text_to_image/clip/', methods=["POST"])
def text_to_image_clip_api():
    try:
        data = request.get_json()
        args = load_instance_from_json(data)

        img_bs64 = args["image_b64"]
        img_pil = bs64_to_pil(img_bs64)

        result_dict = text_to_image_clip(img_pil=img_pil,
                                        return_dict=True)
    except Exception as e:
        return respond(e, {"state": "error " + str(e), "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})
    return respond(None, result_dict)


@app.route('/utils/super_resolution/', methods=["POST"])
def super_resolution_api():
    try:
        check_request_queue_length()

        data = request.get_json()
        args = load_instance_from_json(data)

        request_id = args["request_id"]

        if request_id in response_dict.keys():
            raise DupledRequestKeyError("dupled request id")
        else:
            with lock:
                response_dict[request_id] = None

        img_bs64 = args["image_b64"]
        img_pil = bs64_to_pil(img_bs64)
        
        enqueue_request({"process_function":super_resolution, 
                        "request_id":request_id,
                        "params":{
                            "img_pil":img_pil,
                            "return_dict":True,
                            }})
    except Exception as e:
        return respond(e, {"state": "error " + str(e), "type":"super_resolution", "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})
    return respond(None, {"state":"queued", "type":"super_resolution", "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})

    
@app.route('/diffusion/outpaint/', methods=["POST"])
def outpaint_api():
    try:
        check_request_queue_length()

        data = request.get_json()
        args = load_instance_from_json(data)

        request_id = args["request_id"]

        if request_id in response_dict.keys():
            raise DupledRequestKeyError("dupled request id")
        else:
            with lock:
                response_dict[request_id] = None
        
        img_bs64 = args["image_b64"]
        img_pil = bs64_to_pil(img_bs64)

        mask_bs64 = args["mask_b64"]
        mask_pil = bs64_to_pil(mask_bs64)

        enqueue_request({"process_function":outpaint, 
                        "request_id":request_id,
                        "params":{
                            "img_pil":img_pil,
                            "mask_pil":mask_pil,
                            "num_per_image":args.get("num_per_image", 1),
                            "text":args.get("text", ""),
                            "return_dict":True,
                            }})
    except Exception as e:
        return respond(e, {"state": "error " + str(e), "type":"outpaint", "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})
    return respond(None, {"state":"queued", "type":"outpaint", "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})


@app.route('/diffusion/composition/', methods=["POST"])
def composition_api():
    try:
        check_request_queue_length()

        data = request.get_json()
        args = load_instance_from_json(data)

        request_id = args["request_id"]

        if request_id in response_dict.keys():
            raise DupledRequestKeyError("dupled request id")
        else:
            with lock:
                response_dict[request_id] = None
        
        img_bs64 = args["image_b64"]
        img_pil = bs64_to_pil(img_bs64)

        mask_bs64 = args["mask_b64"]
        mask_pil = bs64_to_pil(mask_bs64)

        enqueue_request({"process_function":composition, 
                        "request_id":request_id,
                        "params":{
                            "img_pil":img_pil,
                            "mask_pil":mask_pil,
                            "num_per_image":args.get("num_per_image", 1),
                            "return_dict":True,
                            }})
    except Exception as e:
        return respond(e, {"state": "error " + str(e), "type":"composition", "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})
    return respond(None, {"state":"queued", "type":"composition", "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})

    
@app.route('/diffusion/augmentation/style/', methods=["POST"])
def augmentation_base_style_api():
    try:
        check_request_queue_length()

        data = request.get_json()
        args = load_instance_from_json(data)

        request_id = args["request_id"]
        
        if request_id in response_dict.keys():
            raise DupledRequestKeyError("dupled request id")
        else:
            with lock:
                response_dict[request_id] = None
        
        img_base_bs64 = args["image_b64_base"]
        img_base_pil = bs64_to_pil(img_base_bs64)

        img_style_bs64 = args["image_b64_style"]
        img_style_pil = bs64_to_pil(img_style_bs64)

        enqueue_request({"process_function":augmentation_base_style, 
                        "request_id":request_id,
                        "params":{
                            "img_base_pil":img_base_pil,
                            "img_style_pil":img_style_pil,
                            "num_per_image":args.get("num_per_image", 1),
                            "return_dict":True,
                            }})
    except Exception as e:
        return respond(e, {"state": "error " + str(e), "type":"augmentation_style", "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})
    return respond(None, {"state":"queued", "type":"augmentation_style", "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})


@app.route('/diffusion/augmentation/text/', methods=["POST"])
def augmentation_base_text_api():
    try:
        check_request_queue_length()

        data = request.get_json()
        args = load_instance_from_json(data)

        request_id = args["request_id"]

        if request_id in response_dict.keys():
            raise DupledRequestKeyError("dupled request id")
        else:
            with lock:
                response_dict[request_id] = None
        
        img_bs64 = args["image_b64"]
        img_pil = bs64_to_pil(img_bs64)

        color = args.get("color", "")
        concept = args.get("concept", "")

        enqueue_request({"process_function":augmentation_base_text, 
                        "request_id":request_id,
                        "params":{
                            "img_pil":img_pil,
                            "color":color,
                            "concept":concept,
                            "num_per_image":args.get("num_per_image", 1),
                            "return_dict":True,
                            }})
    except Exception as e:
        return respond(e, {"state": "error " + str(e), "type":"augmentation_text", "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})
    return respond(None, {"state":"queued", "type":"augmentation_text", "request_qsize":request_queue.qsize(), "response_dict_size":len([True for cur in response_dict if response_dict[cur] is not None])})


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
    parser.add_argument(
        "--request_queue_max_len",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--response_dict_max_len",
        type=int,
        default=10,
    )
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    request_queue_max_len = args.request_queue_max_len
    response_dict_max_len = args.response_dict_max_len

    set_root_model_path(root_model_dir_path=args.root_model_path, root_model_diffusion_dir_path=args.root_model_diffusion_path)
    prepare_ai_models()

    app.run(host=args.host, debug=args.debug, port=args.port)

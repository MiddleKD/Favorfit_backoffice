## Favorfit Backoffice 환경 세팅

- clone back office repository
  ```bash
  git clone https://github.com/MiddleKD/Favorfit_backoffice.git
  ```

- prepare virtual environment and requirements
  ```bash
  cd Favorfit_backoffice
  pip install -r requirements.txt
  ```

- clone additional repositories
  ```bash
  git clone https://github.com/MiddleKD/Favorfit_diffusion.git
  git clone https://github.com/MiddleKD/Favorfit_recommend_template.git
  git clone https://github.com/MiddleKD/Favorfit_color_enhancement.git
  git clone https://github.com/MiddleKD/Favorfit_image_to_text.git
  git clone https://github.com/MiddleKD/Favorfit_super_resolution.git
  git clone https://github.com/MiddleKD/Favorfit_remove_bg.git
  git clone https://github.com/MiddleKD/postprocess_remove_bg.git
  ```

## Run server
- RUNNING
- args
  - `root_model_path`: Model parent path
  - `root_model_diffusion_path`: Diffusion model path in parent
  - `host`: HOST ex:0.0.0.0
  - `debug`: debug mode on off(bool)
  - `port`: PORT ex:8000
  - `request_queue_max_len`: request_queue max length. if you exceed this queue length. It will return `TooMuchRequestQueueError`
  - `response_dict_max_len`: response_dict max length. if you exceed this dict length. It will clear response dict
  ```python
  python3 api_app.py
  ``` 

## Need to know
- `clip-interrogator==0.6.0`과 `torch==1.9.1`에 버전 충돌이 있습니다.
  - clip-interrogator파일에서 backend.mps 문구가 들어가는 부분을 삭제하면 됩니다.
  - backend.mps를 `torch 1.9.1`에서 지원 안 해서 생기는 문제입니다.

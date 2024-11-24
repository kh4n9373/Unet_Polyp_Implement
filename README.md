# PHAM TRAN TUAN KHANG - 20225503

### Complete folder structure 
```
github/
├── __pycache__/
├── checkpoints-20241123T123919Z-001/
│   └── checkpoints/
│       └── model/
│           └── model.ckpt
├── predicted/
├── .gitignore
├── image.jpeg
├── infer.py
├── model.py
├── README.md
└── requirements.txt
```
### Usage
```
git clone https://github.com/kh4n9373/Unet_Polyp_Implement.git

cd Unet_Polyp_Implement

<!-- Download checkpoint from this link https://drive.google.com/drive/folders/156NOV5fsMjb4qn3-o55mwXl_X-VeqiHZ?usp=sharing >
<!-- conda create -n UnetImplementTest python=3.10
conda activate UnetImplementTest
pip install -r requirements.txt -->

<!-- test prediction -->
python3 infer.py --image_path image.jpeg

<!-- check predicted/ folder -->
```


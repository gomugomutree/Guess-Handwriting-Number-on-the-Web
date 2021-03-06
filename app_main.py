from flask import Flask, request, jsonify, render_template
from PIL import Image, ImageOps
from utils import load_model

import torch 
import numpy as np
import os

app = Flask(__name__) # Flask 객체를 생성

# canvas그림판에서의 이미지를 데이터로 변환
def imgToData(data):
    canvas_img = Image.open(data)
    canvas_img = canvas_img.resize((28, 28), Image.LANCZOS)
    img = Image.new("L", canvas_img.size, (255))
    img.paste(canvas_img, canvas_img)
    img = ImageOps.invert(img)
    return img

# route 데코레이터로 URL '/'를 호출(연결)하면 함수()home를 실행한다고 알림 
@app.route('/')
def home():
    # templates폴더에 'htmlcode.html'을 찾아 클라이언트에 응답
    return render_template('htmlcode.html')

@app.route('/classification', methods=['POST'])
def getAnswer(): 
    img = imgToData(request.files["image"])
    # input = (np.asarray(img, dtype = np.float32)).reshape(1,784)
    input = (np.array(img, dtype='float')).reshape(28, 28)

    model = load_model()
    pred = model.predict(input)
    pred_result = float(torch.argmax(pred, dim=-1))
    return jsonify(prediction = str(pred_result))

if __name__=='__main__':
    app.run(host='61.82.220.92', port=5000, debug=True)
    # app.run(port=5000)
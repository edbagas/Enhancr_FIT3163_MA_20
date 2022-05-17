from flask import Flask,render_template,request
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


app = Flask(__name__, template_folder='C:\\Users\\Lenovo\\Desktop\\flask_app\\templates',static_folder='C:\\Users\\Lenovo\\Desktop\\flask_app\\static')

#The app starts here

@app.route('/')
def landing_page():
    return render_template('landing_page.html')

@app.route('/login_page')
def login_page():
    return render_template('login_page.html')

@app.route('/signup')
def sign_up_page():
    return render_template('sign_up_page.html')
    
@app.route('/index')
def index():
    return render_template('index.html')

#Get the uploaded image through POST
#and use our model to enhance it 
@app.route('/',methods=['POST'])
def enhance():
    imagefile = request.files['imagefile']
    image_path="C:\\Users\\Lenovo\\Desktop\\flask_app\\images\\"+imagefile.filename
    imagefile.save(image_path)
    
    weights_file="C:\\Users\\Lenovo\\Desktop\\flask_app\\app\\best.pth"
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()
    image = pil_image.open(image_path).convert('RGB')

    image = np.array(image).astype(np.float32)
    ycbcr = convert_rgb_to_ycbcr(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    psnr = calc_psnr(y, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.uint8)
    output = pil_image.fromarray(output)
    # output.save(image_path.replace('.', '_srcnn_x{}.'.format(2)))

    return render_template('index.html',output=output.show())

if __name__=='__main__':
    app.run(port=5000,debug=True)
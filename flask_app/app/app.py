from crypt import methods
from flask import Flask,render_template,request
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os
import dicom2jpg
import glob

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr


app = Flask(__name__, template_folder=os.getcwd()+'/templates',static_folder=os.getcwd()+'/static')

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
@app.route('/', methods=['GET','POST'])
def enhance():
    imagefile = request.files['imagefile']
    image_path=os.getcwd()+"/static/"+imagefile.filename
    imagefile.save(image_path)
    filename, file_extension = os.path.splitext(image_path)
    os.rename(image_path, os.getcwd()+"/static/CT_before"+file_extension)
    image_path = os.getcwd()+"/static/CT_before"+file_extension
    if file_extension == '.dcm':
        img_array = dicom2jpg.dicom2img(image_path)
        img_data = pil_image.fromarray(img_array)
        img_data.save(os.getcwd()+'/static/CT_before.png')

    weights_file=os.getcwd()+"/app/best.pth"
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
    image = pil_image.open(os.getcwd()+'/static/CT_before.png').convert('RGB')

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
    output.save(os.getcwd()+'/static/CT_after.png')
    # output.save(image_path.replace('.', '_srcnn_x{}.'.format(2)))

    # return render_template('output.html',output=output.show())
    return render_template('output.html')

if __name__=='__main__':
    app.run(port=5000,debug=True)
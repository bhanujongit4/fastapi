import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, request, send_file
from flask_cors import CORS
import io

# SRVGGNetCompact model definition
class SRVGGNetCompact(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(SRVGGNetCompact, self).__init__()
        self.body = nn.ModuleList()
        self.body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))

        if act_type == 'relu':
            activation = nn.ReLU(inplace=True)
        elif act_type == 'prelu':
            activation = nn.PReLU(num_parameters=num_feat)
        elif act_type == 'leakyrelu':
            activation = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.body.append(activation)

        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(activation)

        self.body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body:
            out = layer(out)
        out = self.upsampler(out)
        base = torch.nn.functional.interpolate(x, scale_factor=4, mode='nearest')
        return out + base

# Load the saved model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SRVGGNetCompact().to(device)
model.load_state_dict(torch.load('gandu7 (1).pth', map_location=device))
model.eval()

# Flask App setup
app = Flask(__name__)
CORS(app, resources={r"/infer": {"origins": "http://localhost:3000"}})

@app.route('/infer', methods=['POST'])
def infer():
    # Get the input image from the request
    file = request.files['image']
    input_image = preprocess_image(file)

    # Use the loaded model to generate the output image
    with torch.no_grad():
        output_tensor = model(input_image)

    # Normalize the output tensor and convert it back to an image
    output_tensor = torch.clamp(output_tensor, 0, 1)  # Ensure values are between 0 and 1
    output_image = transforms.ToPILImage()(output_tensor.squeeze(0).cpu())

    # Convert the output image to a BytesIO object and send it as a response
    img_io = io.BytesIO()
    output_image.save(img_io, 'PNG')
    img_io.seek(0)

    # Send the image as a response
    return send_file(img_io, mimetype='image/png')

def preprocess_image(input_file):
    # Preprocess the input image
    image = Image.open(input_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize to match the model's expected input size
        transforms.ToTensor()
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)
    return input_tensor

if __name__ == '__main__':
    app.run(debug=True)
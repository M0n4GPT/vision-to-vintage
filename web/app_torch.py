import os
import time
import random

from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

import torch
from torchvision import transforms
from PIL import Image

# ── CONFIG ───────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
STYLE_DIR     = os.path.join(BASE_DIR, "style")
MODEL_PATH    = os.path.join(BASE_DIR, "models", "stylizer_ddp.pt")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── ADAIN FUNCTIONS ───────────────────────────────────────────────────
def calc_mean_std(feat, eps=1e-5):
    B, C = feat.size()[:2]
    var = feat.view(B, C, -1).var(dim=2) + eps
    std = var.sqrt().view(B, C, 1, 1)
    mean = feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
    return mean, std

def adain(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)

# ── LOAD SCRIPTED MODULE ───────────────────────────────────────────────
# This loads the full wrapper; we extract the underlying style-transfer model
scripted = torch.jit.load(MODEL_PATH, map_location=device)
# The wrapper stored your trained StyleTransferModel as 'model'
model_scripted = scripted.model
model_scripted.eval()

# ── IMAGE TRANSFORMS ───────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])
to_pil = transforms.ToPILImage()

def tensor_to_image(t: torch.Tensor) -> Image.Image:
    return to_pil(t.clamp(0, 1).cpu())

# ── STYLE METADATA PARSING ─────────────────────────────────────────────
def parse_style_name(fname: str):
    name, _ = os.path.splitext(fname)
    author, title = name.split(',', 1)
    return author.replace('_', ' '), title.replace('_', ' ')

# ── FLASK ROUTES ──────────────────────────────────────────────────────
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    file = request.files.get('file')
    if not file:
        return 'No file uploaded', 400

    # Save uploaded content image
    ts = int(time.time() * 1000)
    fname = secure_filename(file.filename)
    content_path = os.path.join(app.config['UPLOAD_FOLDER'], f"content_{ts}_{fname}")
    file.save(content_path)

    # Load and preprocess content
    content_img = Image.open(content_path).convert('RGB')
    input_c = preprocess(content_img).unsqueeze(0).to(device)

    # Pick random style image
    style_files = sorted([f for f in os.listdir(STYLE_DIR)
                          if os.path.isfile(os.path.join(STYLE_DIR, f))])
    style_file = random.choice(style_files)
    author, title = parse_style_name(style_file)
    style_img = Image.open(os.path.join(STYLE_DIR, style_file)).convert('RGB')
    input_s = preprocess(style_img).unsqueeze(0).to(device)

    # Extract features and run AdaIN + decode
    with torch.no_grad():
        c_feat = model_scripted.enc(input_c)
        s_feat = model_scripted.enc(input_s)
        t      = adain(c_feat, s_feat)
        out_tensor = model_scripted.dec(t)

    # Postprocess and save output
    out_img    = tensor_to_image(out_tensor.squeeze(0))
    out_fname  = f"stylized_{ts}.jpg"
    out_path   = os.path.join(app.config['UPLOAD_FOLDER'], out_fname)
    out_img.save(out_path)

    img_url = f"/uploads/{out_fname}?t={ts}"
    return f"""
    <p><strong>Style:</strong> {author}, {title}</p>
    <img src=\"{img_url}\" style=\"max-width:100%;\" />
    <p><a href=\"{img_url}\" download>Download result</a></p>
    """

@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)

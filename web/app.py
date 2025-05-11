import os
import time
import random

from flask import (
    Flask, request,
    render_template, send_from_directory
)
from werkzeug.utils import secure_filename

import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from PIL import Image

#  CONFIG 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STYLE_DIR     = os.path.join(BASE_DIR, 'style')
# HUB_URL      = "https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2"
LOCAL_MODEL_PATH = os.path.join(BASE_DIR, 'models/arbitrary-image-stylization-v1-tensorflow1-256-v2')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.makedirs(STYLE_DIR,    exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

#  LOAD TF‑HUB MODEL 
# hub_model = hub.load(HUB_URL)
hub_model = hub.load(LOCAL_MODEL_PATH)


#  HELPERS 
def load_img(path, max_dim=512):
    """Load an image from disk, resize so longest side <= max_dim, normalize [0,1], batch it."""
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape    = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale    = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    return img[tf.newaxis, :]

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = tf.cast(tensor, tf.uint8)[0].numpy()
    return Image.fromarray(tensor)

# TODO: subject to change based on file name
def parse_style_name(style_path):
    """Turns 'Van_Gogh,Starry_Night.jpg' → ('Starry_Night', 'Van_Gogh')"""
    fn, _ = os.path.splitext(os.path.basename(style_path))
    author, title = fn.split(",", 1)
    return title, author

# ROUTES 
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html")

    # file upload error message
    file = request.files.get("file")
    if not file:
        return "No file uploaded", 400

    # save content image
    filename     = secure_filename(file.filename)
    content_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(content_path)

    # list of all possible authors
    ALL_AUTHORS = [
        "Claude_Monet",
        "Edvard_Munch",
        "Henri_Matisse",
        "Jean-Auguste_Dominique_Ingres",
        "Johannes_Vermeer",
        "da_Vinci",
        "Michelangelo",
        "Pablo_Picasso",
        "Vincent_van_Gogh",
        "Piet_Mondriaan",
    ]

    # pick a random style
    style_files = [
        os.path.join(STYLE_DIR, f)
        for f in os.listdir(STYLE_DIR)
        if os.path.isfile(os.path.join(STYLE_DIR, f))
    ]
    style_path = random.choice(style_files)

    # load into tensors
    c_img = load_img(content_path)
    s_img = load_img(style_path)

    # run & time it
    start = time.time()
    out_tensor = hub_model(tf.constant(c_img), tf.constant(s_img))[0]
    elapsed = time.time() - start

    # convert & save stylized output
    out_img      = tensor_to_image(out_tensor)
    out_filename = f"stylized_{filename}"
    out_path     = os.path.join(app.config["UPLOAD_FOLDER"], out_filename)
    out_img.save(out_path)

    # parse for display
    title, author = parse_style_name(style_path)
    style_label   = f"{title} by {author},"
    ts = int(time.time() * 1000)
    img_url = f"/uploads/{out_filename}?t={ts}"

    # 1) determine correct author
    correct_author = author  # from parse_style_name

    # 2) pick 4 wrong ones
    wrong = [a for a in ALL_AUTHORS if a != correct_author]
    choices = random.sample(wrong, 4) + [correct_author]
    random.shuffle(choices)

    # 3) build HTML: image + time + choice buttons + hidden span with actual
    buttons_html = "".join(
        f'<button class="author-choice btn btn-outline-primary m-1" '
        f'data-author="{c}">{c.replace("_"," ")}</button>'
        for c in choices
    )

    html = f'''
    <p><strong>Elapsed:</strong> {elapsed:.2f}s</p>
    <img src="{img_url}" alt="Stylized" class="img-preview" style="margin-top:1rem;">
    <div class="mt-3">
        <p><strong>Who’s the artist?</strong></p>
        {buttons_html}
        <div id="guess-feedback" class="mt-2"></div>
    </div>
    <span id="actual-author" data-author="{correct_author}" hidden></span>
    '''

    return html

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

# MAIN 
if __name__ == "__main__":
    # debug=False in production
    app.run(host="0.0.0.0", port=8001, debug=True)

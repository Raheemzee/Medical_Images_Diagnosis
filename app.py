from flask import Flask, render_template, request
from fastai.vision.all import *
import os
import uuid
import gc
import torch

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ✅ Ensure upload folder ALWAYS exists (Render fix)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# Lazy model loader
# =========================
_loaded_models = {}

def get_model(task):
    """
    Load only ONE model into memory at a time.
    """

    if task not in _loaded_models and len(_loaded_models) > 0:
        _loaded_models.clear()
        gc.collect()
        torch.cuda.empty_cache()

    if task not in _loaded_models:
        if task == "mammogram":
            _loaded_models[task] = load_learner("models/mammogram_model.pkl")

        elif task == "xray":
            _loaded_models[task] = load_learner("models/Chest_Xray.pkl")

        elif task == "eye":
            _loaded_models[task] = load_learner("models/Eyes_Defects_model.pkl")

        else:
            return None

    return _loaded_models[task]


@app.route("/", methods=["GET", "POST"])
def index():
    results = []

    if request.method == "POST":
        task = request.form.get("task")
        images = request.files.getlist("images")

        if task and images:
            model = get_model(task)

            for image in images:
                filename = f"{uuid.uuid4().hex}.jpg"
                image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                image.save(image_path)

                img = PILImage.create(image_path)

                pred, idx, probs = model.predict(img)

                results.append({
                    "prediction": str(pred),
                    "confidence": f"{max(probs).item() * 100:.2f}%",
                    "image_path": image_path
                })

                del img
                gc.collect()

            torch.cuda.empty_cache()

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)

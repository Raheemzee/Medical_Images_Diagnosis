from flask import Flask, render_template, request
from fastai.vision.all import *
import os
import uuid
import gc
import torch

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# =========================
# Lazy model loader
# =========================
_loaded_models = {}

def get_model(task):
    """
    Load model only when needed.
    Keeps at most one model in memory.
    """

    # If another model is already loaded, remove it
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
    prediction = None
    confidence = None
    image_path = None

    if request.method == "POST":
        task = request.form.get("task")
        image = request.files.get("image")

        if image and task:
            filename = f"{uuid.uuid4().hex}.jpg"
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            image.save(image_path)

            img = PILImage.create(image_path)

            model = get_model(task)

            if model is not None:
                pred, idx, probs = model.predict(img)
                prediction = str(pred)
                confidence = f"{max(probs).item() * 100:.2f}%"

            # Explicit cleanup (important on free tier)
            del img
            gc.collect()
            torch.cuda.empty_cache()

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image_path=image_path
    )


if __name__ == "__main__":
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)

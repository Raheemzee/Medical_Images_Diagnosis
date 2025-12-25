from flask import Flask, render_template, request
from fastai.vision.all import *
import os
import uuid
import gc
import torch

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# =========================
# Lazy model loader
# =========================
_loaded_models = {}

def get_model(task):
    if task not in _loaded_models:
        _loaded_models.clear()
        gc.collect()

        torch.set_num_threads(1)
        torch.backends.cudnn.enabled = False

        model_path = {
            "mammogram": "models/mammogram_model.pkl",
            "xray": "models/Chest_Xray.pkl",
            "eye": "models/Eyes_Defects_model.pkl"
        }.get(task)

        if not model_path:
            return None

        learn = load_learner(model_path, cpu=True)
        learn.model.eval()

        _loaded_models[task] = learn

    return _loaded_models[task]


@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    error = None

    if request.method == "POST":
        task = request.form.get("task")
        images = request.files.getlist("images")

        if not task or not images:
            error = "Please select a task and upload at least one image."
        else:
            try:
                model = get_model(task)

                for image in images:
                    filename = f"{uuid.uuid4().hex}.jpg"
                    image_path = os.path.join(UPLOAD_FOLDER, filename)
                    image.save(image_path)

                    try:
                        img = PILImage.create(image_path)
                        pred, idx, probs = model.predict(img)

                        results.append({
                            "prediction": str(pred),
                            "confidence": f"{max(probs).item() * 100:.2f}%",
                            "image_path": image_path
                        })

                        del img
                        gc.collect()

                    except Exception as e:
                        results.append({
                            "prediction": "Error",
                            "confidence": "N/A",
                            "image_path": image_path
                        })

                torch.cuda.empty_cache()

            except Exception as e:
                error = "Model failed to run due to limited server resources."

    return render_template("index.html", results=results, error=error)


if __name__ == "__main__":
    app.run(debug=True)

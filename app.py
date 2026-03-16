from flask import Flask, render_template, request
import torch, os, csv
from PIL import Image
from utils import preprocess_image
from model import DiseaseEnsemble, StageEnsemble

app = Flask(__name__)

# Correct upload folder
UPLOAD_FOLDER = "C:/Users/Admin/OneDrive/trails of pro/ScalpAI_Ensemble_Project_v2/static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
disease_model = DiseaseEnsemble().to(device)
stage_model = StageEnsemble().to(device)

# Load weights
disease_weights = "C:/Users/Admin/OneDrive/trails of pro/ScalpAI_Ensemble_Project_v2/saved_models/disease_model.pth"
stage_weights = "C:/Users/Admin/OneDrive/trails of pro/ScalpAI_Ensemble_Project_v2/saved_models/stage_model.pth"

if os.path.exists(disease_weights):
    disease_model.load_state_dict(torch.load(disease_weights, map_location=device))

if os.path.exists(stage_weights):
    stage_model.load_state_dict(torch.load(stage_weights, map_location=device))

disease_model.eval()
stage_model.eval()


# Read CSV
def read_csv(path):
    data = {}
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data[row["class"]] = row
    return data


disease_info = read_csv(
    "C:/Users/Admin/OneDrive/trails of pro/ScalpAI_Ensemble_Project_v2/disease_info.csv"
)

stage_info = read_csv(
    "C:/Users/Admin/OneDrive/trails of pro/ScalpAI_Ensemble_Project_v2/stage_info.csv"
)


@app.route("/")
def index():
    return render_template("index.html")


# Disease prediction
@app.route("/predict/disease", methods=["POST"])
def predict_disease():

    file = request.files["image"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = preprocess_image(path).to(device)

    outputs = disease_model(img)
    _, pred = torch.max(outputs, 1)

    label = list(disease_info.keys())[pred.item()]
    info = disease_info[label]

    return render_template(
        "result.html",
        title=label,
        description=info["description"],
        steps=info["steps"],
        image="static/uploads/" + file.filename
    )


# Stage prediction
@app.route("/predict/stage", methods=["POST"])
def predict_stage():

    file = request.files["image"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    img = preprocess_image(path).to(device)

    outputs = stage_model(img)
    _, pred = torch.max(outputs, 1)

    label = list(stage_info.keys())[pred.item()]
    info = stage_info[label]

    return render_template(
        "result.html",
        title=label,
        description=info["description"],
        steps=info["steps"],
        image="static/uploads/" + file.filename
    )


if __name__ == "__main__":
    app.run(debug=True)
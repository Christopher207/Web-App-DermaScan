from flask import Flask, render_template, request, jsonify
import cv2
import torch
import numpy as np
import base64
import io
import torch
from PIL import Image
from io import BytesIO
import albumentations as A
from albumentations.pytorch import ToTensorV2
from unet_model import UNet_RELAN_ASPP
app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = UNet_RELAN_ASPP(out_channels=3).to(device)

# Si guardaste solo los pesos con `torch.save(model.state_dict(), 'model.pth')`:
model.load_state_dict(torch.load("Modelo_Unet_RELAN_ASPP_200_epocas.pth",map_location=device))
model.eval()


def base64_to_image(base64_str):
    header, encoded = base64_str.split(",", 1)
    decoded = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(decoded)).convert('RGB')
    return np.array(img)

def image_to_base64(img):
    _, buffer = cv2.imencode('.png', img)
    b64_img = base64.b64encode(buffer).decode('utf-8')
    return "data:image/png;base64," + b64_img

@app.route('/')
def index():
    return render_template('index.html')

# --- Colores para la máscara (si se necesitan en el backend para el overlay) ---
COLOR_SPOTS = [255, 0, 0]   # Rojo para Manchas
COLOR_WRINKLES = [0, 0, 255] # Azul para Arrugas

@app.route('/segment', methods=['POST'])
def segment():
    data = request.json
    img_b64 = data['image']

    # Decodificar imagen base64
    if "base64," in img_b64:
        img_b64_clean = img_b64.split("base64,")[-1]
    else:
        img_b64_clean = img_b64

    image_bytes = base64.b64decode(img_b64_clean)
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    original_w, original_h = image.size
    img_np = np.array(image)

    val_transforms = A.Compose([
        A.Resize(256, 256, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    transformed = val_transforms(image=img_np)
    input_tensor = transformed['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    color_mask = np.zeros((256, 256, 3), dtype=np.uint8)
    color_mask[pred_mask == 1] = COLOR_SPOTS
    color_mask[pred_mask == 2] = COLOR_WRINKLES

    color_mask_pil = Image.fromarray(color_mask).resize((original_w, original_h), resample=Image.NEAREST)
    color_mask_resized = np.array(color_mask_pil)

    overlay = (0.7 * img_np + 0.3 * color_mask_resized).astype(np.uint8)
    overlay_img = Image.fromarray(overlay)

    # Codificar en base64 para la respuesta
    buffered = BytesIO()
    overlay_img.save(buffered, format="PNG")
    b64_result = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({"result": f"data:image/png;base64,{b64_result}"})

@app.route('/index')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('Acerca-de.html')

@app.route('/contact')
def contact():
    return render_template('Contacto.html')

@app.route('/application')
def application():
    return render_template('App.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)
#
"""HairStyle AI - face shape based hairstyle suggestions. Developed by Moin."""
import io, os, socket
import numpy as np
from flask import Flask, request, jsonify, render_template, Response
from PIL import Image, ImageOps

app = Flask(__name__)
BASE = os.path.dirname(os.path.abspath(__file__))
_mesh = None

def mesh():
    global _mesh
    if _mesh is None:
        import mediapipe as mp
        _mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
    return _mesh

def d(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))

def classify(p):
    """p(i) -> (x, y) of face-mesh landmark i. Returns a face shape name."""
    length = d(p(10), p(152))      # forehead top -> chin
    cheek = d(p(234), p(454))      # cheekbone width
    jaw = d(p(172), p(397))        # jaw width
    fore = d(p(103), p(332))       # forehead width
    ratio, j, f = length / cheek, jaw / cheek, fore / cheek
    if ratio >= 1.38: return "oblong"
    if f < 0.72 and j < 0.72: return "diamond"
    if f - j >= 0.10: return "heart"
    if ratio < 1.22: return "square" if j >= 0.82 else "round"
    return "oval"

SHAPES = {
 "oval": "Balanced proportions. Almost every style suits you, so pick by taste.",
 "round": "Width and length are similar with soft edges. Height and volume on top add length.",
 "square": "Strong jaw and forehead. Soft texture and side-swept styles soften the angles.",
 "oblong": "Face is noticeably longer than wide. Fringes and side volume add width.",
 "heart": "Wider forehead, narrower chin. Styles with fullness near the jaw balance it.",
 "diamond": "Wide cheekbones, narrow forehead and chin. Add width at the forehead and jaw.",
}
STYLES = {  # key: (name, tip)
 "crop": ("Classic Short Crop", "Clean, low maintenance, and sharpens the jawline."),
 "quiff": ("Textured Quiff", "Lifts the top to add height and balance the face."),
 "side": ("Side Part", "Smart and neat; the asymmetry flatters most face shapes."),
 "buzz": ("Buzz / Fade Cut", "Minimal upkeep and highlights bone structure."),
 "tex": ("Messy Textured Top", "Adds soft volume and a relaxed, modern look."),
 "layers": ("Long Layers", "Frames the face and adds movement."),
 "bob": ("Chin-Length Bob", "Adds width at the jaw and looks sleek."),
 "pixie": ("Pixie Cut", "Bold and easy; draws attention to the eyes."),
 "waves": ("Soft Waves", "Adds fullness and softens strong angles."),
 "bun": ("High Bun", "Adds height; great for lengthening a round face."),
 "bangs": ("Side Bangs / Fringe", "Shortens a long face and highlights the eyes."),
 "curls": ("Curly Volume", "Adds width and texture around the face."),
}
PICKS = {
 "male": {"oval": ["quiff","side","tex","crop"], "round": ["quiff","tex","side","crop"],
          "square": ["crop","side","buzz","tex"], "oblong": ["crop","side","curls","tex"],
          "heart": ["side","tex","crop","curls"], "diamond": ["tex","side","crop","curls"]},
 "female": {"oval": ["layers","bob","waves","bangs"], "round": ["layers","pixie","waves","bun"],
          "square": ["waves","layers","bangs","bob"], "oblong": ["bob","bangs","waves","curls"],
          "heart": ["bob","layers","bangs","waves"], "diamond": ["bob","layers","bangs","pixie"]},
}
HAIR = {
 "crop": '<path d="M42 112Q38 40 100 38Q162 40 158 112Q150 72 100 68Q50 72 42 112Z"/>',
 "quiff": '<path d="M40 115Q28 28 100 20Q176 28 160 115Q150 62 100 56Q60 56 40 115Z"/>',
 "side": '<path d="M40 118Q36 36 100 34Q165 36 160 118Q150 72 112 64Q70 80 40 118Z"/>',
 "buzz": '<path d="M44 105Q46 48 100 46Q154 48 156 105Q150 80 100 76Q50 80 44 105Z"/>',
 "tex": '<path d="M38 120Q30 34 100 28Q170 34 162 120Q155 80 135 70L120 80L100 64L80 80L62 70Q45 80 38 120Z"/>',
 "layers": '<path d="M36 205Q18 60 100 30Q182 60 164 205L148 190Q150 100 100 66Q50 100 52 190Z"/>',
 "bob": '<path d="M38 170Q26 40 100 30Q174 40 162 170L146 170Q148 80 100 66Q52 80 54 170Z"/>',
 "pixie": '<path d="M44 100Q40 40 100 36Q160 40 156 100Q150 76 120 70Q80 90 44 100Z"/>',
 "waves": '<path d="M36 205Q16 130 34 100Q30 40 100 28Q170 40 166 100Q184 130 164 205Q150 170 150 130Q148 80 100 64Q52 80 50 130Q50 170 36 205Z"/>',
 "bun": '<path d="M44 105Q46 48 100 46Q154 48 156 105Q150 76 100 70Q50 76 44 105Z"/><circle cx="100" cy="28" r="20"/>',
 "bangs": '<path d="M36 175Q24 40 100 30Q176 40 164 175L148 175Q150 110 100 100Q50 110 52 175Z"/><path d="M50 104Q100 66 150 104L150 80Q100 44 50 80Z"/>',
 "curls": "".join(f'<circle cx="{x}" cy="{y}" r="20"/>' for x, y in
          [(52,92),(70,60),(100,48),(130,60),(148,92),(42,122),(158,122)]),
}
COLORS = ["#3b2a20", "#5a3825", "#1f1a17", "#7a4a2b", "#2d2a35"]

@app.get("/style/<key>.svg")
def style_svg(key):
    if key not in HAIR: return "", 404
    c = COLORS[list(HAIR).index(key) % len(COLORS)]
    s = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 240">'
         f'<rect width="200" height="240" fill="#eef1f6"/>'
         f'<ellipse cx="100" cy="125" rx="58" ry="72" fill="#f1c9a5"/>'
         f'<circle cx="80" cy="120" r="5" fill="#333"/><circle cx="120" cy="120" r="5" fill="#333"/>'
         f'<path d="M84 160Q100 172 116 160" stroke="#a0522d" stroke-width="4" fill="none"/>'
         f'<g fill="{c}">{HAIR[key]}</g></svg>')
    return Response(s, mimetype="image/svg+xml")

def style_img(key):
    # Drop your own photo at static/styles/<key>.jpg to replace the illustration.
    return f"/static/styles/{key}.jpg" if os.path.exists(
        os.path.join(BASE, "static", "styles", f"{key}.jpg")) else f"/style/{key}.svg"

@app.get("/")
def index():
    return render_template("index.html")

@app.post("/analyze")
def analyze():
    f = request.files.get("photo")
    gender = request.form.get("gender", "female")
    if not f: return jsonify(error="Please take or choose a photo first."), 400
    try:
        img = ImageOps.exif_transpose(Image.open(io.BytesIO(f.read()))).convert("RGB")
    except Exception:
        return jsonify(error="That file could not be read as an image."), 400
    img.thumbnail((800, 800))
    arr = np.array(img)
    res = mesh().process(arr)
    if not res.multi_face_landmarks:
        return jsonify(error="No face found. Use good light, look straight at the camera, and keep hair off the forehead."), 422
    lm = res.multi_face_landmarks[0].landmark
    h, w = arr.shape[:2]
    shape = classify(lambda i: (lm[i].x * w, lm[i].y * h))
    picks = PICKS.get(gender, PICKS["female"])[shape]
    return jsonify(shape=shape, desc=SHAPES[shape], styles=[
        {"name": STYLES[k][0], "tip": STYLES[k][1], "img": style_img(k)} for k in picks])

if __name__ == "__main__":
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect(("8.8.8.8", 80)); ip = s.getsockname()[0]
    except Exception: ip = "127.0.0.1"
    print(f"\nOpen this on your phone (same Wi-Fi):  http://{ip}:5000\n")
    app.run(host="0.0.0.0", port=5000)

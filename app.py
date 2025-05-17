import cv2
import numpy as np
import img2pdf
from flask import Flask, request, send_file
from flask_cors import CORS  # Importation du module flask-cors
import tempfile
import os

app = Flask(__name__)
# Activer CORS pour toutes les routes
CORS(app, resources={r"/*": {"origins": "*"}})

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]    # top-right
    rect[3] = pts[np.argmax(diff)]    # bottom-left
    return rect

def is_document_shape(approx, image_shape):
    if len(approx) != 4:
        return False

    # Vérifie la surface (doit être au moins 20% de l'image)
    area = cv2.contourArea(approx)
    image_area = image_shape[0] * image_shape[1]
    if area < 0.2 * image_area:
        return False

    # Vérifie si c'est à peu près un rectangle (angles ~90°)
    pts = approx.reshape(4, 2)
    def angle(p1, p2, p3):
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p3 - p2)
        c = np.linalg.norm(p3 - p1)
        return np.arccos((a**2 + b**2 - c**2) / (2*a*b)) * 180 / np.pi

    angles = []
    for i in range(4):
        angle_deg = angle(pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4])
        angles.append(angle_deg)

    return all(70 < a < 110 for a in angles)

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth-1, 0],
        [maxWidth-1, maxHeight-1],
        [0, maxHeight-1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

def process_image(image_path):
    image = cv2.imread(image_path)
    orig = image.copy()
    ratio = image.shape[0] / 1000.0
    resized = cv2.resize(image, (int(image.shape[1] / ratio), 1000))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 200)

    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    document_found = False
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if is_document_shape(approx, resized.shape):
            doc_cnts = approx
            document_found = True
            break

    if not document_found:
        raise Exception("Aucun document détecté (forme rectangulaire insuffisante)")

    scanned = four_point_transform(orig, doc_cnts.reshape(4, 2) * ratio)
    tmp_img = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(tmp_img.name, scanned)

    tmp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    with open(tmp_pdf.name, "wb") as f:
        f.write(img2pdf.convert(tmp_img.name))

    os.unlink(tmp_img.name)
    return tmp_pdf.name

@app.route('/scan', methods=['POST', 'OPTIONS'])
def scan_document():
    # Gestion explicite des requêtes OPTIONS pour CORS
    if request.method == 'OPTIONS':
        return {'status': 'ok'}, 200
    
    if 'file' not in request.files:
        return {"error": "Aucun fichier trouvé"}, 400
    file = request.files['file']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        try:
            pdf_path = process_image(tmp.name)
            return send_file(
                pdf_path, 
                as_attachment=True, 
                download_name="scanned.pdf",
                mimetype='application/pdf'
            )
        except Exception as e:
            return {"error": str(e)}, 500
        finally:
            os.unlink(tmp.name)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
from flask import Flask, request, send_file, render_template
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from PIL import Image
import io

app = Flask(__name__)
CORS(app)
limiter = Limiter(key_func=get_remote_address, app=app)

# Configure maximum file size (16 mb)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('upload.html')

def load_image(file):
    try:
        if not allowed_file(file.filename) or not file.mimetype.startswith('image/'):
            return None
        img = Image.open(file)
        img.verify()  # Verify image integrity
        file.seek(0)  # Reset file pointer after verification
        return Image.open(file)
    except Exception:
        return None

@app.route('/resize', methods=['POST'])
@limiter.limit('100 per day')
def resize_image():
    try:
        if 'file' not in request.files:
            return {'error': 'No file part'}, 400
        file = request.files['file']
        {'error': 'No selected file'}, 400

        img = load_image(file)
        if img is None:
            return {'error': 'Invalid image file'}, 400

        width = int(request.form.get('width', 100))
        height = int(request.form.get('height', 100))

        if width <= 0 or height <= 0:
            return {'error': 'Invalid dimensions'}, 400

        resized_img = img.resize((width, height))

        img_io = io.BytesIO()
        resized_img.save(img_io, format=img.format or 'JPEG')
        img_io.seek(0)

        return send_file(
            img_io,
            mimetype=f'image/{img.format.lower() if img.format else "jpeg"}'
        )

    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/convert', methods=['POST'])
@limiter.limit('100 per day')
def convert_image():
    try:
        if 'file' not in request.files:
            return {'error': 'No file part'}, 400

        file = request.files['file']
        if file.filename == '':
            return {'error': 'No selected file'}, 400

        img = load_image(file)
        if img is None:
            return {'error': 'Invalid image file'}, 400

        target_format = request.form.get('format', 'PNG').upper()
        if target_format not in ['PNG', 'JPEG', 'GIF']:
            return {'error': 'Unsupported format'}, 400

        img_io = io.BytesIO()
        if target_format == 'JPEG':
            img = img.convert('RGB')  # Remove alpha channel for JPEG compatibility
        img.save(img_io, format=target_format)
        img_io.seek(0)

        return send_file(img_io, mimetype=f'image/{target_format.lower()}')

    except Exception as e:
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(debug=True)

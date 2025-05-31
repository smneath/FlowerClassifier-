import pytest
from app import app
import io
from PIL import Image

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_resize_endpoint(client):
    # Create a test image in memory
    img_io = io.BytesIO()
    Image.new('RGB', (100, 100)).save(img_io, 'JPEG')
    img_io.seek(0)

    data = {
        'file': (img_io, 'test.jpg'),
        'width': '50',
        'height': '50'
    }
    response = client.post('/resize', data=data)
    assert response.status_code == 200


def test_invalid_file(client):
    data = {
        'file': (io.BytesIO(b'invalid'), 'test.txt')
    }
    response = client.post('/resize', data=data)
    assert response.status_code == 400
    
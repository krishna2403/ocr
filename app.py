from flask import Flask, render_template, request, redirect, url_for
from ocr import ocr

app = Flask(__name__)

@app.route('/')
@app.route('/<text>')
def index(text=None):
    return render_template('index.html', text=text)

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    text = ''
    if request.method == 'POST':
        img = request.files['img']
        if(img):
            img.save('image.png')
            text = ocr.predict()
        else:
            return redirect(url_for('index', text='File not selected or Invalid File'))
    return redirect(url_for('index', text=text))

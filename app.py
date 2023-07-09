import os
from flask import Flask, render_template, request, session
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import io
import base64
import openai

app = Flask(__name__)
app.secret_key = 'b1SWA8dA9T7zjWh8J1CslI1s'

# set up OpenAI API credentials
openai.api_key = os.environ.get('OPENAI_API_KEY')

# load image captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


UPLOAD_FOLDER = 'static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'image' in request.files:
            # Get the image file from the request
            file = request.files['image']

            # save the uploaded file to the server-side upload folder
            filename = 'uploaded_image.png'
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #  file and convert it to a PIL Image object
            img = Image.open(file).convert('RGB')

            # generate the image caption
            inputs = processor(img, return_tensors="pt")
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # store the generated caption in a session variable
            session['caption'] = caption

        else:
            # get the uploaded image from the server-side upload folder
            filename = os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.png')
            img = Image.open(filename).convert('RGB')

        # converting the image to base64 encoding for display in HTML
        img_data = io.BytesIO()
        img.save(img_data, format='PNG')
        img_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')

        # get the temperature value from the request
        temperature = float(request.form.get('temperature', 0.7))

        # get the generated caption from the session variable
        caption = session.get('caption', '')

        # generate multiple Instagram captions using OpenAI API
        prompt = f"What are some good Instagram captions for this picture of {caption}?"
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=150,
            n=3,
            stop=None,
            temperature=temperature,
        )
        instagram_captions = [choice.text.strip() for choice in response.choices]

        return render_template('home.html', caption=caption, img_base64=img_base64, instagram_captions=instagram_captions, temperature=temperature)

    return render_template('home.html')


if __name__ == '__main__':
    app.run()
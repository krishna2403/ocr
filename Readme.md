## Run this project on a local machine

Used python version 3.8.10 for this project

Create and activate a python virtual environment in the root folder of the project

```sh
python -m venv env
.\env\Scripts\activate
```

Install the requirements

```sh
pip install -r requirements.txt
```

Start the flask server

```sh
flask run
```

Browse to [http://localhost:5000](http://localhost:5000) to access the web version


## Train the model

After creating and activating a python virtual environment and installing the requirements, run the below command in the root folder of the project

```sh
python ocr/model.py
```

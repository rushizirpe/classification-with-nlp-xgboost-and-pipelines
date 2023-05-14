
# Wine Variety Classification Using NLP
This is a simple Flask API that provides wine variety classification based on their descriptions. The API uses an XGBoost model and an NLTK tokenizer to perform the classification.


# Installation
To run the notebook, you will need to have Python 3 and Jupyter installed. You can install the required Python packages by running the following command in your terminal:

`pip install -r requirements.txt`

This will install all the required packages listed in the requirements.txt file.

# Usage

To start the Flask server, run the following command in your terminal:

`
python app.py
`

This will start the server at `http://localhost:5000`. You can then make requests to the API using a tool like curl or Postman.

The API accepts POST requests to the `/predict` endpoint with the following payload:

`
{
    "description": "This is a wine with a rich aroma of dark fruits and spices, with a long finish."
}
`
The API will respond with a JSON object containing the predicted wine variety as:

`
{
    "variety": "Cabernet Sauvignon"
}
`

<!--
# Data
The dataset used in this analysis is the Wine Reviews dataset from Kaggle. The dataset contains over 130,000 wine reviews with descriptions and ratings.


# License
This notebook is released under the MIT license. Please see the LICENSE file for more information.
-->

# Author
This project was created by Rushi Zirpe. Feel free to contact me with any questions or feedback.

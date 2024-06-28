from flask import Flask, request, render_template
import pickle

# Load the trained pipeline from the pickle file
with open('pipeline.pkl', 'rb') as file:
    pipeline = pickle.load(file)

app = Flask(__name__,template_folder="templates")

# Create the Flask app
app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

# Define a route for song recommendation
@app.route('/recommend', methods=['POST'])
def recommend_song():
    # Get the data from the request
    data = request.get_json()

    # Preprocess the data (if necessary)
    # ...

    # Use the pipeline to make song recommendations
    recommendations = pipeline.predict(data)

    # Return the recommendations as a response
    return recommendations

# Run the Flask app
if __name__ == '__main__':
    app.run()

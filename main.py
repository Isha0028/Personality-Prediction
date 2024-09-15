# from flask import Flask, render_template, request
# from test_utils import *

# app = Flask(__name__)

# @app.route('/')
# def home():
# 	return render_template('index.html')

# @app.route('/prediction', methods=["GET", "POST"])
# def prediction():
# 	if request.method == "POST" and "handle" in request.form:
# 		username = request.form['username']
# 		model_prediction, user_tweets = get_prediction(username)
# 		return render_template('prediction.html', username=username, predicted_type=model_prediction, tweets=user_tweets)

# app.run(debug=True)


from flask import Flask, render_template, request
from test_utils import *

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    if request.method == "POST" and "handle" in request.form:
        handle = request.form['handle']
        try:
            personality = twit(handle)
            jobs = recomend(personality)
            characteristics = charcter(personality)
            return render_template('prediction.html', username=handle, predicted_type=personality, jobs=jobs, characteristics=characteristics)
        except ValueError as e:
            return render_template('error.html', error=str(e))
        except Exception as e:
            return render_template('error.html', error="An unexpected error occurred: " + str(e))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

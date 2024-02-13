from flask import Flask, render_template, request, jsonify

from model import final_result

app=Flask(__name__)

@app.get('/')
def index_get():
    return render_template('index.html')

@app.post('/predict')
def predict():
    text=request.get_json().get('message')
    print(text)
    response=final_result(text)
    answer=response['result']
    message={'answer':answer}
    return jsonify(message)

if __name__=='__main__':
    app.run(debug=True)
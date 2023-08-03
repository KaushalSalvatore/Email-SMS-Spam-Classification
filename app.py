from flask import Flask, request, render_template,jsonify
import pickle
from artifacts.utils import transform_text

app = Flask(__name__)
msg_preprocessor = pickle.load(open('artifacts/sms_model/vectorizer.pkl','rb'))
msg_model = pickle.load(open('artifacts/sms_model/model.pkl','rb'))

email_preprocessor = pickle.load(open('artifacts/email_model/vectorizer.pkl','rb'))
email_model = pickle.load(open('artifacts/email_model/model.pkl','rb'))

# 0 represent “not spam” and 1 represents “spam”.

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/smsprediction',methods=['GET','POST'])
def smsprediction():
    if request.method=='GET':
        return render_template('sms_detection.html')
    else:
        transform_sms = transform_text(request.form.get('message'))
        convert_to_vector = msg_preprocessor.transform([transform_sms])
        result = msg_model.predict(convert_to_vector)[0]
        print(result)
        return render_template('result.html',result=result)
    

    
@app.route('/emailprediction',methods=['GET','POST'])
def emailprediction():
    if request.method=='GET':
        return render_template('email_detection.html')
    else:
        transform_sms = transform_text(request.form.get('message'))
        convert_to_vector = email_preprocessor.transform([transform_sms])
        result = email_model.predict(convert_to_vector)[0]
        print(result)
        return render_template('result.html',result=result)

if __name__ == '__main__': 
   app.run(debug=True)
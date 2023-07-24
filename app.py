from flask import Flask, request, render_template,jsonify
from flask_cors import CORS,cross_origin
import pickle
from artifacts.utils import transform_text

app = Flask(__name__)
msg_preprocessor = pickle.load(open('artifacts/vectorizer.pkl','rb'))
msg_model = pickle.load(open('artifacts/model.pkl','rb'))


@app.route('/',methods=['GET','POST'])
@cross_origin()
def smsprediction():
    if request.method=='GET':
        return render_template('sms_detection.html')
    else:
        transform_sms = transform_text(request.form.get('message'))
        print(transform_sms)
        convert_to_vector = msg_preprocessor.transform([transform_sms])
        print(convert_to_vector)
        result = msg_model.predict(convert_to_vector)[0]
        print(result)
        return render_template('sms_detection.html')



if __name__ == '__main__': 
   app.run(debug=True)
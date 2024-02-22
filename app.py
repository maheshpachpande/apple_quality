from flask import Flask, render_template, request
from src.pipeline.pipeline_prediction import CustomData, PredictPipeline

application = Flask(__name__)
app=application

@app.route('/')
def index():
    return render_template('index.html')

@ app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            Size=float(request.form.get('Size')),
            Weight=float(request.form.get('Weight')),
            Sweetness=float(request.form.get('Sweetness')),
            Crunchiness=float(request.form.get('Crunchiness')),
            Juiciness=float(request.form.get('Juiciness')),
            Ripeness=float(request.form.get('Ripeness')),
            Acidity=float(request.form.get('Acidity'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results)
    
if __name__=="__main__":
    app.run(host='0.0.0.0', debug=True)

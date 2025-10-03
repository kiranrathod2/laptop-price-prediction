from flask import Flask, render_template, request, redirect
import pandas as pd
import pickle as pk

app = Flask(__name__)

linear_model = pk.load(open("linear.pkl","rb"))
poly_features = pk.load(open("poly_features.pkl","rb"))
poly_model    = pk.load(open("poly_model.pkl","rb"))
poly_scaler = pk.load(open("poly_scaler.pkl","rb"))
ridge_model  = pk.load(open("ridge.pkl","rb"))
lasso_model  = pk.load(open("lasso.pkl","rb"))
d_t_model  = pk.load(open("decision tree.pkl","rb"))
scaler_linear = pk.load(open("scaler_linear.pkl","rb"))
encoders = pk.load(open("encoders.pkl","rb"))

@app.route("/")
def homePage():
    return render_template("index.html", companies=encoders["Company"].classes_,
                                            types=encoders["TypeName"].classes_,
                                            cpus=encoders["Cpu"].classes_,
                                            gpus=encoders["Gpu Name"].classes_,
                                            gpu_brands=encoders["Gpu Brand"].classes_,
                                            osys=encoders["OpSys"].classes_) 

@app.route("/predict", methods = ["POST"])
def predict():
    if request.method == "POST":
        company = request.form['company']
        typename = request.form['typename']
        inches = float(request.form['inches'])
        ips = 1 if request.form['ips'] == 'Yes' else 0
        touchscreen = 1 if request.form['touchscreen'] == 'Yes' else 0
        width = int(request.form['width'])
        height = int(request.form['height'])
        cpu = request.form['cpu']
        gen = float(request.form['gen'])
        ram = int(request.form['ram'])
        storage_type = request.form['storage_type']
        storage_value = int(request.form['storage_value'])
        gpu_brand = request.form['gpu_brand']
        gpu = request.form['gpu']
        opsys = request.form['opsys']
        weight = float(request.form['weight'])
        reg_model = request.form['model']

        SSD, HDD, Hybrid, Flash = 0,0,0,0
        if storage_type == 'SSD': SSD = storage_value
        elif storage_type == 'HDD': HDD = storage_value
        elif storage_type == 'Hybrid': Hybrid = storage_value
        elif storage_type == 'Flash': Flash = storage_value

        company   = encoders["Company"].transform([company])[0]
        typename  = encoders["TypeName"].transform([typename])[0]
        cpu       = encoders["Cpu"].transform([cpu])[0]
        gpu_brand = encoders["Gpu Brand"].transform([gpu_brand])[0]
        gpu       = encoders["Gpu Name"].transform([gpu])[0]
        opsys     = encoders["OpSys"].transform([opsys])[0]

        laptop = pd.DataFrame([[company, typename, inches, ips, touchscreen,
                                width, height, cpu, gen, ram, gpu_brand, 
                                SSD, HDD, Hybrid, Flash, gpu, opsys, weight]],
                               columns=['Company','TypeName','Inches','IPS','Touchscreen',
                                        'width','height','Cpu','Gen','Ram','Gpu Brand',
                                        'SSD','HDD','Hybrid','Flash Storage','Gpu Name',
                                        'OpSys','Weight'])

        if reg_model == "Linear Regression":
            scaled_linear = scaler_linear.transform(laptop)
            prediction = linear_model.predict(scaled_linear)[0]
        elif reg_model == "Polynomial Regression":
            laptop_scaled = poly_scaler.transform(laptop)
            laptop_poly = poly_features.transform(laptop_scaled)
            prediction = poly_model.predict(laptop_poly)[0]
        elif reg_model == "Decision Tree Regression":
            prediction = d_t_model.predict(laptop)[0]
        elif reg_model == "Ridge Regression":
            prediction = ridge_model.predict(laptop)[0]
        elif reg_model == "Lasso Regression":
            scaled_linear = scaler_linear.transform(laptop)
            prediction = lasso_model.predict(scaled_linear)[0]

        prediction = max(5000, int(prediction))  
        return render_template('result.html', prediction = prediction)
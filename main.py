from flask import Flask ,render_template,request
import pickle
app=Flask(__name__)


file=open('model.pkl','rb')
clf=pickle.load(file)
file.close()


@app.route('/', methods=['GET', 'POST']) 
def hello_world():
    if request.method == 'POST':
        myDict=request.form
        fever=int(myDict['fever'])
        age=int(myDict['age'])
        pain=int(myDict['pain'])
        runnyNose=int(myDict['runnyNose'])
        diffBreath=int(myDict['diffBreath'])
        #code for infrence
        #inputFeatures=[100,1,22,-1 ,1]
        inputFeatures=[fever,pain,age,runnyNose,diffBreath]
        infProb=clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html',inf=round(infProb*100))
    return render_template('index.html')

@app.route("/about")
def about():
    return render_template('about.html')
	
@app.route("/contact")
def contact():
    return render_template('contact.html')



if __name__ == '__main__':
    app.run(debug=True)
    
from flask import Flask, request
from dim_reduction import dataset
app = Flask(__name__)

@app.route('/recommend', methods=['GET'])
def recommend():
    if request.method == 'GET':
        t = dataset(class_num= 5,cluster="gaussian")
        temp = t.K_NN_for_user()
        t.means()
        return temp
        
if __name__ == '__main__':
    app.run()
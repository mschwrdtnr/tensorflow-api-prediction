from flask import Flask, request, jsonify
import sys

# initialize flask application
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    if request.method == "GET":
        return "Working Fine"

@app.route("/api/v1.0/csharp_python_restfulapi_json", methods=["POST"])
def csharp_python_restfulapi_json():
    """
    simple c# test to call python restful api web service
    """
    try:                
#         get request json object
        request_json = request.get_json()      
#         convert to response json object 
        response = jsonify(request_json)
        response.status_code = 200  
    except:
        exception_message = sys.exc_info()[1]
        response = json.dumps({"content":exception_message})
        response.status_code = 400
    return response

if __name__ == "__main__":
    # run flask application in debug mode
    app.run(debug=True)

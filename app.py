from flask import Flask, render_template, request, redirect, send_from_directory
from advArt import trainPatch
import tempfile
import os
from werkzeug.utils import secure_filename
from multiprocessing import Process
from tensorboard import program
import socket


def availablePorts():
    for port in range(5000, 8081):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            res = sock.connect_ex(('127.0.0.1', port))
            if res != 0:
                sock.close()
                return port
            

app = Flask(__name__)
app.secret_key="secret"
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
session = {}
session["tb"] = False

def sortName(fileName):
    if session["eval"]:
        return int(fileName[7:-4])
    else:
        return int(fileName[:-4])

def getImages():
    imgPath = os.path.join("artImg", session["exp"], "patch")
    epoch = 0
    imgCount = 0
    patches = []
    if os.path.exists(imgPath):
        for name in os.listdir(imgPath):
            if os.path.isfile(os.path.join(imgPath, name)):
                imgCount += 1
                patches.append(name)
        if imgCount != 0:
            epoch = (imgCount-1) *10 + 1
    patches.sort(key=sortName, reverse=True)
    return epoch,imgCount,patches

@app.route("/")
def configuration():
    return render_template("configuration.html")

@app.route("/run", methods=["POST"])
def run():
    form = request.form
    args = {}
    # print(request.files)
    target = request.files["target_u"]
    target_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(target.filename))
    target.save(target_path)
    args["target"] = target_path
    if request.files["resume_u"].filename == "":
        args["resume"] = None
    else:
        resume = request.files["resume_u"]
        resume_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(resume.filename))
        resume.save(resume_path)
        args["resume"] = resume_path
        
    # print(form)
    for entry in form:
        # Convert every input to correct type
        suffix = entry[-2:]
        if entry == "targetClass_i":
            if int(form[entry]) < 0:
                args[entry[:-2]] = None;
            else:
                args[entry[:-2]] = int(form[entry])
        # Suffix _b for boolean
        elif suffix == "_b":
            if form[entry] == "true":
                args[entry[:-2]] = True
            else:
                args[entry[:-2]] = False
        # Suffix _i for int
        elif suffix == "_i":
            args[entry[:-2]] = int(form[entry])
        # Suffix _f for float
        elif suffix == "_f":
            args[entry[:-2]] = float(form[entry])
        # Suffix _s for string
        elif suffix == "_s":
            args[entry[:-2]] = form[entry]
        # Suffix _u for file upload
    # print(args)
    # Train the patch async
    session["exp"] = args["exp"]
    session["eval"] = args["eval"]
    session["detail"] = args["saveDetail"]
    training = Process(target=trainPatch, args=(args,))
    training.start()
    session["process"] = training
    return redirect("/running")

@app.route("/running")
def running():
    if session["tb"] == False:
        tb = program.TensorBoard()
        port = availablePorts()
        tb.configure(argv=[None, '--logdir', "advArt_log", "--port", str(port)])
        session["tb"] = tb.launch()
    if session["process"].is_alive() == False:
        return redirect("/close")
    epoch, imgCount, patches = getImages()
    return render_template("running.html", tbURL=session["tb"], name=session["exp"], patches=patches, epoch=epoch, imgCount=imgCount, eval=session["eval"])


@app.route("/patch/<path:filename>")
def patch(filename):
    imgPath = os.path.join("artImg", session["exp"], "patch")
    return send_from_directory(imgPath, filename)

@app.route("/combine/<path:filename>")
def combine(filename):
    imgPath = os.path.join("artImg", session["exp"], "combine")
    return send_from_directory(imgPath, filename)

@app.route("/close", methods=["POST", "GET"])
def close():
    if session.get("process"):
        session["process"].terminate()
    mAP = ''
    if session.get("eval"):
        file = open(os.path.join("artImg", session["exp"], "result.txt"), 'r')
        mAP = file.read()
    _, imgCount, patches = getImages()
    return render_template("close.html", mAP=mAP, tbURL=session["tb"], name=session["exp"], patches=patches, imgCount=imgCount, eval=session["eval"])



if __name__ == "__main__":
    port = availablePorts()
    app.run("127.0.0.1", port, debug=True)
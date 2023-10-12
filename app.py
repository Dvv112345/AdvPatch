from flask import Flask, render_template, request, redirect, send_from_directory
from advArt import trainPatch
import tempfile
import os
from werkzeug.utils import secure_filename
from multiprocessing import Process
from tensorboard import program
import socket

            

app = Flask(__name__)
app.secret_key="secret"
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
session = {}
session["tb"] = False
args = {}
descriptions = {
    "exp": ["Experiment name: ", "s", "test"],
    "eval": ["Run evaluation, not training: ", "b", "False"],
    "a": ["Weight for detection loss function: ", "f", "1"],
    "b": ["Weight for total variation loss function: ", "f", "0.5"],
    "c": ["Weight for total similarity function: ", "f", "1"],
    "lr": ["Learning rate: ", "f", "0.001"],
    "epoch": ["Maximum epoch: ", "i", "100000"],
    "batch": ["Batch size: ", "i", "8"],
    "resize": ["Size of the patch (in pixels): ", "i", "400"],
    "patchSize": ["Size of the patch relative to bounding box: ", "f", "0.6"],
    "targetClass": ["Target class (only for YOLOv3, -1 means None): ", "i", "-1"],
    "target": ["Target image: ", "u", "None"],
    "imgSize": ["Size of target image (in pixels): ", "i", "416"],
    "resume": ["Start from a patch: ", "u", "None"],
    "dataset": ["Path of the dataset: ", "s", "dataset/inria/Train/pos"],
    "label": ["Path of the labels: ", "s", "dataset/inria/Train/pos/yolo-labels_yolov4"],
    "model": ["Detector model to target: ", "o", {"v3": "YOLOv3", "v7": "YOLOv7", "faster": "Faster-RCNN"}, "s"],
    "tiny": ["Use tiny configuration (only for YOLOv3): ", "b", "False"],
    "imageFilter": ["Filter dateset so bounding boxes are at least a given proportion of the image (0 - 1): ", "f", "0.1"],
    "piecewise": ["Piecewise threshold for detection loss function (0 - 1): ", "f", "0"],
    "startImage": ["Patch start as the target image: ", "b", "False"],
    "saveDetail": ["Save image per batch: ", "b", "False"],
    "noise": ["Use noise transformation: ", "b", "False"],
    "rotate": ["Use rotate transformation: ", "b", "False"],
    "wrinkle": ["Use wrinkle transformation: ", "b", "False"],
    "blur": ["Use blur transformation: ", "b", "False"],
    "persp": ["Use perspective transformation: ", "b", "False"],
    "region": ["Only modify part of the patch: ", "b", "False"],
    "regionX1": ["Left edge of the region to be modified in pixels from the left edge of the patch: ", "i", "0"],
    "regionX2": ["Right edge of the region to be modified in pixels from the left edge of the patch: ", "i", "0"],
    "regionY1": ["Top edge of the region to be modified in pixels from the top edge of the patch: ", "i", "0"],
    "regionY2": ["Bottom edge of the region to be modified in pixels from the top edge of the patch: ", "i", "0"],
    "simWeight": ["Assign a weight to a region for the similarity loss function (-1 means off): ", "i", "-1"],
    "simX1": ["Left edge of the region of similarity weight from the left edge of the patch: ", "i", "0"],
    "simX2": ["Right edge of the region of similarity weight from the left edge of the patch: ", "i", "0"],
    "simY1": ["Top edge of the region of similarity weight from the top edge of the patch: ", "i", "0"],
    "simY2": ["Bottom edge of the region of similarity weight from the top edge of the patch: ", "i", "0"],
    "note": ["Note: ", "s", "NA"]
}

def availablePorts():
    for port in range(5000, 8081):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            res = sock.connect_ex(('127.0.0.1', port))
            if res != 0:
                sock.close()
                return port

def sortName(fileName):
    return str(len(fileName)) + fileName
    # if args["saveDetail"]:
    #     return int(fileName[7:-4])
    # else:
    #     return int(fileName[:-4])

def getImages():
    imgPath = os.path.join("artImg", args["exp"], "patch")
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
    return render_template("configuration.html", descriptions=descriptions)

@app.route("/run", methods=["POST"])
def run():
    form = request.form
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
    return render_template("running.html", tbURL=session["tb"], name=args["exp"], patches=patches, epoch=epoch, imgCount=imgCount, eval=args["eval"], args=args, descriptions=descriptions)


@app.route("/patch/<path:filename>")
def patch(filename):
    imgPath = os.path.join("artImg", args["exp"], "patch")
    return send_from_directory(imgPath, filename)

@app.route("/combine/<path:filename>")
def combine(filename):
    imgPath = os.path.join("artImg", args["exp"], "combine")
    return send_from_directory(imgPath, filename)

@app.route("/close", methods=["POST", "GET"])
def close():
    if session.get("process"):
        session["process"].terminate()
    mAP = ''
    if args.get("eval"):
        file = open(os.path.join("artImg", args["exp"], "result.txt"), 'r')
        mAP = file.read()
    _, imgCount, patches = getImages()
    return render_template("close.html", mAP=mAP, tbURL=session["tb"], name=args["exp"], patches=patches, imgCount=imgCount, eval=args["eval"], args=args, descriptions=descriptions)



if __name__ == "__main__":
    port = availablePorts()
    app.run("127.0.0.1", port, debug=False)
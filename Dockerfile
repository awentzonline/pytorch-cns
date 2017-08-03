from python:3

RUN apt-get update && apt-get -y install python-numpy python-scipy libsdl1.2-dev libsdl-gfx1.2-dev libsdl-image1.2-dev cmake swig
RUN pip install http://download.pytorch.org/whl/cu75/torch-0.1.12.post2-cp36-cp36m-linux_x86_64.whl
RUN pip install torchvision

ADD . /app
WORKDIR /app
RUN pip install -e .
RUN pip install -r requirements.examples.txt

CMD ["python", "-u", "examples/atari_pixrnn_gpa.py", "--num-agents=10", "--clear-store", "--redis-params={\"host\": \"redis\"}"]

from flask.app import Flask
from flask_socketio import SocketIO, emit

from chat import *

app = Flask(__name__)
app.config
io = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def home():
    return { "status": 200 }

@io.on('clientMessage')
def client_message_handler(msg):
    resp = chat(msg)
    # print('resp ---> ', resp)
    emit('serverMessage', resp, json=True)

if __name__ == "__main__":
    # io.run(app, debug=True)
    io.run(app)


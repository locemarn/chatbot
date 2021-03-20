from flask.app import Flask
from flask_socketio import SocketIO, emit

app = Flask(__name__)
app.config
io = SocketIO(app, cors_allowed_origins="*")

def socket_routes():
	@app.route("/")
	def home():
		print('fsfdfsdfd')
		return {"status": 200}

	@io.on('clientMessage')
	def client_message_handler(msg):
		print('msg -->', msg)
		emit('serverMessage', msg, json=True)

if __name__ == "__main__":
    io.run(app, debug=True)
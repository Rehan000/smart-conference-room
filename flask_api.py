# Import required modules and packages
import redis
from flask import Flask
from flask import request

app = Flask(__name__)


# Function to get stream number and send to redis stream
@app.route('/', methods=['GET'])
def get_stream():
    args = request.args
    redis_client.xadd(name="Stream_Change",
                      fields={
                          "Stream_Num": args.get("stream"),
                      },
                      maxlen=10,
                      approximate=False)
    redis_client.execute_command(f'XTRIM Stream_Change MAXLEN 10')
    print("Stream Change Sent!")
    return args


if __name__ == '__main__':
    # Initialize redis client
    redis_client = redis.Redis(host='127.0.0.1')

    # Run flask app
    app.run(debug=True)

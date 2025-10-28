from flask import Flask, send_file
import os

app = Flask(__name__)

@app.route('/')
def index():
    return send_file('simple_chat_demo.html')

@app.route('/simple-demo')
def simple_demo():
    return send_file('simple_chat_demo.html')

@app.route('/react-demo') 
def react_demo():
    return """
    <html>
    <head><title>React Demo</title></head>
    <body>
        <h1>React 앱 링크</h1>
        <p><a href="http://localhost:3000" target="_blank">React 앱 열기</a></p>
        <p><a href="/simple-demo">간단한 채팅 데모</a></p>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)
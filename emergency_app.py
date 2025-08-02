# Emergency simple app for Railway startup issues
from flask import Flask, jsonify
import os

app = Flask(__name__)

@app.route('/health')
def health():
    return 'OK', 200

@app.route('/healthz')
def healthz():
    return jsonify({'status': 'ok'}), 200

@app.route('/api/status')
def status():
    return jsonify({
        'status': 'healthy',
        'service': 'Trading Analysis Emergency',
        'version': '6.1-emergency'
    }), 200

@app.route('/')
def home():
    return '''
    <h1>🚀 Trading Analysis Pro</h1>
    <p>Emergency mode - basic functionality</p>
    <p>Health: <a href="/health">/health</a></p>
    '''

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)

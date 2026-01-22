import os
import time
from flask import Flask, jsonify, request
from bitcoin_puzzle_pool_integrated import PoolCoordinator

app = Flask(__name__)
pool = PoolCoordinator()


@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "uptime_seconds": int(time.time() - pool.start_time)})


@app.route("/api/v1/initialize/<int:puzzle>", methods=["POST"])
def init_puzzle(puzzle):
    return jsonify(pool.initialize_puzzle(puzzle))


@app.route("/api/v1/record/<int:puzzle>/<user_id>", methods=["POST"])
def record(puzzle, user_id):
    return jsonify(pool.record_solution(puzzle, user_id))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
import os
import time
from flask import Flask, jsonify, request
from bitcoin_puzzle_pool_integrated import PoolCoordinator

app = Flask(__name__)
pool = PoolCoordinator()


@app.route("/api/v1/health")
def health():
    return jsonify({"status": "ok", "uptime_seconds": int(time.time() - pool.start_time)})


@app.route("/api/v1/initialize/<int:puzzle>", methods=["POST"])
def init_puzzle(puzzle):
    res = pool.initialize_puzzle(puzzle)
    return jsonify(res)


@app.route("/api/v1/record/<int:puzzle>/<user_id>", methods=["POST"])
def record(puzzle, user_id):
    res = pool.record_solution(puzzle, user_id)
    return jsonify(res)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
"""
Flask API Server for Bitcoin Puzzle Pool
Deploy on Digital Ocean with Gunicorn
"""

from flask import Flask, jsonify, request
from bitcoin_puzzle_pool_integrated import (
    PoolCoordinator, SubscriptionTier, BITCOIN_PUZZLES, PuzzleStatus
)
import json
from datetime import datetime

app = Flask(__name__)
pool = PoolCoordinator()

# ============================================================================
# USER MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/api/v1/users/register', methods=['POST'])
def register_user():
    """Register new user account"""
    data = request.json
    username = data.get('username')
    email = data.get('email')
    
    if not username or not email:
        return jsonify({'error': 'Username and email required'}), 400
    
    try:
        user = pool.create_user(username, email)
        return jsonify({
            'status': 'success',
            'user_id': user.user_id,
            'username': user.username,
            'email': user.email,
            'tier': user.subscription_tier.value,
            'referral_code': user.referral_code
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/users/<user_id>/stats', methods=['GET'])
def get_user_stats(user_id):
    """Get user statistics"""
    try:
        stats = pool.user_manager.get_user_stats(user_id)
        return jsonify(stats), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/users/<user_id>/subscribe', methods=['POST'])
def subscribe_user(user_id):
    """Upgrade user subscription"""
    data = request.json
    tier = data.get('tier')
    
    if tier not in [t.value for t in SubscriptionTier]:
        return jsonify({'error': 'Invalid subscription tier'}), 400
    
    try:
        result = pool.subscribe_user(user_id, SubscriptionTier(tier))
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/users/<user_id>/credits/purchase', methods=['POST'])
def purchase_credits(user_id):
    """Purchase compute credits"""
    data = request.json
    amount_keys = data.get('amount_keys')
    
    if not amount_keys or amount_keys <= 0:
        return jsonify({'error': 'Invalid amount'}), 400
    
    try:
        result = pool.purchase_credits(user_id, amount_keys)
        return jsonify(result), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# PUZZLE & WORK ENDPOINTS
# ============================================================================

@app.route('/api/v1/puzzles', methods=['GET'])
def get_puzzles():
    """Get all puzzles"""
    puzzles_list = []
    for num, puzzle in BITCOIN_PUZZLES.items():
        puzzles_list.append({
            'puzzle_number': num,
            'reward_btc': puzzle.reward_btc,
            'status': puzzle.status.value,
            'bit_length': puzzle.bit_length,
            'address': puzzle.address
        })
    
    return jsonify({
        'total': len(puzzles_list),
        'puzzles': sorted(puzzles_list, key=lambda x: x['puzzle_number'])
    }), 200


@app.route('/api/v1/puzzles/<int:puzzle_number>', methods=['GET'])
def get_puzzle(puzzle_number):
    """Get specific puzzle details"""
    if puzzle_number not in BITCOIN_PUZZLES:
        return jsonify({'error': 'Puzzle not found'}), 404
    
    puzzle = BITCOIN_PUZZLES[puzzle_number]
    return jsonify({
        'puzzle_number': puzzle_number,
        'reward_btc': puzzle.reward_btc,
        'status': puzzle.status.value,
        'bit_length': puzzle.bit_length,
        'range_start': puzzle.range_start,
        'range_end': puzzle.range_end,
        'address': puzzle.address,
        'solver': puzzle.solver_user_id,
        'solving_date': puzzle.solving_date
    }), 200


@app.route('/api/v1/puzzles/<int:puzzle_number>/initialize', methods=['POST'])
def initialize_puzzle(puzzle_number):
    """Initialize puzzle solving"""
    try:
        result = pool.initialize_puzzle(puzzle_number)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/work/assign', methods=['POST'])
def assign_work():
    """Assign work to a worker"""
    data = request.json
    worker_id = data.get('worker_id')
    puzzle_number = data.get('puzzle_number')
    
    if not worker_id:
        return jsonify({'error': 'Worker ID required'}), 400
    
    try:
        work_unit = pool.distributor.assign_work(worker_id, puzzle_number)
        if work_unit:
            return jsonify({
                'unit_id': work_unit.unit_id,
                'puzzle_number': work_unit.puzzle_number,
                'chunk_id': work_unit.chunk_id,
                'range_start': work_unit.key_range_start,
                'range_end': work_unit.key_range_end,
                'difficulty': work_unit.difficulty
            }), 200
        else:
            return jsonify({'error': 'No work available'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/work/submit', methods=['POST'])
def submit_work():
    """Submit work result"""
    data = request.json
    unit_id = data.get('unit_id')
    result = data.get('result')
    
    if not unit_id or not result:
        return jsonify({'error': 'Unit ID and result required'}), 400
    
    try:
        success = pool.distributor.submit_result(unit_id, result)
        if success:
            return jsonify({'status': 'accepted'}), 200
        else:
            return jsonify({'error': 'Invalid unit ID'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# REWARD & SOLUTION ENDPOINTS
# ============================================================================

@app.route('/api/v1/solutions/record', methods=['POST'])
def record_solution():
    """Record puzzle solution"""
    data = request.json
    puzzle_number = data.get('puzzle_number')
    solver_user_id = data.get('solver_user_id')
    
    if not puzzle_number or not solver_user_id:
        return jsonify({'error': 'Puzzle number and solver user ID required'}), 400
    
    try:
        result = pool.record_solution(puzzle_number, solver_user_id)
        return jsonify(result), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# ANALYTICS ENDPOINTS
# ============================================================================

@app.route('/api/v1/pool/analytics', methods=['GET'])
def get_pool_analytics():
    """Get pool analytics"""
    try:
        analytics = pool.get_pool_analytics()
        return jsonify(analytics), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/v1/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': int(pool.start_time)
    }), 200


# ============================================================================
# INFO ENDPOINTS
# ============================================================================

@app.route('/', methods=['GET'])
def index():
    """API documentation"""
    return jsonify({
        'name': 'Bitcoin Puzzle Pool API',
        'version': '1.0.0',
        'status': 'running',
        'endpoints': {
            'users': [
                'POST /api/v1/users/register',
                'GET /api/v1/users/<user_id>/stats',
                'POST /api/v1/users/<user_id>/subscribe',
                'POST /api/v1/users/<user_id>/credits/purchase'
            ],
            'puzzles': [
                'GET /api/v1/puzzles',
                'GET /api/v1/puzzles/<puzzle_number>',
                'POST /api/v1/puzzles/<puzzle_number>/initialize'
            ],
            'work': [
                'POST /api/v1/work/assign',
                'POST /api/v1/work/submit'
            ],
            'rewards': [
                'POST /api/v1/solutions/record'
            ],
            'monitoring': [
                'GET /api/v1/pool/analytics',
                'GET /api/v1/health'
            ]
        }
    }), 200


@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    # Development only - use Gunicorn for production
    app.run(host='0.0.0.0', port=5000, debug=False)

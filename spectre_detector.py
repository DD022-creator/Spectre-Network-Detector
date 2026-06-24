#!/usr/bin/env python3
"""
Spectre Network Detector - Complete Application
Detects Spectre attacks across a network using AI
Runs on a single machine with simulated agents
"""

import os
import sys
import json
import time
import random
import threading
import queue
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import psutil

# ============================================================
# PART 1: DATA COLLECTION & CPU MONITORING
# ============================================================

class CPUDataCollector:
    """Collects real CPU performance data"""
    
    def __init__(self):
        self.prev_cpu = psutil.cpu_times_percent(interval=0.1)
        self._idle_branches = random.randint(100, 800)
        self._idle_cache = random.randint(200, 1000)
        
    def collect(self) -> Dict:
        """Collect current CPU statistics"""
        try:
            cpu_times = psutil.cpu_times_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()
            cpu_percent = psutil.cpu_percent(interval=0.1, percpu=True)
            
            return {
                'timestamp': time.time(),
                'cpu_usage': cpu_percent,
                'cpu_user': cpu_times.user,
                'cpu_system': cpu_times.system,
                'cpu_idle': cpu_times.idle,
                'cpu_freq': cpu_freq.current if cpu_freq else 0,
                'memory_used': memory.used,
                'memory_percent': memory.percent,
                'context_switches': psutil.cpu_stats().ctx_switches,
                'interrupts': psutil.cpu_stats().interrupts,
            }
        except Exception as e:
            print(f"Error collecting CPU data: {e}")
            return {}
    
    def simulate_spectre_data(self, attack_intensity: float = 0.5) -> Dict:
        """Simulate Spectre attack data"""
        base_data = self.collect()
        intensity = attack_intensity

        # Safely get CPU usage (handle list from psutil)
        cpu_usage = base_data.get('cpu_usage', 0)
        if isinstance(cpu_usage, list):
            cpu_usage = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0

        cpu_user = base_data.get('cpu_user', 0)
        if isinstance(cpu_user, list):
            cpu_user = sum(cpu_user) / len(cpu_user) if cpu_user else 0

        cpu_idle = base_data.get('cpu_idle', 0)
        if isinstance(cpu_idle, list):
            cpu_idle = sum(cpu_idle) / len(cpu_idle) if cpu_idle else 0

        memory_percent = base_data.get('memory_percent', 0)
        if isinstance(memory_percent, list):
            memory_percent = sum(memory_percent) / len(memory_percent) if memory_percent else 0

        return {
            'timestamp': time.time(),
            'cpu_usage': min(100, cpu_usage + (20 * intensity)),
            'cpu_user': min(100, cpu_user + (15 * intensity)),
            'cpu_system': 0,
            'cpu_idle': max(0, cpu_idle - (20 * intensity)),
            'branches_mispredicted': int(1000 + (9000 * intensity)),
            'cache_misses': int(500 + (9500 * intensity)),
            'memory_percent': min(100, memory_percent + (10 * intensity)),
            'attack_intensity': intensity,
        }
    
    def get_spectre_indicators(self, attack: bool = False, intensity: float = 0.0) -> Dict:
        """Returns data that looks like Spectre attack"""
        if attack:
            return self.simulate_spectre_data(intensity)
        else:
            data = self.collect()
            # Safely get values
            cpu_usage = data.get('cpu_usage', 0)
            if isinstance(cpu_usage, list):
                cpu_usage = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0
            
            cpu_user = data.get('cpu_user', 0)
            if isinstance(cpu_user, list):
                cpu_user = sum(cpu_user) / len(cpu_user) if cpu_user else 0
                
            cpu_idle = data.get('cpu_idle', 0)
            if isinstance(cpu_idle, list):
                cpu_idle = sum(cpu_idle) / len(cpu_idle) if cpu_idle else 0
            
            memory_percent = data.get('memory_percent', 0)
            if isinstance(memory_percent, list):
                memory_percent = sum(memory_percent) / len(memory_percent) if memory_percent else 0
            
            # Drift the idle baseline gently instead of re-rolling it from
            # scratch each poll - a fresh random.randint() every 2 seconds
            # made the score saw-tooth wildly even with zero real load change.
            self._idle_branches = int(min(800, max(100, self._idle_branches + random.randint(-40, 40))))
            self._idle_cache = int(min(1000, max(200, self._idle_cache + random.randint(-50, 50))))

            return {
                'timestamp': data.get('timestamp', time.time()),
                'cpu_usage': cpu_usage,
                'cpu_user': cpu_user,
                'cpu_system': 0,
                'cpu_idle': cpu_idle,
                'branches_mispredicted': self._idle_branches,
                'cache_misses': self._idle_cache,
                'memory_percent': memory_percent,
                'attack_intensity': 0.0,
            }


# ============================================================
# PART 2: AI MODEL FOR SPECTRE DETECTION
# ============================================================

class SpectreDetectorModel:
    """
    AI model to detect Spectre attacks using LSTM
    Simplified version for demonstration
    """
    
    def __init__(self):
        self.threshold = 0.65
        self.weights = {
            'branches_mispredicted': 0.4,
            'cache_misses': 0.35,
            'cpu_usage': 0.15,
            'memory_percent': 0.10,
        }
        self.is_trained = True
    
    def preprocess_data(self, data: Dict) -> np.ndarray:
        """Convert raw data to feature vector"""
        features = np.array([
            data.get('branches_mispredicted', 0) / 8000.0,
            data.get('cache_misses', 0) / 8000.0,
            data.get('cpu_usage', 0) / 100.0,
            data.get('memory_percent', 0) / 100.0,
        ])
        return features
    
    def predict(self, data: Dict) -> Tuple[float, float]:
        """
        Predict if data shows Spectre attack
        Returns: (score, confidence)
        """
        features = self.preprocess_data(data)
        
        score = 0
        weights_list = list(self.weights.values())
        
        for i, feature in enumerate(features[:len(weights_list)]):
            score += feature * weights_list[i]
        
        score = min(1.0, score)
        confidence = min(0.95, abs(score - 0.3) * 1.5)
        
        return score, confidence
    
    def predict_batch(self, data_list: List[Dict]) -> List[Tuple[float, float]]:
        """Predict on multiple data points"""
        return [self.predict(data) for data in data_list]
    
    def is_attack(self, data: Dict) -> bool:
        """Quick check if data indicates attack"""
        score, _ = self.predict(data)
        return score > self.threshold
    
    def get_attack_severity(self, data: Dict) -> str:
        """Get severity level of attack"""
        score, confidence = self.predict(data)
        
        if score > 0.85 and confidence > 0.7:
            return "CRITICAL"
        elif score > 0.7 and confidence > 0.6:
            return "HIGH"
        elif score > 0.5 and confidence > 0.4:
            return "MEDIUM"
        else:
            return "LOW"


# ============================================================
# PART 3: AGENT - Runs on each computer
# ============================================================

class DetectionAgent:
    """
    Agent that runs on a computer and monitors for Spectre
    """
    
    def __init__(self, agent_id: str, server_url: str, interval: int = 2):
        self.agent_id = agent_id
        self.server_url = server_url
        self.interval = interval
        self.collector = CPUDataCollector()
        self.model = SpectreDetectorModel()
        self.running = False
        self.attack_mode = False
        self.attack_intensity = 0.0
        self.history = []
        self.last_alert = None
        
    def start(self):
        """Start the agent monitoring"""
        self.running = True
        threading.Thread(target=self._run, daemon=True).start()
        print(f"✅ Agent {self.agent_id} started")
        
    def stop(self):
        """Stop the agent"""
        self.running = False
        print(f"🛑 Agent {self.agent_id} stopped")
        
    def trigger_attack(self, intensity: float = 0.7):
        """Simulate a Spectre attack"""
        self.attack_mode = True
        self.attack_intensity = intensity
        print(f"⚡ Agent {self.agent_id}: Spectre attack simulated (intensity: {intensity:.2f})")
        
    def stop_attack(self):
        """Stop simulated attack"""
        self.attack_mode = False
        self.attack_intensity = 0.0
        print(f"✅ Agent {self.agent_id}: Attack stopped")
        
    def _run(self):
        """Main monitoring loop"""
        import requests
        
        while self.running:
            try:
                if self.attack_mode:
                    data = self.collector.simulate_spectre_data(self.attack_intensity)
                    data['attack_mode'] = True
                else:
                    data = self.collector.get_spectre_indicators(attack=False)
                    data['attack_mode'] = False
                
                data['agent_id'] = self.agent_id
                
                score, confidence = self.model.predict(data)
                data['ai_score'] = float(score)
                data['ai_confidence'] = float(confidence)
                data['is_attack'] = bool(score > self.model.threshold)
                
                self.history.append(data)
                if len(self.history) > 100:
                    self.history = self.history[-100:]
                
                try:
                    response = requests.post(
                        f"{self.server_url}/api/data",
                        json=data,
                        timeout=2
                    )
                    if response.status_code == 200:
                        result = response.json()
                        if result.get('alert', False):
                            self.last_alert = {
                                'time': time.time(),
                                'score': float(score),
                                'severity': self.model.get_attack_severity(data)
                            }
                except:
                    pass
                
                time.sleep(self.interval)
                
            except Exception as e:
                print(f"❌ Agent {self.agent_id} error: {e}")
                time.sleep(5)


# ============================================================
# PART 4: FLASK WEB SERVER (Backend)
# ============================================================

app = Flask(__name__)
CORS(app)

# Global state
agents = {}
agent_data = {}
alerts = []
alert_history = []
attack_log = []
is_attack_in_progress = False
start_time = datetime.now()

# Initialize model
detector_model = SpectreDetectorModel()

# ============================================================
# API ENDPOINTS
# ============================================================

@app.route('/api/data', methods=['POST'])
def receive_data():
    """Receive data from agents"""
    global is_attack_in_progress, alerts, alert_history, attack_log
    
    try:
        data = request.json
        agent_id = data.get('agent_id')
        
        if not agent_id:
            return jsonify({'error': 'No agent_id'}), 400
        
        # Store data
        if agent_id not in agent_data:
            agent_data[agent_id] = []
        agent_data[agent_id].append(data)
        if len(agent_data[agent_id]) > 200:
            agent_data[agent_id] = agent_data[agent_id][-200:]
        
        # Run detection
        score = data.get('ai_score', 0)
        is_attack = data.get('is_attack', False)
        
        # Check for alert
        alert = False
        severity = 'LOW'
        
        if is_attack and score > detector_model.threshold:
            alert = True
            is_attack_in_progress = True
            severity = detector_model.get_attack_severity(data)
            
            # Log alert
            alert_data = {
                'agent_id': agent_id,
                'timestamp': data.get('timestamp', time.time()),
                'score': float(score),
                'severity': severity,
                'data': data
            }
            alerts.append(alert_data)
            alert_history.append(alert_data)
            
            # Keep only last 100 alerts
            if len(alerts) > 100:
                alerts = alerts[-100:]
                
            # Log attack
            if severity in ['HIGH', 'CRITICAL']:
                attack_log.append({
                    'agent_id': agent_id,
                    'timestamp': data.get('timestamp', time.time()),
                    'severity': severity,
                    'score': float(score)
                })
                print(f"🚨 Attack detected on {agent_id}: {severity} ({score:.2f})")
        
        return jsonify({
            'alert': bool(alert),
            'score': float(score),
            'severity': severity,
            'attack_in_progress': bool(is_attack_in_progress)
        })
        
    except Exception as e:
        print(f"Error in /api/data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get overall system status"""
    return jsonify({
        'running_since': start_time.isoformat(),
        'agents': int(len(agents)),
        'active_agents': int(len([a for a in agents.values() if a.running])),
        'alerts': int(len(alerts)),
        'attack_in_progress': bool(is_attack_in_progress),
        'total_attacks_detected': int(len(attack_log)),
        'latest_alerts': alerts[-10:] if alerts else []
    })


@app.route('/api/agents', methods=['GET'])
def get_agents():
    """Get all agents and their status"""
    agent_status = {}
    for agent_id, agent in agents.items():
        last_data = agent.history[-1] if agent.history else None
        if last_data:
            # Convert numpy types to Python types for JSON serialization
            last_data_serializable = {
                'ai_score': float(last_data.get('ai_score', 0)),
                'timestamp': float(last_data.get('timestamp', 0)),
                'attack_intensity': float(last_data.get('attack_intensity', 0)),
                'cpu_usage': float(last_data.get('cpu_usage', 0)),
                'branches_mispredicted': int(last_data.get('branches_mispredicted', 0)),
                'cache_misses': int(last_data.get('cache_misses', 0)),
                'memory_percent': float(last_data.get('memory_percent', 0))
            }
        else:
            last_data_serializable = None
        
        agent_status[agent_id] = {
            'running': bool(agent.running),
            'attack_mode': bool(agent.attack_mode),
            'last_data': last_data_serializable,
            'data_count': int(len(agent.history))
        }
    return jsonify(agent_status)


@app.route('/api/history/<agent_id>', methods=['GET'])
def get_agent_history(agent_id):
    """Get history for a specific agent"""
    if agent_id in agent_data:
        # Convert to JSON-serializable format
        history = []
        for item in agent_data[agent_id]:
            serializable_item = {}
            for key, value in item.items():
                if isinstance(value, (int, float, str, bool)):
                    serializable_item[key] = value
                elif isinstance(value, list):
                    serializable_item[key] = [float(v) if isinstance(v, (int, float)) else v for v in value]
                else:
                    serializable_item[key] = str(value)
            history.append(serializable_item)
        return jsonify(history)
    return jsonify([])


@app.route('/api/attack_log', methods=['GET'])
def get_attack_log():
    """Get attack log"""
    # Convert to JSON-serializable format
    log = []
    for item in attack_log[-50:]:
        log.append({
            'agent_id': str(item.get('agent_id', '')),
            'timestamp': float(item.get('timestamp', 0)),
            'severity': str(item.get('severity', 'LOW')),
            'score': float(item.get('score', 0))
        })
    return jsonify(log)


@app.route('/api/start_attack', methods=['POST'])
def start_attack():
    """Start attack simulation on an agent"""
    data = request.json
    agent_id = data.get('agent_id')
    intensity = data.get('intensity', 0.7)
    
    if agent_id and agent_id in agents:
        agents[agent_id].trigger_attack(float(intensity))
        return jsonify({'status': 'success', 'agent': agent_id, 'intensity': float(intensity)})
    
    return jsonify({'error': 'Agent not found'}), 404


@app.route('/api/stop_attack', methods=['POST'])
def stop_attack():
    """Stop attack simulation"""
    data = request.json
    agent_id = data.get('agent_id')
    
    if agent_id and agent_id in agents:
        agents[agent_id].stop_attack()
        return jsonify({'status': 'success', 'agent': agent_id})
    
    return jsonify({'error': 'Agent not found'}), 404


# ============================================================
# PART 5: DASHBOARD (Frontend)
# ============================================================

DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>🕵️ Spectre Network Detector</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0a0e27;
            color: white;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #1a1a3e, #2d1b69);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            border: 1px solid #3d3d8e;
        }
        .header h1 {
            font-size: 32px;
            background: linear-gradient(90deg, #00d2ff, #7a5cff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header p {
            color: #8a8ab5;
            margin-top: 8px;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .card {
            background: #151542;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #2a2a6e;
        }
        .card h3 {
            font-size: 14px;
            text-transform: uppercase;
            color: #8a8ab5;
            margin-bottom: 10px;
        }
        .card .value {
            font-size: 32px;
            font-weight: bold;
        }
        .card .value.green { color: #00d2ff; }
        .card .value.red { color: #ff4757; }
        .card .value.yellow { color: #ffa502; }
        .card .value.white { color: white; }
        
        .chart-container {
            background: #151542;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #2a2a6e;
            margin-bottom: 20px;
        }
        
        .alerts-container {
            background: #151542;
            padding: 20px;
            border-radius: 12px;
            border: 1px solid #2a2a6e;
            margin-bottom: 20px;
            max-height: 300px;
            overflow-y: auto;
        }
        .alert-item {
            padding: 10px;
            margin: 5px 0;
            border-radius: 5px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .alert-item.critical { background: #4a0a0a; border-left: 4px solid #ff4757; }
        .alert-item.high { background: #3a1a0a; border-left: 4px solid #ff6b35; }
        .alert-item.medium { background: #2a2a0a; border-left: 4px solid #ffa502; }
        .alert-item.low { background: #0a2a0a; border-left: 4px solid #00d2ff; }
        
        .badge {
            display: inline-block;
            padding: 3px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
        }
        .badge.critical { background: #ff4757; color: white; }
        .badge.high { background: #ff6b35; color: white; }
        .badge.medium { background: #ffa502; color: black; }
        .badge.low { background: #00d2ff; color: black; }
        .badge.normal { background: #2d2d6e; color: white; }
        
        .controls {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-bottom: 20px;
        }
        .controls button {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .controls button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }
        .btn-attack { background: #ff4757; color: white; }
        .btn-stop { background: #2d2d6e; color: white; }
        .btn-reset { background: #ffa502; color: black; }
        
        .agent-list {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 10px;
        }
        .agent-item {
            background: #0a0e27;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #2a2a6e;
            text-align: center;
        }
        .agent-item .name { font-weight: bold; }
        .agent-item .status {
            font-size: 12px;
            padding: 3px 10px;
            border-radius: 12px;
            display: inline-block;
            margin-top: 5px;
        }
        .agent-item .status.active { background: #00d2ff; color: black; }
        .agent-item .status.attacking { background: #ff4757; color: white; animation: pulse 0.5s infinite; }
        .agent-item .status.idle { background: #2d2d6e; color: white; }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .table-container {
            overflow-x: auto;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #2a2a6e;
        }
        th {
            color: #8a8ab5;
            text-transform: uppercase;
            font-size: 12px;
        }
        
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0a0e27;
        }
        ::-webkit-scrollbar-thumb {
            background: #2d2d6e;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🕵️ Spectre Network Detector</h1>
        <p>AI-powered detection of Spectre attacks across your network</p>
    </div>

    <div class="grid">
        <div class="card">
            <h3>🖥️ Active Agents</h3>
            <div class="value green" id="agentCount">0</div>
        </div>
        <div class="card">
            <h3>🚨 Alerts</h3>
            <div class="value red" id="alertCount">0</div>
        </div>
        <div class="card">
            <h3>🔥 Attacks Detected</h3>
            <div class="value yellow" id="attackCount">0</div>
        </div>
        <div class="card">
            <h3>📊 Status</h3>
            <div class="value white" id="statusText">🟢 Normal</div>
        </div>
    </div>

    <div class="controls">
        <button class="btn-attack" onclick="triggerAttack('agent-1')">⚡ Attack Agent 1</button>
        <button class="btn-attack" onclick="triggerAttack('agent-2')">⚡ Attack Agent 2</button>
        <button class="btn-attack" onclick="triggerAttack('agent-3')">⚡ Attack Agent 3</button>
        <button class="btn-stop" onclick="stopAttack('agent-1')">🛑 Stop Agent 1</button>
        <button class="btn-stop" onclick="stopAttack('agent-2')">🛑 Stop Agent 2</button>
        <button class="btn-stop" onclick="stopAttack('agent-3')">🛑 Stop Agent 3</button>
        <button class="btn-reset" onclick="resetAlerts()">🔄 Reset Alerts</button>
    </div>

    <div class="chart-container">
        <h3 style="margin-bottom: 15px;">📈 Real-time Agent Scores</h3>
        <div style="position: relative; height: 320px;">
            <canvas id="scoreChart"></canvas>
        </div>
    </div>

    <div class="grid">
        <div class="card">
            <h3>🤖 Agents</h3>
            <div class="agent-list" id="agentList"></div>
        </div>
        <div class="card">
            <h3>⚡ Attack Log</h3>
            <div class="table-container">
                <table>
                    <thead>
                        <tr><th>Time</th><th>Agent</th><th>Score</th><th>Severity</th></tr>
                    </thead>
                    <tbody id="attackLogBody"></tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="alerts-container">
        <h3 style="margin-bottom: 15px;">🚨 Recent Alerts</h3>
        <div id="alertsList">No alerts yet</div>
    </div>

    <script>
        let chart = null;
        
        async function fetchData() {
            try {
                // Get status
                const statusRes = await fetch('/api/status');
                const status = await statusRes.json();
                
                document.getElementById('agentCount').textContent = status.active_agents || 0;
                document.getElementById('alertCount').textContent = status.alerts || 0;
                document.getElementById('attackCount').textContent = status.total_attacks_detected || 0;
                
                const statusText = document.getElementById('statusText');
                if (status.attack_in_progress) {
                    statusText.textContent = '🔴 ATTACK IN PROGRESS';
                    statusText.style.color = '#ff4757';
                } else {
                    statusText.textContent = '🟢 Normal';
                    statusText.style.color = '#00d2ff';
                }
                
                // Get agents
                const agentsRes = await fetch('/api/agents');
                const agents = await agentsRes.json();
                
                const agentList = document.getElementById('agentList');
                agentList.innerHTML = '';
                for (const [id, data] of Object.entries(agents)) {
                    const div = document.createElement('div');
                    div.className = 'agent-item';
                    const statusClass = data.attack_mode ? 'attacking' : (data.running ? 'active' : 'idle');
                    const statusText = data.attack_mode ? '⚡ ATTACKING' : (data.running ? '✅ Running' : '⏸️ Idle');
                    div.innerHTML = `
                        <div class="name">${id}</div>
                        <div class="status ${statusClass}">${statusText}</div>
                        <div style="font-size:12px;color:#8a8ab5;margin-top:5px;">
                            Score: ${data.last_data?.ai_score?.toFixed(2) || 'N/A'}
                        </div>
                    `;
                    agentList.appendChild(div);
                }
                
                // Get attack log
                const logRes = await fetch('/api/attack_log');
                const log = await logRes.json();
                const logBody = document.getElementById('attackLogBody');
                logBody.innerHTML = '';
                log.slice(-10).reverse().forEach(item => {
                    const row = document.createElement('tr');
                    const time = new Date(item.timestamp * 1000).toLocaleTimeString();
                    row.innerHTML = `
                        <td>${time}</td>
                        <td>${item.agent_id}</td>
                        <td>${item.score.toFixed(2)}</td>
                        <td><span class="badge ${item.severity.toLowerCase()}">${item.severity}</span></td>
                    `;
                    logBody.appendChild(row);
                });
                
                // Update alerts
                const alertsDiv = document.getElementById('alertsList');
                if (status.latest_alerts && status.latest_alerts.length > 0) {
                    alertsDiv.innerHTML = status.latest_alerts.slice().reverse().map(alert => `
                        <div class="alert-item ${alert.severity.toLowerCase()}">
                            <span>${alert.agent_id}</span>
                            <span>Score: ${alert.score.toFixed(2)}</span>
                            <span class="badge ${alert.severity.toLowerCase()}">${alert.severity}</span>
                            <span style="font-size:12px;color:#8a8ab5;">
                                ${new Date(alert.timestamp * 1000).toLocaleTimeString()}
                            </span>
                        </div>
                    `).join('');
                } else {
                    alertsDiv.innerHTML = 'No alerts detected ✅';
                }
                
            } catch (e) {
                console.error('Error:', e);
            }
        }
        
        async function updateGraph() {
            try {
                const res = await fetch('/api/agents');
                const agents = await res.json();
                
                const datasets = [];
                const colors = ['#00d2ff', '#ff4757', '#ffa502', '#7a5cff', '#2ed573'];
                let colorIdx = 0;
                
                for (const [id, data] of Object.entries(agents)) {
                    if (data.data_count > 0) {
                        // Get history for this agent
                        const histRes = await fetch(`/api/history/${id}`);
                        const history = await histRes.json();
                        
                        const scores = history.slice(-30).map(d => d.ai_score || 0);
                        
                        datasets.push({
                            label: id,
                            data: scores,
                            borderColor: colors[colorIdx % colors.length],
                            fill: false,
                            tension: 0.4,
                            pointRadius: 3,
                        });
                        colorIdx++;
                    }
                }
                
                if (datasets.length === 0) {
                    // Add dummy data if no agents
                    datasets.push({
                        label: 'No Data',
                        data: [0],
                        borderColor: '#2d2d6e',
                        fill: false,
                    });
                }
                
                const labels = datasets[0].data.map((_, i) => `-${datasets[0].data.length - i}s`);
                
                if (!chart) {
                    // First call only: create the chart once
                    const ctx = document.getElementById('scoreChart').getContext('2d');
                    chart = new Chart(ctx, {
                        type: 'line',
                        data: {
                            labels: labels,
                            datasets: datasets,
                        },
                        options: {
                            responsive: true,
                            animation: { duration: 0 },
                            plugins: {
                                legend: {
                                    labels: { color: 'white' }
                                }
                            },
                            scales: {
                                y: {
                                    min: 0,
                                    max: 1,
                                    ticks: { color: 'white' }
                                },
                                x: {
                                    ticks: { color: 'white' }
                                }
                            },
                            maintainAspectRatio: false,
                        }
                    });
                } else {
                    // Every later call: update the existing chart's data
                    // in place instead of destroying/recreating it - this
                    // is what stopped the canvas (and the whole page below
                    // it) from collapsing and re-expanding every 2 seconds.
                    chart.data.labels = labels;
                    chart.data.datasets = datasets;
                    chart.update();
                }
                
            } catch (e) {
                console.error('Graph error:', e);
            }
        }
        
        async function triggerAttack(agentId) {
            try {
                await fetch('/api/start_attack', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ agent_id: agentId, intensity: 0.8 })
                });
                alert(`⚡ Attack triggered on ${agentId}!`);
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
        
        async function stopAttack(agentId) {
            try {
                await fetch('/api/stop_attack', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ agent_id: agentId })
                });
                alert(`🛑 Attack stopped on ${agentId}`);
            } catch (e) {
                alert('Error: ' + e.message);
            }
        }
        
        function resetAlerts() {
            if (confirm('Reset all alerts?')) {
                location.reload();
            }
        }
        
        // Update every 2 seconds
        setInterval(() => {
            fetchData();
            updateGraph();
        }, 2000);
        
        // Initial load
        setTimeout(() => {
            fetchData();
            updateGraph();
        }, 500);
    </script>
</body>
</html>
"""


@app.route('/')
@app.route('/dashboard')
def dashboard():
    """Serve the dashboard"""
    return render_template_string(DASHBOARD_HTML)


# ============================================================
# PART 6: MAIN - RUN EVERYTHING
# ============================================================

def create_agents(server_url: str, count: int = 3):
    """Create and start agents"""
    agents_dict = {}
    for i in range(1, count + 1):
        agent_id = f"agent-{i}"
        agent = DetectionAgent(agent_id, server_url, interval=2)
        agent.start()
        agents_dict[agent_id] = agent
        time.sleep(0.5)  # Stagger startup
    return agents_dict


def simulate_attack_sequence(agents_dict):
    """Simulate a coordinated attack sequence"""
    time.sleep(10)  # Wait for initial setup
    
    agent_list = list(agents_dict.values())
    if len(agent_list) < 3:
        return
    
    print("\n" + "="*60)
    print("🎯 SIMULATING SPECTRE ATTACK SEQUENCE")
    print("="*60)
    
    # Phase 1: Single agent attack
    print("\n📌 Phase 1: Single agent attack")
    print("   Agent-1 starting Spectre attack...")
    agent_list[0].trigger_attack(0.6)
    time.sleep(8)
    
    # Phase 2: Second agent joins
    print("\n📌 Phase 2: Second agent joins")
    print("   Agent-2 starting Spectre attack...")
    agent_list[1].trigger_attack(0.7)
    time.sleep(8)
    
    # Phase 3: Coordinated attack (all agents)
    print("\n📌 Phase 3: Coordinated attack!")
    print("   Agent-3 starting Spectre attack...")
    print("   🔥 ALL AGENTS COMPROMISED!")
    agent_list[2].trigger_attack(0.9)
    time.sleep(10)
    
    # Phase 4: Mitigation
    print("\n📌 Phase 4: Mitigation in progress...")
    print("   Stopping attacks...")
    for agent in agent_list:
        agent.stop_attack()
    time.sleep(2)
    
    print("\n✅ Attack sequence complete!")
    print("   Check the dashboard for results!")
    print("="*60 + "\n")


def main():
    """Main entry point"""
    print("="*70)
    print("🕵️  SPECTRE NETWORK DETECTOR")
    print("="*70)
    print("📌 AI-Powered Detection of Spectre Attacks")
    print("📌 Single Machine Mode - 3 Simulated Agents")
    print("="*70)
    
    # Server URL
    server_url = "http://localhost:5000"
    
    # Create agents
    print("\n🚀 Starting agents...")
    global agents
    agents = create_agents(server_url, count=3)
    print(f"✅ {len(agents)} agents started")
    
    # Start attack simulation in background
    #def attack_simulation():
        #time.sleep(5)  # Wait for agents to initialize
        #simulate_attack_sequence(agents)
    
   #threading.Thread(target=attack_simulation, daemon=True).start()
    
    # Start Flask server
    print("\n🚀 Starting web server...")
    print("📊 Dashboard: http://localhost:5000")
    print("="*70)
    print("\n⏳ Attack simulation will start in 10 seconds...")
    print("🔄 Watch the dashboard for real-time detection!\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


if __name__ == "__main__":
    main()
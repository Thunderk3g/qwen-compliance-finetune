import { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Terminal, Activity, Play, Square, Layers, Cpu } from 'lucide-react';

const API_BASE = 'http://localhost:5000/api';

export default function App() {
  const [isRunning, setIsRunning] = useState(false);
  const [metrics, setMetrics] = useState<any[]>([]);
  const [terminalLog, setTerminalLog] = useState("");
  const terminalEndRef = useRef<HTMLDivElement>(null);

  // Poll status, metrics and logs
  useEffect(() => {
    const interval = setInterval(() => {
      axios.get(`${API_BASE}/status`).then(res => setIsRunning(res.data.is_running)).catch(() => setIsRunning(false));
      axios.get(`${API_BASE}/metrics`).then(res => setMetrics(res.data)).catch(() => {});
      axios.get(`${API_BASE}/terminal`).then(res => setTerminalLog(res.data.log)).catch(() => {});
    }, 2000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    terminalEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [terminalLog]);

  const handleStart = async () => {
    try { await axios.post(`${API_BASE}/start`); } catch (e) { console.error(e); }
  };

  const handleStop = async () => {
    try { await axios.post(`${API_BASE}/stop`); } catch (e) { console.error(e); }
  };

  const currentLoss = metrics.length > 0 ? metrics[metrics.length - 1].loss : 0;
  const currentEpoch = metrics.length > 0 ? metrics[metrics.length - 1].epoch : 0;

  return (
    <div className="min-h-screen bg-slate-950 text-white p-6 font-sans selection:bg-blue-500/30">
      
      {/* Header */}
      <header className="flex justify-between items-center mb-8 border-b border-slate-800 pb-4">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-indigo-400 bg-clip-text text-transparent flex items-center gap-3">
            <Layers className="text-blue-500" />
            Unsloth Qwen-3B Studio
          </h1>
          <p className="text-slate-400 mt-1">Real-time LLM Fine-tuning Dashboard</p>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-slate-900 border border-slate-800">
            <div className={`w-2.5 h-2.5 rounded-full ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-slate-600'}`} />
            <span className="text-sm font-medium text-slate-300">{isRunning ? 'Training Active' : 'Idle'}</span>
          </div>
          
          {isRunning ? (
            <button
              onClick={handleStop}
              className="flex items-center gap-2 bg-red-500/10 text-red-400 hover:bg-red-500/20 border border-red-500/20 px-5 relative overflow-hidden py-2.5 rounded-lg transition-all"
            >
              <Square size={16} fill="currentColor" /> Stop Training
            </button>
          ) : (
            <button
              onClick={handleStart}
              className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white shadow-[0_0_20px_rgba(37,99,235,0.3)] hover:shadow-[0_0_25px_rgba(37,99,235,0.5)] border border-blue-500 px-6 py-2.5 rounded-lg font-medium transition-all"
            >
              <Play size={16} fill="currentColor" /> Start Fine-Tuning
            </button>
          )}
        </div>
      </header>

      {/* Main Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        
        {/* Left Col: Analytics */}
        <div className="lg:col-span-2 flex flex-col gap-6">
          
          {/* Stats Row */}
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-blue-500/5 rounded-full blur-3xl -mr-10 -mt-10" />
              <div className="flex items-center gap-3 text-slate-400 mb-2">
                <Activity size={18} className="text-blue-400" />
                <span className="font-medium text-sm tracking-wide">TRAINING LOSS</span>
              </div>
              <div className="text-4xl font-bold bg-gradient-to-br from-white to-slate-400 bg-clip-text text-transparent">
                {currentLoss ? currentLoss.toFixed(4) : "0.0000"}
              </div>
            </div>

            <div className="bg-slate-900 border border-slate-800 rounded-xl p-5 relative overflow-hidden">
              <div className="absolute top-0 right-0 w-32 h-32 bg-indigo-500/5 rounded-full blur-3xl -mr-10 -mt-10" />
              <div className="flex items-center gap-3 text-slate-400 mb-2">
                <Cpu size={18} className="text-indigo-400" />
                <span className="font-medium text-sm tracking-wide">CURRENT EPOCH</span>
              </div>
              <div className="text-4xl font-bold bg-gradient-to-br from-white to-slate-400 bg-clip-text text-transparent">
                {currentEpoch ? currentEpoch.toFixed(2) : "0.00"}
              </div>
            </div>
          </div>

          {/* Chart */}
          <div className="bg-slate-900 border border-slate-800 rounded-xl p-6 h-[400px]">
            <h3 className="text-lg font-semibold mb-6 flex items-center gap-2">
              <div className="w-1.5 h-6 bg-blue-500 rounded-full" />
              Loss Convergence
            </h3>
            {metrics.length > 0 ? (
              <ResponsiveContainer width="100%" height="85%">
                <LineChart data={metrics}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="step" stroke="#475569" tick={{fill: '#64748b'}} tickLine={false} axisLine={false} />
                  <YAxis stroke="#475569" tick={{fill: '#64748b'}} tickLine={false} axisLine={false} domain={['auto', 'auto']} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#1e293b', borderRadius: '8px', boxShadow: '0 10px 15px -3px rgba(0, 0, 0, 0.5)' }}
                    itemStyle={{ color: '#60a5fa' }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="loss" 
                    stroke="#3b82f6" 
                    strokeWidth={3}
                    dot={{ fill: '#3b82f6', strokeWidth: 2, r: 4 }}
                    activeDot={{ r: 6, fill: '#fff', stroke: '#3b82f6' }}
                    animationDuration={500}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="w-full h-[85%] flex items-center justify-center flex-col text-slate-500">
                <Activity size={48} className="mb-4 opacity-20" />
                <p>Awaiting training data...</p>
              </div>
            )}
          </div>
        </div>

        {/* Right Col: Terminal View */}
        <div className="bg-slate-900 border border-slate-800 rounded-xl flex flex-col h-[600px] lg:h-auto overflow-hidden">
          <div className="bg-slate-950 px-4 py-3 border-b border-slate-800 flex items-center justify-between">
            <h3 className="font-semibold text-slate-300 flex items-center gap-2 text-sm uppercase tracking-wider">
              <Terminal size={15} /> std_out
            </h3>
            <div className="flex gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-red-500/20 border border-red-500/50" />
              <div className="w-2.5 h-2.5 rounded-full bg-yellow-500/20 border border-yellow-500/50" />
              <div className="w-2.5 h-2.5 rounded-full bg-green-500/20 border border-green-500/50" />
            </div>
          </div>
          
          <div className="flex-1 p-4 font-mono text-xs text-slate-300 overflow-y-auto w-full break-words leading-relaxed">
            {terminalLog ? (
              <pre className="whitespace-pre-wrap">{terminalLog}</pre>
            ) : (
              <p className="text-slate-600 italic">Waiting for process attachment...</p>
            )}
            <div ref={terminalEndRef} />
          </div>
        </div>
      </div>
    </div>
  );
}

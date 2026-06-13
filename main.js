const { app, BrowserWindow } = require('electron')
const path = require('path')
const { spawn } = require('child_process')
const net = require('net')

require('dotenv').config({ path: path.join(__dirname, '.env') })

const pyProcs = []

function portBusy(port) {
  return new Promise((resolve) => {
    const socket = new net.Socket()
    socket.setTimeout(800)
    socket
      .once('connect', () => { socket.destroy(); resolve(true) })
      .once('timeout', () => { socket.destroy(); resolve(false) })
      .once('error', () => resolve(false))
      .connect(port, '127.0.0.1')
  })
}

function findPython() {
  const candidates = [
    path.join(__dirname, '.venv', 'bin', 'python3'),
    path.join(__dirname, '.venv', 'bin', 'python'),
    'python3',
    'python',
  ]
  const fs = require('fs')
  for (const p of candidates) {
    try {
      if (p.includes('/') && fs.existsSync(p)) return p
    } catch (_) {}
    if (!p.includes('/')) return p
  }
  return 'python3'
}

function startPy(script, label) {
  const python = findPython()
  const proc = spawn(python, [path.join(__dirname, 'python', script)], {
    cwd: path.join(__dirname, 'python'),
    env: { ...process.env },
    stdio: ['ignore', 'pipe', 'pipe'],
  })
  proc.stdout.on('data', (d) => console.log(`[${label}]`, d.toString().trim()))
  proc.stderr.on('data', (d) => console.error(`[${label}]`, d.toString().trim()))
  proc.on('exit', (code) => console.log(`[${label}] stopped (${code})`))
  pyProcs.push(proc)
}

function hasEnv(name) {
  return Boolean((process.env[name] || '').trim())
}

app.whenReady().then(() => {
  const calPort = parseInt(process.env.GOOGLE_CALENDAR_PORT || '8768', 10)
  const fcPort = parseInt(process.env.FIRECRAWL_PORT || '8769', 10)

  Promise.all([
    portBusy(8765),
    portBusy(8766),
    portBusy(calPort),
    portBusy(fcPort),
  ]).then(([cvUp, sttUp, calUp, fcUp]) => {
    if (!cvUp) startPy('cv_engine.py', 'cv')
    else console.log('[cv] already running on :8765')

    if (!sttUp) startPy('stt_service.py', 'stt')
    else console.log('[stt] already running on :8766')

    if (hasEnv('GOOGLE_CALENDAR_CLIENT_ID') && hasEnv('GOOGLE_CALENDAR_CLIENT_SECRET')) {
      if (!calUp) startPy('calendar_service.py', 'calendar')
      else console.log(`[calendar] already running on :${calPort}`)
    }

    if (hasEnv('FIRECRAWL_API_KEY')) {
      if (!fcUp) startPy('firecrawl_service.py', 'firecrawl')
      else console.log(`[firecrawl] already running on :${fcPort}`)
    }
  })

  const win = new BrowserWindow({
    width: 1280,
    height: 720,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
    },
  })
  win.loadFile('index.html')
})

app.on('before-quit', () => {
  pyProcs.forEach((p) => {
    try { p.kill() } catch (_) {}
  })
})
